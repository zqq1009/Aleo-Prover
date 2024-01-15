use std::{sync::Arc, time::Duration};

use aleo_stratum::{
    codec::{ResponseParams, StratumCodec},
    message::StratumMessage,
};
use futures_util::sink::SinkExt;
use json_rpc_types::Id;
use snarkvm::{console::types::Address, console::network::Testnet3};
use tokio::{
    net::TcpStream,
    sync::{
        mpsc,
        mpsc::{Receiver, Sender},
        Mutex,
    },
    task,
    time::{sleep, timeout},
};
use tokio_stream::StreamExt;
use tokio_util::codec::Framed;
use tracing::{debug, error, info, warn};

use crate::prover::ProverEvent;

pub struct Client {
    pub address: Address<Testnet3>,             // 客户端地址
    server: String,                             // 服务器地址
    sender: Arc<Sender<StratumMessage>>,        // 发送消息的通道
    receiver: Arc<Mutex<Receiver<StratumMessage>>>,   // 接收消息的通道
}

impl Client {
    pub fn init(address: Address<Testnet3>, server: String) -> Arc<Self> {
        let (sender, receiver) = mpsc::channel(1024);   // 创建消息通道
        Arc::new(Self {
            address,
            server,
            sender: Arc::new(sender),
            receiver: Arc::new(Mutex::new(receiver)),
        })
    }

    pub fn sender(&self) -> Arc<Sender<StratumMessage>> {
        self.sender.clone()   // 返回发送消息的通道
    }

    pub fn receiver(&self) -> Arc<Mutex<Receiver<StratumMessage>>> {
        self.receiver.clone()   // 返回接收消息的通道
    }
}

pub fn start(prover_sender: Arc<Sender<ProverEvent>>, client: Arc<Client>) {
    task::spawn(async move {
        let receiver = client.receiver();   // 获取接收消息的通道
        let mut id = 1;                     // 消息ID
        loop {
            info!("Connecting to server...");
            match timeout(Duration::from_secs(5), TcpStream::connect(&client.server)).await {   // 连接服务器
                Ok(socket) => match socket {
                    Ok(socket) => {
                        info!("Connected to {}", client.server);
                        let mut framed = Framed::new(socket, StratumCodec::default());   // 创建帧化套接字
                        let mut pool_address: Option<String> = None;   // 矿池地址
                        let handshake = StratumMessage::Subscribe(
                            Id::Num(id),   // 订阅消息的ID
                            format!("HarukaProver/{}", env!("CARGO_PKG_VERSION")),   // 客户端名称
                            "AleoStratum/2.0.0".to_string(),   // 使用的协议版本
                            None,
                        );
                        id += 1;
                        if let Err(e) = framed.send(handshake).await {   // 发送握手消息给服务器
                        error!("Error sending handshake: {}", e);
                        } else {
                            debug!("Sent handshake");
                        }
                        match framed.next().await {   // 接收服务器的响应
                            None => {
                                error!("Unexpected end of stream");
                                sleep(Duration::from_secs(5)).await;
                                continue;
                            }
                            Some(Ok(message)) => match message {
                                StratumMessage::Response(_, params, _) => {
                                    match params {
                                        Some(ResponseParams::Array(array)) => {
                                            if let Some(address) = array.get(2) {   // 获取矿池地址
                                                if let Some(address) = address.downcast_ref::<String>() {
                                                    pool_address = Some(address.clone());
                                                } else {
                                                    error!("Invalid type for address");
                                                    sleep(Duration::from_secs(5)).await;
                                                    continue;
                                                }
                                            } else {
                                                error!("Invalid handshake response");
                                                sleep(Duration::from_secs(5)).await;
                                                continue;
                                            }
                                        }
                                        None => {
                                            error!("No handshake response");
                                            sleep(Duration::from_secs(5)).await;
                                            continue;
                                        }
                                        _ => {
                                            error!("Invalid handshake response");
                                            sleep(Duration::from_secs(5)).await;
                                            continue;
                                        }
                                    }
                                    info!("Handshake successful");
                                }
                                _ => {
                                    error!("Unexpected message: {:?}", message.name());
                                }
                            },
                            Some(Err(e)) => {
                                error!("Error receiving handshake: {}", e);
                                sleep(Duration::from_secs(5)).await;
                                continue;
                            }
                        }
                        // 发送授权消息给服务器
                        let authorization =
                            StratumMessage::Authorize(Id::Num(id), client.address.to_string(), "".to_string());
                        id += 1;
                        if let Err(e) = framed.send(authorization).await {
                            error!("Error sending authorization: {}", e);
                        } else {
                            debug!("Sent authorization");
                        }
                        // 接收服务器的响应
                        match framed.next().await {
                            None => {
                                error!("Unexpected end of stream");
                                sleep(Duration::from_secs(5)).await;
                                continue;
                            }
                            Some(Ok(message)) => match message {
                                StratumMessage::Response(_, _, _) => {
                                    info!("Authorization successful");
                                }
                                _ => {
                                    error!("Unexpected message: {:?}", message.name());
                                }
                            },
                            Some(Err(e)) => {
                                error!("Error receiving authorization: {}", e);
                                sleep(Duration::from_secs(5)).await;
                                continue;
                            }
                        }
                        // 获取接收消息的互斥锁
                        let receiver = &mut *receiver.lock().await;
                        loop {
                            tokio::select! {
                                Some(message) = receiver.recv() => {
                                    let name = message.name();
                                    debug!("Sending {} to server", name);
                                    if let Err(e) = framed.send(message).await {   // 发送消息给服务器
                                        error!("Error sending {}: {:?}", name, e);
                                    }
                                }
                                result = framed.next() => match result {
                                    Some(Ok(message)) => {
                                        debug!("Received {} from server", message.name());
                                        match message {
                                            StratumMessage::Response(_, result, error) => {
                                                match result {
                                                    Some(params) => {
                                                        match params {
                                                            ResponseParams::Bool(result) => {
                                                                if result {
                                                                    // 将结果发送给证明者
                                                                    if let Err(e) = prover_sender.send(ProverEvent::Result(result, None)).await {
                                                                        error!("Error sending share result to prover: {}", e);
                                                                    } else {
                                                                        debug!("Sent share result to prover");
                                                                    }
                                                                } else {
                                                                    error!("Unexpected result: {}", result);
                                                                }
                                                            }
                                                            _ => {
                                                                error!("Unexpected response params");
                                                            }
                                                        }
                                                    }
                                                    None => {
                                                        let error = error.unwrap();
                                                        // 将错误信息发送给证明者
                                                        if let Err(e) = prover_sender.send(ProverEvent::Result(false, Some(error.message.to_string()))).await {
                                                            error!("Error sending share result to prover: {}", e);
                                                        } else {
                                                            debug!("Sent share result to prover");
                                                        }
                                                    }
                                                }
                                            }
                                            StratumMessage::Notify(job_id, epoch_challenge, address, _) => {
                                                let job_id_bytes = hex::decode(job_id).expect("Failed to decode job_id");
                                                if job_id_bytes.len() != 4 {
                                                    error!("Unexpected job_id length: {}", job_id_bytes.len());
                                                    continue;
                                                }
                                                let epoch = u32::from_le_bytes(job_id_bytes[0..4].try_into().unwrap());

                                                // 将新任务发送给证明者
                                                if let Err(e) = prover_sender.send(ProverEvent::NewWork(epoch, epoch_challenge, address.unwrap_or_else(|| pool_address.clone().expect("No pool address defined")))).await {
                                                    error!("Error sending work to prover: {}", e);
                                                } else {
                                                    debug!("Sent work to prover");
                                                }
                                            }
                                            StratumMessage::SetTarget(difficulty_target) => {
                                                // 将新的难度目标发送给证明者
                                                if let Err(e) = prover_sender.send(ProverEvent::NewTarget(difficulty_target)).await {
                                                    error!("Error sending difficulty target to prover: {}", e);
                                                } else {
                                                    debug!("Sent difficulty target to prover");
                                                }
                                            }
                                            _ => {
                                                debug!("Unhandled message: {}", message.name());
                                            }
                                        }
                                    }
                                    Some(Err(e)) => {
                                        warn!("Failed to read the message: {:?}", e);
                                    }
                                    None => {
                                        error!("Disconnected from server");
                                        sleep(Duration::from_secs(5)).await;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to connect to operator: {}", e);
                        sleep(Duration::from_secs(5)).await;
                    }
                },
                Err(_) => {
                    error!("Failed to connect to operator: Timed out");
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    });
}

