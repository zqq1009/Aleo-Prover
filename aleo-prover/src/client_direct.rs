use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use futures_util::sink::SinkExt;
use snarkos_node_executor::{NodeType, Status};
use snarkos_node_messages::{
    ChallengeRequest,
    ChallengeResponse,
    Data,
    MessageCodec,
    Ping,
    Pong,
    PuzzleRequest,
    PuzzleResponse,
};
use snarkvm::{
    console::account::address::Address,
    prelude::{Block, FromBytes, Network, Testnet3},
};
use tokio::{
    net::{TcpListener, TcpStream},
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

type Message = snarkos_node_messages::Message<Testnet3>; // 定义了一个类型别名，用于表示消息类型

pub struct DirectClient {
    pub address: Address<Testnet3>, // 地址
    server: String, // 服务器地址
    sender: Arc<Sender<Message>>, // 发送消息的通道发送端
    receiver: Arc<Mutex<Receiver<Message>>>, // 接收消息的通道接收端
}

impl DirectClient {
    pub fn init(address: Address<Testnet3>, server: String) -> Arc<Self> {
        let (sender, receiver) = mpsc::channel(1024); // 创建一个带有缓冲区的多生产者单消费者通道
        Arc::new(Self {
            address,
            server,
            sender: Arc::new(sender),
            receiver: Arc::new(Mutex::new(receiver)),
        })
    }

    pub fn sender(&self) -> Arc<Sender<Message>> {
        self.sender.clone()
    }

    pub fn receiver(&self) -> Arc<Mutex<Receiver<Message>>> {
        self.receiver.clone()
    }
}

pub fn start(prover_sender: Arc<Sender<ProverEvent>>, client: Arc<DirectClient>) {
    task::spawn(async move {
        let receiver = client.receiver(); // 获取接收端
        let genesis_header = *Block::<Testnet3>::from_bytes_le(Testnet3::genesis_bytes()) // 获取创世区块头
            .unwrap()
            .header();
        let connected = Arc::new(AtomicBool::new(false)); // 表示是否连接到服务器的原子布尔值
        let client_sender = client.sender(); // 获取发送端

        let connected_req = connected.clone(); // 克隆一份连接状态
        task::spawn(async move {
            loop {
                sleep(Duration::from_secs(Testnet3::ANCHOR_TIME as u64)).await; // 等待一段时间后执行下面的代码
                if connected_req.load(Ordering::SeqCst) { // 如果已连接到服务器
                    if let Err(e) = client_sender.send(Message::PuzzleRequest(PuzzleRequest {})).await { // 发送谜题请求消息
                    error!("Failed to send puzzle request: {}", e);
                    }
                }
            }
        });

        // incoming socket
        task::spawn(async move {
            let (_, listener) = match TcpListener::bind("0.0.0.0:4140").await { // 在指定端口上进行监听
                Ok(listener) => {
                    let local_ip = listener.local_addr().expect("Could not get local ip"); // 获取本地 IP 地址
                    info!("Listening on {}", local_ip); // 输出监听地址信息
                    (local_ip, listener)
                }
                Err(e) => {
                    panic!("Unable to listen on port 4140: {:?}", e);
                }
            };
            loop {
                match listener.accept().await { // 接收连接请求
                    Ok((stream, peer_addr)) => {
                        info!("New connection from: {}", peer_addr); // 输出新连接的地址信息
                        drop(stream); // 关闭连接
                    }
                    Err(e) => {
                        error!("Error accepting connection: {:?}", e);
                    }
                }
            }
        });

        debug!("Created coinbase puzzle request task");
        loop {
            info!("Connecting to server...");
            match timeout(Duration::from_secs(5), TcpStream::connect(&client.server)).await { // 尝试连接到服务器
                Ok(socket) => match socket {
                    Ok(socket) => {
                        info!("Connected to {}", client.server); // 输出连接成功的服务器地址
                        let mut framed = Framed::new(socket, MessageCodec::default()); // 创建帧化流
                        let challenge_request = Message::ChallengeRequest(ChallengeRequest { // 创建挑战请求消息
                            version: Message::VERSION,
                            fork_depth: 4096,
                            node_type: NodeType::Prover,
                            status: Status::Ready,
                            listener_port: 4140,
                        });
                        if let Err(e) = framed.send(challenge_request).await { // 发送挑战请求消息
                        error!("Error sending challenge request: {}", e);
                        } else {
                            debug!("Sent challenge request");
                        }
                        let receiver = &mut *receiver.lock().await;
                        loop {
                            tokio::select! {
                                Some(message) = receiver.recv() => { // 接收来自通道的消息
                                    let m = message.clone();
                                    let name = m.name(); // 获取消息的名称
                                    debug!("Sending {} to beacon", name);
                                    if let Err(e) = framed.send(message).await { // 发送消息到服务器
                                        error!("Error sending {}: {:?}", name, e);
                                    }
                                }
                                result = framed.next() => match result {
                                    Some(Ok(message)) => {
                                        debug!("Received {} from beacon", message.name());
                                        match message {
                                            Message::ChallengeRequest(ChallengeRequest {
                                                version,
                                                fork_depth: _,
                                                node_type,
                                                ..
                                            }) => {
                                                if version < Message::VERSION {
                                                    error!("Peer is running an older version of the protocol");
                                                    sleep(Duration::from_secs(5)).await;
                                                    break;
                                                }
                                                if node_type != NodeType::Beacon && node_type != NodeType::Validator {
                                                    error!("Peer is not a beacon or validator");
                                                    sleep(Duration::from_secs(5)).await;
                                                    break;
                                                }
                                                let response = Message::ChallengeResponse(ChallengeResponse {
                                                    header: Data::Object(genesis_header),
                                                });
                                                if let Err(e) = framed.send(response).await { // 发送挑战响应消息
                                                    error!("Error sending challenge response: {:?}", e);
                                                } else {
                                                    debug!("Sent challenge response");
                                                }
                                            }
                                            Message::ChallengeResponse(message) => {
                                                let block_header = match message.header.deserialize().await { // 反序列化区块头
                                                    Ok(block_header) => block_header,
                                                    Err(error) => {
                                                        error!("Error deserializing block header: {:?}", error);
                                                        sleep(Duration::from_secs(5)).await;
                                                        break;
                                                    }
                                                };
                                                match block_header == genesis_header {
                                                    true => {
                                                        let message = Message::Ping(Ping { // 创建 Ping 消息
                                                            version: Message::VERSION,
                                                            fork_depth: 4096,
                                                            node_type: NodeType::Prover,
                                                            status: Status::Ready,
                                                        });
                                                        if let Err(e) = framed.send(message).await { // 发送 Ping 消息
                                                            error!("Error sending ping: {:?}", e);
                                                        } else {
                                                            debug!("Sent ping");
                                                        }
                                                    }
                                                    false => {
                                                        error!("Peer has a different genesis block");
                                                        sleep(Duration::from_secs(5)).await;
                                                        break;
                                                    }
                                                }
                                            }
                                            Message::Ping(_) => {
                                                let pong = Message::Pong(Pong { is_fork: None });
                                                if let Err(e) = framed.send(pong).await { // 发送 Pong 消息
                                                    error!("Error sending pong: {:?}", e);
                                                } else {
                                                    debug!("Sent pong");
                                                }
                                            }
                                            Message::Pong(_) => {
                                                connected.store(true, Ordering::SeqCst); // 更新连接状态
                                                if let Err(e) = client.sender().send(Message::PuzzleRequest(PuzzleRequest {})).await { // 发送谜题请求消息
                                                    error!("Failed to send puzzle request: {}", e);
                                                }
                                            }
                                            Message::PuzzleResponse(PuzzleResponse {
                                                epoch_challenge, block
                                            }) => {
                                                let block = match block.deserialize().await { // 反序列化区块
                                                    Ok(block) => block,
                                                    Err(error) => {
                                                        error!("Error deserializing block: {:?}", error);
                                                        sleep(Duration::from_secs(5)).await;
                                                        break;
                                                    }
                                                };
                                                // 发送新的目标给 prover
                                                if let Err(e) = prover_sender.send(ProverEvent::NewTarget(block.proof_target())).await {
                                                    error!("Error sending new target to prover: {}", e);
                                                } else {
                                                    debug!("Sent new target to prover");
                                                }
                                                // 发送新的工作给 prover
                                                if let Err(e) = prover_sender.send(ProverEvent::NewWork(epoch_challenge.epoch_number(), epoch_challenge, client.address)).await {
                                                    error!("Error sending new work to prover: {}", e);
                                                } else {
                                                    debug!("Sent new work to prover");
                                                }
                                            }
                                            Message::Disconnect(message) => {
                                                error!("Peer disconnected: {:?}", message.reason);
                                                sleep(Duration::from_secs(5)).await;
                                                break;
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
                                        error!("Disconnected from beacon");
                                        connected.store(false, Ordering::SeqCst); // 更新连接状态
                                        sleep(Duration::from_secs(5)).await;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to connect to beacon: {}", e);
                        sleep(Duration::from_secs(5)).await;
                    }
                },
                Err(_) => {
                    error!("Failed to connect to beacon: Timed out");
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    });
}
