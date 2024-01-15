use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use futures_util::sink::SinkExt;
use rand::{rngs::OsRng, Rng};
use snarkos_account::Account;
use snarkos_node_messages::{
    ChallengeRequest,
    ChallengeResponse,
    Data,
    MessageCodec,
    NodeType,
    Ping,
    Pong,
    PuzzleRequest,
    PuzzleResponse,
};
use snarkvm::{
    prelude::{FromBytes, Network, Testnet3},
    synthesizer::Block,
};
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
use tracing::{debug, error, info, trace, warn};

use crate::ServerMessage;

pub struct Node {
    operator: String,
    sender: Arc<Sender<SnarkOSMessage>>,
    receiver: Arc<Mutex<Receiver<SnarkOSMessage>>>,
}

pub(crate) type SnarkOSMessage = snarkos_node_messages::Message<Testnet3>;

impl Node {
    // 初始化节点
    pub fn init(operator: String) -> Self {
        let (sender, receiver) = mpsc::channel(1024); // 创建消息通道，sender用于发送消息，receiver用于接收消息
        Self {
            operator,
            sender: Arc::new(sender),
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    // 获取节点的接收器
    pub fn receiver(&self) -> Arc<Mutex<Receiver<SnarkOSMessage>>> {
        self.receiver.clone()
    }

    // 获取节点的发送器
    pub fn sender(&self) -> Arc<Sender<SnarkOSMessage>> {
        self.sender.clone()
    }
}


//与 SnarkOS 网络中的运营者建立连接，接收来自网络中其它节点的消息，并向其它节点发送消息。
pub fn start(node: Node, server_sender: Sender<ServerMessage>) {
    let receiver = node.receiver(); // 获取节点的接收器
    let sender = node.sender(); // 获取节点的发送器
    task::spawn(async move { // 创建异步任务
        let genesis_header = *Block::<Testnet3>::from_bytes_le(Testnet3::genesis_bytes())
            .unwrap()
            .header(); // 获取创始块的区块头信息

        let connected = Arc::new(AtomicBool::new(false)); // 标识是否与其它节点连接
        let peer_sender = sender.clone(); // 克隆节点的发送器
        let peer_sender_ping = sender.clone(); // 克隆节点的发送器

        let connected_req = connected.clone(); // 克隆连接状态
        let connected_ping = connected.clone(); // 克隆连接状态
        task::spawn(async move { // 创建异步任务
            loop { // 循环
                sleep(Duration::from_secs(15)).await; // 暂停一段时间
                if connected_req.load(Ordering::SeqCst) { // 如果已经连接
                    if let Err(e) = peer_sender.send(SnarkOSMessage::PuzzleRequest(PuzzleRequest {})).await { // 向其它节点发送谜题请求
                    error!("Failed to send puzzle request: {}", e); // 发送失败
                    }
                }
            }
        });
        task::spawn(async move { // 创建异步任务
            loop { // 循环
                sleep(Duration::from_secs(5)).await; // 暂停一段时间
                if connected_ping.load(Ordering::SeqCst) { // 如果已经连接
                    if let Err(e) = peer_sender_ping
                        .send(SnarkOSMessage::Ping(Ping {
                            version: SnarkOSMessage::VERSION,
                            node_type: NodeType::Prover,
                            block_locators: None,
                        }))
                        .await
                    {
                        error!("Failed to send ping: {}", e); // 发送 ping 失败
                    }
                }
            }
        });

        let rng = &mut OsRng; // 生成随机数
        let random_account = Account::new(rng).unwrap(); // 创建一个随机账户
        loop { // 循环
        info!("Connecting to operator..."); // 输出日志信息
            match timeout(Duration::from_secs(5), TcpStream::connect(&node.operator)).await { // 连接运营者
                Ok(socket) => match socket { // 如果成功
                    Ok(socket) => { // 如果连接成功
                    info!("Connected to {}", node.operator); // 输出日志信息
                        let mut framed: Framed<TcpStream, MessageCodec<Testnet3>> =
                            Framed::new(socket, Default::default()); // 创建帧
                        let challenge = SnarkOSMessage::ChallengeRequest(ChallengeRequest { // 创建 ChallengeRequest 请求
                            version: SnarkOSMessage::VERSION,
                            listener_port: 4140,
                            node_type: NodeType::Prover,
                            address: random_account.address(),
                            nonce: rng.gen(),
                        });
                        if let Err(e) = framed.send(challenge).await { // 发送 ChallengeRequest 请求
                        error!("Error sending challenge request: {}", e); // 发送失败
                        } else {
                            trace!("Sent challenge request"); // 输出日志信息
                        }
                        let receiver = &mut *receiver.lock().await; // 获取接收器
                        loop { // 循环
                        tokio::select! { // 选择不同的异步任务
                                Some(message) = receiver.recv() => { // 接收到消息
                                    trace!("Sending {} to validator", message.name()); // 输出日志信息
                                    if let Err(e) = framed.send(message.clone()).await { // 发送消息
                                        error!("Error sending {}: {:?}", message.name(), e); // 发送失败
                                    }
                                }
                                result = framed.next() => match result { // 获取从帧中接收到的消息
                                    Some(Ok(message)) => { // 如果接收到了消息
                                        trace!("Received {} from validator", message.name()); // 输出日志信息
                                        match message { // 处理不同的消息类型
                                            SnarkOSMessage::ChallengeRequest(ChallengeRequest { // 处理 ChallengeRequest 请求
                                                version,
                                                listener_port: _,
                                                node_type,
                                                address: _,
                                                nonce,
                                            }) => {
                                                if version < SnarkOSMessage::VERSION { // 判断 SnarkOSMessage 的版本是否正确
                                                    error!("Peer is running an older version of the protocol"); // 输出日志信息
                                                    sleep(Duration::from_secs(25)).await; // 暂停一段时间
                                                    break; // 退出循环
                                                }
                                                if node_type != NodeType::Beacon && node_type != NodeType::Validator { // 判断节点类型是否正确
                                                    error!("Peer is not a beacon or validator"); // 输出日志信息
                                                    sleep(Duration::from_secs(25)).await; // 暂停一段时间
                                                    break; // 退出循环
                                                }
                                                let response = SnarkOSMessage::ChallengeResponse(ChallengeResponse { // 创建 ChallengeResponse 响应
                                                    genesis_header,
                                                    signature: Data::Object(random_account.sign_bytes(&nonce.to_le_bytes(), rng).unwrap()),
                                                });
                                                if let Err(e) = framed.send(response).await { // 发送 ChallengeResponse 响应
                                                    error!("Error sending challenge response: {:?}", e); // 发送失败
                                                } else {
                                                    debug!("Sent challenge response"); // 输出日志信息
                                                }
                                            }
                                            SnarkOSMessage::ChallengeResponse(message) => { // 处理 ChallengeResponse 响应
                                                match message.genesis_header == genesis_header { // 判断响应中的区块头与本地是否一致
                                                    true => { // 如果一致
                                                        let was_connected = connected.load(Ordering::SeqCst); // 获取当前连接状态
                                                        connected.store(true, Ordering::SeqCst); // 设置连接状态为已连接
                                                        if !was_connected {
                                                            if let Err(e) = sender.send(SnarkOSMessage::PuzzleRequest(PuzzleRequest {})).await {
                                                                error!("Failed to send puzzle request: {}", e);
                                                            }
                                                        }
                                                    }
                                                    false => { // 如果不一致
                                                        error!("Peer has a different genesis block"); // 输出日志信息
                                                        sleep(Duration::from_secs(25)).await; // 暂停一段时间
                                                        break; // 退出循环
                                                    }
                                                }
                                            }
                                            SnarkOSMessage::Ping(..) => { // 处理 Ping 请求
                                                let pong = SnarkOSMessage::Pong(Pong { is_fork: None }); // 创建 Pong 响应
                                                if let Err(e) = framed.send(pong).await { // 发送 Pong 响应
                                                    error!("Error sending pong: {:?}", e); // 发送失败
                                                } else {
                                                    debug!("Sent pong"); // 输出日志信息
                                                }
                                                let message = SnarkOSMessage::Ping(Ping { // 创建 Ping 请求
                                                    version: SnarkOSMessage::VERSION,
                                                    node_type: NodeType::Prover,
                                                    block_locators: None,
                                                });
                                                if let Err(e) = framed.send(message).await { // 发送 Ping 请求
                                                    error!("Error sending ping: {:?}", e); // 发送失败
                                                } else {
                                                    debug!("Sent ping"); // 输出日志信息
                                                }
                                            }
                                            SnarkOSMessage::PuzzleResponse(PuzzleResponse { // 处理 PuzzleResponse 响应
                                                epoch_challenge, block_header
                                            }) => {
                                                let block_header = match block_header.deserialize().await { // 将序列化后的区块头反序列化
                                                    Ok(block_header) => block_header, // 反序列化成功
                                                    Err(error) => { // 反序列化失败
                                                        error!("Error deserializing block header: {:?}", error); // 输出日志信息
                                                        connected.store(false, Ordering::SeqCst); // 设置连接状态为未连接
                                                        sleep(Duration::from_secs(25)).await; // 暂停一段时间
                                                        break; // 退出循环
                                                    }
                                                };
                                                let epoch_number = epoch_challenge.epoch_number(); // 获取当前时代的编号
                                                if let Err(e) = server_sender.send(ServerMessage::NewEpochChallenge( // 将新的时代谜题发送给服务器
                                                    epoch_challenge, block_header.proof_target()
                                                )).await {
                                                    error!("Error sending new block template to pool server: {}", e); // 发送失败
                                                } else {
                                                    trace!("Sent new epoch challenge {} to pool server", epoch_number); // 输出日志信息
                                                }
                                            }
                                            SnarkOSMessage::Disconnect(message) => { // 处理 Disconnect 消息
                                                error!("Peer disconnected: {:?}", message.reason); // 输出日志信息
                                                connected.store(false, Ordering::SeqCst); // 设置连接状态为未连接
                                                sleep(Duration::from_secs(25)).await; // 暂停一段时间
                                                break; // 退出循环
                                            }
                                            _ => { // 处理其它消息类型
                                                debug!("Unhandled message: {}", message.name()); // 输出日志信息
                                            }
                                        }
                                    }
                                    Some(Err(e)) => { // 如果接收到的消息出错
                                        warn!("Failed to read the message: {:?}", e); // 输出日志信息
                                    }
                                    None => { // 如果连接断开
                                        error!("Disconnected from operator"); // 输出日志信息
                                        connected.store(false, Ordering::SeqCst); // 设置连接状态为未连接
                                        sleep(Duration::from_secs(25)).await; // 暂停一段时间
                                        break; // 退出循环
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => { // 如果连接失败
                    error!("Failed to connect to operator: {}", e); // 输出日志信息
                        sleep(Duration::from_secs(25)).await; // 暂停一段时间
                    }
                },
                Err(_) => { // 如果连接超时
                error!("Failed to connect to operator: Timed out"); // 输出日志信息
                    sleep(Duration::from_secs(25)).await; // 暂停一段时间
                }
            }
        }
    });
}
