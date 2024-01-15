use std::{collections::BTreeMap, sync::Arc, time::Duration};

use futures_util::sink::SinkExt;
use rand::{thread_rng, Rng};
use snarkos::{
    environment::Prover,
    helpers::{NodeType, State},
    Data,
    Message,
};
use snarkos_storage::BlockLocators;
use snarkvm::{
    dpc::{testnet2::Testnet2, Address, BlockHeader},
    traits::Network,
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
use tracing::{debug, error, info, warn};

use crate::{prover::ProverEvent, Client};

pub struct Node {
    address: Address<Testnet2>,   // 节点地址
    operator: String,             // 操作者信息
    router: Arc<Sender<SendMessage>>,  // 发送消息通道
    receiver: Arc<Mutex<Receiver<SendMessage>>>,  // 接收消息通道
}

#[derive(Debug)]
pub struct SendMessage {
    pub(crate) message: Message<Testnet2, Prover<Testnet2>>,  // 发送的消息
}

impl Node {
    // 初始化节点，返回一个Arc指针
    pub fn init(address: Address<Testnet2>, operator: String) -> Arc<Self> {
        let (router_tx, router_rx) = mpsc::channel(1024);  // 创建发送和接收通道
        Arc::new(Self {
            address,
            operator,
            router: Arc::new(router_tx),
            receiver: Arc::new(Mutex::new(router_rx)),
        })
    }

    // 获取发送通道
    pub fn router(&self) -> Arc<Sender<SendMessage>> {
        self.router.clone()
    }

    // 获取接收通道
    pub fn receiver(&self) -> Arc<Mutex<Receiver<SendMessage>>> {
        self.receiver.clone()
    }
}

// 启动节点
pub fn start(prover_router: Arc<Sender<ProverEvent>>, client: Arc<Client>) {
    task::spawn(async move {  // 异步执行
        let receiver = client.receiver();  // 获取接收通道
        loop {  // 循环不停地运行
        info!("Connecting to operator...");  // 输出日志信息
            match timeout(Duration::from_secs(5), TcpStream::connect(&client.pool)).await {  // 连接操作者
                Ok(socket) => match socket {
                    Ok(socket) => {
                        info!("Connected to {}", client.pool);  // 输出日志信息
                        let mut framed = Framed::new(socket, Message::<Testnet2, Prover<Testnet2>>::PeerRequest);  // 创建Framed实例
                        let challenge = Message::ChallengeRequest(  // 创建ChallengeRequest消息
                                                                    12,
                                                                    Testnet2::ALEO_MAXIMUM_FORK_DEPTH,
                                                                    NodeType::Prover,
                                                                    State::Ready,
                                                                    4132,
                                                                    thread_rng().gen(),
                                                                    0,
                        );
                        if let Err(e) = framed.send(challenge).await {  // 发送ChallengeRequest消息
                        error!("Error sending challenge request: {}", e);  // 输出日志信息
                        } else {
                            debug!("Sent challenge request");  // 输出日志信息
                        }
                        let receiver = &mut *receiver.lock().await;  // 获取接收通道锁
                        loop {  // 循环不停地运行
                        tokio::select! {
                                Some(message) = receiver.recv() => {  // 接收消息
                                    let message = message.clone();  // 克隆消息
                                    debug!("Sending {} to operator", message.name());  // 输出日志信息
                                    if let Err(e) = framed.send(message.clone()).await {  // 发送消息
                                        error!("Error sending {}: {:?}", message.name(), e);  // 输出日志信息
                                    }
                                }
                                result = framed.next() => match result {  // 获取Framed实例中的下一个元素
                                    Some(Ok(message)) => {  // 如果成功获取到元素
                                        debug!("Received {} from operator", message.name());  // 输出日志信息
                                        match message {
                                            Message::ChallengeRequest(..) => {  // 如果收到ChallengeRequest消息
                                                let resp = Message::ChallengeResponse(Data::Object(Testnet2::genesis_block().header().clone()));  // 创建ChallengeResponse消息
                                                if let Err(e) = framed.send(resp).await {  // 发送ChallengeResponse消息
                                                    error!("Error sending challenge response: {:?}", e);  // 输出日志信息
                                                } else {
                                                    debug!("Sent challenge response");  // 输出日志信息
                                                }
                                            }
                                            Message::ChallengeResponse(..) => {  // 如果收到ChallengeResponse消息
                                                let ping = Message::<Testnet2, Prover<Testnet2>>::Ping(  // 创建Ping消息
                                                    12,
                                                    Testnet2::ALEO_MAXIMUM_FORK_DEPTH,
                                                    NodeType::Prover,
                                                    State::Ready,
                                                    Testnet2::genesis_block().hash(),
                                                    Data::Object(Testnet2::genesis_block().header().clone()),
                                                );
                                                if let Err(e) = framed.send(ping).await {  // 发送Ping消息
                                                    error!("Error sending ping: {:?}", e);  // 输出日志信息
                                                } else {
                                                    debug!("Sent ping");  // 输出日志信息
                                                }
                                            }
                                            Message::Ping(..) => {  // 如果收到Ping消息
                                                let mut locators: BTreeMap<u32, (<Testnet2 as Network>::BlockHash, Option<BlockHeader<Testnet2>>)> = BTreeMap::new();
                                                locators.insert(0, (Testnet2::genesis_block().hash(), None));
                                                let resp = Message::<Testnet2, Prover<Testnet2>>::Pong(None, Data::Object(BlockLocators::<Testnet2>::from(locators).unwrap_or_default()));  // 创建Pong消息
                                                if let Err(e) = framed.send(resp).await {  // 发送Pong消息
                                                    error!("Error sending pong: {:?}", e);  // 输出日志信息
                                                } else {
                                                    debug!("Sent pong");  // 输出日志信息
                                                }
                                            }
                                            Message::Pong(..) => {  // 如果收到Pong消息
                                                let register = Message::<Testnet2, Prover<Testnet2>>::PoolRegister(client.address);  // 创建PoolRegister消息
                                                if let Err(e) = framed.send(register).await {  // 发送PoolRegister消息
                                                    error!("Error sending pool register: {:?}", e);  // 输出日志信息
                                                } else {
                                                    debug!("Sent pool register");  // 输出日志信息
                                                }
                                            }
                                            Message::PoolRequest(share_difficulty, block_template) => {  // 如果收到PoolRequest消息
                                                if let Ok(block_template) = block_template.deserialize().await {  // 反序列化BlockTemplate消息
                                                    if let Err(e) = prover_router.send(ProverWork::new(share_difficulty, block_template)).await {  // 发送ProverWork消息
                                                        error!("Error sending work to prover: {:?}", e);  // 输出日志信息
                                                    } else {
                                                        debug!("Sent work to prover");  // 输出日志信息
                                                    }
                                                } else {
                                                    error!("Error deserializing block template");  // 输出日志信息
                                                }
                                            }
                                            Message::UnconfirmedBlock(..) => {}  // 如果收到UnconfirmedBlock消息
                                            _ => {
                                                debug!("Unhandled message: {}", message.name());  // 输出日志信息
                                            }
                                        }
                                    }
                                    Some(Err(e)) => {
                                        warn!("Failed to read the message: {:?}", e);  // 输出日志信息
                                    }
                                    None => {
                                        error!("Disconnected from operator");  // 输出日志信息
                                        sleep(Duration::from_secs(5)).await;  // 等待5秒
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to connect to operator: {}", e);  // 输出日志信息
                        sleep(Duration::from_secs(5)).await;  // 等待5秒
                    }
                },
                Err(_) => {
                    error!("Failed to connect to operator: Timed out");  // 输出日志信息
                    sleep(Duration::from_secs(5)).await;  // 等待5秒
                }
            }
        }
    });
}
