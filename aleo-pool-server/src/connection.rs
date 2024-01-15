use std::{
    net::SocketAddr,
    str::FromStr,
    time::{Duration, Instant},
};

use aleo_stratum::{
    codec::{BoxedType, ResponseParams, StratumCodec},
    message::StratumMessage,
};
use anyhow::{anyhow, Result};
use futures_util::SinkExt;
use semver::Version;
use snarkvm::{
    prelude::{ Environment, Testnet3, FromBytes},
    console::types::Address,
};
use snarkvm_algorithms::polycommit::kzg10::{KZGCommitment, KZGProof};
use tokio::{
    net::TcpStream,
    sync::mpsc::{channel, Sender},
    task,
    time::timeout,
};
use tokio_stream::StreamExt;
use tokio_util::codec::Framed;
use tracing::{error, info, trace, warn};

use crate::server::ServerMessage;

pub struct Connection {
    user_agent: String,                     // 用户代理
    address: Option<Address<Testnet3>>,     // 地址
    version: Version,                       // 版本
    last_received: Option<Instant>,         // 最后接收时间
}

static PEER_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);   // 握手超时时间
static PEER_COMM_TIMEOUT: Duration = Duration::from_secs(180);       // 通信超时时间

static MIN_SUPPORTED_VERSION: Version = Version::new(2, 0, 0);       // 最低支持版本
static MAX_SUPPORTED_VERSION: Version = Version::new(2, 0, 0);       // 最高支持版本

impl Connection {
    pub async fn init(
        stream: TcpStream,  // TCP流，用于与客户端通信
        peer_addr: SocketAddr,  // 客户端的地址
        server_sender: Sender<ServerMessage>,  // 用于向服务器发送消息的通道
        pool_address: Address<Testnet3>,  // 连接到的矿池地址
    ) {
        // 创建异步任务，调用 `Connection::run` 方法处理连接
        task::spawn(Connection::run(stream, peer_addr, server_sender, pool_address));
    }

    pub async fn run(
        stream: TcpStream,
        peer_addr: SocketAddr,
        server_sender: Sender<ServerMessage>,
        pool_address: Address<Testnet3>,
    ) {
        // 使用自定义的 StratumCodec 实现对数据的解析和封装
        let mut framed = Framed::new(stream, StratumCodec::default());

        // 创建一个异步通道，用于在连接中发送和接收消息
        let (sender, mut receiver) = channel(1024);

        // 创建一个 Connection 结构体，表示当前连接的状态和属性
        let mut conn = Connection {
            user_agent: "Unknown".to_string(),
            address: None,
            version: Version::new(0, 0, 0),
            last_received: None,
        };

        // Handshake（握手过程）
        // 调用 handshake 方法进行握手，获取用户代理和协议版本
        if let Ok((user_agent, version)) = Connection::handshake(&mut framed, pool_address.to_string()).await {
            conn.user_agent = user_agent;
            conn.version = version;
        } else {
            if let Err(e) = server_sender.send(ServerMessage::ProverDisconnected(peer_addr)).await {
                error!("Failed to send ProverDisconnected message to server: {}", e);
            }
            return;
        }

        // 调用 authorize 方法进行授权，获取连接的地址
        if let Ok(address) = Connection::authorize(&mut framed).await {
            conn.address = Some(address);
            if let Err(e) = server_sender
                .send(ServerMessage::ProverAuthenticated(
                    peer_addr,
                    conn.address.unwrap(),
                    sender,
                ))
                .await
            {
                error!("Failed to send ProverAuthenticated message to server: {}", e);
            }
        } else {
            if let Err(e) = server_sender.send(ServerMessage::ProverDisconnected(peer_addr)).await {
                error!("Failed to send ProverDisconnected message to server: {}", e);
            }
            return;
        }

        // 更新最后接收消息的时间
        conn.last_received = Some(Instant::now());

        info!("Peer {:?} authenticated as {}", peer_addr, conn.address.unwrap());

        //不断监听来自客户端和服务器的消息
        loop {
            tokio::select! {//同时监听多个异步事件
                Some(msg) = receiver.recv() => { //收到来自通道的消息，执行对应的处理逻辑
                    if let Some(instant) = conn.last_received { //如果上一次接收消息的时间存在，即表示已经有消息传输过来了
                        if instant.elapsed() > PEER_COMM_TIMEOUT { //如果两次消息的时间差超过了 PEER_COMM_TIMEOUT，认为连接已经超时。
                            warn!("Peer {:?} timed out", peer_addr);
                            break; //跳出循环，结束任务
                        }
                    }
                    //记录调试日志，表示正在向对端发送消息
                    trace!("Sending message {} to peer {:?}", msg.name(), peer_addr);
                    //使用 framed 发送消息
                    if let Err(e) = framed.send(msg).await {
                        error!("Failed to send message to peer {:?}: {:?}", peer_addr, e);
                    }
                },

                //收到来自对端的消息，执行对应的处理逻辑。
                result = framed.next() => match result {
                    Some(Ok(msg)) => { //如果消息正常，执行对应的处理逻辑。
                        trace!("Received message {} from peer {:?}", msg.name(), peer_addr);
                        //更新最后接收消息的时间。
                        conn.last_received = Some(Instant::now());
                        match msg {
                            //处理客户端提交证明的消息。
                            StratumMessage::Submit(id, _worker_name, job_id, nonce, commitment, proof) => {
                                //解码作业号
                                let job_bytes = hex::decode(job_id.clone());
                                if job_bytes.is_err() {
                                    warn!("Failed to decode job_id {} from peer {:?}", job_id, peer_addr);
                                    break;
                                }
                                //如果解码出来的字节数不等于 4，表示作业号格式不正确，记录警告日志。
                                if job_bytes.clone().unwrap().len() != 4 {
                                    warn!("Invalid job_id {} from peer {:?}", job_id, peer_addr);
                                    break;
                                }
                                //解析出 epoch_number
                                let epoch_number = u32::from_le_bytes(job_bytes.unwrap().try_into().unwrap());
                                //解码 nonce
                                let nonce_bytes = hex::decode(nonce.clone());
                                if nonce_bytes.is_err() {
                                    warn!("Failed to decode nonce {} from peer {:?}", nonce, peer_addr);
                                    break;
                                }
                                //解析出 nonce
                                let nonce = u64::from_le_bytes(nonce_bytes.unwrap().try_into().unwrap());
                                //解码 commitment
                                let commitment_bytes = hex::decode(commitment.clone());
                                if commitment_bytes.is_err() {
                                    warn!("Failed to decode commitment {} from peer {:?}", commitment, peer_addr);
                                    break;
                                }
                                //解析出 commitment
                                let commitment = KZGCommitment::<<Testnet3 as Environment>::PairingCurve>::from_bytes_le(&commitment_bytes.unwrap()[..]);
                                //如果解析失败，表示 commitment 格式不正确，记录警告日志
                                if commitment.is_err() {
                                    warn!("Invalid commitment from peer {:?}", peer_addr);
                                    break;
                                }
                                //解码 proof
                                let proof_bytes = hex::decode(proof.clone());
                                if proof_bytes.is_err() {
                                warn!("Failed to decode proof {} from peer {:?}", proof, peer_addr);
                                    break;
                                }
                                //解析出 proof
                                let proof = KZGProof::<<Testnet3 as Environment>::PairingCurve>::from_bytes_le(&proof_bytes.unwrap());
                                if proof.is_err() {
                                    warn!("Invalid proof from peer {:?}", peer_addr);
                                    break;
                                }
                                //将解析出来的证明信息发送给服务器
                                if let Err(e) = server_sender.send(ServerMessage::ProverSubmit(id, peer_addr, epoch_number, nonce, commitment.unwrap(), proof.unwrap())).await {
                                    error!("Failed to send ProverSubmit message to server: {}", e);
                                }
                            }
                            _ => {//其他类型的消息，表示收到非法消息，记录警告日志
                                warn!("Received unexpected message from peer {:?}: {:?}", peer_addr, msg.name());
                                break;
                            }
                        }
                    }
                    Some(Err(e)) => {
                        warn!("Failed to read message from peer: {:?}", e);
                        break;
                    }
                    None => {
                        info!("Peer {:?} disconnected", peer_addr);
                        break;
                    }
                },
                _ = tokio::time::sleep(PEER_COMM_TIMEOUT) => {
                    // 超时
                    info!("Peer {:?} timed out", peer_addr);
                    break;
                },
            }
        }
        if let Err(e) = server_sender.send(ServerMessage::ProverDisconnected(peer_addr)).await {
            error!("Failed to send ProverDisconnected message to server: {}", e);
        }
    }

    pub async fn handshake(
        framed: &mut Framed<TcpStream, StratumCodec>,
        pool_address: String,
    ) -> Result<(String, Version)> {
        // 获取对端的地址信息
        let peer_addr = framed.get_ref().peer_addr()?;

        // 使用超时机制等待对端发送消息
        match timeout(PEER_HANDSHAKE_TIMEOUT, framed.next()).await {
            Ok(Some(Ok(message))) => {
                trace!("Received message {} from peer {:?}", message.name(), peer_addr);

                // 处理收到的消息
                match message {
                    StratumMessage::Subscribe(id, user_agent, protocol_version, _) => {
                        // 解析协议版本号
                        let split: Vec<&str> = protocol_version.split('/').collect();
                        if split.len() != 2 {
                            warn!(
                            "Invalid protocol version {} from peer {:?}",
                            protocol_version, peer_addr
                        );
                            return Err(anyhow!("Invalid protocol version"));
                        }

                        // 检查协议名称是否正确
                        if split[0] != "AleoStratum" {
                            warn!("Invalid protocol name {} from peer {:?}", split[0], peer_addr);
                            return Err(anyhow!("Invalid protocol name"));
                        }

                        // 解析协议版本号
                        let version = Version::parse(split[1]).map_err(|e| {
                            warn!(
                            "Invalid protocol version {} from peer {:?}: {:?}",
                            split[1], peer_addr, e
                        );
                            e
                        })?;

                        // 检查协议版本是否支持
                        if version < MIN_SUPPORTED_VERSION || version > MAX_SUPPORTED_VERSION {
                            warn!("Unsupported protocol version {} from peer {:?}", version, peer_addr);
                            return Err(anyhow!("Unsupported protocol version"));
                        }

                        // 构造响应消息的参数
                        let response_params: Vec<Box<dyn BoxedType>> = vec![
                            Box::new(Option::<String>::None),
                            Box::new(Option::<String>::None),
                            Box::new(Some(pool_address)),
                        ];

                        // 发送响应消息给对端
                        framed
                            .send(StratumMessage::Response(
                                id,
                                Some(ResponseParams::Array(response_params)),
                                None,
                            ))
                            .await?;

                        // 返回用户代理和协议版本号
                        Ok((user_agent, version))
                    }
                    _ => {
                        warn!("Peer {:?} sent {} before handshake", peer_addr, message.name());
                        Err(anyhow!("Unexpected message before handshake"))
                    }
                }
            }
            Ok(Some(Err(e))) => {
                warn!("Error reading from peer {:?}: {}", peer_addr, e);
                Err(anyhow!("Error reading from peer"))
            }
            Ok(None) => {
                warn!("Peer {:?} disconnected before authorization", peer_addr);
                Err(anyhow!("Peer disconnected before authorization"))
            }
            Err(e) => {
                warn!("Peer {:?} timed out on handshake: {}", peer_addr, e);
                Err(anyhow!("Peer timed out on handshake"))
            }
        }
    }

    pub async fn authorize(framed: &mut Framed<TcpStream, StratumCodec>) -> Result<Address<Testnet3>> {
        // 获取对端的地址信息
        let peer_addr = framed.get_ref().peer_addr()?;

        // 使用超时机制等待对端发送消息
        match timeout(PEER_HANDSHAKE_TIMEOUT, framed.next()).await {
            Ok(Some(Ok(message))) => {
                trace!("Received message {} from peer {:?}", message.name(), peer_addr);

                // 处理收到的消息
                match message {
                    StratumMessage::Authorize(id, address, _) => {
                        // 解析地址
                        let address = Address::<Testnet3>::from_str(address.as_str()).map_err(|e| {
                            warn!("Invalid address {} from peer {:?}: {:?}", address, peer_addr, e);
                            e
                        })?;

                        // 发送授权响应给对端
                        framed
                            .send(StratumMessage::Response(id, Some(ResponseParams::Bool(true)), None))
                            .await?;

                        // 返回解析得到的地址
                        Ok(address)
                    }
                    _ => {
                        warn!("Peer {:?} sent {} before authorizing", peer_addr, message.name());
                        Err(anyhow!("Unexpected message before authorization"))
                    }
                }
            }
            Ok(Some(Err(e))) => {
                warn!("Error reading from peer {:?}: {}", peer_addr, e);
                Err(anyhow!("Error reading from peer"))
            }
            Ok(None) => {
                warn!("Peer {:?} disconnected before authorization", peer_addr);
                Err(anyhow!("Peer disconnected before authorization"))
            }
            Err(e) => {
                warn!("Peer {:?} timed out on authorize: {}", peer_addr, e);
                Err(anyhow!("Peer timed out on authorize"))
            }
        }
    }
}