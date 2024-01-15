use std::{
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter},
    net::SocketAddr,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use aleo_stratum::{codec::ResponseParams, message::StratumMessage};
use anyhow::ensure;
use blake2::Digest;
use flurry::HashSet as FlurryHashSet;
use json_rpc_types::{Error, ErrorCode, Id};
use snarkos_node_router_messages::{UnconfirmedSolution};
use snarkvm::{
    ledger::narwhal::Data,
    prelude::{ ToBytes, Testnet3, Environment, UniversalSRS },
    ledger::coinbase::{ CoinbasePuzzle, EpochChallenge, PuzzleConfig, PartialSolution, ProverSolution, PuzzleCommitment},
    console::types::Address,
};
use snarkvm_fields::PrimeField;

// use snarkvm_console_network::{Testnet3, Network};

use snarkvm_algorithms::{
    cfg_into_iter,
    crypto_hash::sha256d_to_u64,
    fft::DensePolynomial,
    polycommit::kzg10::{KZGCommitment, KZGProof, KZG10},
};
use snarkvm_curves::PairingEngine;
use snarkvm_utilities::serialize::CanonicalSerialize;
use speedometer::Speedometer;
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{
        mpsc::{channel, Sender},
        RwLock,
    },
    task,
};
use tracing::{debug, error, info, trace, warn};

use rayon::prelude::*;


// use crate::{connection::Connection, validator_peer::SnarkOSMessage, AccountingMessage};
use crate::{connection::Connection, validator_dummy::SnarkOSMessage, AccountingMessage};

pub struct ProverState {
    peer_addr: SocketAddr,          // 对等方地址
    address: Address<Testnet3>,     // 地址
    speed_2m: Speedometer,          // 2分钟速度计
    speed_5m: Speedometer,          // 5分钟速度计
    speed_15m: Speedometer,         // 15分钟速度计
    speed_30m: Speedometer,         // 30分钟速度计
    speed_1h: Speedometer,          // 1小时速度计
    current_target: u64,            // 当前目标值
    next_target: u64,               // 下一个目标值
}

impl ProverState {
    // 创建新的ProverState实例
    pub fn new(peer_addr: SocketAddr, address: Address<Testnet3>) -> Self {
        Self {
            peer_addr,
            address,
            speed_2m: Speedometer::init(Duration::from_secs(120)),
            speed_5m: Speedometer::init_with_cache(Duration::from_secs(60 * 5), Duration::from_secs(30)),
            speed_15m: Speedometer::init_with_cache(Duration::from_secs(60 * 15), Duration::from_secs(30)),
            speed_30m: Speedometer::init_with_cache(Duration::from_secs(60 * 30), Duration::from_secs(30)),
            speed_1h: Speedometer::init_with_cache(Duration::from_secs(60 * 60), Duration::from_secs(30)),
            current_target: 512,
            next_target: 512,
        }
    }

    // 添加份额，并更新速度和下一个目标值
    pub async fn add_share(&mut self, value: u64) {
        let now = Instant::now();
        self.speed_2m.event(value).await;
        self.speed_5m.event(value).await;
        self.speed_15m.event(value).await;
        self.speed_30m.event(value).await;
        self.speed_1h.event(value).await;
        self.next_target = ((self.speed_2m.speed().await * 20.0) as u64).max(1);
        debug!("add_share took {} us", now.elapsed().as_micros());
    }

    // 获取下一个目标值，并更新当前目标值.如果下一个目标值不在当前目标值的0.9到1.1之间，则将下一个目标值赋值给当前目标值。
    pub async fn next_target(&mut self) -> u64 {
        if self.next_target < ((self.current_target as f64) * 0.9) as u64
            || self.next_target > ((self.current_target as f64) * 1.1) as u64
        {
            self.current_target = self.next_target;
        }
        self.current_target
    }

    // 获取当前目标值
    pub fn current_target(&self) -> u64 {
        self.current_target
    }

    // 获取地址
    pub fn address(&self) -> Address<Testnet3> {
        self.address
    }

    // 获取速度，返回一个包含5分钟速度、15分钟速度、30分钟速度和1小时速度的Vec<f64>。
    // noinspection DuplicatedCode
    pub async fn speed(&mut self) -> Vec<f64> {
        vec![
            self.speed_5m.speed().await,
            self.speed_15m.speed().await,
            self.speed_30m.speed().await,
            self.speed_1h.speed().await,
        ]
    }
}

// 实现 Display trait 用于格式化输出 ProverState
impl Display for ProverState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let addr_str = self.address.to_string();
        write!(
            f,
            "{} ({}...{})",
            self.peer_addr,
            &addr_str[0..11],
            &addr_str[addr_str.len() - 6..]
        )
    }
}

// 定义 PoolState 结构体
struct PoolState {
    speed_1m: Speedometer,      // 1分钟速度计
    speed_5m: Speedometer,      // 5分钟速度计
    speed_15m: Speedometer,     // 15分钟速度计
    speed_30m: Speedometer,     // 30分钟速度计
    speed_1h: Speedometer,      // 1小时速度计
    current_global_target_modifier: f64,// 当前全局目标修正器
    next_global_target_modifier: f64,   // 下一个全局目标修正器
}

impl PoolState {
    // 创建新的 PoolState 实例
    pub fn new() -> Self {
        Self {
            speed_1m: Speedometer::init(Duration::from_secs(60)),  // 初始化速度计
            speed_5m: Speedometer::init_with_cache(Duration::from_secs(60 * 5), Duration::from_secs(30)),
            speed_15m: Speedometer::init_with_cache(Duration::from_secs(60 * 15), Duration::from_secs(30)),
            speed_30m: Speedometer::init_with_cache(Duration::from_secs(60 * 30), Duration::from_secs(30)),
            speed_1h: Speedometer::init_with_cache(Duration::from_secs(60 * 60), Duration::from_secs(30)),
            current_global_target_modifier: 1.0,
            next_global_target_modifier: 1.0,
        }
    }

    // 添加份额，并更新速度和下一个全局目标修正器
    pub async fn add_share(&mut self, value: u64) {
        let now = Instant::now();
        self.speed_1m.event(1).await;  // 更新速度计
        self.speed_5m.event(value).await;
        self.speed_15m.event(value).await;
        self.speed_30m.event(value).await;
        self.speed_1h.event(value).await;
        self.next_global_target_modifier = (self.speed_1m.speed().await / 200.0).max(1f64);  // 计算下一个全局目标修正器
        // todo: make adjustable through admin api
        debug!("pool state add_share took {} us", now.elapsed().as_micros());
    }

    // 获取下一个全局目标修正器，并更新当前全局目标修正器
    pub async fn next_global_target_modifier(&mut self) -> f64 {
        self.current_global_target_modifier = self.next_global_target_modifier;
        if self.current_global_target_modifier > 1.0 {
            info!(
                "Current global target modifier: {}",
                self.current_global_target_modifier
            );
        }
        self.current_global_target_modifier
    }

    // 获取当前全局目标修正器
    pub fn current_global_target_modifier(&self) -> f64 {
        self.current_global_target_modifier
    }

    // 获取速度
    // noinspection DuplicatedCode
    pub async fn speed(&mut self) -> Vec<f64> {
        vec![
            self.speed_5m.speed().await,
            self.speed_15m.speed().await,
            self.speed_30m.speed().await,
            self.speed_1h.speed().await,
        ]
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum ServerMessage {
    //表示有新的 Prover 连接到服务器，并包含 TCP 流和对等方地址。
    ProverConnected(TcpStream, SocketAddr),
    //表示 Prover 成功通过身份验证，并包含对等方地址、Prover 地址和发送回复消息的通道。
    ProverAuthenticated(SocketAddr, Address<Testnet3>, Sender<StratumMessage>),
    //表示某个 Prover 断开了与服务器的连接，并包含对等方地址。
    ProverDisconnected(SocketAddr),
    //表示 Prover 提交了一个份额，并包含份额 ID、对等方地址、起始位置、份额值、KZG 承诺和 KZG 证
    ProverSubmit(
        Id,
        SocketAddr,
        u32,
        u64,
        KZGCommitment<<Testnet3 as Environment>::PairingCurve>,
        KZGProof<<Testnet3 as Environment>::PairingCurve>,
    ),
    //表示开始新的挑战周期，并包含挑战和时间戳。
    NewEpochChallenge(EpochChallenge<Testnet3>, u64),
    Exit,
}

impl ServerMessage {
    fn name(&self) -> &'static str {
        match self {
            ServerMessage::ProverConnected(..) => "ProverConnected",
            ServerMessage::ProverAuthenticated(..) => "ProverAuthenticated",
            ServerMessage::ProverDisconnected(..) => "ProverDisconnected",
            ServerMessage::ProverSubmit(..) => "ProverSubmit",
            ServerMessage::NewEpochChallenge(..) => "NewEpochChallenge",
            ServerMessage::Exit => "Exit",
        }
    }
}

impl Display for ServerMessage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

pub struct Server {
    sender: Sender<ServerMessage>,  // 用于向Server发送消息的通道
    // validator_sender: Arc<Sender<SnarkOSMessage>>,
    validator_sender: Arc<Sender<String>>, // 用于发送验证消息的通道
    accounting_sender: Sender<AccountingMessage>,// 用于发送会计消息的通道
    pool_address: Address<Testnet3>,// 指定服务器地址
    connected_provers: RwLock<HashSet<SocketAddr>>,// 已连接的prover的Socket地址集合
    authenticated_provers: Arc<RwLock<HashMap<SocketAddr, Sender<StratumMessage>>>>,// 已认证的prover的Socket地址和消息发送者的映射
    pool_state: Arc<RwLock<PoolState>>,// 共享的池状态
    prover_states: Arc<RwLock<HashMap<SocketAddr, RwLock<ProverState>>>>,// prover的Socket地址和其状态的映射
    prover_address_connections: Arc<RwLock<HashMap<Address<Testnet3>, HashSet<SocketAddr>>>>, // prover地址和连接的Socket地址的映射
    coinbase_puzzle: CoinbasePuzzle<Testnet3>,// coinbase难题
    latest_epoch_number: AtomicU32,// 最新的epoch编号
    latest_epoch_challenge: Arc<RwLock<Option<EpochChallenge<Testnet3>>>>, // 最新的epoch挑战
    latest_proof_target: AtomicU64,// 最新的证明目标
    nonce_seen: Arc<FlurryHashSet<u64>>,// 见过的nonce的集合
}

impl Server {
    // 初始化Server实例
// 初始化服务器实例
    pub async fn init(
        port: u16,  // 监听端口
        address: Address<Testnet3>,  // 服务器地址
        //validator_sender: Arc<Sender<SnarkOSMessage>>,
        validator_sender: Arc<Sender<String>>,  // 用于发送验证消息的通道
        accounting_sender: Sender<AccountingMessage>,  // 用于发送会计消息的通道
    ) -> Arc<Server> {
        // 创建消息通道，用于将消息发送给Server实例
        let (sender, mut receiver) = channel(1024);

        // 监听指定端口
        let (_, listener) = match TcpListener::bind(format!("0.0.0.0:{}", port)).await {
            Ok(listener) => {
                let local_ip = listener.local_addr().expect("Could not get local ip");
                info!("Listening on {}", local_ip);
                (local_ip, listener)
            }
            Err(e) => {
                panic!("Unable to start the server: {:?}", e);
            }
        };

        // 加载universal SRS，并创建CoinbasePuzzle实例
        info!("Initializing universal SRS");
        let srs = UniversalSRS::<Testnet3>::load().expect("Failed to load SRS");
        info!("Universal SRS initialized");

        info!("Initializing coinbase verifying key");
        let coinbase_puzzle = CoinbasePuzzle::<Testnet3>::trim(&srs, PuzzleConfig { degree: (1 << 13) - 1 })
            .expect("Failed to load coinbase verifying key");
        info!("Coinbase verifying key initialized");

        // 创建Server实例
        let server = Arc::new(Server {
            sender,
            validator_sender,
            accounting_sender,
            pool_address: address,
            connected_provers: Default::default(),
            authenticated_provers: Default::default(),
            pool_state: Arc::new(RwLock::new(PoolState::new())),
            prover_states: Default::default(),
            prover_address_connections: Default::default(),
            coinbase_puzzle,
            latest_epoch_number: AtomicU32::new(0),
            latest_epoch_challenge: Default::default(),
            latest_proof_target: AtomicU64::new(u64::MAX),
            nonce_seen: Arc::new(FlurryHashSet::with_capacity(10 << 20)),
        });

        // 定期清除nonce集合
        {
            let nonce = server.nonce_seen.clone();
            let mut ticker = tokio::time::interval(Duration::from_secs(60));
            task::spawn(async move {
                loop {
                    ticker.tick().await;
                    nonce.pin().clear()
                }
            });
        }

        // 监听连接请求并处理
        let s = server.clone();
        task::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        info!("New connection from: {}", peer_addr);
                        if let Err(e) = s.sender.send(ServerMessage::ProverConnected(stream, peer_addr)).await {
                            error!("Error accepting connection: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Error accepting connection: {:?}", e);
                    }
                }
            }
        });

        // 处理来自消息通道的消息
        let s = server.clone();
        task::spawn(async move {
            let server = s.clone();
            while let Some(msg) = receiver.recv().await {
                let server = server.clone();
                task::spawn(async move {
                    server.process_message(msg).await;
                });
            }
        });

        // 返回Server实例
        server
    }


    fn seen_nonce(nonce_seen: Arc<FlurryHashSet<u64>>, nonce: u64) -> bool {
        !nonce_seen.pin().insert(nonce)
    }

    fn clear_nonce(&self) {
        self.nonce_seen.pin().clear()
    }

    pub fn sender(&self) -> Sender<ServerMessage> {
        self.sender.clone()
    }

    // 处理接收到的消息
    pub async fn process_message(&self, msg: ServerMessage) {
        // 处理接收到的消息
        trace!("Received message: {}", msg);
        match msg {
            // 处理服务器消息中的"ProverConnected"情况
            ServerMessage::ProverConnected(stream, peer_addr) => {
                // 在连接的验证者集合中插入验证者的地址
                self.connected_provers.write().await.insert(peer_addr);
                // 初始化连接
                Connection::init(stream, peer_addr, self.sender.clone(), self.pool_address).await;
            }
            // 处理服务器消息中的"ProverAuthenticated"情况
            ServerMessage::ProverAuthenticated(peer_addr, address, sender) => {
                self.authenticated_provers
                    .write()
                    .await
                    .insert(peer_addr, sender.clone());
                self.prover_states
                    .write()
                    .await
                    .insert(peer_addr, ProverState::new(peer_addr, address).into());
                // 更新验证者连接的地址集合
                let mut pac_write = self.prover_address_connections.write().await;
                if let Some(address) = pac_write.get_mut(&address) {
                    address.insert(peer_addr);
                } else {
                    pac_write.insert(address, HashSet::from([peer_addr]));
                }
                drop(pac_write);
                // 向验证者发送初始目标值
                if let Err(e) = sender.send(StratumMessage::SetTarget(512)).await {
                    error!("Error sending initial target to prover: {}", e);
                }
                // 如果存在最新的时期挑战，则向验证者发送通知
                if let Some(epoch_challenge) = self.latest_epoch_challenge.read().await.as_ref() {
                    let job_id = hex::encode(self.latest_epoch_number.load(Ordering::SeqCst).to_le_bytes());
                    if let Err(e) = sender
                        .send(StratumMessage::Notify(
                            job_id,
                            hex::encode(epoch_challenge.to_bytes_le().unwrap()),
                            None,
                            true,
                        ))
                        .await
                    {
                        error!(
                            "Error sending initial epoch challenge to prover {} ({}): {}",
                            peer_addr, address, e
                        );
                    }
                }
            }
            // 处理服务器消息中的"ProverDisconnected"情况
            ServerMessage::ProverDisconnected(peer_addr) => {
                // 移除验证者状态
                let state = self.prover_states.write().await.remove(&peer_addr);
                let address = match state {
                    Some(state) => Some(state.read().await.address()),
                    None => None,
                };
                // 从验证者连接的地址集合中移除对应的地址
                if address.is_some() {
                    let mut pac_write = self.prover_address_connections.write().await;
                    let pac = pac_write.get_mut(&address.unwrap());
                    if let Some(pac) = pac {
                        pac.remove(&peer_addr);
                        if pac.is_empty() {
                            pac_write.remove(&address.unwrap());
                        }
                    }
                }
                // 从连接的验证者集合中移除验证者
                self.connected_provers.write().await.remove(&peer_addr);
                // 从已验证的验证者集合中移除验证者
                self.authenticated_provers.write().await.remove(&peer_addr);
            }
            // 处理服务器消息中的"NewEpochChallenge"情况
            ServerMessage::NewEpochChallenge(epoch_challenge, proof_target) => {
                // 获取最新的时期编号
                let latest_epoch = self.latest_epoch_number.load(Ordering::SeqCst);
                // 如果最新的时期编号小于接收到的时期挑战的编号，或者最新的时期编号为0且接收到的时期编号也为0，则更新最新的时期挑战
                if latest_epoch < epoch_challenge.epoch_number()
                    || (epoch_challenge.epoch_number() == 0 && latest_epoch == 0)
                {
                    info!("New epoch challenge: {}", epoch_challenge.epoch_number());
                    self.latest_epoch_number
                        .store(epoch_challenge.epoch_number(), Ordering::SeqCst);
                    self.latest_epoch_challenge
                        .write()
                        .await
                        .replace(epoch_challenge.clone());
                    self.clear_nonce();
                }
                // 如果接收到的时期编号小于最新的时期编号，则直接返回
                if epoch_challenge.epoch_number() < latest_epoch {
                    return;
                }
                // 更新证明目标值
                info!("Updating target to {}", proof_target);
                self.latest_proof_target.store(proof_target, Ordering::SeqCst);
                // 发送会计消息
                if let Err(e) = self
                    .accounting_sender
                    .send(AccountingMessage::SetN(proof_target * 5))
                    .await
                {
                    error!("Error sending accounting message: {}", e);
                }
                // 获取全局难度调整器
                let global_difficulty_modifier = self.pool_state.write().await.next_global_target_modifier().await;
                debug!("Global difficulty modifier: {}", global_difficulty_modifier);
                // 获取作业编号和时期挑战的十六进制表示
                let job_id = hex::encode(epoch_challenge.epoch_number().to_le_bytes());
                let epoch_challenge_hex = hex::encode(epoch_challenge.to_bytes_le().unwrap());
                // 遍历已验证的验证者，给每个验证者发送目标值和时期挑战
                for (peer_addr, sender) in self.authenticated_provers.read().await.clone().iter() {
                    let states = self.prover_states.read().await;
                    let prover_state = match states.get(peer_addr) {
                        Some(state) => state,
                        None => {
                            error!("Prover state not found for peer: {}", peer_addr);
                            continue;
                        }
                    };

                    let prover_display = format!("{}", prover_state.read().await);
                    let current_difficulty = prover_state.read().await.current_target();
                    let mut next_difficulty =
                        (prover_state.write().await.next_target().await as f64 * global_difficulty_modifier) as u64;
                    drop(states);
                    if next_difficulty > proof_target {
                        next_difficulty = proof_target;
                    }
                    // 发送目标值给验证者
                    if current_difficulty != next_difficulty {
                        if let Err(e) = sender.send(StratumMessage::SetTarget(next_difficulty)).await {
                            error!("Error sending difficulty target to prover {}: {}", prover_display, e);
                        }
                    }
                    // 发送时期挑战给验证者
                    if let Err(e) = sender
                        .send(StratumMessage::Notify(
                            job_id.clone(),
                            epoch_challenge_hex.clone(),
                            None,
                            true,
                        ))
                        .await
                    {
                        error!("Error sending block template to prover {}: {}", prover_display, e);
                    }
                }
            }
            // 处理服务器消息中的"ProverSubmit"情况
            ServerMessage::ProverSubmit(id, peer_addr, epoch_number, nonce, commitment, proof) => {
                // 复制所有需要用到的变量
                let prover_states = self.prover_states.clone();
                let pool_state = self.pool_state.clone();
                let authenticated_provers = self.authenticated_provers.clone();
                let latest_epoch_number = self.latest_epoch_number.load(Ordering::SeqCst);
                let current_global_difficulty_modifier = self.pool_state.read().await.current_global_target_modifier();
                let latest_epoch_challenge = self.latest_epoch_challenge.clone();
                let accounting_sender = self.accounting_sender.clone();
                let validator_sender = self.validator_sender.clone();
                let seen_nonce = self.nonce_seen.clone();
                let global_proof_target = self.latest_proof_target.load(Ordering::SeqCst);
                let pool_address = self.pool_address;
                let coinbase_puzzle = self.coinbase_puzzle.clone();
                // 开启一个异步任务来处理验证者提交的证明
                task::spawn(async move {
                    // 定义函数，用来发送结果消息给验证者
                    async fn send_result(
                        sender: &Sender<StratumMessage>,
                        id: Id,
                        result: bool,
                        error_code: Option<ErrorCode>,
                        desc: Option<String>,
                    ) {
                        if result {
                            if let Err(e) = sender
                                .send(StratumMessage::Response(id, Some(ResponseParams::Bool(true)), None))
                                .await
                            {
                                error!("Error sending result to prover: {}", e);
                            }
                        } else if let Err(e) = sender
                            .send(StratumMessage::Response(
                                id,
                                None,
                                Some(Error::with_custom_msg(error_code.unwrap(), desc.unwrap().as_str())),
                            ))
                            .await
                        {
                            error!("Error sending result to prover: {}", e);
                        }
                    }
                    // 获取已验证的验证者集合和验证者状态集合
                    let provers = authenticated_provers.read().await;
                    let states = prover_states.read().await;
                    // 获取验证者的发送者
                    let sender = match provers.get(&peer_addr) {
                        Some(sender) => sender,
                        None => {
                            error!("Sender not found for peer: {}", peer_addr);
                            return;
                        }
                    };
                    // 获取验证者状态
                    let prover_state = match states.get(&peer_addr) {
                        Some(state) => state,
                        None => {
                            error!("Received solution from unknown prover: {}", peer_addr);
                            send_result(
                                sender,
                                id,
                                false,
                                Some(ErrorCode::from_code(24)),
                                Some("Unknown prover".to_string()),
                            )
                            .await;
                            return;
                        }
                    };
                    // 获取验证者的显示信息
                    let prover_display = format!("{}", prover_state.read().await);
                    // 获取最新的时期挑战
                    let epoch_challenge = match latest_epoch_challenge.read().await.clone() {
                        Some(template) => template,
                        None => {
                            warn!(
                                "Received solution from prover {} while no epoch challenge is available",
                                prover_display
                            );
                            send_result(
                                sender,
                                id,
                                false,
                                Some(ErrorCode::from_code(21)),
                                Some("No epoch challenge".to_string()),
                            )
                            .await;
                            return;
                        }
                    };
                    // 如果提交的时期编号与最新的时期编号不一致，则返回错误消息
                    if epoch_number != latest_epoch_number {
                        info!(
                            "Received stale solution from prover {} with epoch number: {} (expected {})",
                            prover_display, epoch_number, latest_epoch_number
                        );
                        send_result(
                            sender,
                            id,
                            false,
                            Some(ErrorCode::from_code(21)),
                            Some("Stale solution".to_string()),
                        )
                            .await;
                        return;
                    }
                    // 如果提交的nonce已经出现过，则返回错误消息
                    if Server::seen_nonce(seen_nonce, nonce) {
                        warn!("Received duplicate nonce from prover {}", prover_display);
                        send_result(
                            sender,
                            id,
                            false,
                            Some(ErrorCode::from_code(22)),
                            Some("Duplicate nonce".to_string()),
                        )
                        .await;
                        return;
                    }
                    // 计算验证者的目标值
                    let mut prover_target =
                        (prover_state.read().await.current_target() as f64 * current_global_difficulty_modifier) as u64;
                    if prover_target > global_proof_target {
                        prover_target = global_proof_target;
                    }
                    // 计算提交的证明的难度
                    let proof_difficulty = match &commitment.to_bytes_le() {
                        Ok(bytes) => u64::MAX / sha256d_to_u64(bytes),
                        Err(e) => {
                            warn!("Received invalid solution from prover {}: {}", prover_display, e);
                            send_result(
                                sender,
                                id,
                                false,
                                Some(ErrorCode::from_code(23)),
                                Some("Invalid solution".to_string()),
                            )
                            .await;
                            return;
                        }
                    };
                    // 如果证明的难度小于验证者的目标值，则返回错误消息
                    if proof_difficulty < prover_target {
                        warn!(
                            "Received solution with difficulty {} from prover {} (expected {})",
                            proof_difficulty, prover_display, prover_target
                        );
                        send_result(
                            sender,
                            id,
                            false,
                            Some(ErrorCode::from_code(23)),
                            Some("Difficulty target not met".to_string()),
                        )
                        .await;
                        return;
                    }
                    // 构造验证者多项式
                    let polynomial = match prover_polynomial(&epoch_challenge, pool_address, nonce) {
                        Ok(polynomial) => polynomial,
                        Err(e) => {
                            warn!(
                                "Failed to construct prover polynomial from prover {}: {}",
                                prover_display, e
                            );
                            send_result(
                                sender,
                                id,
                                false,
                                Some(ErrorCode::from_code(20)),
                                Some("Invalid polynomial".to_string()),
                            )
                            .await;
                            return;
                        }
                    };
                    // 对提交的证明进行哈希
                    let point = match hash_commitment(&commitment) {
                        Ok(point) => point,
                        Err(e) => {
                            warn!("Failed to hash commitment from prover {}: {}", prover_display, e);
                            send_result(
                                sender,
                                id,
                                false,
                                Some(ErrorCode::from_code(20)),
                                Some("Invalid commitment".to_string()),
                            )
                            .await;
                            return;
                        }
                    };
                    // 验证提交的证明
                    let product_eval_at_point =
                        polynomial.evaluate(point) * epoch_challenge.epoch_polynomial().evaluate(point);
                    match KZG10::check(
                        coinbase_puzzle.coinbase_verifying_key(),
                        &commitment,
                        point,
                        product_eval_at_point,
                        &proof,
                    ) {
                        Ok(true) => {
                            debug!("Verified proof from prover {}", prover_display);
                        }
                        _ => {
                            warn!("Failed to verify proof from prover {}", prover_display);
                            send_result(
                                sender,
                                id,
                                false,
                                Some(ErrorCode::from_code(20)),
                                Some("Invalid proof".to_string()),
                            )
                            .await;
                            return;
                        }
                    }
                    // 更新验证者的份额
                    prover_state.write().await.add_share(prover_target).await;
                    // 更新池的份额
                    pool_state.write().await.add_share(prover_target).await;
                    // 发送会计消息
                    if let Err(e) = accounting_sender
                        .send(AccountingMessage::NewShare(
                            prover_state.read().await.address().to_string(),
                            proof_difficulty.min(global_proof_target * 2),
                        ))
                        .await
                    {
                        error!("Failed to send accounting message: {}", e);
                    }
                    // 发送成功结果给验证者
                    send_result(sender, id, true, None, None).await;
                    drop(provers);
                    drop(states);
                    debug!(
                        "Received valid proof from prover {} with difficulty {}",
                        prover_display, proof_difficulty
                    );
                    // TODO: testnet3 rewards
                    // 发送未确认的解决方案给操作方
                    if proof_difficulty >= global_proof_target {
                        info!(
                            "Received unconfirmed solution from prover {} with difficulty {} (target {})",
                            prover_display, proof_difficulty, global_proof_target
                        );
                        // TODO: dummy operator
                        // if let Err(e) = validator_sender
                        //     .send(SnarkOSMessage::UnconfirmedSolution(UnconfirmedSolution {
                        //         puzzle_commitment: PuzzleCommitment::new(commitment),
                        //         solution: Data::Object(ProverSolution::<Testnet3>::new(
                        //             PartialSolution::<Testnet3>::new(pool_address, nonce, commitment),
                        //             proof,
                        //         )),
                        //     }))
                        //     .await
                        // {
                        //     error!("Failed to report unconfirmed block to operator: {}", e);
                        // }
                        if let Err(e) = validator_sender
                            .send("xiaoyu1998 receive a solution".to_string())
                            .await
                        {
                            error!("Failed to report unconfirmed block to operator: {}", e);
                        }
                        if let Err(e) = {
                            accounting_sender
                                .send(AccountingMessage::NewSolution(PuzzleCommitment::new(commitment)))
                                .await
                        } {
                            error!("Failed to send accounting message: {}", e);
                        }
                    }
                });
            }
            ServerMessage::Exit => {}
        }
    }


    pub async fn online_provers(&self) -> u32 {
        // 获取在线验证者数量
        self.authenticated_provers.read().await.len() as u32
    }

    pub async fn online_addresses(&self) -> u32 {
        // 获取在线地址数量
        self.prover_address_connections.read().await.len() as u32
    }

    pub async fn pool_speed(&self) -> Vec<f64> {
        // 获取所有在线验证者的速度
        self.pool_state.write().await.speed().await
    }

    pub async fn address_prover_count(&self, address: Address<Testnet3>) -> u32 {
        // 获取指定地址上的在线验证者数量
        self.prover_address_connections
            .read()
            .await
            .get(&address)
            .map(|prover_connections| prover_connections.len() as u32)
            .unwrap_or(0)
    }

    pub async fn address_speed(&self, address: Address<Testnet3>) -> Vec<f64> {
        // 获取指定地址上所有在线验证者的平均速度
        let mut speed = vec![0.0, 0.0, 0.0, 0.0];
        let prover_connections_lock = self.prover_address_connections.read().await;
        let prover_connections = prover_connections_lock.get(&address);
        if prover_connections.is_none() {
            return speed;
        }
        for prover_connection in prover_connections.unwrap() {
            if let Some(prover_state) = self.prover_states.read().await.get(prover_connection) {
                let mut prover_state_lock = prover_state.write().await;
                prover_state_lock
                    .speed()
                    .await
                    .iter()
                    .zip(speed.iter_mut())
                    .for_each(|(s, speed)| {
                        *speed += s;
                    });
            }
        }
        speed
    }
}

fn prover_polynomial(
    epoch_challenge: &EpochChallenge<Testnet3>,  // 传入当前的 EpochChallenge 对象
    address: Address<Testnet3>,  // 验证者的地址
    nonce: u64,  // 随机数，用于生成验证者多项式
) -> anyhow::Result<DensePolynomial<<<Testnet3 as Environment>::PairingCurve as PairingEngine>::Fr>> {
    // 返回值为 Result 类型，包含 DensePolynomial 对象和错误信息
    let input = {
        let mut bytes = [0u8; 76];  // 定义一个长度为 76 的字节数组
        bytes[..4].copy_from_slice(&epoch_challenge.epoch_number().to_bytes_le()?);  // 将 EpochChallenge 的 epoch_number 转为小端序字节数组并复制到前四个字节
        bytes[4..36].copy_from_slice(&epoch_challenge.epoch_block_hash().to_bytes_le()?);  // 将 EpochChallenge 的 epoch_block_hash 转为小端序字节数组并复制到第 4 到 35 个字节
        bytes[36..68].copy_from_slice(&address.to_bytes_le()?);  // 将验证者地址转为小端序字节数组并复制到第 36 到 67 个字节
        bytes[68..].copy_from_slice(&nonce.to_le_bytes());  // 将随机数转为小端序字节数组并复制到第 68 到 75 个字节
        bytes
    };
    Ok(hash_to_polynomial::<
        <<Testnet3 as Environment>::PairingCurve as PairingEngine>::Fr,
    >(&input, epoch_challenge.degree()))  // 使用 hash_to_polynomial 函数生成验证者多项式
}

// 将输入哈希成一个系数向量，然后构造一个密度多项式并返回
fn hash_to_polynomial<F: PrimeField>(input: &[u8], degree: u32) -> DensePolynomial<F> {
    // Hash the input into coefficients.
    // 哈希输入得到系数向量
    let coefficients = hash_to_coefficients(input, degree + 1);
    // Construct the polynomial from the coefficients.
    // 通过系数向量构造密度多项式
    DensePolynomial::from_coefficients_vec(coefficients)
}

// 将输入哈希成一个长度为 num_coefficients 的系数向量并返回
fn hash_to_coefficients<F: PrimeField>(input: &[u8], num_coefficients: u32) -> Vec<F> {
    // Hash the input.
    let hash = blake2::Blake2s256::digest(input);
    // Hash with a counter and return the coefficients.
    cfg_into_iter!(0..num_coefficients)
        .map(|counter| {
            let mut input_with_counter = [0u8; 36];
            input_with_counter[..32].copy_from_slice(&hash);
            input_with_counter[32..].copy_from_slice(&counter.to_le_bytes());
            F::from_bytes_le_mod_order(&blake2::Blake2b512::digest(input_with_counter))
        })
        .collect()
}

// 对 KZGCommitment 进行哈希，返回哈希值
fn hash_commitment<E: PairingEngine>(commitment: &KZGCommitment<E>) -> anyhow::Result<E::Fr> {
    // Convert the commitment into bytes.
    let mut bytes = Vec::with_capacity(96);
    commitment.serialize_uncompressed(&mut bytes)?;
    ensure!(bytes.len() == 96, "Invalid commitment byte length for hashing");

    // Return the hash of the commitment.
    Ok(E::Fr::from_bytes_le_mod_order(&blake2::Blake2b512::digest(&bytes)))
}
