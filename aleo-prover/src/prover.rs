use std::{
    collections::VecDeque,
    str::FromStr,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use aleo_stratum::message::StratumMessage;
use ansi_term::Colour::{Cyan, Green, Red};
use anyhow::Result;
use json_rpc_types::Id;
use rand::{thread_rng, RngCore};
use rayon::{ThreadPool, ThreadPoolBuilder};
use snarkvm::{
    console::types::Address,
    prelude::{Testnet3, FromBytes, ToBytes},
    synthesizer::snark::UniversalSRS,
    ledger::coinbase::{CoinbasePuzzle, EpochChallenge, PuzzleConfig},
};
use snarkvm_algorithms::crypto_hash::sha256d_to_u64;
use tokio::{sync::mpsc, task};
use tracing::{debug, error, info, warn};

// use cactus_timer::{start_timer, end_timer};

use crate::client::Client;

pub struct Prover {
    thread_pools: Arc<Vec<Arc<ThreadPool>>>,
    cuda: Option<Vec<i16>>,
    _cuda_jobs: Option<u8>,
    sender: Arc<mpsc::Sender<ProverEvent>>,
    client: Arc<Client>,
    current_epoch: Arc<AtomicU32>,
    total_proofs: Arc<AtomicU32>,
    valid_shares: Arc<AtomicU32>,
    invalid_shares: Arc<AtomicU32>,
    current_proof_target: Arc<AtomicU64>,
    coinbase_puzzle: CoinbasePuzzle<Testnet3>,
}

#[allow(clippy::large_enum_variant)]
pub enum ProverEvent {
    NewTarget(u64),
    NewWork(u32, String, String),
    Result(bool, Option<String>),
}

impl Prover {
    pub async fn init(
        threads: u16,                     // 线程数
        thread_pool_size: u8,             // 线程池大小
        client: Arc<Client>,              // 客户端
        cuda: Option<Vec<i16>>,           // CUDA GPU 数组（可选）
        cuda_jobs: Option<u8>,            // CUDA 任务数（可选）
    ) -> Result<Arc<Self>> {
        let mut thread_pools: Vec<Arc<ThreadPool>> = Vec::new();  // 线程池数组
        let pool_count;                                           // 线程池数量
        let pool_threads;                                         // 每个线程池的线程数
        if cuda.is_none() {                                       // 如果没有使用 CUDA
            if threads < thread_pool_size as u16 {                // 如果线程数小于线程池大小
                pool_count = 1;                                   // 线程池数量为1
                pool_threads = thread_pool_size as u16;           // 每个线程池的线程数为线程池大小
            } else {
                pool_count = threads / thread_pool_size as u16;   // 线程池数量为线程数除以线程池大小
                pool_threads = thread_pool_size as u16;           // 每个线程池的线程数为线程池大小
            }
        } else {                                                  // 如果使用了 CUDA
            pool_threads = thread_pool_size as u16;               // 每个线程池的线程数为线程池大小
            pool_count = (cuda_jobs.unwrap_or(1) * cuda.clone().unwrap().len() as u8) as u16;  // 线程池数量为 CUDA 任务数乘以 CUDA GPU 数组的长度
        }

        // 遍历从 0 到 pool_count 的范围
        for index in 0..pool_count {
            
            // 创建一个线程池构建器
            let builder = ThreadPoolBuilder::new()
                .stack_size(8 * 1024 * 1024)  // 设置线程栈大小为 8MB
                .num_threads(pool_threads as usize);  // 设置线程数为 pool_threads
            
            // 根据是否存在 cuda，选择不同的线程名
            let pool = if cuda.is_none() {
                // 没有 cuda，使用 "ap-cpu-{}-{}" 格式的线程名
                builder.thread_name(move |idx| format!("ap-cpu-{}-{}", index, idx))
            } else {
                // 存在 cuda，使用 "ap-cuda-{}-{}" 格式的线程名
                builder.thread_name(move |idx| format!("ap-cuda-{}-{}", index, idx))
            }
            .build()?;  // 构建线程池
            
            // 将构建好的线程池放入 thread_pools 向量中，并使用 Arc 进行引用计数
            thread_pools.push(Arc::new(pool));
        }


        // 输出线程池信息，包括线程池数量和每个线程池中的线程数
        info!(
            "Created {} prover thread pools with {} threads in each pool",
            thread_pools.len(),
            pool_threads
        );

        // 创建一个大小为 1024 的消息发送器和一个消息接收器
        let (sender, mut receiver) = mpsc::channel(1024);

        // 输出正在初始化 universal SRS 的信息
        info!("Initializing universal SRS");

        // 加载 universal SRS，并将其赋值给变量 srs
        let srs = UniversalSRS::<Testnet3>::load().expect("Failed to load SRS");

        // 输出 universal SRS 初始化完成的信息
        info!("Universal SRS initialized");

        // 输出正在初始化 coinbase proving key 的信息
        info!("Initializing coinbase proving key");

        // 基于 srs 创建一个 CoinbasePuzzle 实例（即 coinbase proving key）
        // 并根据给定的 puzzle 配置进行修剪
        let coinbase_puzzle = CoinbasePuzzle::<Testnet3>::trim(&srs, PuzzleConfig { degree: (1 << 10) - 1 })
            .expect("Failed to load coinbase proving key");

        // 输出 coinbase proving key 初始化完成的信息
        info!("Coinbase proving key initialized");


        let prover = Arc::new(Self {
            thread_pools: Arc::new(thread_pools),
            cuda,
            _cuda_jobs: cuda_jobs,
            sender: Arc::new(sender),
            client,
            current_epoch: Default::default(),
            total_proofs: Default::default(),
            valid_shares: Default::default(),
            invalid_shares: Default::default(),
            current_proof_target: Default::default(),
            coinbase_puzzle,
        });

        // 创建一个 prover 的克隆，并将其赋值给变量 p
        let p = prover.clone();

        // 使用 task::spawn 创建一个异步任务
        let _ = task::spawn(async move {
            // 当从接收器接收到消息时，进行循环处理
            while let Some(msg) = receiver.recv().await {
                match msg {
                    ProverEvent::NewTarget(target) => {
                        // 如果收到的消息是 NewTarget，则调用 p 的 new_target 方法
                        p.new_target(target);
                    }
                    ProverEvent::NewWork(epoch_number, epoch_challenge, address) => {
                        // 如果收到的消息是 NewWork，则输出相应的信息，并进行一些处理
                        info!("epoch {} {}", epoch_number, epoch_challenge);

                        // 将 epoch_challenge 的字符串形式转换为字节数组，并解码为 EpochChallenge 类型
                        let epoch_challenge = EpochChallenge::<Testnet3>::from_bytes_le(
                            &*hex::decode(epoch_challenge.as_bytes()).unwrap(),
                        ).unwrap();

                        // 将 epoch_number、epoch_challenge 和 address 转换为相应的类型，并调用 p 的 new_work 方法
                        p.new_work(
                            epoch_number,
                            epoch_challenge,
                            Address::<Testnet3>::from_str(&address).unwrap(),
                        ).await;
                    }
                    ProverEvent::Result(success, error) => {
                        // 如果收到的消息是 Result，则调用 p 的 result 方法
                        p.result(success, error).await;
                    }
                }
            }
        });

        // 输出相应的调试信息
        debug!("Created prover message handler");


        // 克隆 total_proofs，并将其赋值给 total_proofs 变量
        let total_proofs = prover.total_proofs.clone();

        // 使用 task::spawn 创建一个异步任务
        let _ = task::spawn(async move {
            // 定义一个计算证明率的函数，接受当前数量、过去数量和时间间隔作为参数，并返回相应的字符串
            fn calculate_proof_rate(now: u32, past: u32, interval: u32) -> Box<str> {
                // 如果时间间隔小于1，则返回"---"
                if interval < 1 {
                    return Box::from("---");
                }
                // 如果当前数量小于等于过去数量，或者过去数量为0，则返回"---"
                if now <= past || past == 0 {
                    return Box::from("---");
                }
                // 计算证明率
                let rate = (now - past) as f64 / (interval * 60) as f64;
                Box::from(format!("{:.2}", rate))
            }

            // 创建一个长度为60的循环队列 log，并将每个元素初始化为0
            let mut log = VecDeque::<u32>::from(vec![0; 60]);

            // 进入无限循环
            loop {
                // 每隔60秒执行一次
                tokio::time::sleep(Duration::from_secs(60)).await;

                // 获取当前的总证明数量
                let proofs = total_proofs.load(Ordering::SeqCst);

                // 将当前的总证明数量添加到 log 的末尾
                log.push_back(proofs);

                // 获取过去60秒、55秒、45秒、30秒和1分钟的证明数量
                let m1 = *log.get(59).unwrap_or(&0);
                let m5 = *log.get(55).unwrap_or(&0);
                let m15 = *log.get(45).unwrap_or(&0);
                let m30 = *log.get(30).unwrap_or(&0);
                let m60 = log.pop_front().unwrap_or_default();

                // 输出相应的信息，包括总证明数量以及各个时间段的证明率
                info!(
                    "{}",
                    Cyan.normal().paint(format!(
                        "Total solutions: {} (1m: {} c/s, 5m: {} c/s, 15m: {} c/s, 30m: {} c/s, 60m: {} c/s)",
                        proofs,
                        calculate_proof_rate(proofs, m1, 1),
                        calculate_proof_rate(proofs, m5, 5),
                        calculate_proof_rate(proofs, m15, 15),
                        calculate_proof_rate(proofs, m30, 30),
                        calculate_proof_rate(proofs, m60, 60),
                    ))
                );
            }
        });
        debug!("Created proof rate calculator");

        Ok(prover)
    }

    pub fn sender(&self) -> Arc<mpsc::Sender<ProverEvent>> {
        self.sender.clone()
    }

    // 定义一个异步函数 result，接受 success 和 msg 作为参数
    async fn result(&self, success: bool, msg: Option<String>) {
        // 如果 success 为真
        if success {
            // 将 valid_shares 的值增加1，并将结果赋值给 valid_minus_1
            let valid_minus_1 = self.valid_shares.fetch_add(1, Ordering::SeqCst);
            // 计算 valid 的值为 valid_minus_1 + 1
            let valid = valid_minus_1 + 1;
            // 获取 invalid_shares 的值
            let invalid = self.invalid_shares.load(Ordering::SeqCst);
            // 如果 msg 存在
            if let Some(msg) = msg {
                // 输出分享被接受的信息，包括分享消息、有效分享数量和总分享数量的百分比
                info!(
                    "{}",
                    Green.normal().paint(format!(
                        "Share accepted: {}  {} / {} ({:.2}%)",
                        msg,
                        valid,
                        valid + invalid,
                        (valid as f64 / (valid + invalid) as f64) * 100.0
                    ))
                );
            } else {
                // 输出分享被接受的信息，包括有效分享数量和总分享数量的百分比
                info!(
                    "{}",
                    Green.normal().paint(format!(
                        "Share accepted  {} / {} ({:.2}%)",
                        valid,
                        valid + invalid,
                        (valid as f64 / (valid + invalid) as f64) * 100.0
                    ))
                );
            }
        } else {
            // 将 invalid_shares 的值增加1，并将结果赋值给 invalid_minus_1
            let invalid_minus_1 = self.invalid_shares.fetch_add(1, Ordering::SeqCst);
            // 计算 invalid 的值为 invalid_minus_1 + 1
            let invalid = invalid_minus_1 + 1;
            // 获取 valid_shares 的值
            let valid = self.valid_shares.load(Ordering::SeqCst);
            // 如果 msg 存在
            if let Some(msg) = msg {
                // 输出分享被拒绝的信息，包括分享消息、有效分享数量和总分享数量的百分比
                info!(
                    "{}",
                    Red.normal().paint(format!(
                        "Share rejected: {}  {} / {} ({:.2}%)",
                        msg,
                        valid,
                        valid + invalid,
                        (valid as f64 / (valid + invalid) as f64) * 100.0
                    ))
                );
            } else {
                // 输出分享被拒绝的信息，包括有效分享数量和总分享数量的百分比
                info!(
                    "{}",
                    Red.normal().paint(format!(
                        "Share rejected  {} / {} ({:.2}%)",
                        valid,
                        valid + invalid,
                        (valid as f64 / (valid + invalid) as f64) * 100.0
                    ))
                );
            }
        }
    }

    fn new_target(&self, proof_target: u64) {
        self.current_proof_target.store(proof_target, Ordering::SeqCst);
        info!("New proof target: {}", proof_target);
    }

    async fn new_work(&self, epoch_number: u32, epoch_challenge: EpochChallenge<Testnet3>, address: Address<Testnet3>) {
        // 获取上一个 epoch 的编号
        let last_epoch_number = self.current_epoch.load(Ordering::SeqCst);
        // 如果新的 epoch 编号小于等于上一个 epoch 编号且不为0，则返回
        if epoch_number <= last_epoch_number && epoch_number != 0 {
            return;
        }
        // 存储当前的 epoch 编号
        self.current_epoch.store(epoch_number, Ordering::SeqCst);
        // 输出接收到新工作的信息，包括 epoch 编号
        info!("Received new work: epoch {}", epoch_number);
        // 克隆当前的 proof target
        let current_proof_target = self.current_proof_target.clone();
    
        // 复制各个变量以便在后续的异步任务中使用
        let current_epoch = self.current_epoch.clone();
        let client = self.client.clone();
        let thread_pools = self.thread_pools.clone();
        let total_proofs = self.total_proofs.clone();
        let cuda = self.cuda.clone();
        let coinbase_puzzle = self.coinbase_puzzle.clone();
    
        // 启动一个异步任务
        task::spawn(async move {
            // 启动一个子任务，用于创建多个线程进行工作
            let _ = task::spawn(async move {
                let mut joins = Vec::new();
                // 如果使用了 CUDA，只使用第一个 GPU，输出警告信息
                if let Some(_) = cuda {
                    warn!("This version of the prover is only using the first GPU");
                }
                // 遍历线程池
                for (_, tp) in thread_pools.iter().enumerate() {
                    // 克隆各个变量以便在每个线程中使用
                    let current_proof_target = current_proof_target.clone();
                    let current_epoch = current_epoch.clone();
                    let client = client.clone();
                    let epoch_challenge = epoch_challenge.clone();
                    let address = address.clone();
                    let total_proofs = total_proofs.clone();
                    let tp = tp.clone();
                    let coinbase_puzzle = coinbase_puzzle.clone();
                    // 启动一个异步任务
                    joins.push(task::spawn(async move {
                        loop {
                            // 克隆各个变量以便在每次循环中使用
                            let current_proof_target = current_proof_target.clone();
                            let epoch_challenge = epoch_challenge.clone();
                            let address = address.clone();
                            let tp = tp.clone();
                            let coinbase_puzzle = coinbase_puzzle.clone();
                            // 如果当前的 epoch 编号与传入的 epoch 编号不一致，则终止循环
                            if epoch_number != current_epoch.load(Ordering::SeqCst) {
                                debug!(
                                    "Terminating stale work: current {} latest {}",
                                    epoch_number,
                                    current_epoch.load(Ordering::SeqCst)
                                );
                                break;
                            }
                            // 生成随机数 nonce
                            let nonce = thread_rng().next_u64();
                            // 使用 spawn_blocking 函数在阻塞上下文中执行 CPU 密集型计算
                            if let Ok(Ok(solution)) = task::spawn_blocking(move || {
                                tp.install(|| {
                                    // 执行 coinbase_puzzle 的 prove 方法，生成 proof
                                    coinbase_puzzle.prove(
                                        &epoch_challenge,
                                        address,
                                        nonce,
                                        Option::from(current_proof_target.load(Ordering::SeqCst)),
                                    )
                                })
                            })
                            .await
                            {
                                // 如果当前的 epoch 编号与传入的 epoch 编号不一致，则终止循环
                                if epoch_number != current_epoch.load(Ordering::SeqCst) {
                                    debug!(
                                        "Terminating stale work: current {} latest {}",
                                        epoch_number,
                                        current_epoch.load(Ordering::SeqCst)
                                    );
                                    break;
                                }
                                // 确保满足分享难度目标
                                let proof_difficulty =
                                    u64::MAX / sha256d_to_u64(&*solution.commitment().to_bytes_le().unwrap());
    
                                // 输出找到解决方案的信息，包括当前的 epoch 编号和证明的难度
                                info!(
                                    "Solution found for epoch {} with difficulty {}",
                                    epoch_number, proof_difficulty
                                );
    
                                // 发送一个 `PoolResponse` 给操作员
                                let message = StratumMessage::Submit(
                                    Id::Num(0),
                                    client.address.to_string(),
                                    hex::encode(epoch_number.to_le_bytes()),
                                    hex::encode(nonce.to_bytes_le().unwrap()),
                                    hex::encode(solution.commitment().to_bytes_le().unwrap()),
                                    hex::encode(solution.proof().to_bytes_le().unwrap()),
                                );
                                if let Err(error) = client.sender().send(message).await {
                                    error!("Failed to send PoolResponse: {}", error);
                                }
                                // 增加总共的证明数量
                                total_proofs.fetch_add(1, Ordering::SeqCst);
                            } else {
                                total_proofs.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }));
                }
                // 等待所有子任务完成
                futures::future::join_all(joins).await;
            });
        });
    }    
}
