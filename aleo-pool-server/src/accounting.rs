use std::{
    collections::{HashMap, VecDeque},
    fs::create_dir_all,
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
    time::{Duration, Instant},
};

use anyhow::{anyhow, Error, Result};
use cache::Cache;
use dirs::home_dir;
use parking_lot::RwLock;
use savefile::{load_file, save_file};
use savefile_derive::Savefile;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
// use snarkvm::prelude::{PuzzleCommitment, Testnet3};
use snarkvm::prelude::{Testnet3};
use snarkvm::prelude::coinbase::{PuzzleCommitment};

use tokio::{
    sync::{
        mpsc::{channel, Sender},
        RwLock as TokioRwLock,
    },
    task,
    time::sleep,
};
use tracing::{debug, error, info};

#[cfg(feature = "db")]
use crate::db::DB;
use crate::{
    accounting::AccountingMessage::{NewShare, NewSolution},
    AccountingMessage::{Exit, SetN},
};

// 定义一个 PayoutModel trait，包含 add_share 方法。
trait PayoutModel {
    fn add_share(&mut self, share: Share);
}

// 定义 Share 结构体，用于保存共享数据。
#[derive(Clone, Savefile)]
struct Share {
    value: u64, // 共享数据的数值
    owner: String, // 共享数据的拥有者
}

impl Share {
    // 实现 Share 结构体的初始化方法，返回一个新的 Share 实例。
    pub fn init(value: u64, owner: String) -> Self {
        Share { value, owner }
    }
}

// 定义 PPLNS 结构体，实现 PayoutModel trait，用于实现 PPLNS 分配模型。
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Savefile)]
struct PPLNS {
    queue: VecDeque<Share>, // 保存共享数据的队列
    current_n: Arc<RwLock<u64>>, // 当前分配数量的读写锁
    n: Arc<RwLock<u64>>, // 总分配数量的读写锁
}

impl PPLNS {
    // 创建新的 PPLNS 实例，并从本地磁盘中加载已保存的数据，如果没有数据，则返回初始状态。
    pub fn load() -> Self {
        let home = home_dir(); // 获取当前用户的主目录
        if home.is_none() {
            panic!("No home directory found");
        }
        create_dir_all(home.as_ref().unwrap().join(".aleo_pool_testnet3_2")).unwrap(); // 创建存储数据的目录
        let db_path = home.unwrap().join(".aleo_pool_testnet3_2/state"); // 获取存储数据的文件路径
        if !db_path.exists() { // 如果文件不存在，则返回初始状态的 PPLNS 实例
            return PPLNS {
                queue: VecDeque::new(),
                current_n: Default::default(),
                n: Default::default(),
            };
        }
        load_file::<PPLNS, PathBuf>(db_path, 0).unwrap() // 否则从本地磁盘中加载已保存的数据，并返回 PPLNS 实例
    }

    // 将当前 PPLNS 实例的状态保存到本地磁盘中。
    pub fn save(&self) -> std::result::Result<(), Error> {
        let home = home_dir(); // 获取当前用户的主目录
        if home.is_none() {
            panic!("No home directory found");
        }
        let db_path = home.unwrap().join(".aleo_pool_testnet3_2/state"); // 获取存储数据的文件路径
        save_file(db_path, 0, self).map_err(|e| anyhow!("Failed to save PPLNS state: {}", e)) // 将 PPLNS 实例的状态保存到本地磁盘中
    }

    // 设置总分配数量，并根据已有共享数据情况，更新当前分配数量。
    pub fn set_n(&mut self, n: u64) {
        let start = Instant::now();
        let mut current_n = self.current_n.write(); // 获取当前分配数量的写锁
        let mut self_n = self.n.write(); // 获取总分配数量的写锁
        if n < *self_n { // 如果新设置的总分配数量小于旧分配数量
            while *current_n > n { // 从当前分配数量开始，循环移除共享数据
                let share = self.queue.pop_front().unwrap(); // 移除队列中的第一个共享数据，并返回该数据
                *current_n -= share.value; // 减去共享数据的值
            }
        }
        *self_n = n; // 更新总分配数量
        debug!("set_n took {} us", start.elapsed().as_micros());
    }
}

// 实现 PayoutModel trait 中的 add_share 方法，用于添加共享数据，并根据当前分配数量和总分配数量的比较，自动移除旧的共享数据。
impl PayoutModel for PPLNS {
    fn add_share(&mut self, share: Share) {
        let start = Instant::now();
        self.queue.push_back(share.clone()); // 将共享数据添加到队列尾部
        let mut current_n = self.current_n.write(); // 获取当前分配数量的写锁
        let self_n = self.n.read(); // 获取总分配数量的读锁
        *current_n += share.value; // 更新当前分配数量
        while *current_n > *self_n { // 当当前分配数量大于总分配数量时，循环移除队列头部的共享数据
            let share = self.queue.pop_front().unwrap(); // 移除队列头部的共享数据，并返回该数据
            *current_n -= share.value; // 减去共享数据的值
        }
        debug!("add_share took {} us", start.elapsed().as_micros()); // 记录运行时间
        debug!("n: {} / {}", *current_n, self_n); // 输出当前分配数量和总分配数量
    }
}

// 定义 Null 结构体，用于缓存当前轮次的数据。
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
struct Null {}

// 定义 AccountingMessage 枚举类型，包含 NewShare、SetN、NewSolution 和 Exit 四个成员。
pub enum AccountingMessage {
    NewShare(String, u64), // 添加新的共享数据
    SetN(u64), // 设置总分配数量
    NewSolution(PuzzleCommitment<Testnet3>), // 发现新的解决方案
    Exit, // 退出程序
}

#[cfg(feature = "db")]
static PAY_INTERVAL: Duration = Duration::from_secs(60); // 定义 PAY_INTERVAL 常量，表示支付间隔为 60 秒。

// 定义 Accounting 结构体，用于管理共享数据、分配模型和支付方式等。
#[allow(clippy::type_complexity)]
pub struct Accounting {
    pplns: Arc<TokioRwLock<PPLNS>>, // 共享数据的读写锁
    #[cfg(feature = "db")]
    database: Arc<DB>, // 数据库实例
    sender: Sender<AccountingMessage>, // AccountingMessage 的发送者
    round_cache: TokioRwLock<Cache<Null, (u32, HashMap<String, u64>)>>, // 缓存当前轮次的数据
    exit_lock: Arc<AtomicBool>, // 程序退出的原子锁
}

//结构体 Accounting 实现。该结构体用于记录和处理矿池中的分享和解，并根据 PPLNS 算法来计算每个矿工的收益。
//如果开启了数据库功能，它还可以将解储存到数据库中，并定期进行支付循环来向矿工支付收益。
//该结构体提供了一些方法来获取当前轮的信息、获取通道发送器和等待退出信号等。
impl Accounting {
    // 初始化函数
    pub fn init() -> Arc<Accounting> {
        // 如果开启了数据库功能，则初始化数据库
        #[cfg(feature = "db")]
            let database = Arc::new(DB::init());

        // 初始化 pplns（PPLNS 算法） 和通道
        let pplns = Arc::new(TokioRwLock::new(PPLNS::load()));
        let (sender, mut receiver) = channel(1024);

        // 创建 accounting 实例
        let accounting = Accounting {
            pplns,
            #[cfg(feature = "db")]
            database,
            sender,
            round_cache: TokioRwLock::new(Cache::new(Duration::from_secs(10))),
            exit_lock: Arc::new(AtomicBool::new(false)),
        };

        // 处理来自通道的数据
        let pplns = accounting.pplns.clone();
        #[cfg(feature = "db")]
            let database = accounting.database.clone();
        let exit_lock = accounting.exit_lock.clone();
        task::spawn(async move {
            while let Some(request) = receiver.recv().await {
                match request {
                    // 如果是新的分享，则添加到 pplns 中
                    NewShare(address, value) => {
                        pplns.write().await.add_share(Share::init(value, address.clone()));
                        debug!("Recorded share from {} with value {}", address, value);
                    }
                    // 如果是设置 n 值，则设置 pplns 的 n 值
                    SetN(n) => {
                        pplns.write().await.set_n(n);
                        debug!("Set N to {}", n);
                    }
                    // 如果是新的解，则将该解的信息储存到数据库中（如果有开启数据库功能）
                    NewSolution(commitment) => {
                        let pplns = pplns.read().await.clone();
                        let (_, address_shares) = Accounting::pplns_to_provers_shares(&pplns);

                        #[cfg(feature = "db")]
                        if let Err(e) = database.save_solution(commitment, address_shares).await {
                            error!("Failed to save block reward : {}", e);
                        } else {
                            info!("Recorded solution {}", commitment);
                        }
                    }
                    // 如果是退出信号，则关闭通道，保存 pplns 并设置 exit_lock 标志位
                    Exit => {
                        receiver.close();
                        let _ = pplns.read().await.save();
                        exit_lock.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
        });

        // 定时备份 pplns（每间隔 60 秒）
        // backup pplns
        let pplns = accounting.pplns.clone();
        task::spawn(async move {
            loop {
                sleep(Duration::from_secs(60)).await;
                if let Err(e) = pplns.read().await.save() {
                    error!("Unable to backup pplns: {}", e);
                }
            }
        });

        let res = Arc::new(accounting);

        // 如果开启了数据库功能，则开启支付循环
        // payout routine
        #[cfg(feature = "db")]
        task::spawn(Accounting::payout_loop(res.clone()));

        res
    }

    // 获取通道发送器
    pub fn sender(&self) -> Sender<AccountingMessage> {
        self.sender.clone()
    }

    // 等待退出信号
    pub async fn wait_for_exit(&self) {
        while !self.exit_lock.load(std::sync::atomic::Ordering::SeqCst) {
            sleep(Duration::from_millis(100)).await;
        }
    }

    // 将当前的 pplns 转换为所需的格式，并返回
    fn pplns_to_provers_shares(pplns: &PPLNS) -> (u32, HashMap<String, u64>) {
        // 生成空的地址-份额哈希表
        let mut address_shares = HashMap::new();

        // 遍历 pplns 的队列，将每个分享加入对应的地址份额中
        let time = Instant::now();
        pplns.queue.iter().for_each(|share| {
            if let Some(shares) = address_shares.get_mut(&share.owner) {
                *shares += share.value;
            } else {
                address_shares.insert(share.clone().owner, share.value);
            }
        });
        debug!("PPLNS to Provers shares took {} us", time.elapsed().as_micros());

        // 返回地址份额哈希表以及地址数量
        (address_shares.len() as u32, address_shares)
    }

    // 获取当前轮的信息
    pub async fn current_round(&self) -> Value {
        let pplns = self.pplns.clone().read().await.clone();
        let cache = self.round_cache.read().await.get(Null {});
        let (provers, shares) = match cache {
            // 如果已经缓存了当前轮的信息，则从缓存中获取
            Some(cache) => cache,
            // 如果没有缓存则重新计算，并存入缓存中
            None => {
                let result = Accounting::pplns_to_provers_shares(&pplns);
                self.round_cache.write().await.set(Null {}, result.clone());
                result
            }
        };
        // 返回当前轮信息
        json!({
            "n": pplns.n,
            "current_n": pplns.current_n,
            "provers": provers,
            "shares": shares,
        })
    }

    // 检查一个解是否合法，并将结果储存到数据库中（如果有开启数据库功能）
    #[cfg(feature = "db")]
    async fn check_solution(&self, commitment: &String) -> Result<bool> {
        let client = reqwest::Client::new();

        let result = &client
            .get(format!("http://127.0.0.1:8001/commitment?commitment={}", commitment))
            .send()
            .await?
            .json::<Value>()
            .await?;
        let is_valid = result.as_null().is_none();
        if is_valid {
            self.database
                .set_solution_valid(
                    commitment,
                    true,
                    Some(result["height"].as_u64().ok_or_else(|| anyhow!("height"))? as u32),
                    Some(result["reward"].as_u64().ok_or_else(|| anyhow!("reward"))?),
                )
                .await?;
        } else {
            self.database.set_solution_valid(commitment, false, None, None).await?;
        }
        Ok(is_valid)
    }

    // 支付循环（如果有开启数据库功能）
    #[cfg(feature = "db")]
    async fn payout_loop(self: Arc<Accounting>) {
        'forever: loop {
            info!("Running payout loop");
            // 获取应支付的解列表
            let blocks = self.database.get_should_pay_solutions().await;
            if blocks.is_err() {
                error!("Unable to get should pay blocks: {}", blocks.unwrap_err());
                sleep(PAY_INTERVAL).await;
                continue;
            }
            for (id, commitment) in blocks.unwrap() {
                // 检查解是否合法
                let valid = self.check_solution(&commitment).await;
                if valid.is_err() {
                    error!("Unable to check solution: {}", valid.unwrap_err());
                    sleep(PAY_INTERVAL).await;
                    continue 'forever;
                }
                let valid = valid.unwrap();
                // 如果解合法，则支付
                if valid {
                    match self.database.pay_solution(id).await {
                        Ok(_) => {
                            info!("Paid solution {}", commitment);
                        }
                        Err(e) => {
                            error!("Unable to pay solution {}: {}", id, e);
                            sleep(PAY_INTERVAL).await;
                            continue 'forever;
                        }
                    }
                }
            }

            // 等待支付循环的下一次执行时间
            sleep(PAY_INTERVAL).await;
        }
    }
}
