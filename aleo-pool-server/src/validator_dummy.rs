use std::{
    sync::{
        Arc,
    },
    time::Duration,
};

use rand::{rngs::OsRng, Rng};
use snarkvm::{
    prelude::{Testnet3, Network},
    ledger::coinbase::EpochChallenge,
};

use tokio::{
    sync::{
        mpsc,
        mpsc::{Receiver, Sender},
        Mutex,
    },
    task,
    time::sleep,
};
use tracing::{debug, error, info, trace, warn};

use crate::ServerMessage;

pub struct Node {
    operator: String, //用于保存节点操作者名称的字符串
    sender: Arc<Sender<String>>, //一个用于发送消息的 Sender 实例的 Arc 引用计数器
    receiver: Arc<Mutex<Receiver<String>>>, //一个用于接收消息的 Receiver 实例的 Mutex 包装的 Arc 引用计数器
}

impl Node {
    // 初始化节点，接收操作者名称
    pub fn init(operator: String) -> Self {
        let (sender, receiver) = mpsc::channel(1024); // 创建用于发送和接收消息的通道
        Self {
            operator, // 节点操作者名称
            sender: Arc::new(sender), // 发送者通道的Arc引用
            receiver: Arc::new(Mutex::new(receiver)), // 接收者通道的Mutex包装的Arc引用
        }
    }

    // 返回接收者通道的Arc引用
    pub fn receiver(&self) -> Arc<Mutex<Receiver<String>>> {
        self.receiver.clone()
    }

    // 返回发送者通道的Arc引用
    pub fn sender(&self) -> Arc<Sender<String>> {
        self.sender.clone()
    }
}

// 定义SnarkOSMessage类型别名
pub(crate) type SnarkOSMessage = snarkos_node_router_messages::Message<Testnet3>;

// 启动节点
pub fn start(node: Node, server_sender: Sender<ServerMessage>) {
    let receiver = node.receiver(); // 获取接收者通道的Arc引用
    task::spawn(async move {

        // 启动周期性任务，发送新的EpochChallenge给矿池服务器
        task::spawn(async move {
            let mut epoch_number = 100; // 从100号Epoch开始
            let proof_target = 100; // 设置目标证明数

            loop {
                epoch_number += 1; // 每次循环增加1

                let rng = &mut OsRng; // 创建一个操作系统随机数生成器
                let epoch_block_hash: <Testnet3 as Network>::BlockHash = rng.gen(); // 生成一个随机的Epoch块哈希
                let epoch_challenge = EpochChallenge::<Testnet3>::new(epoch_number, epoch_block_hash, (1 << 10) - 1 ).unwrap(); // 创建新的EpochChallenge

                // 将新的EpochChallenge发送给矿池服务器
                if let Err(e) = server_sender.send(ServerMessage::NewEpochChallenge(
                    epoch_challenge, proof_target
                )).await {
                    error!("Error sending new block template to pool server: {}", e);
                } else {
                    trace!("Sent new epoch challenge {} to pool server", epoch_number);
                }

                sleep(Duration::from_secs(120)).await; // 等待120秒，即两分钟
            }
        });

        loop {
            let receiver = &mut *receiver.lock().await; // 获取接收者通道的锁，并获取可变引用
            loop {
                tokio::select! {
                    Some(message) = receiver.recv() => { // 接收到消息时执行以下代码块
                        trace!("dummy Sending {} to validator {}", message, node.operator);
                    }

                }
            }
        }
    });
}
