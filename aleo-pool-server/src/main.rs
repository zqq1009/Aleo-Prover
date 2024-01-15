mod accounting;
mod api;
mod connection;
mod server;
// mod validator_peer;
mod validator_dummy;

#[cfg(feature = "db")]
mod db;

use std::sync::Arc;

use clap::Parser;
use futures::stream::StreamExt;
use rand::seq::SliceRandom;
use signal_hook::consts::{SIGABRT, SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGTSTP, SIGUSR1};
use signal_hook_tokio::Signals;
use snarkvm::{
    prelude::Testnet3,
    console::types::Address,
};
use tokio::sync::mpsc::Sender;
use tracing::{debug, error, info, warn};
use tracing_log::{log, LogTracer};
use tracing_subscriber::{layer::SubscriberExt, EnvFilter};

// use crate::validator_peer::Node;
use crate::validator_dummy::Node;
use crate::{
    accounting::{Accounting, AccountingMessage},
    //    operator_peer::Node,
    server::{Server, ServerMessage},
};

#[derive(Debug, Parser)]
#[clap(name = "pool_server", about = "Aleo mining pool server")]
struct Opt {
    /// Validator node address
    #[clap(short, long)]
    validator: Option<String>,

    /// Mining pool address
    #[clap(short, long)]
    address: Address<Testnet3>,

    /// Port to listen for incoming provers
    #[clap(short, long)]
    port: u16,

    /// API port
    #[clap(short, long = "api-port")]
    api_port: u16,

    /// Enable debug logging
    #[clap(short, long)]
    debug: bool,

    /// Enable trace logging
    #[clap(short, long)]
    trace: bool,

    /// Output log to file
    #[clap(long)]
    log: Option<String>,
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();  // 加载环境变量文件

    let opt = Opt::parse();  // 解析命令行参数

    // 根据命令行参数设置日志级别
    let tracing_level = if opt.trace {
        tracing::Level::TRACE
    } else if opt.debug {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    let _ = LogTracer::init_with_filter(log::LevelFilter::Info);  // 初始化日志记录器

    let filter = EnvFilter::from_default_env()
        .add_directive(tracing_level.into())  // 设置追踪日志级别
        .add_directive("hyper=info".parse().unwrap())
        .add_directive("warp=info".parse().unwrap())
        .add_directive("warp=warn".parse().unwrap())
        .add_directive("tokio_util=info".parse().unwrap())
        .add_directive("api".parse().unwrap());

    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(filter)  // 使用过滤器设置日志级别
        .finish();
    // .with(
    //     tracing_subscriber::fmt::Layer::default()
    //         .with_ansi(true)
    //         .with_writer(std::io::stdout),
    // );
    // 将日志记录器设置为全局默认值
    if let Some(log) = opt.log {
        let file = std::fs::File::create(log).unwrap();
        let file = tracing_subscriber::fmt::layer().with_writer(file).with_ansi(false);
        tracing::subscriber::set_global_default(subscriber.with(file))
            .expect("unable to set global default subscriber");
    } else {
        tracing::subscriber::set_global_default(subscriber).expect("unable to set global default subscriber");
    }

    // 配置线程池
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)  // 设置线程栈大小
        .num_threads(num_cpus::get())  // 设置线程数量，使用系统可用的CPU核心数
        .build_global()
        .unwrap();

    let validator = match opt.validator {
        Some(validator) => validator,  // 如果命令行参数中指定了验证器地址，则使用该地址
        None => {
            // 如果没有指定验证器地址，则随机选择一个默认地址
            let bootstrap = [
                "164.92.111.59:4133",
                "159.223.204.96:4133",
                "167.71.219.176:4133",
                "157.245.205.209:4133",
                "134.122.95.106:4133",
                "161.35.24.55:4133",
                "138.68.103.139:4133",
                "207.154.215.49:4133",
                "46.101.114.158:4133",
                "138.197.190.94:4133",
            ];
            bootstrap.choose(&mut rand::thread_rng()).unwrap().to_string()
        }
    };
    let port = opt.port;  // 监听的端口号
    let address = opt.address;  // 指定的矿池地址
    let accounting = Accounting::init();  // 初始化会计模块
    let node = Node::init(validator);  // 初始化验证节点
    let server = Server::init(port, address, node.sender(), accounting.sender()).await;  // 初始化服务器模块

    // 启动验证节点模块
    //validator_peer::start(node, server.sender());

    // 启动虚拟验证节点模块
    validator_dummy::start(node, server.sender());

    api::start(opt.api_port, accounting.clone(), server.clone());  // 启动API模块

    // 注册信号处理函数，处理退出信号
    match Signals::new([SIGABRT, SIGTERM, SIGHUP, SIGINT, SIGQUIT, SIGUSR1, SIGTSTP]) {
        Ok(signals) => {
            tokio::spawn(handle_signals(signals, accounting.clone(), server.sender()));  // 在新的tokio任务中处理信号
        }
        Err(err) => {
            error!("Unable to register signal handlers: {:?}", err);
            std::process::exit(1);
        }
    }
    // 等待未来的永久任务
    std::future::pending::<()>().await;
}


async fn handle_signals(mut signals: Signals, accounting: Arc<Accounting>, server_sender: Sender<ServerMessage>) {
    while let Some(signal) = signals.next().await {
        info!("Received signal: {:?}", signal);
        // 获取会计模块的消息发送器
        let accounting_sender = accounting.sender();
        match signal {
            SIGABRT => {
                // 收到SIGABRT信号时，尝试在终止之前拯救状态
                info!("Trying to salvage states before aborting...");

                // 向会计模块发送退出消息
                let _ = accounting_sender.send(AccountingMessage::Exit).await;

                // 等待会计模块退出
                accounting.wait_for_exit().await;

                // 向服务器模块发送退出消息
                let _ = server_sender.send(ServerMessage::Exit).await;
                // 强制终止进程
                std::process::abort();
            }
            SIGTERM | SIGINT | SIGHUP | SIGQUIT => {
                // 收到SIGTERM、SIGINT、SIGHUP或SIGQUIT信号时，保存状态后退出
                info!("Saving states before exiting...");

                // 向会计模块发送退出消息
                let _ = accounting_sender.send(AccountingMessage::Exit).await;

                // 等待会计模块退出
                accounting.wait_for_exit().await;

                // 向服务器模块发送退出消息
                let _ = server_sender.send(ServerMessage::Exit).await;
                // 正常退出进程
                std::process::exit(0);
            }
            SIGUSR1 => {
                debug!("Should do something useful here...");
            }
            SIGTSTP => {
                warn!("Suspending is not supported");
            }
            // 不可能到达的分支
            _ => unreachable!(),
        }
    }
}

