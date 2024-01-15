use std::{convert::Infallible, net::SocketAddr, sync::Arc};

use serde_json::json;
use snarkvm::{console::types::Address, prelude::Testnet3};
use tokio::task;
use tracing::info;
use warp::{
    addr::remote,
    get,
    head,
    path,
    reply,
    reply::{json, Json},
    serve,
    Filter,
    Reply,
};

use crate::{Accounting, Server};

pub fn start(port: u16, accounting: Arc<Accounting>, server: Arc<Server>) {
    task::spawn(async move {
        // 定义不同的API路径，并与对应的处理函数绑定
        let current_round = path("current_round")
            .and(use_accounting(accounting.clone()))
            .then(current_round)
            .boxed();

        let pool_stats = path("stats").and(use_server(server.clone())).then(pool_stats).boxed();

        let address_stats = path!("stats" / String)
            .and(use_server(server.clone()))
            .then(address_stats)
            .boxed();

        let admin_current_round = path!("admin" / "current_round")
            .and(remote())
            .and(use_accounting(accounting.clone()))
            .then(admin_current_round)
            .boxed();

        // 将所有的路径处理函数合并成一个路由
        let endpoints = current_round
            .or(address_stats)
            .or(pool_stats)
            .or(admin_current_round)
            .boxed();

        // 定义HTTP请求方法，将路由和日志中间件绑定在一起
        let routes = get()
            .or(head())
            .unify()
            .and(endpoints)
            .with(warp::log("aleo_pool_server::api"));
        info!("Starting API server on port {}", port);

        // 启动HTTP服务
        serve(routes).run(([0, 0, 0, 0], port)).await;
    });
}

// 根据提供的Accounting实例创建一个过滤器
fn use_accounting(
    accounting: Arc<Accounting>,
) -> impl Filter<Extract = (Arc<Accounting>,), Error = Infallible> + Clone {
    warp::any().map(move || accounting.clone())
}

// 根据提供的Server实例创建一个过滤器
fn use_server(server: Arc<Server>) -> impl Filter<Extract = (Arc<Server>,), Error = Infallible> + Clone {
    warp::any().map(move || server.clone())
}

// 处理/pool/stats路径的请求，返回当前池子的统计信息
async fn pool_stats(server: Arc<Server>) -> Json {
    json(&json!({
        "online_addresses": server.online_addresses().await,
        "online_provers": server.online_provers().await,
        "speed": server.pool_speed().await,
    }))
}

// 处理/stats/{address}路径的请求，返回指定地址的统计信息
async fn address_stats(address: String, server: Arc<Server>) -> impl Reply {
    if let Ok(address) = address.parse::<Address<Testnet3>>() {
        let speed = server.address_speed(address).await;
        let prover_count = server.address_prover_count(address).await;
        Ok(reply::with_status(
            json(&json!({
                "online_provers": prover_count,
                "speed": speed,
            })),
            warp::http::StatusCode::OK,
        ))
    } else {
        Ok(reply::with_status(
            json(&json!({
                "error": "invalid address"
            })),
            warp::http::StatusCode::BAD_REQUEST,
        ))
    }
}

// 处理/current_round路径的请求，返回当前轮次的信息
async fn current_round(accounting: Arc<Accounting>) -> Json {
    let data = accounting.current_round().await;

    json(&json! ({
        "n": data["n"],
        "current_n": data["current_n"],
        "provers": data["provers"],
    }))
}

// 处理/admin/current_round路径的请求，返回管理者权限下的当前轮次信息
async fn admin_current_round(addr: Option<SocketAddr>, accounting: Arc<Accounting>) -> impl Reply {
    let addr = addr.unwrap();
    if addr.ip().is_loopback() {  // 只允许本地回环地址访问该接口
        let pplns = accounting.current_round().await;
        Ok(reply::with_status(json(&pplns), warp::http::StatusCode::OK))
    } else {
        Ok(reply::with_status(
            json(&"Method Not Allowed"),
            warp::http::StatusCode::METHOD_NOT_ALLOWED,
        ))
    }
}

