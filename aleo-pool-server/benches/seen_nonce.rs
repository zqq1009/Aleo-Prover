#[macro_use]
extern crate criterion;

use std::sync::Arc;
use criterion::Criterion;
use flurry::HashSet;
use rand::thread_rng;
use snarkvm::dpc::testnet2::Testnet2;
use snarkvm::prelude::Network;
use snarkvm::utilities::UniformRand;

fn fake_nonce() -> String {  // 定义函数，用于生成假nonce
    let nonce: <Testnet2 as Network>::PoSWNonce = UniformRand::rand(&mut thread_rng());  // 生成随机nonce
    nonce.to_string()  // 将nonce转换为字符串并返回
}

fn seen_nonce_benchmark(c: &mut Criterion) {  // 定义函数，用于测试seen_nonce的性能
    let nonce_seen = Arc::new(HashSet::with_capacity(10 << 20));  // 创建一个容量为10M的HashSet，并将其封装在Arc指针中
    c.bench_function("seen_nonce", |b| b.iter(|| nonce_seen.pin().insert(fake_nonce())));  // 对插入新nonce进行基准测试
}

criterion_group!(nonce,seen_nonce_benchmark);  // 将seen_nonce_benchmark函数添加到名为"nonce"的基准测试组中
criterion_main!(nonce);  // 运行名为"nonce"的基准测试组

