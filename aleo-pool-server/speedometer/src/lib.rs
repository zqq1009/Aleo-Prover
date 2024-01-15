use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use tokio::sync::RwLock;

pub struct Speedometer {
    storage: RwLock<VecDeque<(Instant, u64)>>,  // 存储事件的队列，使用RwLock保证线程安全
    interval: Duration,  // 统计速度的时间间隔
    cached: bool,  // 是否开启缓存
    cache_interval: Option<Duration>,  // 缓存持续时间
    cache_instant: Option<Instant>,  // 上一次缓存更新时间
    cache_value: f64,  // 缓存的速度值
}

impl Speedometer {
    pub fn init(interval: Duration) -> Self {  // 创建新的Speedometer实例
        Self {
            storage: RwLock::new(VecDeque::new()),
            interval,
            cached: false,
            cache_interval: None,
            cache_instant: None,
            cache_value: 0.0,
        }
    }

    pub fn init_with_cache(interval: Duration, cache_interval: Duration) -> Self {  // 创建新的Speedometer实例，并开启缓存功能
        Self {
            storage: RwLock::new(VecDeque::new()),
            interval,
            cached: true,
            cache_interval: Some(cache_interval),
            cache_instant: Some(Instant::now() - cache_interval),  // 初始化上一次缓存更新时间为当前时间减去缓存持续时间
            cache_value: 0.0,
        }
    }

    pub async fn event(&self, value: u64) {  // 记录事件
        let mut storage = self.storage.write().await;  // 获取写锁
        storage.push_back((Instant::now(), value));  // 将事件加入队列尾部
        while storage.front().map_or(false, |t| t.0.elapsed() > self.interval) {  // 如果队列头部的事件已经过期，则将其移出队列
            storage.pop_front();
        }
    }

    pub async fn speed(&mut self) -> f64 {  // 计算速度
        if self.cached && self.cache_instant.unwrap().elapsed() < self.cache_interval.unwrap() {  // 如果缓存未过期，则直接返回缓存值
            return self.cache_value;
        }
        let mut storage = self.storage.write().await;  // 获取写锁
        while storage.front().map_or(false, |t| t.0.elapsed() > self.interval) {  // 如果队列头部的事件已经过期，则将其移出队列
            storage.pop_front();
        }
        drop(storage);  // 释放写锁
        let events = self.storage.read().await.iter().fold(0, |acc, t| acc + t.1);  // 统计事件总数
        let speed = events as f64 / self.interval.as_secs_f64();  // 计算速度
        if self.cached {  // 如果开启了缓存，则更新缓存
            self.cache_instant = Some(Instant::now());
            self.cache_value = speed;
        }
        speed  // 返回速度值
    }

    #[allow(dead_code)]
    pub async fn reset(&self) {  // 清空事件队列
        self.storage.write().await.clear();
    }
}

