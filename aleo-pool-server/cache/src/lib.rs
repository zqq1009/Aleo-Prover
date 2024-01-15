use std::{
    collections::HashMap,  // 导入HashMap类型
    hash::Hash,  // 导入Hash trait
    time::{Duration, Instant},  // 导入Duration和Instant类型
};

pub struct Cache<K: Eq + Hash + Clone, V: Clone> {
    duration: Duration,  // 缓存的持续时间
    instants: HashMap<K, Instant>,  // 存储key对应的value插入时间
    values: HashMap<K, V>,  // 存储key对应的value值
}

impl<K: Eq + Hash + Clone, V: Clone> Cache<K, V> {
    pub fn new(duration: Duration) -> Self {  // 构造函数，创建一个新的Cache实例
        Cache {
            duration,
            instants: Default::default(),  // 初始化instants为默认值
            values: Default::default(),  // 初始化values为默认值
        }
    }

    pub fn get(&self, key: K) -> Option<V> {  // 根据key获取对应的value
        let instant = self.instants.get(&key)?;  // 获取key对应的插入时间
        if instant.elapsed() > self.duration {  // 如果插入时间已超过duration，则返回None
            return None;
        }
        self.values.get(&key).cloned()  // 否则返回key对应的value值
    }

    pub fn set(&mut self, key: K, value: V) {  // 设置缓存，将key-value对加入缓存中
        self.values.insert(key.clone(), value);  // 将key-value对插入values中
        self.instants.insert(key, Instant::now());  // 将key插入instants中，并记录插入时间
    }
}
