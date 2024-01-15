use std::{marker::PhantomData, sync::Arc};

use anyhow::Result;
use bincode::Options;
use dirs::home_dir;
use rocksdb::{DBWithThreadMode, SingleThreaded, DB};
use serde::{de::DeserializeOwned, Serialize};
use tracing::error;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub enum StorageType {
    PPLNS,
}

impl StorageType {
    // 获取存储类型对应的前缀
    pub fn prefix(&self) -> &'static [u8; 1] {
        match self {
            StorageType::PPLNS => &[0], // 如果是 PPLNS 存储类型，则返回长度为1的字节数组[0]
        }
    }
}

pub struct Storage {
    db: Arc<DB>, // 数据库实例
}

//扩展了 Storage 结构体的功能。
impl Storage {
    // load() 方法用于加载存储实例
    // 它首先获取用户的家目录，然后构建数据库文件路径。
    // 接下来，创建 RocksDB 的配置选项，并设置了一些选项，例如是否在不存在数据库时创建、压缩类型、使用 fsync 等。通过设置前缀提取器，将存储类型的前缀长度设置为1。
    // 然后，通过设置比较器函数，可以控制键值对的排序。
    // 最后，打开数据库并返回存储实例。
    pub fn load() -> Storage {
        // 获取用户的家目录
        let home = home_dir();
        if home.is_none() {
            panic!("No home directory found");
        }
        // 构建数据库文件路径
        let db_path = home.unwrap().join(".aleo_pool_testnet3/state.db");

        // 创建 RocksDB 的配置选项
        let mut db_options = rocksdb::Options::default();
        db_options.create_if_missing(true);
        db_options.set_compression_type(rocksdb::DBCompressionType::Zstd);
        db_options.set_use_fsync(true);
        // 设置前缀提取器，将存储类型的前缀长度设置为1
        db_options.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(1));
        // 设置比较器函数，使得 [1] 前缀的键值对排在其他键值对之前
        db_options.set_comparator("comparator_v1", |a, b| {
            if !(a[0] == 1 && b[0] == 1) {
                a.cmp(b)
            } else if a == [1] {
                std::cmp::Ordering::Less
            } else if b == [1] {
                std::cmp::Ordering::Greater
            } else {
                a.cmp(b).reverse()
            }
        });

        // 打开数据库
        let db = DB::open(&db_options, db_path.to_str().unwrap()).expect("Failed to open DB");

        Storage { db: Arc::new(db) } // 返回存储实例
    }

    // init_data() 方法用于初始化存储数据。
    // 它接受一个存储类型参数 storage_type，并根据当前的存储实例创建一个 StorageData 实例，将数据库和存储类型作为参数传递。
    // 返回创建的 StorageData 实例。
    pub fn init_data<K: Serialize + DeserializeOwned, V: Serialize + DeserializeOwned>(
        &self,
        storage_type: StorageType,
    ) -> StorageData<K, V> {
        StorageData {
            db: self.db.clone(),
            storage_type,
            _p: PhantomData,
        }
    }
}

#[derive(Clone)]
// 定义一个具有泛型参数的结构体，用于存储数据
pub struct StorageData<K: Serialize + DeserializeOwned, V: Serialize + DeserializeOwned> {
    db: Arc<DB>, // RocksDB 实例
    storage_type: StorageType, // 存储类型
    _p: PhantomData<(K, V)>, // 用于占位符的类型
}

impl<K: Serialize + DeserializeOwned, V: Serialize + DeserializeOwned> StorageData<K, V> {
    // 获取指定键的值
    pub fn get(&self, key: &K) -> Result<Option<V>> {
        // 创建序列化选项
        let options = bincode::config::DefaultOptions::new()
            .with_big_endian() // 使用大端序
            .with_fixint_encoding() // 使用固定长度整数编码
            .allow_trailing_bytes(); // 允许多余字节

        // 构建键值对的键
        let mut key_buf = vec![self.storage_type.prefix()[0]]; // 将存储类型的前缀加入键中
        key_buf.reserve(options.serialized_size(&key)? as usize); // 根据键计算所需空间并预留空间
        options.serialize_into(&mut key_buf, key)?; // 序列化键并将其添加到键中

        // 从数据库获取值
        match self.db.get(key_buf)? {
            Some(value) => Ok(Some(options.deserialize(&value)?)), // 如果存在值，则反序列化值并返回
            None => Ok(None), // 如果不存在值，则返回 None
        }
    }

    // 将指定的键值对写入数据库
    pub fn put(&self, key: &K, value: &V) -> Result<()> {
        // 创建序列化选项
        let options = bincode::config::DefaultOptions::new()
            .with_big_endian() // 使用大端序
            .with_fixint_encoding() // 使用固定长度整数编码
            .allow_trailing_bytes(); // 允许多余字节

        // 构建键值对的键和值
        let mut key_buf = vec![self.storage_type.prefix()[0]]; // 将存储类型的前缀加入键中
        key_buf.reserve(options.serialized_size(&key)? as usize); // 根据键计算所需空间并预留空间
        options.serialize_into(&mut key_buf, key)?; // 序列化键并将其添加到键中
        let value_buf = options.serialize(value)?; // 序列化值

        // 将键值对写入数据库
        self.db.put(key_buf, value_buf)?;

        Ok(()) // 返回成功标志
    }

    // 获取迭代器
    #[allow(dead_code)]
    pub fn iter(&self) -> StorageIter<'_, K, V> {
        StorageIter {
            storage_type: self.storage_type.clone(), // 存储类型
            iter: self.db.prefix_iterator(self.storage_type.prefix()), // 前缀迭代器
            _p: Default::default(),
        }
    }
}

// 定义一个迭代器，用于遍历数据库中存储的所有键值对
pub struct StorageIter<'a, K: Serialize + DeserializeOwned, V: Serialize + DeserializeOwned> {
    storage_type: StorageType, // 存储类型
    iter: rocksdb::DBIteratorWithThreadMode<'a, DBWithThreadMode<SingleThreaded>>, // 前缀迭代器
    _p: PhantomData<(K, V)>, // 用于占位符的类型
}

impl<'a, K: Serialize + DeserializeOwned, V: Serialize + DeserializeOwned> Iterator for StorageIter<'a, K, V> {
    type Item = (K, V); // 迭代器返回值类型为包含键值对的元组

    // 获取下一个键值对
    fn next(&mut self) -> Option<Self::Item> {
        // 创建序列化选项
        let options = bincode::config::DefaultOptions::new()
            .with_big_endian() // 使用大端序
            .with_fixint_encoding() // 使用固定长度整数编码
            .allow_trailing_bytes(); // 允许多余字节

        // 获取下一个键值对的原始字节数组
        match self.iter.next()? {
            Ok((raw_key, raw_value)) => {
                // 如果键的前缀与存储类型相同，则解析键值对
                if raw_key[0] == self.storage_type.prefix()[0] {
                    // 反序列化键和值
                    let key = options.deserialize::<K>(&raw_key[1..]); // 键的第一个字节为存储类型前缀，因此从第二个字节开始反序列化
                    let value = options.deserialize::<V>(&raw_value);

                    // 检查反序列化是否成功
                    match key.and_then(|k| value.map(|v| (k, v))) {
                        Ok(item) => Some(item), // 如果反序列化成功，则返回键值对
                        Err(e) => {
                            error!("Failed to deserialize key or value: {:?}", e); // 如果反序列化失败，则输出错误日志
                            error!("Key: {:?}", &raw_key);
                            error!("Value: {:?}", &raw_value);
                            None // 返回空值
                        }
                    }
                } else {
                    None // 如果键的前缀与存储类型不同，则返回空值
                }
            }
            Err(e) => {
                error!("Failed to iterate: {:?}", e); // 输出错误日志
                None // 返回空值
            }
        }
    }
}
