use std::{collections::HashMap, env};

use anyhow::Result;
use deadpool_postgres::{
    ClientWrapper,
    Config,
    Hook,
    HookError,
    HookErrorCause,
    Manager,
    ManagerConfig,
    Pool,
    RecyclingMethod,
    Runtime,
};
use snarkvm::prelude:{PuzzleCommitment, Testnet3};
use tokio_postgres::NoTls;
use tracing::warn;

pub struct DB {
    connection_pool: Pool,
}

impl DB {
    pub fn init() -> DB {
        // 初始化数据库连接
        let mut cfg = Config::new();
        cfg.host = Some(env::var("DB_HOST").expect("No database host defined")); // 设置数据库主机
        cfg.port = Some(
            env::var("DB_PORT")
                .unwrap_or_else(|_| "5432".to_string())
                .parse::<u16>()
                .expect("Invalid database port"),
        ); // 设置数据库端口，默认为5432
        cfg.dbname = Some(env::var("DB_DATABASE").expect("No database name defined")); // 设置数据库名称
        cfg.user = Some(env::var("DB_USER").expect("No database user defined")); // 设置数据库用户名
        cfg.password = Some(env::var("DB_PASSWORD").expect("No database password defined")); // 设置数据库密码
        let schema = env::var("DB_SCHEMA").unwrap_or_else(|_| {
            warn!("Using schema public as default");
            "public".to_string()
        }); // 获取数据库架构，默认为"public"
        cfg.manager = Some(ManagerConfig {
            recycling_method: RecyclingMethod::Verified,
        });
        // 设置数据库连接池管理配置，使用验证的回收方法
        // 创建数据库连接池
        // This is almost like directly using deadpool, but we really need the hooks
        // The helper methods from deadpool_postgres helps as well
        let pool = Pool::builder(Manager::from_config(
            cfg.get_pg_config().expect("Invalid database config"),
            NoTls,
            cfg.get_manager_config(),
        ))
            .config(cfg.get_pool_config())
            .post_create(Hook::async_fn(move |client: &mut ClientWrapper, _| {
                let schema = schema.clone(); // 克隆数据库架构
                Box::pin(async move {
                    client
                        .simple_query(&format!("set search_path = {}", schema))
                        .await
                        .map_err(|e| HookError::Abort(HookErrorCause::Backend(e)))?; // 在连接创建后设置数据库架构
                    Ok(())
                })
            }))
            .runtime(Runtime::Tokio1)
            .build()
            .expect("Failed to create database connection pool"); // 构建数据库连接池

        DB { connection_pool: pool } // 返回包含连接池的数据库对象
    }

    pub async fn save_solution(
        &self,
        commitment: PuzzleCommitment<Testnet3>,
        shares: HashMap<String, u64>,
    ) -> Result<()> {
        let mut conn = self.connection_pool.get().await?; // 从连接池获取连接
        let transaction = conn.transaction().await?; // 开启事务

        let solution_id: i32 = transaction
            .query_one(
                "INSERT INTO solution (commitment) VALUES ($1) RETURNING id",
                &[&commitment.to_string()],
            )
            .await?
            .try_get("id")?; // 插入解决方案并返回插入的id

        let stmt = transaction
            .prepare_cached("INSERT INTO share (solution_id, address, share) VALUES ($1, $2, $3)")
            .await?; // 准备插入共享数据的SQL语句
        for (address, share) in shares {
            transaction
                .query(&stmt, &[&solution_id, &address, &(share as i64)])
                .await?; // 插入共享数据
        }

        transaction.commit().await?; // 提交事务
        Ok(())
    }

    pub async fn set_solution_valid(
        &self,
        commitment: &String,
        valid: bool,
        height: Option<u32>,
        reward: Option<u64>,
    ) -> Result<()> {
        let mut conn = self.connection_pool.get().await?; // 从连接池获取连接
        let transaction = conn.transaction().await?; // 开启事务
        let stmt = transaction
            .prepare_cached("UPDATE solution SET valid = $1, checked = checked + 1 WHERE commitment = $2")
            .await?; // 准备更新解决方案有效性的SQL语句
        transaction.query(&stmt, &[&valid, commitment]).await?; // 更新解决方案有效性和检查次数
        if valid {
            transaction
                .query(
                    "UPDATE solution SET height = $1, reward = $2 WHERE commitment = $3",
                    &[&(height.unwrap() as i64), &(reward.unwrap() as i64), commitment],
                )
                .await?; // 如果解决方案有效，则更新高度和奖励
        }
        transaction.commit().await?; // 提交事务
        Ok(())
    }

    pub async fn get_should_pay_solutions(&self) -> Result<Vec<(i32, String)>> {
        let conn = self.connection_pool.get().await?; // 从连接池获取连接
        let stmt = conn
            .prepare_cached(
                "SELECT * FROM solution WHERE paid = false AND ((valid = false AND checked < 3) OR valid = true) \
                 ORDER BY id",
            )
            .await?; // 准备查询待支付解决方案的SQL语句
        let rows = conn.query(&stmt, &[]).await?; // 执行查询
        Ok(rows
            .into_iter()
            .map(|row| {
                let id: i32 = row.get("id");
                let commitment: String = row.get("commitment");
                (id, commitment)
            })
            .collect()) // 将查询结果转换为包含id和commitment的元组列表并返回
    }

    // pub async fn set_checked_blocks(&self, latest_height: u32) -> Result<()> {
    //     let conn = self.connection_pool.get().await?;
    //     let stmt = conn
    //         .prepare_cached("UPDATE block SET checked = true WHERE height <= $1 AND checked = false")
    //         .await?;
    //     conn.query(&stmt, &[&((latest_height as i64).saturating_sub(4100))])
    //         .await?;
    //     Ok(())
    // }

    pub async fn pay_solution(&self, solution_id: i32) -> Result<()> {
        let conn = self.connection_pool.get().await?; // 从连接池获取连接
        let stmt = conn.prepare("CALL pay_solution($1)").await?; // 准备调用支付解决方案的SQL语句
        conn.query(&stmt, &[&solution_id]).await?; // 调用支付解决方案存储过程
        Ok(())
    }
}
