use std::io;

use bytes::BytesMut;
use downcast_rs::{impl_downcast, DowncastSync};
use erased_serde::Serialize as ErasedSerialize;
use json_rpc_types::{Id, Request, Response, Version};
use serde::{ser::SerializeSeq, Deserialize, Serialize};
use serde_json::Value;
use tokio_util::codec::{AnyDelimiterCodec, Decoder, Encoder};

use crate::message::StratumMessage;

// 定义 StratumCodec 结构体，包含 AnyDelimiterCodec 类型的 codec 字段
pub struct StratumCodec {
    codec: AnyDelimiterCodec,
}

impl Default for StratumCodec {
    fn default() -> Self {
        Self {
            // Notify is ~400 bytes and submit is ~1750 bytes. 4096 should be enough for all messages
            // TODO: verify again
            codec: AnyDelimiterCodec::new_with_max_length(vec![b'\n'], vec![b'\n'], 4096),
        }
    }
}

// 定义 SubscribeParams 和 NotifyParams 结构体，用于反序列化 Stratum 协议中的参数
#[derive(Serialize, Deserialize)]
struct NotifyParams(String, String, Option<String>, bool);

#[derive(Serialize, Deserialize)]
struct SubscribeParams(String, String, Option<String>);

// 定义 BoxedType trait 和 ResponseParams 枚举类型，用于序列化和反序列化 Stratum 协议中的参数
pub trait BoxedType: ErasedSerialize + Send + DowncastSync {}
erased_serde::serialize_trait_object!(BoxedType);
impl_downcast!(sync BoxedType);

impl BoxedType for String {}
impl BoxedType for Option<u64> {}
impl BoxedType for Option<String> {}

pub enum ResponseParams {
    Bool(bool),
    Array(Vec<Box<dyn BoxedType>>),
    Null,
}

// 实现 ResponseParams 的序列化
impl Serialize for ResponseParams {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
    {
        match self {
            ResponseParams::Bool(b) => serializer.serialize_bool(*b),
            ResponseParams::Array(v) => {
                let mut seq = serializer.serialize_seq(Some(v.len()))?;
                for item in v {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            ResponseParams::Null => serializer.serialize_none(),
        }
    }
}

// 实现 ResponseParams 的反序列化
impl<'de> Deserialize<'de> for ResponseParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Bool(b) => Ok(ResponseParams::Bool(b)),
            Value::Array(a) => {
                let mut vec: Vec<Box<dyn BoxedType>> = Vec::new();
                a.iter().for_each(|v| match v {
                    Value::Null => vec.push(Box::new(None::<String>)),
                    Value::String(s) => vec.push(Box::new(s.clone())),
                    Value::Number(n) => vec.push(Box::new(n.as_u64())),
                    _ => {}
                });
                Ok(ResponseParams::Array(vec))
            }
            Value::Null => Ok(ResponseParams::Null),
            _ => Err(serde::de::Error::custom("invalid response params")),
        }
    }
}

// 实现 StratumCodec 的编码器 Encoder trait
impl Encoder<StratumMessage> for StratumCodec {
    type Error = io::Error;

    fn encode(&mut self, item: StratumMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        let bytes = match item {
            StratumMessage::Subscribe(id, user_agent, protocol_version, session_id) => {
                let request = Request {
                    jsonrpc: Version::V2,
                    method: "mining.subscribe",
                    params: Some(SubscribeParams(user_agent, protocol_version, session_id)),
                    id: Some(id),
                };
                serde_json::to_vec(&request).unwrap_or_default()
            }
            StratumMessage::Authorize(id, worker_name, worker_password) => {
                let request = Request {
                    jsonrpc: Version::V2,
                    method: "mining.authorize",
                    params: Some(vec![worker_name, worker_password]),
                    id: Some(id),
                };
                serde_json::to_vec(&request).unwrap_or_default()
            }
            StratumMessage::SetTarget(difficulty_target) => {
                let request = Request {
                    jsonrpc: Version::V2,
                    method: "mining.set_target",
                    params: Some(vec![difficulty_target]),
                    id: None,
                };
                serde_json::to_vec(&request).unwrap_or_default()
            }
            StratumMessage::Notify(job_id, epoch_challenge, address, clean_jobs) => {
                let request = Request {
                    jsonrpc: Version::V2,
                    method: "mining.notify",
                    params: Some(NotifyParams(job_id, epoch_challenge, address, clean_jobs)),
                    id: None,
                };
                serde_json::to_vec(&request).unwrap_or_default()
            }
            StratumMessage::Submit(id, worker_name, job_id, nonce, commitment, proof) => {
                let request = Request {
                    jsonrpc: Version::V2,
                    method: "mining.submit",
                    params: Some(vec![worker_name, job_id, nonce, commitment, proof]),
                    id: Some(id),
                };
                serde_json::to_vec(&request).unwrap_or_default()
            }
            StratumMessage::Response(id, result, error) => match error {
                Some(error) => {
                    let response = Response::<(), ()>::error(Version::V2, error, Some(id));
                    serde_json::to_vec(&response).unwrap_or_default()
                }
                None => {
                    let response = Response::<Option<ResponseParams>, ()>::result(Version::V2, result, Some(id));
                    serde_json::to_vec(&response).unwrap_or_default()
                }
            },
        };
        let string =
            std::str::from_utf8(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        self.codec
            .encode(string, dst)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        Ok(())
    }
}

// 定义 unwrap_str_value、unwrap_bool_value、unwrap_u64_value 函数，用于获取 Value 类型的值并转换为 Rust 中的 String、bool、u64 类型
fn unwrap_str_value(value: &Value) -> Result<String, io::Error> {
    match value {
        Value::String(s) => Ok(s.clone()),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Param is not str")),
    }
}

// 将 Value 类型的参数解析为 bool 类型的值
fn unwrap_bool_value(value: &Value) -> Result<bool, io::Error> {
    match value {
        Value::Bool(b) => Ok(*b), // 如果参数是布尔类型，则返回布尔值
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Param is not bool")), // 参数不是布尔类型，返回错误
    }
}

// 将 Value 类型的参数解析为 u64 类型的值
fn unwrap_u64_value(value: &Value) -> Result<u64, io::Error> {
    match value {
        Value::Number(n) => Ok(n
            .as_u64()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Param is not u64"))?), // 如果参数是数字类型并且可以转换为u64类型，则返回u64值，否则返回错误
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Param is not u64")), // 参数不是数字类型，返回错误
    }
}

impl Decoder for StratumCodec {
    type Error = io::Error;
    type Item = StratumMessage;
    // 解码器的 decode 方法，将字节流解码为 StratumMessage 类型的消息
    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        // 使用self.codec的decode方法将字节流解码为字符串，并将可能的错误转换为io::Error类型的错误
        let string = self
            .codec
            .decode(src)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        // 如果解码后得到的字符串为空，则返回Ok(None)
        if string.is_none() {
            return Ok(None);
        }
        let bytes = string.unwrap();
        // 将字符串解析为serde_json::Value类型的JSON值，并将可能的错误转换为io::Error类型的错误
        let json = serde_json::from_slice::<serde_json::Value>(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        // 如果解析后得到的JSON值不是对象类型，则返回错误
        if !json.is_object() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Not an object"));
        }
        // 将JSON对象强制转换为serde_json::Map类型的对象
        let object = json.as_object().unwrap();
        let result = if object.contains_key("method") {
            let request = serde_json::from_value::<Request<Vec<Value>>>(json)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let id = request.id;
            let method = request.method.as_str();
            let params = match request.params {
                Some(params) => params,
                // 如果参数为空，则返回错误
                None => return Err(io::Error::new(io::ErrorKind::InvalidData, "No params")),
            };
            match method {
                "mining.subscribe" => {
                    // 如果参数长度不为3，则返回错误
                    if params.len() != 3 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid params"));
                    }
                    let user_agent = unwrap_str_value(&params[0])?;
                    let protocol_version = unwrap_str_value(&params[1])?;
                    let session_id = match &params[2] {
                        Value::String(s) => Some(s),
                        // 如果参数不是字符串或null，则返回错误
                        Value::Null => None,
                        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid params")),
                    };
                    StratumMessage::Subscribe(
                        id.unwrap_or(Id::Num(0)),
                        user_agent,
                        protocol_version,
                        session_id.cloned(),
                    )
                }
                "mining.authorize" => {
                    // 如果参数长度不为2，则返回错误
                    if params.len() != 2 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid params"));
                    }
                    let worker_name = unwrap_str_value(&params[0])?;
                    let worker_password = unwrap_str_value(&params[1])?;
                    StratumMessage::Authorize(id.unwrap_or(Id::Num(0)), worker_name, worker_password)
                }
                "mining.set_target" => {
                    // 如果参数长度不为1，则返回错误
                    if params.len() != 1 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid params"));
                    }
                    let difficulty_target = unwrap_u64_value(&params[0])?;
                    StratumMessage::SetTarget(difficulty_target)
                }
                "mining.notify" => {
                    // 如果参数长度不为4，则返回错误
                    if params.len() != 4 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid params"));
                    }
                    let job_id = unwrap_str_value(&params[0])?;
                    let epoch_challenge = unwrap_str_value(&params[1])?;
                    let address = match &params[2] {
                        Value::String(s) => Some(s),
                        // 如果参数不是字符串或null，则返回错误
                        Value::Null => None,
                        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid params")),
                    };
                    let clean_jobs = unwrap_bool_value(&params[3])?;
                    StratumMessage::Notify(job_id, epoch_challenge, address.cloned(), clean_jobs)
                }
                "mining.submit" => {
                    if params.len() != 5 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid params")); // 如果参数长度不为5，则返回错误
                    }
                    let worker_name = unwrap_str_value(&params[0])?;
                    let job_id = unwrap_str_value(&params[1])?;
                    let nonce = unwrap_str_value(&params[2])?;
                    let commitment = unwrap_str_value(&params[3])?;
                    let proof = unwrap_str_value(&params[4])?;
                    StratumMessage::Submit(id.unwrap_or(Id::Num(0)), worker_name, job_id, nonce, commitment, proof)
                }
                _ => {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "Unknown method")); // 未知的方法，返回错误
                }
            }
        } else {
            let response = serde_json::from_value::<Response<ResponseParams, ()>>(json)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let id = response.id;
            match response.payload {
                Ok(payload) => StratumMessage::Response(id.unwrap_or(Id::Num(0)), Some(payload), None),
                Err(error) => StratumMessage::Response(id.unwrap_or(Id::Num(0)), None, Some(error)),
            }
        };
        Ok(Some(result))
    }
}
