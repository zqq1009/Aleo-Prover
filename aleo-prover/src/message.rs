use std::io::Write; // 引入标准库中的io模块和Write trait
use anyhow::anyhow; // 引入anyhow库，用于方便地创建任意类型的错误信息
use byteorder::{LittleEndian, ReadBytesExt}; // 引入byteorder库，用于处理字节序
use bytes::{Buf, BufMut, BytesMut}; // 引入bytes库，用于处理字节缓冲
use snarkvm::{ // 引入snarkvm库中的各种结构体、枚举和trait
               dpc::{testnet2::Testnet2, Address, BlockTemplate, PoSWProof},
               traits::Network,
               utilities::{FromBytes, ToBytes},
};
use tokio_util::codec::{Decoder, Encoder}; // 引入tokio_util库中的codec模块和Decoder、Encoder trait

// 定义ProverMessage枚举类型
#[allow(clippy::large_enum_variant)]
pub enum ProverMessage {
    // 与stratum协议类似，但增加了一个协议版本字段
    Authorize(Address<Testnet2>, String, u16),
    AuthorizeResult(bool, Option<String>),

    // 合并Notify和SetDifficulty，使之一致
    Notify(BlockTemplate<Testnet2>, u64),
    // 包括块高度以更快地检测到过期的提交
    Submit(u32, <Testnet2 as Network>::PoSWNonce, PoSWProof<Testnet2>),
    // 矿工可能想知道过期率，可选提供一条消息
    SubmitResult(bool, Option<String>),

    Canary, // 不支持的消息类型
}

// 定义ProverMessage的方法
impl ProverMessage {
    #[allow(dead_code)]
    pub fn version() -> &'static u16 {
        &VERSION
    }

    // 获取消息类型的ID
    pub fn id(&self) -> u8 {
        match self {
            ProverMessage::Authorize(..) => 0,
            ProverMessage::AuthorizeResult(..) => 1,
            ProverMessage::Notify(..) => 2,
            ProverMessage::Submit(..) => 3,
            ProverMessage::SubmitResult(..) => 4,

            ProverMessage::Canary => 5,
        }
    }

    // 获取消息类型的名称
    pub fn name(&self) -> &'static str {
        match self {
            ProverMessage::Authorize(..) => "Authorize",
            ProverMessage::AuthorizeResult(..) => "AuthorizeResult",
            ProverMessage::Notify(..) => "Notify",
            ProverMessage::Submit(..) => "Submit",
            ProverMessage::SubmitResult(..) => "SubmitResult",

            ProverMessage::Canary => "Canary",
        }
    }
}

// 实现Encoder trait，将ProverMessage编码为字节流
impl Encoder<ProverMessage> for ProverMessage {
    type Error = anyhow::Error;

    fn encode(&mut self, item: ProverMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        dst.extend_from_slice(&0u32.to_le_bytes()); // 先在字节缓冲中写入一个0值占位，后面再替换为消息长度
        let mut writer = dst.writer(); // 将字节缓冲包装成一个实现了Write trait的写入器
        writer.write_all(&[item.id()])?; // 写入消息类型ID
        match item {
            ProverMessage::Authorize(addr, password, version) => { // Authorize消息类型
                bincode::serialize_into(&mut writer, &addr)?; // 序列化Address到写入器中
                bincode::serialize_into(&mut writer, &password)?; // 序列化密码到写入器中
                writer.write_all(&version.to_le_bytes())?; // 将协议版本编码为小端字节序并写入到写入器中
            }
            ProverMessage::AuthorizeResult(result, message) | ProverMessage::SubmitResult(result, message) => { // AuthorizeResult和SubmitResult消息类型
                writer.write_all(&[match result { // 匹配result是否为true
                    true => 1,
                    false => 0,
                }])?;
                if let Some(message) = message {
                    writer.write_all(&[1])?; // 如果有消息，则写入一个值为1的字节表示有消息
                    bincode::serialize_into(&mut writer, &message)?; // 将消息序列化到写入器中
                } else {
                    writer.write_all(&[0])?; // 否则写入一个值为0的字节表示没有消息
                }
            }
            ProverMessage::Notify(template, difficulty) => { // Notify消息类型
                template.write_le(&mut writer)?; // 序列化BlockTemplate到写入器中
                writer.write_all(&difficulty.to_le_bytes())?; // 将难度编码为小端字节序并写入到写入器中
            }
            ProverMessage::Submit(height, nonce, proof) => { // Submit消息类型
                writer.write_all(&height.to_le_bytes())?; // 将块高度编码为小端字节序并写入到写入器中
                nonce.write_le(&mut writer)?; // 序列化PoSWNonce到写入器中
                proof.write_le(&mut writer)?; // 序列化PoSWProof到写入器中
            }
            ProverMessage::Canary => return Err(anyhow!("Use of unsupported message")), // 不支持的消息类型，返回一个错误
        }
        let msg_len = dst.len() - 4; // 计算消息长度（减去占位用的4个字节）
        dst[..4].copy_from_slice(&(msg_len as u32).to_le_bytes()); // 将占位用的4个字节替换为实际的消息长度（编码为小端字节序）
        Ok(()) // 返回编码成功的结果
    }
}

// 实现Decoder trait，将字节流解码为ProverMessage
impl Decoder for ProverMessage {
    type Error = anyhow::Error;
    type Item = ProverMessage;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if src.len() < 4 { // 如果字节缓冲中的字节数不足4，则返回解码失败
            return Ok(None);
        }
        let length = u32::from_le_bytes(src[..4].try_into().unwrap()) as usize; // 从字节缓冲中读取消息长度，并转换为usize类型
        if length > 1048576 { // 如果消息长度超过了1MB，则返回一个错误
            return Err(anyhow!("Message too long"));
        }
        if src.len() < 4 + length { // 如果字节缓冲中的字节数不足消息长度+4，则返回解码失败
            return Ok(None);
        }

        let reader = &mut src.reader(); // 将字节缓冲包装成一个实现了Read trait的读取器
        reader.read_u32::<LittleEndian>()?; // 跳过占位用的4个字节
        let msg_id = reader.read_u8()?; // 从读取器中读取消息类型ID
        let msg = match msg_id {
            0 => { // Authorize消息类型
                let addr = bincode::deserialize_from(&mut *reader)?; // 从读取器中反序列化Address
                let password = bincode::deserialize_from(&mut *reader)?; // 从读取器中反序列化密码
                let version = reader.read_u16::<LittleEndian>()?; // 从读取器中读取协议版本（小端字节序）
                ProverMessage::Authorize(addr, password, version)
            }
            1 => { // AuthorizeResult消息类型
                let result = reader.read_u8()? == 1; // 判断result是否为true
                let message = if reader.read_u8()? == 1 { // 判断是否有消息
                    Some(bincode::deserialize_from(reader)?) // 如果有，则从读取器中反序列化消息
                } else {
                    None // 否则为空
                };
                ProverMessage::AuthorizeResult(result, message)
            }
            2 => { // Notify消息类型
                let template = BlockTemplate::<Testnet2>::read_le(&mut *reader)?; // 从读取器中反序列化BlockTemplate
                let difficulty = reader.read_u64::<LittleEndian>()?; // 从读取器中读取难度（小端字节序）
                ProverMessage::Notify(template, difficulty)
            }
            3 => { // Submit消息类型
                let height = reader.read_u32::<LittleEndian>()?; // 从读取器中读取块高度（小端字节序）
                let nonce = <Testnet2 as Network>::PoSWNonce::read_le(&mut *reader)?; // 从读取器中反序列化PoSWNonce
                let proof = PoSWProof::<Testnet2>::read_le(&mut *reader)?; // 从读取器中反序列化PoSWProof
                ProverMessage::Submit(height, nonce, proof)
            }
            4 => { // SubmitResult消息类型
                let result = reader.read_u8()? == 1; // 判断result是否为true
                let message = if reader.read_u8()? == 1 { // 判断是否有消息
                    Some(bincode::deserialize_from(reader)?) // 如果有，则从读取器中反序列化消息
                } else {
                    None // 否则为空
                };
                ProverMessage::SubmitResult(result, message)
            }
            _ => { // 不支持的消息类型，返回一个错误
                return Err(anyhow!("Unknown message id: {}", msg_id));
            }
        };
        Ok(Some(msg)) // 返回解码成功的结果
    }
}
