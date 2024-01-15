use std::io::Write;

use anyhow::anyhow;
use byteorder::{LittleEndian, ReadBytesExt};
use bytes::{Buf, BufMut, BytesMut};
use snarkvm::{
    // dpc::{Testnet3::Testnet3, Address, BlockTemplate, PoSWProof},

    prelude::{Testnet3, Environment, Network, address::Address,},
    
    // traits::Network,
    utilities::{FromBytes, ToBytes},
};
use tokio_util::codec::{Decoder, Encoder};

/// Not being used anymore as we are migrating to "standard" stratum+tcp protocol.
#[allow(clippy::large_enum_variant)]
pub enum ProverMessage {
    // 如同stratum，增加了一个额外的协议版本字段
    Authorize(Address<Testnet3>, String, u16),
    AuthorizeResult(bool, Option<String>),

    // // 结合notify和set_difficulty以保持一致性
    // Notify(BlockTemplate<Testnet3>, u64),
    // // 包括块高度以更快地检测陈旧
    // Submit(u32, <Testnet3 as Network>::PoSWNonce, PoSWProof<Testnet3>),
    // 矿工可能想要了解陈旧率，可选择提供一条消息
    SubmitResult(bool, Option<String>),

    Canary,
}

#[allow(dead_code)]
static VERSION: u16 = 1;

impl ProverMessage {
    #[allow(dead_code)]
    pub fn version() -> &'static u16 {
        &VERSION
    }

    pub fn id(&self) -> u8 {
        match self {
            ProverMessage::Authorize(..) => 0,
            ProverMessage::AuthorizeResult(..) => 1,
            // ProverMessage::Notify(..) => 2,
            // ProverMessage::Submit(..) => 3,
            ProverMessage::SubmitResult(..) => 4,

            ProverMessage::Canary => 5,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ProverMessage::Authorize(..) => "Authorize",
            ProverMessage::AuthorizeResult(..) => "AuthorizeResult",
            // ProverMessage::Notify(..) => "Notify",
            // ProverMessage::Submit(..) => "Submit",
            ProverMessage::SubmitResult(..) => "SubmitResult",

            ProverMessage::Canary => "Canary",
        }
    }
}

impl Encoder<ProverMessage> for ProverMessage {
    type Error = anyhow::Error;

    fn encode(&mut self, item: ProverMessage, dst: &mut BytesMut) -> Result<(), Self::Error> {
        // 将0u32转为LE字节序，并将其扩展到dst中
        dst.extend_from_slice(&0u32.to_le_bytes());
        let mut writer = dst.writer();
        // 将消息ID写入writer
        writer.write_all(&[item.id()])?;
        match item {
            ProverMessage::Authorize(addr, password, version) => {
                // 序列化addr并写入writer
                bincode::serialize_into(&mut writer, &addr)?;
                // 序列化password并写入writer
                bincode::serialize_into(&mut writer, &password)?;
                // 将版本号以LE字节序写入writer
                writer.write_all(&version.to_le_bytes())?;
            }
            ProverMessage::AuthorizeResult(result, message) | ProverMessage::SubmitResult(result, message) => {
                // 根据result的值决定写入1或0
                writer.write_all(&[match result {
                    true => 1,
                    false => 0,
                }])?;
                if let Some(message) = message {
                    // 如果有message，写入1，并序列化message并写入writer
                    writer.write_all(&[1])?;
                    bincode::serialize_into(&mut writer, &message)?;
                } else {
                    // 如果没有message，写入0
                    writer.write_all(&[0])?;
                }
            }
            // ProverMessage::Notify(template, difficulty) => {
            //     template.write_le(&mut writer)?;
            //     writer.write_all(&difficulty.to_le_bytes())?;
            // }
            // ProverMessage::Submit(height, nonce, proof) => {
            //     writer.write_all(&height.to_le_bytes())?;
            //     nonce.write_le(&mut writer)?;
            //     proof.write_le(&mut writer)?;
            // }
            ProverMessage::Canary => return Err(anyhow!("Use of unsupported message")),
        }
        // 计算消息长度，并将其以LE字节序写入dst的前4个字节
        let msg_len = dst.len() - 4;
        dst[..4].copy_from_slice(&(msg_len as u32).to_le_bytes());
        Ok(())
    }
}

impl Decoder for ProverMessage {
    type Error = anyhow::Error;
    type Item = ProverMessage;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if src.len() < 4 {
            return Ok(None);
        }
        // 从src的前4个字节解析出消息长度
        let length = u32::from_le_bytes(src[..4].try_into().unwrap()) as usize;
        if length > 1048576 {
            return Err(anyhow!("Message too long"));
        }
        if src.len() < 4 + length {
            return Ok(None);
        }

        let reader = &mut src.reader();
        // 读取并丢弃4个字节的消息长度
        reader.read_u32::<LittleEndian>()?;
        // 读取并返回消息ID
        let msg_id = reader.read_u8()?;
        let msg = match msg_id {
            0 => {
                // 反序列化addr
                let addr = bincode::deserialize_from(&mut *reader)?;
                // 反序列化password
                let password = bincode::deserialize_from(&mut *reader)?;
                // 读取版本号
                let version = reader.read_u16::<LittleEndian>()?;
                ProverMessage::Authorize(addr, password, version)
            }
            1 => {
                // 根据读取的值判断result的值
                let result = reader.read_u8()? == 1;
                let message = if reader.read_u8()? == 1 {
                    // 如果有message，反序列化并返回Some(message)
                    Some(bincode::deserialize_from(reader)?)
                } else {
                    None
                };
                ProverMessage::AuthorizeResult(result, message)
            }
            // 2 => {
            //     let template = BlockTemplate::<Testnet3>::read_le(&mut *reader)?;
            //     let difficulty = reader.read_u64::<LittleEndian>()?;
            //     ProverMessage::Notify(template, difficulty)
            // }
            // 3 => {
            //     let height = reader.read_u32::<LittleEndian>()?;
            //     let nonce = <Testnet3 as Network>::PoSWNonce::read_le(&mut *reader)?;
            //     let proof = PoSWProof::<Testnet3>::read_le(&mut *reader)?;
            //     ProverMessage::Submit(height, nonce, proof)
            // }
            4 => {
                // 根据读取的值判断result的值
                let result = reader.read_u8()? == 1;
                let message = if reader.read_u8()? == 1 {
                    // 如果有message，反序列化并返回Some(message)
                    Some(bincode::deserialize_from(reader)?)
                } else {
                    None
                };
                ProverMessage::SubmitResult(result, message)
            }
            _ => {
                return Err(anyhow!("Unknown message id: {}", msg_id));
            }
        };
        Ok(Some(msg))
    }
}

