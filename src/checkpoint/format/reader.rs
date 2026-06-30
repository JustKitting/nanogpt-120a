use std::io::{BufReader, Read};

use super::{MAGIC, VERSION};
use crate::AppResult;

pub struct CheckpointReader<R: Read> {
    reader: BufReader<R>,
}

pub struct CheckpointTensor {
    pub name: String,
    pub len: usize,
    pub global_scale: f32,
    pub bytes: Vec<u8>,
    pub scales: Vec<u8>,
}

impl<R: Read> CheckpointReader<R> {
    pub fn new(reader: BufReader<R>) -> Self {
        Self { reader }
    }

    pub fn read_header(&mut self) -> AppResult<u32> {
        let mut magic = vec![0_u8; MAGIC.len()];
        self.reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err("invalid GPT2 NVFP4 checkpoint magic".into());
        }

        let version = self.read_u32()?;
        if version != VERSION {
            return Err(format!("unsupported checkpoint version {version}").into());
        }

        self.read_u32()
    }

    pub fn read_tensor(&mut self) -> AppResult<CheckpointTensor> {
        let name_len = self.read_u32()? as usize;
        let len = self.read_u64()? as usize;
        let byte_len = self.read_u64()? as usize;
        let scale_len = self.read_u64()? as usize;

        let mut global_scale = [0_u8; 4];
        self.reader.read_exact(&mut global_scale)?;
        let global_scale = f32::from_le_bytes(global_scale);

        let mut name = vec![0_u8; name_len];
        self.reader.read_exact(&mut name)?;
        let name = String::from_utf8(name)?;

        let mut bytes = vec![0_u8; byte_len];
        self.reader.read_exact(&mut bytes)?;
        let mut scales = vec![0_u8; scale_len];
        self.reader.read_exact(&mut scales)?;

        Ok(CheckpointTensor {
            name,
            len,
            global_scale,
            bytes,
            scales,
        })
    }

    fn read_u32(&mut self) -> AppResult<u32> {
        let mut bytes = [0_u8; 4];
        self.reader.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> AppResult<u64> {
        let mut bytes = [0_u8; 8];
        self.reader.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
}
