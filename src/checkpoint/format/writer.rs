use std::io::{BufWriter, Write};

use super::{MAGIC, VERSION};
use crate::AppResult;

pub struct CheckpointWriter<W: Write> {
    writer: BufWriter<W>,
}

impl<W: Write> CheckpointWriter<W> {
    pub fn new(writer: BufWriter<W>) -> Self {
        Self { writer }
    }

    pub fn write_header(&mut self, tensor_count: u32) -> AppResult {
        self.writer.write_all(MAGIC)?;
        self.write_u32(VERSION)?;
        self.write_u32(tensor_count)
    }

    pub fn write_tensor(
        &mut self,
        name: &str,
        len: usize,
        global_scale: f32,
        bytes: &[u8],
        scales: &[u8],
    ) -> AppResult {
        self.write_u32(name.len() as u32)?;
        self.write_u64(len as u64)?;
        self.write_u64(bytes.len() as u64)?;
        self.write_u64(scales.len() as u64)?;
        self.writer.write_all(&global_scale.to_le_bytes())?;
        self.writer.write_all(name.as_bytes())?;
        self.writer.write_all(bytes)?;
        self.writer.write_all(scales)?;
        Ok(())
    }

    pub fn finish(&mut self) -> AppResult {
        self.writer.flush()?;
        Ok(())
    }

    fn write_u32(&mut self, value: u32) -> AppResult {
        self.writer.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    fn write_u64(&mut self, value: u64) -> AppResult {
        self.writer.write_all(&value.to_le_bytes())?;
        Ok(())
    }
}
