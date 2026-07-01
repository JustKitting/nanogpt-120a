use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, LinearScratch, NEXTLAT_HIDDEN, NEXTLAT_INPUT};

pub struct NextLatScratchBuffers {
    pub output_projection: LinearScratch,
    pub transition: LinearScratch,
    pub input_projection: LinearScratch,
}

impl NextLatScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            output_projection: LinearScratch::new(stream, NEXTLAT_HIDDEN, GPT2_N_EMBD)?,
            transition: LinearScratch::new(stream, NEXTLAT_HIDDEN, NEXTLAT_HIDDEN)?,
            input_projection: LinearScratch::new(stream, NEXTLAT_INPUT, NEXTLAT_HIDDEN)?,
        })
    }
}
