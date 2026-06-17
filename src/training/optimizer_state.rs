use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_QKV, GPT2_VOCAB_SIZE};

pub struct OptimizerStateBuffers {
    step: u32,
    pub(super) token_embedding: AdamState,
    pub(super) ln_f: LayerNormState,
    pub(super) blocks: [BlockState; GPT2_N_LAYER],
}

pub(super) struct BlockState {
    pub(super) ln_1: LayerNormState,
    pub(super) attn_qkv: LinearState,
    pub(super) attn_c_proj: LinearState,
    pub(super) ln_2: LayerNormState,
    pub(super) mlp_up: LinearState,
    pub(super) mlp_down: LinearState,
}

pub(super) struct LayerNormState {
    pub(super) weight: AdamState,
    pub(super) bias: AdamState,
}

pub(super) struct LinearState {
    pub(super) weight_aurora: AuroraState,
    pub(super) bias: AdamState,
}

pub(super) struct AdamState {
    pub(super) first: DeviceBuffer<f32>,
    pub(super) second: DeviceBuffer<f32>,
    pub(super) residual: DeviceBuffer<f32>,
}

pub(super) struct AuroraState {
    pub(super) momentum: DeviceBuffer<f32>,
}

impl OptimizerStateBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            step: 0,
            token_embedding: AdamState::new(stream, GPT2_VOCAB_SIZE * GPT2_N_EMBD)?,
            ln_f: LayerNormState::new(stream)?,
            blocks: block_array(|| BlockState::new(stream))?,
        })
    }

    pub(super) fn advance(&mut self) -> u32 {
        self.step += 1;
        self.step
    }
}

impl BlockState {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            ln_1: LayerNormState::new(stream)?,
            attn_qkv: LinearState::new(stream, GPT2_N_EMBD * GPT2_QKV, GPT2_QKV)?,
            attn_c_proj: LinearState::new(stream, GPT2_N_EMBD * GPT2_N_EMBD, GPT2_N_EMBD)?,
            ln_2: LayerNormState::new(stream)?,
            mlp_up: LinearState::new(stream, GPT2_N_EMBD * GPT2_MLP, GPT2_MLP)?,
            mlp_down: LinearState::new(stream, GPT2_MLP * GPT2_N_EMBD, GPT2_N_EMBD)?,
        })
    }
}

impl LayerNormState {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            weight: AdamState::new(stream, GPT2_N_EMBD)?,
            bias: AdamState::new(stream, GPT2_N_EMBD)?,
        })
    }
}

impl LinearState {
    fn new(stream: &CudaStream, weight_len: usize, bias_len: usize) -> Result<Self, DriverError> {
        Ok(Self {
            weight_aurora: AuroraState::new(stream, weight_len)?,
            bias: AdamState::new(stream, bias_len)?,
        })
    }
}

impl AdamState {
    fn new(stream: &CudaStream, len: usize) -> Result<Self, DriverError> {
        Ok(Self {
            first: DeviceBuffer::zeroed(stream, len)?,
            second: DeviceBuffer::zeroed(stream, len)?,
            residual: DeviceBuffer::zeroed(stream, len)?,
        })
    }
}

impl AuroraState {
    fn new(stream: &CudaStream, len: usize) -> Result<Self, DriverError> {
        Ok(Self {
            momentum: DeviceBuffer::zeroed(stream, len)?,
        })
    }
}

fn block_array<F, T>(mut f: F) -> Result<[T; GPT2_N_LAYER], DriverError>
where
    F: FnMut() -> Result<T, DriverError>,
{
    let values = (0..GPT2_N_LAYER)
        .map(|_| f())
        .collect::<Result<Vec<_>, _>>()?;
    match values.try_into() {
        Ok(array) => Ok(array),
        Err(_) => unreachable!("block array length must match GPT2_N_LAYER"),
    }
}
