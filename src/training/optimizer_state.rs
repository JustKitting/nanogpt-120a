use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::GPT2_N_LAYER;
use rust_kernels_cuda::nvfp4::{Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4DeviceTensor};

use crate::upload::{
    UploadedBlock, UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNvfp4,
};

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
    pub(super) master: DeviceBuffer<f32>,
    pub(super) first: DeviceBuffer<f32>,
    pub(super) second: DeviceBuffer<f32>,
}

pub(super) struct AuroraState {
    pub(super) master: DeviceBuffer<f32>,
    pub(super) momentum: DeviceBuffer<f32>,
}

impl OptimizerStateBuffers {
    pub fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        uploaded: &UploadedModel,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            step: 0,
            token_embedding: AdamState::new(stream, decode, &uploaded.token_embedding)?,
            ln_f: LayerNormState::new(stream, decode, &uploaded.ln_f)?,
            blocks: block_array(|i| BlockState::new(stream, decode, &uploaded.blocks[i]))?,
        })
    }

    pub(super) fn advance(&mut self) -> u32 {
        self.step += 1;
        self.step
    }
}

impl BlockState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        block: &UploadedBlock,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            ln_1: LayerNormState::new(stream, decode, &block.ln_1)?,
            attn_qkv: LinearState::new(stream, decode, &block.attn_qkv)?,
            attn_c_proj: LinearState::new(stream, decode, &block.attn_c_proj)?,
            ln_2: LayerNormState::new(stream, decode, &block.ln_2)?,
            mlp_up: LinearState::new(stream, decode, &block.mlp_up)?,
            mlp_down: LinearState::new(stream, decode, &block.mlp_down)?,
        })
    }
}

impl LayerNormState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        layer_norm: &UploadedLayerNorm,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            weight: AdamState::new(stream, decode, &layer_norm.weight)?,
            bias: AdamState::new(stream, decode, &layer_norm.bias)?,
        })
    }
}

impl LinearState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        linear: &UploadedLinear,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            weight_aurora: AuroraState::new(stream, decode, &linear.weight)?,
            bias: AdamState::new(stream, decode, &linear.bias)?,
        })
    }
}

impl AdamState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        tensor: &UploadedNvfp4,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            master: decode_master(stream, decode, tensor)?,
            first: DeviceBuffer::zeroed(stream, tensor.len)?,
            second: DeviceBuffer::zeroed(stream, tensor.len)?,
        })
    }
}

impl AuroraState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        tensor: &UploadedNvfp4,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            master: decode_master(stream, decode, tensor)?,
            momentum: DeviceBuffer::zeroed(stream, tensor.len)?,
        })
    }
}

fn decode_master(
    stream: &CudaStream,
    decode: &Nvfp4DecodeModule,
    tensor: &UploadedNvfp4,
) -> Result<DeviceBuffer<f32>, DriverError> {
    let mut master = DeviceBuffer::zeroed(stream, tensor.len)?;
    decode.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
        stream,
        input: Nvfp4DeviceTensor {
            bytes: &tensor.bytes,
            scales: &tensor.scales,
            global_scale: &tensor.global_scale,
        },
        output: &mut master,
        rows: 1,
        cols: tensor.len as u32,
    })?;
    Ok(master)
}

fn block_array<F, T>(mut f: F) -> Result<[T; GPT2_N_LAYER], DriverError>
where
    F: FnMut(usize) -> Result<T, DriverError>,
{
    let values = (0..GPT2_N_LAYER)
        .map(|i| f(i))
        .collect::<Result<Vec<_>, _>>()?;
    match values.try_into() {
        Ok(array) => Ok(array),
        Err(_) => unreachable!("block array length must match GPT2_N_LAYER"),
    }
}
