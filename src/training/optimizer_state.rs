use cuda_core::{CudaStream, DeviceBuffer, DriverError, memory};
use gpt2_nvfp4::GPT2_N_LAYER;
use rust_kernels_cuda::nvfp4::{Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs, Nvfp4DeviceTensor};

use crate::upload::{
    UploadedBlock, UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNextLat, UploadedNvfp4,
};

use super::update_skip::{UpdateSkipDecision, UpdateSkipState};

pub struct OptimizerStateBuffers {
    step: u32,
    schedule_free_weight_sum: f32,
    update_skip: UpdateSkipState,
    pub(super) token_embedding: AdamState,
    pub(super) ln_f: LayerNormState,
    pub(super) next_latent: NextLatState,
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

pub(super) struct NextLatState {
    pub(super) norm: LayerNormState,
    pub(super) input_projection: LinearState,
    pub(super) transition: LinearState,
    pub(super) output_projection: LinearState,
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
    pub(super) z_master: DeviceBuffer<f32>,
    pub(super) x_master: DeviceBuffer<f32>,
    pub(super) first: DeviceBuffer<f32>,
    pub(super) second: DeviceBuffer<f32>,
}

pub(super) struct AuroraState {
    pub(super) z_master: DeviceBuffer<f32>,
    pub(super) x_master: DeviceBuffer<f32>,
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
            schedule_free_weight_sum: 0.0,
            update_skip: UpdateSkipState::new(),
            token_embedding: AdamState::new(stream, decode, &uploaded.token_embedding)?,
            ln_f: LayerNormState::new(stream, decode, &uploaded.ln_f)?,
            next_latent: NextLatState::new(stream, decode, &uploaded.next_latent)?,
            blocks: block_array(|i| BlockState::new(stream, decode, &uploaded.blocks[i]))?,
        })
    }

    pub(super) fn advance(&mut self) -> u32 {
        self.step += 1;
        self.step
    }

    pub(super) fn schedule_free_average_coefficient(&mut self, step: u32) -> f32 {
        super::learning_rate::schedule_free_average_coefficient(
            step,
            &mut self.schedule_free_weight_sum,
        )
    }

    pub(super) fn next_step(&self) -> u32 {
        self.step + 1
    }

    pub(super) fn should_skip_update(
        &mut self,
        loss: Option<f32>,
        grad_norm: f32,
    ) -> UpdateSkipDecision {
        self.update_skip.observe(loss, grad_norm)
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

impl NextLatState {
    fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        next_latent: &UploadedNextLat,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            norm: LayerNormState::new(stream, decode, &next_latent.norm)?,
            input_projection: LinearState::new(stream, decode, &next_latent.input_projection)?,
            transition: LinearState::new(stream, decode, &next_latent.transition)?,
            output_projection: LinearState::new(stream, decode, &next_latent.output_projection)?,
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
        let master = decode_master(stream, decode, tensor)?;
        Ok(Self {
            z_master: clone_device(stream, &master)?,
            x_master: master,
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
        let master = decode_master(stream, decode, tensor)?;
        Ok(Self {
            z_master: clone_device(stream, &master)?,
            x_master: master,
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

fn clone_device(
    stream: &CudaStream,
    buffer: &DeviceBuffer<f32>,
) -> Result<DeviceBuffer<f32>, DriverError> {
    let cloned = DeviceBuffer::zeroed(stream, buffer.len())?;
    stream.context().bind_to_thread()?;

    unsafe {
        memory::memcpy_dtod_async(
            cloned.cu_deviceptr(),
            buffer.cu_deviceptr(),
            buffer.num_bytes(),
            stream.cu_stream(),
        )?;
    }

    Ok(cloned)
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
