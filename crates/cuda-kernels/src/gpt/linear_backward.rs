use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};

use crate::mma::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_THREADS_PER_BLOCK,
    Nvfp4FourSixMmaWeightTensor, Nvfp4ProjectionParams, nvfp4_projection_nobias_kernel_body,
    projection_grid_dim,
};
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;
use crate::nvfp4_quant::{MsEdenQuantArgs, Nvfp4QuantModule};

pub const QUARTET_BACKWARD_SCALE_OVERRIDE: f32 = (17.0 / 16.0) * 0.93;

pub struct LinearBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub e_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub e_t_h: Nvfp4RowwiseDeviceTensor<'a>,
    pub input_t_h: Nvfp4FourSixMmaWeightTensor<'a>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct MsEdenOperandScratch<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub global_scale: f32,
}

impl<'a> MsEdenOperandScratch<'a> {
    fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scales: &*self.global_scales,
        }
    }

    fn mma_weight(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor {
            bytes: &*self.bytes,
            scales: &*self.scales,
            global_scale: self.global_scale,
        }
    }
}

pub struct LinearBackwardMsEdenScratch<'a> {
    pub e_h: MsEdenOperandScratch<'a>,
    pub weight_t_h: MsEdenOperandScratch<'a>,
    pub e_t_h: MsEdenOperandScratch<'a>,
    pub input_t_h: MsEdenOperandScratch<'a>,
}

pub struct LinearBackwardMsEdenArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub quant_module: &'a Nvfp4QuantModule,
    pub e: &'a DeviceBuffer<f32>,
    pub weight_t: &'a DeviceBuffer<f32>,
    pub e_t: &'a DeviceBuffer<f32>,
    pub input_t: &'a DeviceBuffer<f32>,
    pub scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub struct LinearBackwardModule {
    module: kernels::LoadedModule,
}

impl LinearBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn backward(&self, args: LinearBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.linear_backward_projection_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_grid_dim(args.token_count, args.input_dim),
                block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.e_h.bytes,
            args.e_h.scales,
            args.e_h.global_scales,
            args.weight_t_h.bytes,
            args.weight_t_h.scales,
            args.dinput,
            Nvfp4ProjectionParams {
                token_count: args.token_count,
                input_dim: args.output_dim,
                output_dim: args.input_dim,
                weight_global_scale: args.weight_t_h.global_scale,
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_NONE,
            },
        )?;

        self.module.linear_backward_projection_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_grid_dim(args.output_dim, args.input_dim),
                block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.e_t_h.bytes,
            args.e_t_h.scales,
            args.e_t_h.global_scales,
            args.input_t_h.bytes,
            args.input_t_h.scales,
            args.dweight,
            Nvfp4ProjectionParams {
                token_count: args.output_dim,
                input_dim: args.token_count,
                output_dim: args.input_dim,
                weight_global_scale: args.input_t_h.global_scale,
                bias_global_scale: 0.0,
                residual_add: 0,
                activation: NVFP4_PROJECTION_ACTIVATION_NONE,
            },
        )
    }

    pub fn backward_ms_eden(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        let scratch = args.scratch;
        quantize_operand(
            args.quant_module,
            QuantizeOperandArgs {
                stream: args.stream,
                x: args.e,
                scratch: &mut *scratch.e_h.bytes,
                scales: &mut *scratch.e_h.scales,
                global_scales: &mut *scratch.e_h.global_scales,
                chunk_amax: &mut *scratch.e_h.chunk_amax,
                row_count: args.token_count,
                row_len: args.output_dim,
                global_scale: scratch.e_h.global_scale,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed,
            },
        )?;
        quantize_operand(
            args.quant_module,
            QuantizeOperandArgs {
                stream: args.stream,
                x: args.weight_t,
                scratch: &mut *scratch.weight_t_h.bytes,
                scales: &mut *scratch.weight_t_h.scales,
                global_scales: &mut *scratch.weight_t_h.global_scales,
                chunk_amax: &mut *scratch.weight_t_h.chunk_amax,
                row_count: args.input_dim,
                row_len: args.output_dim,
                global_scale: scratch.weight_t_h.global_scale,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed ^ 0x9e37_79b9,
            },
        )?;
        quantize_operand(
            args.quant_module,
            QuantizeOperandArgs {
                stream: args.stream,
                x: args.e_t,
                scratch: &mut *scratch.e_t_h.bytes,
                scales: &mut *scratch.e_t_h.scales,
                global_scales: &mut *scratch.e_t_h.global_scales,
                chunk_amax: &mut *scratch.e_t_h.chunk_amax,
                row_count: args.output_dim,
                row_len: args.token_count,
                global_scale: scratch.e_t_h.global_scale,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed ^ 0x85eb_ca6b,
            },
        )?;
        quantize_operand(
            args.quant_module,
            QuantizeOperandArgs {
                stream: args.stream,
                x: args.input_t,
                scratch: &mut *scratch.input_t_h.bytes,
                scales: &mut *scratch.input_t_h.scales,
                global_scales: &mut *scratch.input_t_h.global_scales,
                chunk_amax: &mut *scratch.input_t_h.chunk_amax,
                row_count: args.input_dim,
                row_len: args.token_count,
                global_scale: scratch.input_t_h.global_scale,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed ^ 0xc2b2_ae35,
            },
        )?;

        self.backward(LinearBackwardArgs {
            stream: args.stream,
            e_h: scratch.e_h.rowwise(),
            weight_t_h: scratch.weight_t_h.mma_weight(),
            e_t_h: scratch.e_t_h.rowwise(),
            input_t_h: scratch.input_t_h.mma_weight(),
            dinput: args.dinput,
            dweight: args.dweight,
            token_count: args.token_count,
            input_dim: args.input_dim,
            output_dim: args.output_dim,
        })
    }
}

struct QuantizeOperandArgs<'a, 'out> {
    stream: &'a CudaStream,
    x: &'a DeviceBuffer<f32>,
    scratch: &'out mut DeviceBuffer<u8>,
    scales: &'out mut DeviceBuffer<u8>,
    global_scales: &'out mut DeviceBuffer<f32>,
    chunk_amax: &'out mut DeviceBuffer<f32>,
    row_count: u32,
    row_len: u32,
    global_scale: f32,
    sign_seed: u32,
    scale_seed: u32,
}

fn quantize_operand(
    module: &Nvfp4QuantModule,
    args: QuantizeOperandArgs<'_, '_>,
) -> Result<(), DriverError> {
    module.fp32_to_nvfp4_ms_eden(MsEdenQuantArgs {
        stream: args.stream,
        x: args.x,
        out_fp4: args.scratch,
        out_scales: args.scales,
        out_global_scales: args.global_scales,
        out_chunk_amax: args.chunk_amax,
        row_count: args.row_count,
        row_len: args.row_len,
        global_scale: args.global_scale,
        scale_override: QUARTET_BACKWARD_SCALE_OVERRIDE,
        sign_seed: args.sign_seed,
        scale_seed: args.scale_seed,
    })
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn linear_backward_projection_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        nvfp4_projection_nobias_kernel_body(
            input_bytes,
            input_scales,
            input_global_scales,
            weight_bytes,
            weight_scales,
            &mut out,
            params,
        );
    }
}
