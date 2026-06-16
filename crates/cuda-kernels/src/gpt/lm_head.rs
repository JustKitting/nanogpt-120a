use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::mma::{
    NVFP4_PROJECTION_M, NVFP4_PROJECTION_N, NVFP4_PROJECTION_THREADS_PER_BLOCK,
    Nvfp4FourSixMmaWeightTensor, Nvfp4ProjectionParams, Nvfp4ProjectionTile,
    nvfp4_projection_accumulate_tile, projection_grid_dim,
};
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LmHeadParams {
    pub token_count: u32,
    pub input_dim: u32,
    pub vocab_size: u32,
    pub weight_global_scale: f32,
}

unsafe impl DeviceCopy for LmHeadParams {}

pub struct LmHeadArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub logits: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub vocab_size: u32,
}

pub struct LmHeadModule {
    module: kernels::LoadedModule,
}

impl LmHeadModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn logits(&self, args: LmHeadArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.lm_head_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: projection_grid_dim(args.token_count, args.vocab_size),
                block_dim: (NVFP4_PROJECTION_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.logits,
            LmHeadParams {
                token_count: args.token_count,
                input_dim: args.input_dim,
                vocab_size: args.vocab_size,
                weight_global_scale: args.weight.global_scale,
            },
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn lm_head_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        mut logits: DisjointSlice<f32>,
        params: LmHeadParams,
    ) {
        let lane = thread::threadIdx_x();
        if lane >= NVFP4_PROJECTION_THREADS_PER_BLOCK {
            return;
        }

        let tile_col = thread::blockIdx_x() * NVFP4_PROJECTION_N;
        let tile_row = thread::blockIdx_y() * NVFP4_PROJECTION_M;
        let group = lane >> 2;
        let thread_in_group = lane & 0x3;
        let tile = Nvfp4ProjectionTile {
            tile_row,
            tile_col,
            group,
            thread_in_group,
        };
        let projection_params = Nvfp4ProjectionParams {
            token_count: params.token_count,
            input_dim: params.input_dim,
            output_dim: params.vocab_size,
            weight_global_scale: params.weight_global_scale,
            bias_global_scale: 0.0,
            residual_add: 0,
            activation: 0,
        };
        let acc = nvfp4_projection_accumulate_tile(
            input_bytes,
            input_scales,
            weight_bytes,
            weight_scales,
            tile,
            &projection_params,
        );

        store_logits(acc, input_global_scales, tile, &params, &mut logits);
    }

    #[inline(always)]
    fn store_logits(
        acc: [f32; 4],
        input_global_scales: &[f32],
        tile: Nvfp4ProjectionTile,
        params: &LmHeadParams,
        logits: &mut DisjointSlice<'_, f32>,
    ) {
        let mut i = 0;
        while i < 4 {
            let row = tile.tile_row + tile.group + if i < 2 { 0 } else { 8 };
            let col = tile.tile_col + tile.thread_in_group * 2 + (i & 1);

            if row < params.token_count && col < params.vocab_size {
                let scale = input_global_scales[row as usize] * params.weight_global_scale;
                let value = acc[i as usize] * scale;
                let index = row as usize * params.vocab_size as usize + col as usize;

                unsafe {
                    *logits.get_unchecked_mut(index) = value;
                }
            }

            i += 1;
        }
    }
}
