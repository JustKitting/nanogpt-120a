use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig, memory};

use super::args::{
    MsEdenDeviceScaleQuantArgs, MsEdenQuantArgs, Nvfp4QuantArgs, Nvfp4QuantRowwiseArgs,
    QuartetBackwardMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenQuantArgs, RowAmaxArgs,
};
use super::config::{GROUP_SIZE_U32, THREADS_PER_BLOCK};
use super::kernels;
use crate::quartet::{QUARTET_MS_EDEN_SCALE_OVERRIDE, quartet_backward_ms_eden_global_scale};

const SCALE_OVERRIDE: f32 = 1.0;

pub struct Nvfp4QuantModule {
    row_amax: kernels::row_amax::module::LoadedModule,
    four_six: kernels::four_six::module::LoadedModule,
    ms_eden: kernels::ms_eden::module::LoadedModule,
}

impl Nvfp4QuantModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            row_amax: kernels::row_amax::module::from_module(module.clone())?,
            four_six: kernels::four_six::module::from_module(module.clone())?,
            ms_eden: kernels::ms_eden::module::from_module(module)?,
        })
    }

    pub fn fp32_to_nvfp4_four_six(&self, args: Nvfp4QuantArgs<'_, '_>) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six(
            args.stream,
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.group_count,
            0,
            0.0,
        )
    }

    pub fn fp32_to_nvfp4_four_six_rowwise(
        &self,
        args: Nvfp4QuantRowwiseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six(
            args.stream,
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.group_count,
            args.row_len,
            0.0,
        )
    }

    pub fn fp32_to_nvfp4_four_six_fixed_global(
        &self,
        args: Nvfp4QuantArgs<'_, '_>,
        global_scale: f32,
    ) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six(
            args.stream,
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.group_count,
            0,
            global_scale,
        )
    }

    pub fn row_amax_f32(&self, args: RowAmaxArgs<'_, '_>) -> Result<(), DriverError> {
        self.row_amax.row_amax_f32_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.row_count, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            args.x,
            args.out,
            args.row_count,
            args.row_len,
        )
    }

    fn tensor_chunk_amax_f32(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        element_count: u32,
    ) -> Result<u32, DriverError> {
        let chunk_count = element_count.div_ceil(kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK);
        self.row_amax.tensor_chunk_amax_f32_kernel(
            stream,
            LaunchConfig {
                grid_dim: (chunk_count, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            x,
            out,
            element_count,
        )?;
        Ok(chunk_count)
    }

    pub fn fp32_to_nvfp4_ms_eden(&self, args: MsEdenQuantArgs<'_, '_>) -> Result<(), DriverError> {
        let element_count = args.row_count * args.dst_row_len;
        self.ms_eden.fp32_to_nvfp4_ms_eden_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (element_count.div_ceil(32), 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            args.x,
            args.out_fp4,
            args.out_scales,
            args.out_global_scales,
            args.out_chunk_amax,
            args.src_row_len,
            args.dst_row_len,
            args.global_scale,
            args.scale_override,
            args.sign_seed,
            args.scale_seed,
        )
    }

    pub fn fp32_to_nvfp4_ms_eden_device_scale(
        &self,
        args: MsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let element_count = args.row_count * args.dst_row_len;
        self.ms_eden.fp32_to_nvfp4_ms_eden_device_scale_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (element_count.div_ceil(32), 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            args.x,
            args.out_fp4,
            args.out_scales,
            args.out_global_scales,
            args.out_chunk_amax,
            args.global_scale,
            args.src_row_len,
            args.dst_row_len,
            args.scale_override,
            args.sign_seed,
            args.scale_seed,
        )
    }

    pub fn fp32_to_nvfp4_quartet_backward_ms_eden(
        &self,
        args: QuartetBackwardMsEdenQuantArgs<'_, '_>,
    ) -> Result<f32, DriverError> {
        let args = args;
        self.row_amax_f32(RowAmaxArgs {
            stream: args.stream,
            x: args.x,
            out: &mut *args.out_chunk_amax,
            row_count: 1,
            row_len: args.row_count * args.src_row_len,
        })?;

        let global_scale = quartet_backward_ms_eden_global_scale(copy_first_f32(
            args.stream,
            args.out_chunk_amax,
        )?);
        self.fp32_to_nvfp4_quartet_backward_ms_eden_with_global_scale(args, global_scale)
    }

    pub fn fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        &self,
        args: QuartetBackwardMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let args = args;
        let element_count = args.row_count * args.src_row_len;
        let chunk_count = self.tensor_chunk_amax_f32(
            args.stream,
            args.x,
            &mut *args.out_chunk_amax,
            element_count,
        )?;

        self.ms_eden
            .quartet_backward_ms_eden_global_scale_from_chunks_kernel(
                args.stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                },
                &*args.out_chunk_amax,
                &mut *args.out_global_scale,
                chunk_count,
            )?;

        self.fp32_to_nvfp4_ms_eden_device_scale(MsEdenDeviceScaleQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: args.out_fp4,
            out_scales: args.out_scales,
            out_global_scales: args.out_global_scales,
            out_chunk_amax: args.out_chunk_amax,
            global_scale: &*args.out_global_scale,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        })
    }

    pub fn fp32_to_nvfp4_quartet_backward_ms_eden_with_amax(
        &self,
        args: QuartetBackwardMsEdenQuantArgs<'_, '_>,
        amax: f32,
    ) -> Result<f32, DriverError> {
        let global_scale = quartet_backward_ms_eden_global_scale(amax);
        self.fp32_to_nvfp4_quartet_backward_ms_eden_with_global_scale(args, global_scale)
    }

    pub fn fp32_to_nvfp4_quartet_backward_ms_eden_with_global_scale(
        &self,
        args: QuartetBackwardMsEdenQuantArgs<'_, '_>,
        global_scale: f32,
    ) -> Result<f32, DriverError> {
        self.launch_quartet_backward_ms_eden(args, global_scale)?;
        Ok(global_scale)
    }

    fn launch_quartet_backward_ms_eden(
        &self,
        args: QuartetBackwardMsEdenQuantArgs<'_, '_>,
        global_scale: f32,
    ) -> Result<(), DriverError> {
        self.fp32_to_nvfp4_ms_eden(MsEdenQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: args.out_fp4,
            out_scales: args.out_scales,
            out_global_scales: args.out_global_scales,
            out_chunk_amax: args.out_chunk_amax,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            global_scale,
            scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_fp32_to_nvfp4_four_six(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        amax: &DeviceBuffer<f32>,
        out_fp4: &mut DeviceBuffer<u8>,
        out_scales: &mut DeviceBuffer<u8>,
        out_global_scale: &mut DeviceBuffer<f32>,
        group_count: u32,
        row_len: u32,
        fixed_global_scale: f32,
    ) -> Result<(), DriverError> {
        let groups_per_block = THREADS_PER_BLOCK / GROUP_SIZE_U32;

        self.four_six.fp32_to_nvfp4_four_six_kernel(
            stream,
            LaunchConfig {
                grid_dim: (group_count.div_ceil(groups_per_block), 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            x,
            amax,
            out_fp4,
            out_scales,
            out_global_scale,
            row_len,
            SCALE_OVERRIDE,
            fixed_global_scale,
        )
    }
}

fn copy_first_f32(stream: &CudaStream, buffer: &DeviceBuffer<f32>) -> Result<f32, DriverError> {
    let mut value = 0.0f32;
    unsafe {
        memory::memcpy_dtoh_async(
            &mut value as *mut f32,
            buffer.cu_deviceptr(),
            std::mem::size_of::<f32>(),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()?;
    Ok(value)
}
