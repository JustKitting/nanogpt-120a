use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::{GRAD_CLIP_VALUES_PER_CHUNK, GradientClipArgs, OptimizerModule};

use crate::common;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn global_clip_scales_all_gradient_buffers_together() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = OptimizerModule::from_module(ptx)?;

    let first = DeviceBuffer::from_host(&stream, &[3.0_f32, 4.0, 0.0, 0.0])?;
    let second = DeviceBuffer::from_host(&stream, &[12.0_f32, 0.0, 0.0, 0.0])?;
    let ptrs = DeviceBuffer::from_host(&stream, &[first.cu_deviceptr(), second.cu_deviceptr()])?;
    let lens = DeviceBuffer::from_host(&stream, &[4_u32, 4])?;
    let chunk_offsets = DeviceBuffer::from_host(&stream, &[0_u32, chunks(4)])?;
    let mut chunk_sums = DeviceBuffer::<f32>::zeroed(&stream, chunks(4) as usize * 2)?;
    let mut scale = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut norm = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    module.clip_gradients(GradientClipArgs {
        stream: &stream,
        ptrs: &ptrs,
        lens: &lens,
        chunk_offsets: &chunk_offsets,
        chunk_sums: &mut chunk_sums,
        scale: &mut scale,
        norm: &mut norm,
        slot_count: 2,
        chunk_count: chunks(4) * 2,
        max_norm: 6.5,
    })?;

    assert_close(&first.to_host_vec(&stream)?, &[1.5, 2.0, 0.0, 0.0]);
    assert_close(&second.to_host_vec(&stream)?, &[6.0, 0.0, 0.0, 0.0]);
    assert_close(&scale.to_host_vec(&stream)?, &[0.5]);
    assert_close(&norm.to_host_vec(&stream)?, &[13.0]);
    Ok(())
}

fn chunks(len: u32) -> u32 {
    len.div_ceil(GRAD_CLIP_VALUES_PER_CHUNK as u32)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((*actual - *expected).abs() <= 1.0e-6);
    }
}
