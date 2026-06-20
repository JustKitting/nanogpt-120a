use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::optimizer::{AdamWUpdateArgs, OptimizerModule};

use crate::common;

const LEN: usize = 32;
const E2M1_ONE_PAIR: u8 = 0x22;
const E4M3_ONE: u8 = 0x38;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_adamw_update_tracks_moments_and_requantizes() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        OptimizerModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let mut bytes = DeviceBuffer::from_host(&stream, &[E2M1_ONE_PAIR; LEN / 2])?;
    let mut scales = DeviceBuffer::from_host(&stream, &[E4M3_ONE; LEN / 16])?;
    let mut z_master = DeviceBuffer::from_host(&stream, &[1.0_f32; LEN])?;
    let mut x_master = DeviceBuffer::from_host(&stream, &[1.0_f32; LEN])?;
    let grad = DeviceBuffer::from_host(&stream, &[0.5_f32; LEN])?;
    let mut first = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut second = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut global_scale = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    module.apply_adamw_update(AdamWUpdateArgs {
        stream: &stream,
        bytes: &mut bytes,
        scales: &mut scales,
        global_scale: &mut global_scale,
        z_master: &mut z_master,
        x_master: &mut x_master,
        grad: &grad,
        first_moment: &mut first,
        second_moment: &mut second,
        amax: &mut amax,
        chunk_amax: &mut chunk_amax,
        len: LEN as u32,
        learning_rate: 0.25,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        beta1_correction: 0.1,
        beta2_correction: 0.05,
        eps: 1.0e-10,
        average_coefficient: 1.0,
    })?;

    let z_master = z_master.to_host_vec(&stream)?;
    let x_master = x_master.to_host_vec(&stream)?;
    let first = first.to_host_vec(&stream)?;
    let second = second.to_host_vec(&stream)?;
    assert!(
        z_master
            .iter()
            .all(|value| (*value - 0.725).abs() <= 1.0e-6)
    );
    assert!(
        x_master
            .iter()
            .all(|value| (*value - 0.725).abs() <= 1.0e-6)
    );
    assert!(first.iter().all(|value| (*value - 0.05).abs() <= 1.0e-6));
    assert!(second.iter().all(|value| (*value - 0.0125).abs() <= 1.0e-6));
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_adamw_update_applies_schedule_free_average() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        OptimizerModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let mut bytes = DeviceBuffer::from_host(&stream, &[E2M1_ONE_PAIR; LEN / 2])?;
    let mut scales = DeviceBuffer::from_host(&stream, &[E4M3_ONE; LEN / 16])?;
    let mut z_master = DeviceBuffer::from_host(&stream, &[1.0_f32; LEN])?;
    let mut x_master = DeviceBuffer::from_host(&stream, &[1.0_f32; LEN])?;
    let grad = DeviceBuffer::from_host(&stream, &[0.5_f32; LEN])?;
    let mut first = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut second = DeviceBuffer::<f32>::zeroed(&stream, LEN)?;
    let mut amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut global_scale = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    module.apply_adamw_update(AdamWUpdateArgs {
        stream: &stream,
        bytes: &mut bytes,
        scales: &mut scales,
        global_scale: &mut global_scale,
        z_master: &mut z_master,
        x_master: &mut x_master,
        grad: &grad,
        first_moment: &mut first,
        second_moment: &mut second,
        amax: &mut amax,
        chunk_amax: &mut chunk_amax,
        len: LEN as u32,
        learning_rate: 0.25,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.95,
        beta1_correction: 0.1,
        beta2_correction: 0.05,
        eps: 1.0e-10,
        average_coefficient: 0.25,
    })?;

    let z_master = z_master.to_host_vec(&stream)?;
    let x_master = x_master.to_host_vec(&stream)?;
    assert!(
        z_master
            .iter()
            .all(|value| (*value - 0.725).abs() <= 1.0e-6)
    );
    assert!(
        x_master
            .iter()
            .all(|value| (*value - 0.93125).abs() <= 1.0e-6)
    );
    Ok(())
}
