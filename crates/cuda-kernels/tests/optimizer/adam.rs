use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::optimizer::{AdamWUpdateArgs, OptimizerModule};

use crate::common;
use crate::common::nvfp4::{E2M1_ONE_PAIR, E4M3_ONE};

const LEN: usize = 32;

struct AdamFixture {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    z_master: DeviceBuffer<f32>,
    x_master: DeviceBuffer<f32>,
    grad: DeviceBuffer<f32>,
    first: DeviceBuffer<f32>,
    second: DeviceBuffer<f32>,
    amax: DeviceBuffer<f32>,
    chunk_amax: DeviceBuffer<f32>,
    global_scale: DeviceBuffer<f32>,
}

impl AdamFixture {
    fn new(stream: &CudaStream) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            bytes: DeviceBuffer::from_host(stream, &[E2M1_ONE_PAIR; LEN / 2])?,
            scales: DeviceBuffer::from_host(stream, &[E4M3_ONE; LEN / 16])?,
            z_master: DeviceBuffer::from_host(stream, &[1.0_f32; LEN])?,
            x_master: DeviceBuffer::from_host(stream, &[1.0_f32; LEN])?,
            grad: DeviceBuffer::from_host(stream, &[0.5_f32; LEN])?,
            first: DeviceBuffer::<f32>::zeroed(stream, LEN)?,
            second: DeviceBuffer::<f32>::zeroed(stream, LEN)?,
            amax: DeviceBuffer::<f32>::zeroed(stream, 1)?,
            chunk_amax: DeviceBuffer::<f32>::zeroed(stream, 1)?,
            global_scale: DeviceBuffer::from_host(stream, &[1.0_f32])?,
        })
    }

    fn apply(
        &mut self,
        stream: &CudaStream,
        module: &OptimizerModule,
        average_coefficient: f32,
    ) -> Result<(), Box<dyn Error>> {
        module.apply_adamw_update(AdamWUpdateArgs {
            stream,
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scale: &mut self.global_scale,
            z_master: &mut self.z_master,
            x_master: &mut self.x_master,
            grad: &self.grad,
            first_moment: &mut self.first,
            second_moment: &mut self.second,
            amax: &mut self.amax,
            chunk_amax: &mut self.chunk_amax,
            len: LEN as u32,
            learning_rate: 0.25,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            beta1_correction: 0.1,
            beta2_correction: 0.05,
            eps: 1.0e-10,
            average_coefficient,
        })?;
        Ok(())
    }
}

fn assert_all_close(values: &[f32], expected: f32) {
    assert!(
        values
            .iter()
            .all(|value| (*value - expected).abs() <= 1.0e-6)
    );
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_adamw_update_tracks_moments_and_requantizes() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = OptimizerModule::from_module(ptx)?;
    let mut fixture = AdamFixture::new(&stream)?;

    fixture.apply(&stream, &module, 1.0)?;
    let z_master = fixture.z_master.to_host_vec(&stream)?;
    let x_master = fixture.x_master.to_host_vec(&stream)?;
    let first = fixture.first.to_host_vec(&stream)?;
    let second = fixture.second.to_host_vec(&stream)?;
    assert_all_close(&z_master, 0.725);
    assert_all_close(&x_master, 0.725);
    assert_all_close(&first, 0.05);
    assert_all_close(&second, 0.0125);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_adamw_update_applies_schedule_free_average() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = OptimizerModule::from_module(ptx)?;
    let mut fixture = AdamFixture::new(&stream)?;

    fixture.apply(&stream, &module, 0.25)?;
    let z_master = fixture.z_master.to_host_vec(&stream)?;
    let x_master = fixture.x_master.to_host_vec(&stream)?;
    assert_all_close(&z_master, 0.725);
    assert_all_close(&x_master, 0.93125);
    Ok(())
}
