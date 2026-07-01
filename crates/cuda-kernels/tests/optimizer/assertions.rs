use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};

use crate::common;

pub fn assert_update_matches(
    stream: &CudaStream,
    x_master: DeviceBuffer<f32>,
    z_master: DeviceBuffer<f32>,
    expected: f32,
) -> Result<(), Box<dyn Error>> {
    let x_master = x_master.to_host_vec(stream)?;
    let z_master = z_master.to_host_vec(stream)?;
    common::assert_all_close(&x_master, expected, 1.0e-7);
    common::assert_all_close(&z_master, expected, 1.0e-7);
    Ok(())
}
