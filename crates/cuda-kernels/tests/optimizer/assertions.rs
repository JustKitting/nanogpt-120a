use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};

pub fn assert_update_matches(
    stream: &CudaStream,
    x_master: DeviceBuffer<f32>,
    z_master: DeviceBuffer<f32>,
    expected: f32,
) -> Result<(), Box<dyn Error>> {
    let x_master = x_master.to_host_vec(stream)?;
    let z_master = z_master.to_host_vec(stream)?;
    let max_x_error = max_abs_error(&x_master, expected);
    let max_z_error = max_abs_error(&z_master, expected);
    assert!(
        x_master.iter().all(|value| value.is_finite())
            && z_master.iter().all(|value| value.is_finite())
            && max_x_error <= 1.0e-7
            && max_z_error <= 1.0e-7,
        "expected={expected:.10e} x0={:.10e} z0={:.10e} max_x_error={max_x_error:.10e} max_z_error={max_z_error:.10e}",
        x_master[0],
        z_master[0]
    );
    Ok(())
}

fn max_abs_error(values: &[f32], expected: f32) -> f32 {
    values
        .iter()
        .map(|value| (*value - expected).abs())
        .fold(0.0_f32, f32::max)
}
