use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantArgs, Nvfp4QuantModule};

mod common;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_to_nvfp4_four_six_writes_quantized_outputs() -> Result<(), Box<dyn Error>> {
    let x = [
        -3.25f32, -2.0, -1.25, -0.5, -0.125, 0.0, 0.25, 0.75, 1.0, 1.5, 2.25, 3.0, 4.0, 5.0, 6.5,
        8.0,
    ];
    let amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];

    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let module =
        Nvfp4QuantModule::from_module(ctx.load_module_from_file(common::ptx_path().as_str())?)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut fp4_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut scales_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut global_scale_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    module.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut fp4_dev,
        out_scales: &mut scales_dev,
        out_global_scale: &mut global_scale_dev,
        group_count: 1,
        scale_override: 1.0,
    })?;

    let fp4 = fp4_dev.to_host_vec(&stream)?;
    let scales = scales_dev.to_host_vec(&stream)?;
    let global_scale = global_scale_dev.to_host_vec(&stream)?;

    assert!(fp4.iter().any(|byte| *byte != 0));
    assert!(scales.iter().any(|byte| *byte != 0));
    assert!((global_scale[0] - 8.0 / (256.0 * 6.0)).abs() <= 1.0e-8);
    Ok(())
}
