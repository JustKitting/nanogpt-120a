use std::error::Error;

use cuda_core::CudaContext;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::Nvfp4TcMatmulModule;

use crate::common;

#[path = "polar_nvfp4/device.rs"]
mod device;
#[path = "polar_nvfp4/math.rs"]
mod math;
#[path = "polar_nvfp4/scratch.rs"]
mod scratch;

const ROWS: usize = 32;
const COLS: usize = 64;
const MAX_ITERATIONS: usize = 7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_rht_polar_estimator_reports_update_error() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(common::gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(common::ptx_path().as_str())?;
    let matmul = Nvfp4TcMatmulModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let polar = device::Nvfp4Polar::new(&stream, &matmul, &quant);

    let source = math::normalized_source(&math::gradient(ROWS, COLS), ROWS, COLS);
    let gram_expected = math::matmul_f16_leaf(&source, &source, ROWS, ROWS, COLS);
    let gram_actual = polar.product(&source, &source, ROWS, ROWS, COLS, 0, 0)?;
    println!(
        "nvfp4_rht_polar first_gram cosine={:.8} rel_l2={:.8e} max_abs={:.8e}",
        math::cosine(&gram_actual, &gram_expected),
        math::relative_l2(&gram_actual, &gram_expected),
        math::max_abs_error(&gram_actual, &gram_expected)
    );
    let source_t = math::transpose(&source, ROWS, COLS);
    let ax_expected = math::matmul_f16_leaf(&gram_expected, &source_t, ROWS, COLS, ROWS);
    let ax_actual = polar.product(&gram_expected, &source_t, ROWS, COLS, ROWS, 0, 1)?;
    println!(
        "nvfp4_rht_polar first_ax_from_expected_gram cosine={:.8} rel_l2={:.8e} max_abs={:.8e}",
        math::cosine(&ax_actual, &ax_expected),
        math::relative_l2(&ax_actual, &ax_expected),
        math::max_abs_error(&ax_actual, &ax_expected)
    );
    let ax_t = math::transpose(&ax_expected, ROWS, COLS);
    let aax_expected = math::matmul_f16_leaf(&gram_expected, &ax_t, ROWS, COLS, ROWS);
    let aax_actual = polar.product(&gram_expected, &ax_t, ROWS, COLS, ROWS, 0, 2)?;
    println!(
        "nvfp4_rht_polar first_aax_from_expected_inputs cosine={:.8} rel_l2={:.8e} max_abs={:.8e}",
        math::cosine(&aax_actual, &aax_expected),
        math::relative_l2(&aax_actual, &aax_expected),
        math::max_abs_error(&aax_actual, &aax_expected)
    );

    for iterations in 1..=MAX_ITERATIONS {
        let expected = math::polar_iterations_f16_leaf(source.clone(), ROWS, COLS, iterations);
        let actual = polar.iterations(source.clone(), ROWS, COLS, iterations)?;
        let cosine = math::cosine(&actual, &expected);
        let rel_l2 = math::relative_l2(&actual, &expected);
        let max_abs = math::max_abs_error(&actual, &expected);
        println!(
            "nvfp4_rht_polar iterations={iterations} cosine={cosine:.8} rel_l2={rel_l2:.8e} max_abs={max_abs:.8e}"
        );
    }

    let expected = math::polar_iterations_f16_leaf(source.clone(), ROWS, COLS, 5);
    for fp4_prefix in 1..=3 {
        let mut actual = source.clone();
        for iter in 0..5 {
            actual = if iter < fp4_prefix {
                polar.step(&actual, ROWS, COLS, iter)?
            } else {
                math::polar_step_f16_leaf(&actual, ROWS, COLS, iter)
            };
        }
        println!(
            "nvfp4_rht_polar hybrid_fp4_prefix={fp4_prefix} cosine={:.8} rel_l2={:.8e} max_abs={:.8e}",
            math::cosine(&actual, &expected),
            math::relative_l2(&actual, &expected),
            math::max_abs_error(&actual, &expected)
        );
    }

    Ok(())
}
