use std::error::Error;

use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::f32_matrix_ops::F32MatrixOpsModule;

use super::reference;
use crate::common;

mod gram_ns;
mod ops;
mod standard;

use ops::DeviceRun;

const ITERATIONS: usize = 5;
const RESET_CASES: [(&str, &[usize]); 5] = [
    ("reset_2", &[2]),
    ("reset_2_4", &[2, 4]),
    ("reset_1_3", &[1, 3]),
    ("reset_2_3_4", &[2, 3, 4]),
    ("reset_every", &[1, 2, 3, 4]),
];

pub fn run_timing_case(rows: usize, cols: usize) -> Result<(), Box<dyn Error>> {
    for iterations in 1..ITERATIONS {
        let resets = if iterations > 2 { &[2][..] } else { &[] };
        run_iteration_case(rows, cols, iterations, resets, "prefix")?;
    }

    for (label, resets) in RESET_CASES {
        run_iteration_case(rows, cols, ITERATIONS, resets, label)?;
    }
    Ok(())
}

fn run_iteration_case(
    rows: usize,
    cols: usize,
    iterations: usize,
    resets: &[usize],
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let (ctx, stream, ptx) = common::cuda_test_context()?;
    let f16 = F16TcMatmulModule::from_module(ptx.clone())?;
    let ops = F32MatrixOpsModule::from_module(ptx)?;
    let source = reference::normalized_polar_source(&reference::gradient(rows, cols), rows, cols);
    let expected_standard = reference::standard_polar(source.clone(), rows, cols, iterations);
    let expected_gram =
        reference::stabilized_gram_ns(source.clone(), rows, cols, iterations, resets);
    let runner = DeviceRun {
        stream: &stream,
        f16: &f16,
        ops: &ops,
        ctx: &ctx,
        rows,
        cols,
    };

    let (standard, standard_ms) = runner.standard(&source, iterations)?;
    let (gram_ns, gram_ns_ms) = runner.gram_ns(&source, iterations, resets)?;

    println!(
        "gram_ns_device rows={rows} cols={cols} iterations={iterations} schedule={label} standard_ms={standard_ms:.6} gram_ns_ms={gram_ns_ms:.6} speedup={:.6}",
        standard_ms / gram_ns_ms,
    );
    println!(
        "gram_ns_device standard_rel_l2={:.8e} gram_ns_rel_l2={:.8e} standard_vs_gram_rel_l2={:.8e} standard_vs_gram_cosine={:.8}",
        reference::relative_l2(&standard, &expected_standard),
        reference::relative_l2(&gram_ns, &expected_gram),
        reference::relative_l2(&gram_ns, &standard),
        reference::cosine(&gram_ns, &standard),
    );

    assert!(standard.iter().all(|value| value.is_finite()));
    assert!(gram_ns.iter().all(|value| value.is_finite()));
    if iterations == ITERATIONS && label == "reset_every" {
        assert!(reference::cosine(&gram_ns, &standard) >= 0.999);
    }
    Ok(())
}
