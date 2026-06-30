use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, sys};
use rust_kernels_cuda::f16_tc_matmul::{
    F16TcMatmulAddRhsTransposeBaseArgs, F16TcMatmulF32Args, F16TcMatmulF32RhsArgs,
    F16TcMatmulModule,
};
use rust_kernels_cuda::f32_matrix_ops::{
    F32AddScaledIdentityArgs, F32Linear2Args, F32MatrixOpsModule,
};

use super::reference;
use crate::common;
use crate::polar_coefficients::coefficients;

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

    let (standard, standard_ms) =
        run_standard(&stream, &ctx, &f16, &ops, &source, rows, cols, iterations)?;
    let (gram_ns, gram_ns_ms) = run_gram_ns(
        &stream, &ctx, &f16, &ops, &source, rows, cols, iterations, resets,
    )?;

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

fn run_standard(
    stream: &CudaStream,
    ctx: &std::sync::Arc<cuda_core::CudaContext>,
    f16: &F16TcMatmulModule,
    ops: &F32MatrixOpsModule,
    source: &[f32],
    rows: usize,
    cols: usize,
    iterations: usize,
) -> Result<(Vec<f32>, f32), Box<dyn Error>> {
    let mut x = DeviceBuffer::from_host(stream, source)?;
    let mut next = DeviceBuffer::<f32>::zeroed(stream, rows * cols)?;
    let mut gram = DeviceBuffer::<f32>::zeroed(stream, rows * rows)?;
    let mut ax = DeviceBuffer::<f32>::zeroed(stream, rows * cols)?;
    let mut base = DeviceBuffer::<f32>::zeroed(stream, rows * cols)?;
    let start = timing_event(ctx)?;
    let end = timing_event(ctx)?;

    start.record(stream)?;
    for iter in 0..iterations {
        let (a, b, c) = coefficients(iter);
        gram_from_x(f16, stream, &x, &mut gram, rows, cols)?;
        matmul_rhs(f16, stream, &gram, &x, &mut ax, rows, cols, rows)?;
        ops.linear2(F32Linear2Args {
            stream,
            a: &x,
            b: &ax,
            out: &mut base,
            len: (rows * cols) as u32,
            a_scale: a,
            b_scale: b,
        })?;
        matmul_add_rhs(
            f16, stream, &gram, &ax, &base, &mut next, rows, cols, rows, 1.0, c,
        )?;
        std::mem::swap(&mut x, &mut next);
    }
    end.record(stream)?;
    let ms = start.elapsed_ms(&end)?;

    Ok((x.to_host_vec(stream)?, ms))
}

fn run_gram_ns(
    stream: &CudaStream,
    ctx: &std::sync::Arc<cuda_core::CudaContext>,
    f16: &F16TcMatmulModule,
    ops: &F32MatrixOpsModule,
    source: &[f32],
    rows: usize,
    cols: usize,
    iterations: usize,
    resets: &[usize],
) -> Result<(Vec<f32>, f32), Box<dyn Error>> {
    let mut x = DeviceBuffer::from_host(stream, source)?;
    let mut x_next = DeviceBuffer::<f32>::zeroed(stream, rows * cols)?;
    let mut r = DeviceBuffer::<f32>::zeroed(stream, rows * rows)?;
    let mut q = DeviceBuffer::<f32>::zeroed(stream, rows * rows)?;
    let mut z = DeviceBuffer::<f32>::zeroed(stream, rows * rows)?;
    let mut tmp = DeviceBuffer::<f32>::zeroed(stream, rows * rows)?;
    let mut q_initialized = false;
    let start = timing_event(ctx)?;
    let end = timing_event(ctx)?;

    start.record(stream)?;
    gram_from_x(f16, stream, &x, &mut r, rows, cols)?;
    for iter in 0..iterations {
        if resets.contains(&iter) {
            matmul_rhs(f16, stream, &q, &x, &mut x_next, rows, cols, rows)?;
            std::mem::swap(&mut x, &mut x_next);
            gram_from_x(f16, stream, &x, &mut r, rows, cols)?;
            q_initialized = false;
        }

        let (a, b, c) = coefficients(iter);
        matmul_add_rhs(f16, stream, &r, &r, &r, &mut z, rows, rows, rows, b, c)?;

        if q_initialized {
            matmul_add_rhs(f16, stream, &q, &z, &q, &mut tmp, rows, rows, rows, a, 1.0)?;
            std::mem::swap(&mut q, &mut tmp);
        } else {
            ops.add_scaled_identity(F32AddScaledIdentityArgs {
                stream,
                src: &z,
                out: &mut q,
                dim: rows as u32,
                scale: a,
            })?;
            q_initialized = true;
        }

        if iter + 1 < iterations && !resets.contains(&(iter + 1)) {
            matmul_add_rhs(f16, stream, &r, &z, &r, &mut tmp, rows, rows, rows, a, 1.0)?;
            matmul_add_rhs(
                f16, stream, &z, &tmp, &tmp, &mut r, rows, rows, rows, a, 1.0,
            )?;
        }
    }
    matmul_rhs(f16, stream, &q, &x, &mut x_next, rows, cols, rows)?;
    end.record(stream)?;
    let ms = start.elapsed_ms(&end)?;

    Ok((x_next.to_host_vec(stream)?, ms))
}

fn gram_from_x(
    f16: &F16TcMatmulModule,
    stream: &CudaStream,
    x: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<(), DriverError> {
    f16.batched_matmul_f32_input(F16TcMatmulF32Args {
        stream,
        a: x,
        b_t: x,
        out,
        batch_count: 1,
        m: rows as u32,
        n: rows as u32,
        k: cols as u32,
    })
}

fn matmul_rhs(
    f16: &F16TcMatmulModule,
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    rhs: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), DriverError> {
    f16.batched_matmul_f32_rhs(F16TcMatmulF32RhsArgs {
        stream,
        a,
        rhs,
        out,
        batch_count: 1,
        m: m as u32,
        n: n as u32,
        k: k as u32,
    })
}

#[allow(clippy::too_many_arguments)]
fn matmul_add_rhs(
    f16: &F16TcMatmulModule,
    stream: &CudaStream,
    a: &DeviceBuffer<f32>,
    rhs: &DeviceBuffer<f32>,
    base: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    base_scale: f32,
    matmul_scale: f32,
) -> Result<(), DriverError> {
    f16.batched_matmul_add_rhs_transposed_base(F16TcMatmulAddRhsTransposeBaseArgs {
        stream,
        a,
        rhs,
        base,
        out,
        batch_count: 1,
        m: m as u32,
        n: n as u32,
        k: k as u32,
        base_scale,
        matmul_scale,
    })
}

fn timing_event(
    ctx: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<cuda_core::CudaEvent, DriverError> {
    ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
}
