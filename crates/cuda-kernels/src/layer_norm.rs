use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm, thread, warp};

type AppResult<T> = Result<T, Box<dyn Error>>;

const ROW_SIZE: usize = 32;
const WARPS_PER_BLOCK: u32 = 8;
const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * ROW_SIZE as u32;

#[cuda_module]
mod kernels {
    use super::*;

    const WARP_MASK: u32 = 0xffff_ffff;
    const ROW_SIZE_F32: f32 = ROW_SIZE as f32;

    #[kernel]
    pub fn layer_norm_warp_f32_kernel(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
        row_count: u32,
        epsilon: f32,
    ) {
        let lane = warp::lane_id() as usize;
        let warp_in_block = thread::threadIdx_x() / ROW_SIZE as u32;
        let warps_per_block = thread::blockDim_x() / ROW_SIZE as u32;
        let row = thread::blockIdx_x() * warps_per_block + warp_in_block;

        if row < row_count {
            let index = row as usize * ROW_SIZE + lane;
            let value = x[index];
            let mean = warp_sum(value) / ROW_SIZE_F32;
            let centered = value - mean;
            let variance = warp_sum(centered * centered) / ROW_SIZE_F32;
            let inv_std = 1.0 / sqrt_f32(variance + epsilon);
            let normalized = centered * inv_std;

            unsafe {
                *out.get_unchecked_mut(index) = fma_f32(normalized, gamma[lane], beta[lane]);
            }
        }
    }

    #[inline(always)]
    fn warp_sum(mut value: f32) -> f32 {
        value += warp::shuffle_xor_f32_sync(WARP_MASK, value, 16);
        value += warp::shuffle_xor_f32_sync(WARP_MASK, value, 8);
        value += warp::shuffle_xor_f32_sync(WARP_MASK, value, 4);
        value += warp::shuffle_xor_f32_sync(WARP_MASK, value, 2);
        value + warp::shuffle_xor_f32_sync(WARP_MASK, value, 1)
    }

    #[inline(always)]
    fn sqrt_f32(x: f32) -> f32 {
        let y: f32;
        unsafe {
            ptx_asm!(
                "sqrt.rn.f32 %0, %1;",
                out("=f") y,
                in("f") x,
                options(register_only),
            );
        }
        y
    }

    #[inline(always)]
    fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
        let y: f32;
        unsafe {
            ptx_asm!(
                "fma.rn.f32 %0, %1, %2, %3;",
                out("=f") y,
                in("f") a,
                in("f") b,
                in("f") c,
                options(register_only),
            );
        }
        y
    }
}

pub fn run_default() -> AppResult<()> {
    let row_count = 2usize;
    let epsilon = 1.0e-5f32;
    let mut x = [0.0f32; ROW_SIZE * 2];
    let mut gamma = [0.0f32; ROW_SIZE];
    let mut beta = [0.0f32; ROW_SIZE];

    for col in 0..ROW_SIZE {
        gamma[col] = 0.75 + col as f32 * 0.01;
        beta[col] = -0.125 + col as f32 * 0.005;
    }

    for row in 0..row_count {
        for col in 0..ROW_SIZE {
            let base = col as f32 - 15.5;
            x[row * ROW_SIZE + col] = base * (0.25 + row as f32 * 0.125) + row as f32 * 0.5;
        }
    }

    let ctx = CudaContext::new(1)?;
    let stream = ctx.new_stream()?;
    let module = kernels::from_module(ctx.load_module_from_file(crate::CUDA_OXIDE_PTX_PATH)?)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let gamma_dev = DeviceBuffer::from_host(&stream, &gamma)?;
    let beta_dev = DeviceBuffer::from_host(&stream, &beta)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len())?;

    module.layer_norm_warp_f32_kernel(
        &stream,
        LaunchConfig {
            grid_dim: ((row_count as u32).div_ceil(WARPS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        },
        &x_dev,
        &gamma_dev,
        &beta_dev,
        &mut out_dev,
        row_count as u32,
        epsilon,
    )?;

    let out = out_dev.to_host_vec(&stream)?;
    let expected = reference_layer_norm(&x, &gamma, &beta, row_count, epsilon);
    let max_abs_error = max_abs_error(&out, &expected);

    println!(
        "layer_norm out0=[{}] max_abs_error={:.8e}",
        first_values(&out, 8),
        max_abs_error
    );
    Ok(())
}

fn reference_layer_norm(
    x: &[f32; ROW_SIZE * 2],
    gamma: &[f32; ROW_SIZE],
    beta: &[f32; ROW_SIZE],
    row_count: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; row_count * ROW_SIZE];
    for row in 0..row_count {
        let base = row * ROW_SIZE;
        let mut sum = 0.0f32;
        for col in 0..ROW_SIZE {
            sum += x[base + col];
        }

        let mean = sum / ROW_SIZE as f32;
        let mut variance_sum = 0.0f32;
        for col in 0..ROW_SIZE {
            let centered = x[base + col] - mean;
            variance_sum += centered * centered;
        }

        let inv_std = 1.0 / (variance_sum / ROW_SIZE as f32 + epsilon).sqrt();
        for col in 0..ROW_SIZE {
            let centered = x[base + col] - mean;
            out[base + col] = (centered * inv_std).mul_add(gamma[col], beta[col]);
        }
    }
    out
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .fold(0.0f32, |max, (actual, expected)| {
            max.max((actual - expected).abs())
        })
}

fn first_values(values: &[f32], count: usize) -> String {
    values
        .iter()
        .take(count)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(" ")
}
