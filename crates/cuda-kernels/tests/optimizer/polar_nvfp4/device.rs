use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::f16_tc_matmul::{F16TcMatmulF32Args, F16TcMatmulModule};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::{Nvfp4TcMatmulArgs, Nvfp4TcMatmulModule};

use super::math::{combine_next, transpose};
use super::scratch::{Scratch, global_scale};

#[path = "device/iterations/mod.rs"]
mod iterations;
#[path = "device/mode.rs"]
mod mode;

pub use mode::{CorrectionStats, GramCorrectionMode};

pub struct Nvfp4Polar<'a> {
    stream: &'a CudaStream,
    f16: &'a F16TcMatmulModule,
    matmul: &'a Nvfp4TcMatmulModule,
    quant: &'a Nvfp4QuantModule,
}

impl<'a> Nvfp4Polar<'a> {
    pub fn new(
        stream: &'a CudaStream,
        f16: &'a F16TcMatmulModule,
        matmul: &'a Nvfp4TcMatmulModule,
        quant: &'a Nvfp4QuantModule,
    ) -> Self {
        Self {
            stream,
            f16,
            matmul,
            quant,
        }
    }

    pub fn step(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let gram = self.product(source, source, rows, rows, cols, iter, 0)?;
        let source_t = transpose(source, rows, cols);
        let ax = self.product(&gram, &source_t, rows, cols, rows, iter, 1)?;
        let ax_t = transpose(&ax, rows, cols);
        let aax = self.product(&gram, &ax_t, rows, cols, rows, iter, 2)?;
        Ok(combine_next(source, &ax, &aax, iter))
    }

    fn step_from_gram(
        &self,
        source: &[f32],
        gram: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let ax = self.f16_product(gram, &transpose(source, rows, cols), rows, cols, rows)?;
        let aax = self.f16_product(gram, &transpose(&ax, rows, cols), rows, cols, rows)?;
        Ok(combine_next(source, &ax, &aax, iter))
    }

    fn gram_form_step_from_gram(
        &self,
        source: &[f32],
        gram: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
        coefficient_safety: f32,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let (mut a, mut b, mut c) = crate::polar_coefficients::coefficients(iter);
        if coefficient_safety != 1.0 {
            a /= coefficient_safety;
            b /= coefficient_safety.powi(3);
            c /= coefficient_safety.powi(5);
        }
        let gram2 = self.f16_product(gram, &transpose(gram, rows, rows), rows, rows, rows)?;
        let mut q = vec![0.0_f32; rows * rows];
        for index in 0..q.len() {
            q[index] = c.mul_add(gram2[index], b * gram[index]);
        }
        for row in 0..rows {
            q[row * rows + row] += a;
        }
        self.f16_product(&q, &transpose(source, rows, cols), rows, cols, rows)
    }

    fn averaged_nvfp4_gram(
        &self,
        source: &[f32],
        rows: usize,
        cols: usize,
        iter: usize,
        samples: usize,
        stats: &mut CorrectionStats,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut sum = vec![0.0_f32; rows * rows];
        for sample in 0..samples {
            stats.nvfp4_gram_count += 1;
            let gram = self.product(source, source, rows, rows, cols, iter, 16 + sample as u32)?;
            for (sum, value) in sum.iter_mut().zip(gram) {
                *sum += value;
            }
        }
        let inv_samples = 1.0 / samples as f32;
        for value in &mut sum {
            *value *= inv_samples;
        }
        Ok(sum)
    }

    pub fn product(
        &self,
        a: &[f32],
        b_t: &[f32],
        m: usize,
        n: usize,
        k: usize,
        iter: usize,
        stage: u32,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let a_dev = DeviceBuffer::from_host(self.stream, a)?;
        let b_t_dev = DeviceBuffer::from_host(self.stream, b_t)?;
        let mut out = DeviceBuffer::<f32>::zeroed(self.stream, m * n)?;
        let mut scratch = Scratch::new(self.stream, m, n, k, global_scale(a), global_scale(b_t))?;

        self.matmul.matmul_ms_eden(Nvfp4TcMatmulArgs {
            stream: self.stream,
            quant_module: self.quant,
            a: &a_dev,
            b_t: &b_t_dev,
            out: &mut out,
            scratch: scratch.args(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            sign_seed: seed(iter, stage, 0),
            scale_seed: seed(iter, stage, 1),
        })?;

        Ok(out.to_host_vec(self.stream)?)
    }

    fn f16_product(
        &self,
        a: &[f32],
        b_t: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let a_dev = DeviceBuffer::from_host(self.stream, a)?;
        let b_t_dev = DeviceBuffer::from_host(self.stream, b_t)?;
        let mut out = DeviceBuffer::<f32>::zeroed(self.stream, m * n)?;

        self.f16.batched_matmul_f32_input(F16TcMatmulF32Args {
            stream: self.stream,
            a: &a_dev,
            b_t: &b_t_dev,
            out: &mut out,
            batch_count: 1,
            m: m as u32,
            n: n as u32,
            k: k as u32,
        })?;

        Ok(out.to_host_vec(self.stream)?)
    }
}

fn row_orthogonality_residual(x: &[f32], rows: usize, cols: usize) -> f32 {
    let mut sum = 0.0_f32;
    for row in 0..rows {
        for col in 0..rows {
            let mut dot = 0.0_f32;
            for k in 0..cols {
                dot += x[row * cols + k] * x[col * cols + k];
            }
            let expected = if row == col { 1.0 } else { 0.0 };
            let diff = dot - expected;
            sum = diff.mul_add(diff, sum);
        }
    }
    sum.sqrt()
}

fn seed(iter: usize, stage: u32, stream: u32) -> u32 {
    0x9e37_79b9_u32
        .wrapping_mul((iter as u32).wrapping_add(1))
        .wrapping_add(stage.wrapping_mul(0x85eb_ca6b))
        .wrapping_add(stream.wrapping_mul(0xc2b2_ae35))
}
