use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tc_matmul::{Nvfp4TcMatmulArgs, Nvfp4TcMatmulModule};

use super::math::{combine_next, transpose};
use super::scratch::{Scratch, global_scale};

pub struct Nvfp4Polar<'a> {
    stream: &'a CudaStream,
    matmul: &'a Nvfp4TcMatmulModule,
    quant: &'a Nvfp4QuantModule,
}

impl<'a> Nvfp4Polar<'a> {
    pub fn new(
        stream: &'a CudaStream,
        matmul: &'a Nvfp4TcMatmulModule,
        quant: &'a Nvfp4QuantModule,
    ) -> Self {
        Self {
            stream,
            matmul,
            quant,
        }
    }

    pub fn iterations(
        &self,
        mut source: Vec<f32>,
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        for iter in 0..iterations {
            source = self.step(&source, rows, cols, iter)?;
        }
        Ok(source)
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
}

fn seed(iter: usize, stage: u32, stream: u32) -> u32 {
    0x9e37_79b9_u32
        .wrapping_mul((iter as u32).wrapping_add(1))
        .wrapping_add(stage.wrapping_mul(0x85eb_ca6b))
        .wrapping_add(stream.wrapping_mul(0xc2b2_ae35))
}
