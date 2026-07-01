use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulF32Args;
use rust_kernels_cuda::nvfp4_tc_matmul::Nvfp4TcMatmulArgs;

use super::super::scratch::{Scratch, global_scale};
use super::Nvfp4Polar;

impl<'a> Nvfp4Polar<'a> {
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

    pub(super) fn f16_product(
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

fn seed(iter: usize, stage: u32, stream: u32) -> u32 {
    0x9e37_79b9_u32
        .wrapping_mul((iter as u32).wrapping_add(1))
        .wrapping_add(stage.wrapping_mul(0x85eb_ca6b))
        .wrapping_add(stream.wrapping_mul(0xc2b2_ae35))
}
