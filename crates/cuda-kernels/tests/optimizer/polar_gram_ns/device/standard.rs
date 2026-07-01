use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f32_matrix_ops::F32Linear2Args;

use super::ops::DeviceRun;
use crate::polar_coefficients::coefficients;

impl<'a> DeviceRun<'a> {
    pub(super) fn standard(
        &self,
        source: &[f32],
        iterations: usize,
    ) -> Result<(Vec<f32>, f32), Box<dyn Error>> {
        let row_len = self.rows * self.cols;
        let row_shape = (self.rows, self.cols, self.rows);
        let mut x = DeviceBuffer::from_host(self.stream, source)?;
        let mut next = DeviceBuffer::<f32>::zeroed(self.stream, row_len)?;
        let mut gram = DeviceBuffer::<f32>::zeroed(self.stream, self.rows * self.rows)?;
        let mut ax = DeviceBuffer::<f32>::zeroed(self.stream, row_len)?;
        let mut base = DeviceBuffer::<f32>::zeroed(self.stream, row_len)?;

        let (_, ms) = self.timed(|| {
            for iter in 0..iterations {
                let (a, b, c) = coefficients(iter);
                self.gram_from_x(&x, &mut gram)?;
                self.matmul_rhs(&gram, &x, &mut ax, row_shape)?;
                self.ops.linear2(F32Linear2Args {
                    stream: self.stream,
                    a: &x,
                    b: &ax,
                    out: &mut base,
                    len: row_len as u32,
                    a_scale: a,
                    b_scale: b,
                })?;
                self.matmul_add_rhs(&gram, &ax, &base, &mut next, row_shape, (1.0, c))?;
                std::mem::swap(&mut x, &mut next);
            }
            Ok(())
        })?;

        Ok((x.to_host_vec(self.stream)?, ms))
    }
}
