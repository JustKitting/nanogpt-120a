use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f32_matrix_ops::F32AddScaledIdentityArgs;

use super::ops::DeviceRun;
use crate::polar_coefficients::coefficients;

impl<'a> DeviceRun<'a> {
    pub(super) fn gram_ns(
        &self,
        source: &[f32],
        iterations: usize,
        resets: &[usize],
    ) -> Result<(Vec<f32>, f32), Box<dyn Error>> {
        let row_len = self.rows * self.cols;
        let square_len = self.rows * self.rows;
        let row_shape = (self.rows, self.cols, self.rows);
        let square_shape = (self.rows, self.rows, self.rows);
        let mut x = DeviceBuffer::from_host(self.stream, source)?;
        let mut x_next = DeviceBuffer::<f32>::zeroed(self.stream, row_len)?;
        let mut r = DeviceBuffer::<f32>::zeroed(self.stream, square_len)?;
        let mut q = DeviceBuffer::<f32>::zeroed(self.stream, square_len)?;
        let mut z = DeviceBuffer::<f32>::zeroed(self.stream, square_len)?;
        let mut tmp = DeviceBuffer::<f32>::zeroed(self.stream, square_len)?;
        let mut q_initialized = false;

        let (_, ms) = self.timed(|| {
            self.gram_from_x(&x, &mut r)?;
            for iter in 0..iterations {
                if resets.contains(&iter) {
                    self.matmul_rhs(&q, &x, &mut x_next, row_shape)?;
                    std::mem::swap(&mut x, &mut x_next);
                    self.gram_from_x(&x, &mut r)?;
                    q_initialized = false;
                }

                let (a, b, c) = coefficients(iter);
                self.matmul_add_rhs(&r, &r, &r, &mut z, square_shape, (b, c))?;

                if q_initialized {
                    self.matmul_add_rhs(&q, &z, &q, &mut tmp, square_shape, (a, 1.0))?;
                    std::mem::swap(&mut q, &mut tmp);
                } else {
                    self.ops.add_scaled_identity(F32AddScaledIdentityArgs {
                        stream: self.stream,
                        src: &z,
                        out: &mut q,
                        dim: self.rows as u32,
                        scale: a,
                    })?;
                    q_initialized = true;
                }

                if iter + 1 < iterations && !resets.contains(&(iter + 1)) {
                    self.matmul_add_rhs(&r, &z, &r, &mut tmp, square_shape, (a, 1.0))?;
                    self.matmul_add_rhs(&z, &tmp, &tmp, &mut r, square_shape, (a, 1.0))?;
                }
            }
            self.matmul_rhs(&q, &x, &mut x_next, row_shape)?;
            Ok(())
        })?;

        Ok((x_next.to_host_vec(self.stream)?, ms))
    }
}
