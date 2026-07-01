use std::{error::Error, sync::Arc};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, sys};
use rust_kernels_cuda::f16_tc_matmul::{
    F16TcMatmulAddRhsTransposeBaseArgs, F16TcMatmulF32Args, F16TcMatmulF32RhsArgs,
    F16TcMatmulModule,
};
use rust_kernels_cuda::f32_matrix_ops::F32MatrixOpsModule;

pub(super) struct DeviceRun<'a> {
    pub(super) stream: &'a CudaStream,
    pub(super) f16: &'a F16TcMatmulModule,
    pub(super) ops: &'a F32MatrixOpsModule,
    pub(super) ctx: &'a Arc<CudaContext>,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl<'a> DeviceRun<'a> {
    pub(super) fn timed<T>(
        &self,
        run: impl FnOnce() -> Result<T, Box<dyn Error>>,
    ) -> Result<(T, f32), Box<dyn Error>> {
        let event_flag = Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
        let start = self.ctx.new_event(event_flag)?;
        let end = self.ctx.new_event(event_flag)?;

        start.record(self.stream)?;
        let value = run()?;
        end.record(self.stream)?;

        Ok((value, start.elapsed_ms(&end)?))
    }

    pub(super) fn gram_from_x(
        &self,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        self.f16.batched_matmul_f32_input(F16TcMatmulF32Args {
            stream: self.stream,
            a: x,
            b_t: x,
            out,
            batch_count: 1,
            m: self.rows as u32,
            n: self.rows as u32,
            k: self.cols as u32,
        })
    }

    pub(super) fn matmul_rhs(
        &self,
        a: &DeviceBuffer<f32>,
        rhs: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        shape: (usize, usize, usize),
    ) -> Result<(), DriverError> {
        let (m, n, k) = shape;
        self.f16.batched_matmul_f32_rhs(F16TcMatmulF32RhsArgs {
            stream: self.stream,
            a,
            rhs,
            out,
            batch_count: 1,
            m: m as u32,
            n: n as u32,
            k: k as u32,
        })
    }

    pub(super) fn matmul_add_rhs(
        &self,
        a: &DeviceBuffer<f32>,
        rhs: &DeviceBuffer<f32>,
        base: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        shape: (usize, usize, usize),
        scales: (f32, f32),
    ) -> Result<(), DriverError> {
        let (m, n, k) = shape;
        let (base_scale, matmul_scale) = scales;
        self.f16
            .batched_matmul_add_rhs_transposed_base(F16TcMatmulAddRhsTransposeBaseArgs {
                stream: self.stream,
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
}
