use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_tc_matmul::{
    nvfp4_tc_matmul_bytes, nvfp4_tc_matmul_chunks, nvfp4_tc_matmul_padded_k, nvfp4_tc_matmul_scales,
};

pub(super) const ROWS: usize = 33;
pub(super) const COLS: usize = 16;
pub(super) const SIGN_SEED: u32 = 0x1c69_b3f5;
pub(super) const SCALE_SEED: u32 = 0x4a7c_15d3;
pub(super) const SCALE_OVERRIDE: f32 = 0.25;

pub(super) struct QuantScratch {
    pub(super) bytes: DeviceBuffer<u8>,
    pub(super) scales: DeviceBuffer<u8>,
    pub(super) global_scales: DeviceBuffer<f32>,
    pub(super) chunk_amax: DeviceBuffer<f32>,
    pub(super) global_scale: DeviceBuffer<f32>,
}

macro_rules! assert_buffer_fields_eq {
    ($stream:expr, $actual:expr, $expected:expr, [$($field:ident),+ $(,)?]) => {{
        $(assert_eq!(
            $actual.$field.to_host_vec($stream)?,
            $expected.$field.to_host_vec($stream)?
        );)+
        Ok(())
    }};
}

impl QuantScratch {
    pub(super) fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_bytes(COLS as u32, ROWS as u32))?,
            scales: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_scales(COLS as u32, ROWS as u32))?,
            global_scales: DeviceBuffer::zeroed(stream, COLS)?,
            chunk_amax: DeviceBuffer::zeroed(
                stream,
                nvfp4_tc_matmul_chunks(COLS as u32, ROWS as u32),
            )?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub(super) fn new_exact(
        stream: &CudaStream,
        row_count: usize,
        row_len: usize,
    ) -> Result<Self, DriverError> {
        let element_count = row_count * row_len;

        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, element_count / 2)?,
            scales: DeviceBuffer::zeroed(stream, element_count / 16)?,
            global_scales: DeviceBuffer::zeroed(stream, row_count)?,
            chunk_amax: DeviceBuffer::zeroed(stream, element_count / 32)?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub(super) fn assert_ms_eden_eq(
        &self,
        stream: &CudaStream,
        expected: &Self,
    ) -> Result<(), DriverError> {
        assert_buffer_fields_eq!(
            stream,
            self,
            expected,
            [bytes, scales, global_scales, chunk_amax]
        )
    }

    pub(super) fn assert_quartet_eq(
        &self,
        stream: &CudaStream,
        expected: &Self,
    ) -> Result<(), DriverError> {
        self.assert_ms_eden_eq(stream, expected)?;
        assert_buffer_fields_eq!(stream, self, expected, [global_scale])
    }

    pub(super) fn assert_no_chunk_quartet_eq(
        &self,
        stream: &CudaStream,
        expected: &Self,
    ) -> Result<(), DriverError> {
        assert_buffer_fields_eq!(
            stream,
            self,
            expected,
            [bytes, scales, global_scales, global_scale]
        )
    }
}

pub(super) struct SourceScratch {
    pub(super) bytes: DeviceBuffer<u8>,
    pub(super) scales: DeviceBuffer<u8>,
    pub(super) global_scale: DeviceBuffer<f32>,
}

impl SourceScratch {
    pub(super) fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, ROWS * COLS / 2)?,
            scales: DeviceBuffer::zeroed(stream, ROWS * COLS / 16)?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub(super) fn tensor(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scale: &self.global_scale,
        }
    }
}

pub(super) struct RowwiseSourceScratch {
    pub(super) bytes: DeviceBuffer<u8>,
    pub(super) scales: DeviceBuffer<u8>,
    pub(super) global_scales: DeviceBuffer<f32>,
}

impl RowwiseSourceScratch {
    pub(super) fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Self::new_for_shape(stream, ROWS, COLS)
    }

    pub(super) fn new_for_shape(
        stream: &CudaStream,
        rows: usize,
        cols: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, rows * cols / 2)?,
            scales: DeviceBuffer::zeroed(stream, rows * cols / 16)?,
            global_scales: DeviceBuffer::zeroed(stream, rows)?,
        })
    }

    pub(super) fn tensor(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor {
            bytes: &self.bytes,
            scales: &self.scales,
            global_scales: &self.global_scales,
        }
    }
}

pub(super) fn padded_rows() -> usize {
    nvfp4_tc_matmul_padded_k(ROWS as u32) as usize
}

pub(super) fn input_matrix() -> Vec<f32> {
    (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            (row as f32 - 9.0) * 0.03125 + (col as f32 - 4.0) * 0.0078125
        })
        .collect()
}

pub(super) fn cpu_amax(x: &[f32]) -> f32 {
    x.iter().fold(0.0_f32, |max, value| max.max(value.abs()))
}
