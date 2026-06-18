use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::linear_backward::LinearBackwardDeviceScaleArgs;
use rust_kernels_cuda::mma::Nvfp4DeviceScaleMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::data::{row_scales, upload_bytes, upload_scales};

pub struct ProjectionTensors {
    e_bytes: DeviceBuffer<u8>,
    e_scales: DeviceBuffer<u8>,
    e_globals: DeviceBuffer<f32>,
    weight_bytes: DeviceBuffer<u8>,
    weight_scales: DeviceBuffer<u8>,
    weight_global: DeviceBuffer<f32>,
    e_t_bytes: DeviceBuffer<u8>,
    e_t_scales: DeviceBuffer<u8>,
    e_t_globals: DeviceBuffer<f32>,
    input_t_bytes: DeviceBuffer<u8>,
    input_t_scales: DeviceBuffer<u8>,
    input_t_global: DeviceBuffer<f32>,
}

impl ProjectionTensors {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            e_bytes: upload_bytes(stream, super::TOKEN_COUNT, super::OUTPUT_DIM, 3)?,
            e_scales: upload_scales(stream, super::TOKEN_COUNT, super::OUTPUT_DIM)?,
            e_globals: DeviceBuffer::from_host(stream, &row_scales(super::TOKEN_COUNT, 0.001))?,
            weight_bytes: upload_bytes(stream, super::INPUT_DIM, super::OUTPUT_DIM, 7)?,
            weight_scales: upload_scales(stream, super::INPUT_DIM, super::OUTPUT_DIM)?,
            weight_global: DeviceBuffer::from_host(stream, &[0.75_f32])?,
            e_t_bytes: upload_bytes(stream, super::OUTPUT_DIM, super::TOKEN_COUNT, 11)?,
            e_t_scales: upload_scales(stream, super::OUTPUT_DIM, super::TOKEN_COUNT)?,
            e_t_globals: DeviceBuffer::from_host(stream, &row_scales(super::OUTPUT_DIM, 0.0005))?,
            input_t_bytes: upload_bytes(stream, super::INPUT_DIM, super::TOKEN_COUNT, 13)?,
            input_t_scales: upload_scales(stream, super::INPUT_DIM, super::TOKEN_COUNT)?,
            input_t_global: DeviceBuffer::from_host(stream, &[1.25_f32])?,
        })
    }

    pub fn args<'a, 'out>(
        &'a self,
        stream: &'a CudaStream,
        dinput: &'out mut DeviceBuffer<f32>,
        dweight: &'out mut DeviceBuffer<f32>,
    ) -> LinearBackwardDeviceScaleArgs<'a, 'out> {
        LinearBackwardDeviceScaleArgs {
            stream,
            e_h: rowwise(&self.e_bytes, &self.e_scales, &self.e_globals),
            weight_t_h: weight(&self.weight_bytes, &self.weight_scales, &self.weight_global),
            e_t_h: rowwise(&self.e_t_bytes, &self.e_t_scales, &self.e_t_globals),
            input_t_h: weight(
                &self.input_t_bytes,
                &self.input_t_scales,
                &self.input_t_global,
            ),
            dinput,
            dweight,
            token_count: super::TOKEN_COUNT as u32,
            input_dim: super::INPUT_DIM as u32,
            output_dim: super::OUTPUT_DIM as u32,
        }
    }
}

fn rowwise<'a>(
    bytes: &'a DeviceBuffer<u8>,
    scales: &'a DeviceBuffer<u8>,
    globals: &'a DeviceBuffer<f32>,
) -> Nvfp4RowwiseDeviceTensor<'a> {
    Nvfp4RowwiseDeviceTensor {
        bytes,
        scales,
        global_scales: globals,
    }
}

fn weight<'a>(
    bytes: &'a DeviceBuffer<u8>,
    scales: &'a DeviceBuffer<u8>,
    global: &'a DeviceBuffer<f32>,
) -> Nvfp4DeviceScaleMmaWeightTensor<'a> {
    Nvfp4DeviceScaleMmaWeightTensor {
        bytes,
        scales,
        global_scale: global,
    }
}
