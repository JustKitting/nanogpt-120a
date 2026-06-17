use cuda_core::{CudaStream, DeviceBuffer, DriverError, memory};

use crate::GPT2_N_LAYER;

use super::{BlockForwardSaved, Gpt2ForwardSaved, LayerNormSaved};

pub struct Gpt2ForwardTape<'a> {
    pub embedding_residual: &'a mut DeviceBuffer<f32>,
    pub blocks: [BlockForwardTape<'a>; GPT2_N_LAYER],
    pub final_norm: LayerNormTape<'a>,
    pub logits: &'a mut DeviceBuffer<f32>,
}

pub struct BlockForwardTape<'a> {
    pub residual_in: &'a mut DeviceBuffer<f32>,
    pub ln_1: LayerNormTape<'a>,
    pub qkv: &'a mut DeviceBuffer<f32>,
    pub attention_out: &'a mut DeviceBuffer<f32>,
    pub residual_after_attention: &'a mut DeviceBuffer<f32>,
    pub ln_2: LayerNormTape<'a>,
    pub mlp_activation: &'a mut DeviceBuffer<f32>,
    pub residual_out: &'a mut DeviceBuffer<f32>,
}

pub struct LayerNormTape<'a> {
    pub residual: &'a mut DeviceBuffer<f32>,
    pub normalized: &'a mut DeviceBuffer<f32>,
    pub normalized_amax: &'a mut DeviceBuffer<f32>,
}

impl<'a> Gpt2ForwardTape<'a> {
    pub fn saved<'t>(&'t self, tokens: &'t DeviceBuffer<u32>) -> Gpt2ForwardSaved<'t> {
        Gpt2ForwardSaved {
            tokens,
            embedding_residual: &*self.embedding_residual,
            blocks: std::array::from_fn(|index| self.blocks[index].saved()),
            final_norm: self.final_norm.saved(),
            logits: &*self.logits,
        }
    }

    pub(crate) fn block(&mut self, index: usize) -> BlockForwardTape<'_> {
        self.blocks[index].reborrow()
    }

    pub(crate) fn save_embedding(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.embedding_residual)
    }

    pub(crate) fn save_logits(
        &mut self,
        stream: &CudaStream,
        logits: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, logits, self.logits)
    }
}

impl<'a> BlockForwardTape<'a> {
    fn saved(&self) -> BlockForwardSaved<'_> {
        BlockForwardSaved {
            residual_in: &*self.residual_in,
            ln_1: self.ln_1.saved(),
            qkv: &*self.qkv,
            attention_out: &*self.attention_out,
            residual_after_attention: &*self.residual_after_attention,
            ln_2: self.ln_2.saved(),
            mlp_activation: &*self.mlp_activation,
            residual_out: &*self.residual_out,
        }
    }

    fn reborrow(&mut self) -> BlockForwardTape<'_> {
        BlockForwardTape {
            residual_in: &mut *self.residual_in,
            ln_1: self.ln_1.reborrow(),
            qkv: &mut *self.qkv,
            attention_out: &mut *self.attention_out,
            residual_after_attention: &mut *self.residual_after_attention,
            ln_2: self.ln_2.reborrow(),
            mlp_activation: &mut *self.mlp_activation,
            residual_out: &mut *self.residual_out,
        }
    }

    pub(crate) fn save_residual_in(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual_in)
    }

    pub(crate) fn save_qkv(
        &mut self,
        stream: &CudaStream,
        qkv: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, qkv, self.qkv)
    }

    pub(crate) fn save_attention_out(
        &mut self,
        stream: &CudaStream,
        out: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, out, self.attention_out)
    }

    pub(crate) fn save_residual_after_attention(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual_after_attention)
    }

    pub(crate) fn save_mlp_activation(
        &mut self,
        stream: &CudaStream,
        activation: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, activation, self.mlp_activation)
    }

    pub(crate) fn save_residual_out(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual_out)
    }
}

impl<'a> LayerNormTape<'a> {
    fn saved(&self) -> LayerNormSaved<'_> {
        LayerNormSaved {
            residual: &*self.residual,
            normalized: &*self.normalized,
            normalized_amax: &*self.normalized_amax,
        }
    }

    fn reborrow(&mut self) -> LayerNormTape<'_> {
        LayerNormTape {
            residual: &mut *self.residual,
            normalized: &mut *self.normalized,
            normalized_amax: &mut *self.normalized_amax,
        }
    }

    pub(crate) fn save(
        &mut self,
        stream: &CudaStream,
        residual: &DeviceBuffer<f32>,
        normalized: &DeviceBuffer<f32>,
        normalized_amax: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        copy_device(stream, residual, self.residual)?;
        copy_device(stream, normalized, self.normalized)?;
        copy_device(stream, normalized_amax, self.normalized_amax)
    }
}

fn copy_device<T>(
    stream: &CudaStream,
    src: &DeviceBuffer<T>,
    dst: &mut DeviceBuffer<T>,
) -> Result<(), DriverError> {
    assert_eq!(src.len(), dst.len());
    stream.context().bind_to_thread()?;

    unsafe {
        memory::memcpy_dtod_async(
            dst.cu_deviceptr(),
            src.cu_deviceptr(),
            src.num_bytes(),
            stream.cu_stream(),
        )
    }
}
