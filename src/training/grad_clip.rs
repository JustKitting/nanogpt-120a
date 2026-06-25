use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_QKV, GPT2_VOCAB_SIZE, NEXTLAT_HIDDEN, NEXTLAT_INPUT};
use rust_kernels_cuda::optimizer::{GRAD_CLIP_VALUES_PER_CHUNK, GradientClipArgs, OptimizerModule};

use super::grads::BackwardBuffers;
use super::next_latent::NextLatGradBuffers;

const GLOBAL_GRAD_CLIP_NORM: f32 = 1.0;

pub(super) struct GradientClipBuffers {
    ptrs: DeviceBuffer<u64>,
    lens: DeviceBuffer<u32>,
    chunk_offsets: DeviceBuffer<u32>,
    chunk_sums: DeviceBuffer<f32>,
    scale: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    slot_count: u32,
    chunk_count: u32,
}

#[derive(Clone, Copy)]
struct HostGradPtr {
    ptr: u64,
    len: u32,
    chunk_offset: u32,
}

impl GradientClipBuffers {
    pub(super) fn new(
        stream: &CudaStream,
        grads: &BackwardBuffers,
        next_latent: &NextLatGradBuffers,
    ) -> Result<Self, DriverError> {
        let rows = parameter_gradients(grads, next_latent);
        let chunk_count = rows
            .last()
            .map(|row| row.chunk_offset + chunks(row.len))
            .unwrap_or(0);

        Ok(Self {
            ptrs: upload(stream, &rows, |row| row.ptr)?,
            lens: upload(stream, &rows, |row| row.len)?,
            chunk_offsets: upload(stream, &rows, |row| row.chunk_offset)?,
            chunk_sums: DeviceBuffer::zeroed(stream, chunk_count as usize)?,
            scale: DeviceBuffer::zeroed(stream, 1)?,
            norm: DeviceBuffer::zeroed(stream, 1)?,
            slot_count: rows.len() as u32,
            chunk_count,
        })
    }

    pub(super) fn clip(
        &mut self,
        stream: &CudaStream,
        optimizer: &OptimizerModule,
    ) -> Result<f32, DriverError> {
        optimizer.clip_gradients(GradientClipArgs {
            stream,
            ptrs: &self.ptrs,
            lens: &self.lens,
            chunk_offsets: &self.chunk_offsets,
            chunk_sums: &mut self.chunk_sums,
            scale: &mut self.scale,
            norm: &mut self.norm,
            slot_count: self.slot_count,
            chunk_count: self.chunk_count,
            max_norm: GLOBAL_GRAD_CLIP_NORM,
        })?;
        Ok(self.norm.to_host_vec(stream)?[0])
    }
}

fn upload<T, F>(
    stream: &CudaStream,
    rows: &[HostGradPtr],
    f: F,
) -> Result<DeviceBuffer<T>, DriverError>
where
    T: DeviceCopy,
    F: Fn(HostGradPtr) -> T,
{
    let values: Vec<T> = rows.iter().copied().map(f).collect();
    DeviceBuffer::from_host(stream, &values)
}

fn parameter_gradients(
    grads: &BackwardBuffers,
    next_latent: &NextLatGradBuffers,
) -> Vec<HostGradPtr> {
    let mut rows = Vec::new();
    push(
        &mut rows,
        &grads.d_lm_head_weight,
        GPT2_VOCAB_SIZE * GPT2_N_EMBD,
    );
    push_layer_norm(&mut rows, &grads.final_norm);

    for block in grads.blocks.iter() {
        push_layer_norm(&mut rows, &block.ln_1);
        push(&mut rows, &block.d_attn_qkv_weight, GPT2_N_EMBD * GPT2_QKV);
        push(&mut rows, &block.d_attn_qkv_bias, GPT2_QKV);
        push(
            &mut rows,
            &block.d_attn_c_proj_weight,
            GPT2_N_EMBD * GPT2_N_EMBD,
        );
        push(&mut rows, &block.d_attn_c_proj_bias, GPT2_N_EMBD);
        push_layer_norm(&mut rows, &block.ln_2);
        push(&mut rows, &block.d_mlp_c_fc_weight, GPT2_N_EMBD * GPT2_MLP);
        push(&mut rows, &block.d_mlp_c_fc_bias, GPT2_MLP);
        push(
            &mut rows,
            &block.d_mlp_c_proj_weight,
            GPT2_MLP * GPT2_N_EMBD,
        );
        push(&mut rows, &block.d_mlp_c_proj_bias, GPT2_N_EMBD);
    }
    push(&mut rows, &next_latent.d_norm_weight, NEXTLAT_INPUT);
    push(&mut rows, &next_latent.d_norm_bias, NEXTLAT_INPUT);
    push(
        &mut rows,
        &next_latent.d_input_projection_weight,
        NEXTLAT_INPUT * NEXTLAT_HIDDEN,
    );
    push(
        &mut rows,
        &next_latent.d_input_projection_bias,
        NEXTLAT_HIDDEN,
    );
    push(
        &mut rows,
        &next_latent.d_transition_weight,
        NEXTLAT_HIDDEN * NEXTLAT_HIDDEN,
    );
    push(&mut rows, &next_latent.d_transition_bias, NEXTLAT_HIDDEN);
    push(
        &mut rows,
        &next_latent.d_output_projection_weight,
        NEXTLAT_HIDDEN * GPT2_N_EMBD,
    );
    push(
        &mut rows,
        &next_latent.d_output_projection_bias,
        GPT2_N_EMBD,
    );

    rows
}

fn push_layer_norm(rows: &mut Vec<HostGradPtr>, grads: &super::grad_block::LayerNormGradBuffers) {
    push(rows, &grads.d_weight, GPT2_N_EMBD);
    push(rows, &grads.d_bias, GPT2_N_EMBD);
}

fn push(rows: &mut Vec<HostGradPtr>, buffer: &DeviceBuffer<f32>, len: usize) {
    let chunk_offset = rows
        .last()
        .map(|row| row.chunk_offset + chunks(row.len))
        .unwrap_or(0);
    rows.push(HostGradPtr {
        ptr: buffer.cu_deviceptr(),
        len: len as u32,
        chunk_offset,
    });
}

fn chunks(len: u32) -> u32 {
    len.div_ceil(GRAD_CLIP_VALUES_PER_CHUNK as u32)
}
