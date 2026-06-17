use std::error::Error;
use std::path::PathBuf;

use cuda_core::CudaContext;
use gpt2_nvfp4::{
    AttentionBackwardModules, BlockAttentionBackwardArgs, BlockAttentionBackwardModules,
    BlockAttentionBackwardSeeds, Gpt2Rng, attention_side_backward,
};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::residual::ResidualBackwardModule;
use rust_kernels_cuda::transpose::TransposeModule;

#[path = "support/attention_core_scratch.rs"]
mod attention_core_scratch;
#[path = "block_attention_backward/buffers/mod.rs"]
mod buffers;
#[path = "block_attention_backward/data.rs"]
mod data;
#[path = "block_attention_backward/scratch.rs"]
mod scratch;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn block_attention_side_backward_runs_full_chain() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(gpu_device_index())?;
    let stream = ctx.new_stream()?;
    let ptx = ctx.load_module_from_file(ptx_path().as_str())?;
    let saved = buffers::SavedBuffers::new(&stream)?;
    let weights = buffers::WeightBuffers::new(&stream)?;
    let mut grads = buffers::GradBuffers::new(&stream)?;
    let mut scratch = scratch::BlockAttentionScratch::new(&stream)?;
    let mut rng = Gpt2Rng::new(0x4154_544e);

    attention_side_backward(BlockAttentionBackwardArgs {
        stream: &stream,
        modules: BlockAttentionBackwardModules {
            residual: &ResidualBackwardModule::from_module(ptx.clone())?,
            layer_norm: &LayerNormBackwardModule::from_module(ptx.clone())?,
            attention: &AttentionModule::from_module(ptx.clone())?,
            f16_tc: &F16TcMatmulModule::from_module(ptx.clone())?,
            linear: AttentionBackwardModules {
                transpose: &TransposeModule::from_module(ptx.clone())?,
                decode: &Nvfp4DecodeModule::from_module(ptx.clone())?,
                linear: &LinearBackwardModule::from_module(ptx.clone())?,
                quant: &Nvfp4QuantModule::from_module(ptx)?,
            },
        },
        saved: saved.block(),
        ln_1: weights.ln_1(),
        projections: weights.projections(),
        grads: grads.block(),
        scratch: scratch.block(),
        seeds: BlockAttentionBackwardSeeds::from_rng(&mut rng),
    })?;

    assert_nonzero_finite(&grads.d_residual_in.to_host_vec(&stream)?);
    assert_nonzero_finite(&grads.d_attention_out.to_host_vec(&stream)?);
    assert_nonzero_finite(&grads.d_qkv.to_host_vec(&stream)?);
    assert_nonzero_finite(&grads.d_attn_qkv_weight.to_host_vec(&stream)?);
    assert_nonzero_finite(&grads.d_attn_c_proj_weight.to_host_vec(&stream)?);
    Ok(())
}

fn assert_nonzero_finite(values: &[f32]) {
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 0.0));
}

fn gpu_device_index() -> usize {
    std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn ptx_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned()
}
