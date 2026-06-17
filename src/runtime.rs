use std::path::PathBuf;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream};
use gpt2_nvfp4::{
    AttentionBackwardModules, BlockAttentionBackwardModules, BlockMlpBackwardModules,
    FinalHeadBackwardModules, Gpt2BackwardModules, MlpBackwardModules,
};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::embedding::EmbeddingModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::lm_head::LmHeadModule;
use rust_kernels_cuda::loss::LossModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::optimizer::OptimizerModule;
use rust_kernels_cuda::residual::ResidualBackwardModule;
use rust_kernels_cuda::transpose::TransposeModule;

use crate::AppResult;

pub struct Runtime {
    pub stream: Arc<CudaStream>,
    pub embedding: EmbeddingModule,
    pub attention: AttentionModule,
    pub f16_tc_matmul: F16TcMatmulModule,
    pub quant: Nvfp4QuantModule,
    pub layer_norm: LayerNormModule,
    pub mlp: MlpModule,
    pub lm_head: LmHeadModule,
    pub loss: LossModule,
    pub transpose: TransposeModule,
    decode: Nvfp4DecodeModule,
    linear: LinearBackwardModule,
    layer_norm_backward: LayerNormBackwardModule,
    residual: ResidualBackwardModule,
    pub optimizer: OptimizerModule,
}

impl Runtime {
    pub fn new() -> AppResult<Self> {
        let ctx = CudaContext::new(gpu_device_index())?;
        let stream = ctx.new_stream()?;
        let ptx = ctx.load_module_from_file(ptx_path().as_str())?;
        Ok(Self {
            stream,
            embedding: EmbeddingModule::from_module(ptx.clone())?,
            attention: AttentionModule::from_module(ptx.clone())?,
            f16_tc_matmul: F16TcMatmulModule::from_module(ptx.clone())?,
            quant: Nvfp4QuantModule::from_module(ptx.clone())?,
            layer_norm: LayerNormModule::from_module(ptx.clone())?,
            mlp: MlpModule::from_module(ptx.clone())?,
            lm_head: LmHeadModule::from_module(ptx.clone())?,
            loss: LossModule::from_module(ptx.clone())?,
            transpose: TransposeModule::from_module(ptx.clone())?,
            decode: Nvfp4DecodeModule::from_module(ptx.clone())?,
            linear: LinearBackwardModule::from_module(ptx.clone())?,
            layer_norm_backward: LayerNormBackwardModule::from_module(ptx.clone())?,
            residual: ResidualBackwardModule::from_module(ptx.clone())?,
            optimizer: OptimizerModule::from_module(ptx)?,
        })
    }

    pub fn backward_modules(&self) -> Gpt2BackwardModules<'_> {
        let linear = AttentionBackwardModules {
            transpose: &self.transpose,
            decode: &self.decode,
            linear: &self.linear,
            quant: &self.quant,
        };
        Gpt2BackwardModules {
            final_head: FinalHeadBackwardModules {
                loss: &self.loss,
                transpose: &self.transpose,
                decode: &self.decode,
                linear: &self.linear,
                quant: &self.quant,
            },
            final_norm: &self.layer_norm_backward,
            attention: BlockAttentionBackwardModules {
                residual: &self.residual,
                layer_norm: &self.layer_norm_backward,
                attention: &self.attention,
                f16_tc: &self.f16_tc_matmul,
                linear,
            },
            mlp: BlockMlpBackwardModules {
                residual: &self.residual,
                layer_norm: &self.layer_norm_backward,
                mlp: MlpBackwardModules {
                    transpose: &self.transpose,
                    decode: &self.decode,
                    linear: &self.linear,
                    quant: &self.quant,
                    mlp: &self.mlp,
                },
            },
        }
    }
}

fn gpu_device_index() -> usize {
    std::env::var("CUDA_DEVICE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn ptx_path() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("rust_kernels_cuda.ptx")
        .to_string_lossy()
        .into_owned()
}
