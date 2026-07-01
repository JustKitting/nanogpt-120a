macro_rules! store_acc4 {
    ($store:ident, $acc:ident, $($args:expr),+ $(,)?) => {
        $store($acc[0], 0, $($args),+);
        $store($acc[1], 1, $($args),+);
        $store($acc[2], 2, $($args),+);
        $store($acc[3], 3, $($args),+);
    };
}

pub mod f16;
pub mod mxf4nvf4;
pub mod projection;
pub mod projection_cta;
pub mod tensors;

pub use f16::mma_m16n8k16_f16_f16_f32;
pub use mxf4nvf4::mma_m16n8k64_scale4x_ue4m3;
pub use projection::{
    NVFP4_PROJECTION_ACTIVATION_NONE, NVFP4_PROJECTION_ACTIVATION_RELU2, NVFP4_PROJECTION_M,
    NVFP4_PROJECTION_N, NVFP4_PROJECTION_THREADS_PER_BLOCK, Nvfp4ProjectionParams,
    Nvfp4ProjectionTile, nvfp4_projection_accumulate_tile, nvfp4_projection_kernel_body,
    nvfp4_projection_nobias_kernel_body, nvfp4_projection_relu2_kernel_body,
    nvfp4_projection_residual_kernel_body, projection_grid_dim,
};
pub use projection_cta::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_M,
    NVFP4_PROJECTION_CTA_N, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionCtaTile,
    nvfp4_projection_cta_kernel_body, nvfp4_projection_cta_kernel_body_at_aligned_row_pair,
    nvfp4_projection_cta_nobias_kernel_body, nvfp4_projection_cta_nobias_kernel_body_at,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair,
    nvfp4_projection_cta_relu2_kernel_body,
    nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair, projection_cta_grid_dim,
    projection_cta_launch_grid_dim, projection_cta_row_pair_grid_dim,
    projection_cta_row_pair_tile_count, projection_cta_shape_aligned,
};
pub(crate) use projection_cta::{ProjectionCtaAPacks, ProjectionCtaAScales, ProjectionCtaBPacks, ProjectionCtaBScales, dispatch_projection_cta_tiles, with_projection_cta_tiles};
pub use tensors::{Nvfp4DeviceScaleMmaWeightTensor, Nvfp4FourSixMmaWeightTensor};
