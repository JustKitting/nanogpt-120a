macro_rules! projection_cta_biased_body_fns {
    (
        $body:ident,
        $row_pair_body:ident,
        $store:ident,
        $store_aligned:ident,
        extra: [$($extra_arg:ident: $extra_ty:ty),+ $(,)?],
        store_args: [$($store_arg:ident),+ $(,)?] $(,)?
    ) => {
        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        pub fn $body(
            input_bytes: &[u8],
            input_scales: &[u8],
            input_global_scales: &[f32],
            weight_bytes: &[u8],
            weight_scales: &[u8],
            $($extra_arg: $extra_ty,)+
            params: $crate::mma::projection::Nvfp4ProjectionParams,
            a_packs: &mut $crate::mma::ProjectionCtaAPacks,
            b_packs: &mut $crate::mma::ProjectionCtaBPacks,
            a_scales: &mut $crate::mma::ProjectionCtaAScales,
            b_scales: &mut $crate::mma::ProjectionCtaBScales,
        ) {
            let thread_id = cuda_device::thread::threadIdx_x();
            if thread_id >= $crate::mma::NVFP4_PROJECTION_CTA_THREADS {
                return;
            }

            let tile = $crate::mma::Nvfp4ProjectionCtaTile::new(thread_id);
            let sources = $crate::mma::projection_cta::ProjectionCtaSources { input_bytes, input_scales, weight_bytes, weight_scales };
            let mut tiles = $crate::mma::projection_cta::ProjectionCtaTiles { a_packs, b_packs, a_scales, b_scales };
            let acc = $crate::mma::projection_cta::accumulate::projection_accumulator(
                sources, tile, &params, &mut tiles,
            );
            $store(acc, input_global_scales, $($store_arg,)+ tile, &params);
        }

        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        pub fn $row_pair_body(
            input_bytes: &[u8],
            input_scales: &[u8],
            input_global_scales: &[f32],
            weight_bytes: &[u8],
            weight_scales: &[u8],
            $($extra_arg: $extra_ty,)+
            params: $crate::mma::projection::Nvfp4ProjectionParams,
            mut tiles: $crate::mma::ProjectionCtaRowPairTiles<'_>,
            tile0: $crate::mma::Nvfp4ProjectionCtaTile,
            tile1: $crate::mma::Nvfp4ProjectionCtaTile,
        ) {
            let sources = $crate::mma::projection_cta::ProjectionCtaSources { input_bytes, input_scales, weight_bytes, weight_scales };
            let (acc0, acc1) =
                $crate::mma::projection_cta::accumulate::projection_accumulator_aligned_row_pair(
                    sources, tile0, tile1, &params, &mut tiles,
                );
            $store_aligned(acc0, input_global_scales, $($store_arg,)+ tile0, &params);
            if tile1.row_base < params.token_count {
                $store_aligned(acc1, input_global_scales, $($store_arg,)+ tile1, &params);
            }
        }
    };
}

mod affine;
mod nobias;
mod relu2;

pub use affine::{
    nvfp4_projection_cta_kernel_body, nvfp4_projection_cta_kernel_body_at_aligned_row_pair,
};
pub use nobias::{
    nvfp4_projection_cta_nobias_kernel_body, nvfp4_projection_cta_nobias_kernel_body_at,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair,
};
pub use relu2::{
    nvfp4_projection_cta_relu2_kernel_body,
    nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair,
};
