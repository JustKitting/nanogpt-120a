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
            a_packs: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_PACKS }>,
            b_packs: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_B_PACKS }>,
            a_scales: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_SCALES }>,
            b_scales: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_B_SCALES }>,
        ) {
            let thread_id = cuda_device::thread::threadIdx_x();
            if thread_id >= $crate::mma::NVFP4_PROJECTION_CTA_THREADS {
                return;
            }

            let tile = $crate::mma::Nvfp4ProjectionCtaTile::new(thread_id);
            let acc = $crate::mma::projection_cta::accumulate::projection_accumulator(
                input_bytes,
                input_scales,
                weight_bytes,
                weight_scales,
                tile,
                &params,
                a_packs,
                b_packs,
                a_scales,
                b_scales,
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
            a_packs: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_PACKS }>,
            a1_packs: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_PACKS }>,
            b_packs: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_B_PACKS }>,
            a_scales: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_SCALES }>,
            a1_scales: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_SCALES }>,
            b_scales: &mut cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_B_SCALES }>,
            tile0: $crate::mma::Nvfp4ProjectionCtaTile,
            tile1: $crate::mma::Nvfp4ProjectionCtaTile,
        ) {
            let (acc0, acc1) =
                $crate::mma::projection_cta::accumulate::projection_accumulator_aligned_row_pair(
                    input_bytes,
                    input_scales,
                    weight_bytes,
                    weight_scales,
                    tile0,
                    tile1,
                    &params,
                    a_packs,
                    a1_packs,
                    b_packs,
                    a_scales,
                    a1_scales,
                    b_scales,
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
