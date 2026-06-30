mod accumulate;
mod body;
mod load;
mod stage;
mod store;
mod tile;

macro_rules! with_projection_cta_tiles {
    ($body:ident; $($arg:expr),+ $(,)?) => {{
        static mut A_PACKS: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_PACKS }> =
            cuda_device::SharedArray::UNINIT;
        static mut B_PACKS: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_B_PACKS }> =
            cuda_device::SharedArray::UNINIT;
        static mut A_SCALES: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_SCALES }> =
            cuda_device::SharedArray::UNINIT;
        static mut B_SCALES: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_B_SCALES }> =
            cuda_device::SharedArray::UNINIT;

        $body(
            $($arg,)*
            unsafe { &mut A_PACKS },
            unsafe { &mut B_PACKS },
            unsafe { &mut A_SCALES },
            unsafe { &mut B_SCALES },
        )
    }};
}

macro_rules! dispatch_projection_cta_tiles {
    ($params:expr, $aligned:ident, $generic:ident; $($arg:expr),+ $(,)?) => {{
        static mut A_PACKS: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_PACKS }> =
            cuda_device::SharedArray::UNINIT;
        static mut A1_PACKS: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_PACKS }> =
            cuda_device::SharedArray::UNINIT;
        static mut B_PACKS: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_B_PACKS }> =
            cuda_device::SharedArray::UNINIT;
        static mut A_SCALES: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_SCALES }> =
            cuda_device::SharedArray::UNINIT;
        static mut A1_SCALES: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_A_SCALES }> =
            cuda_device::SharedArray::UNINIT;
        static mut B_SCALES: cuda_device::SharedArray<u32, { $crate::mma::NVFP4_PROJECTION_CTA_B_SCALES }> =
            cuda_device::SharedArray::UNINIT;

        if $crate::mma::projection_cta_shape_aligned(
            ($params).token_count,
            ($params).input_dim,
            ($params).output_dim,
        ) {
            let (tile0, tile1) =
                $crate::mma::Nvfp4ProjectionCtaTile::row_pair(cuda_device::thread::threadIdx_x());
            $aligned(
                $($arg,)*
                unsafe { &mut A_PACKS },
                unsafe { &mut A1_PACKS },
                unsafe { &mut B_PACKS },
                unsafe { &mut A_SCALES },
                unsafe { &mut A1_SCALES },
                unsafe { &mut B_SCALES },
                tile0,
                tile1,
            )
        } else {
            $generic(
                $($arg,)*
                unsafe { &mut A_PACKS },
                unsafe { &mut B_PACKS },
                unsafe { &mut A_SCALES },
                unsafe { &mut B_SCALES },
            )
        }
    }};
}

pub(crate) use dispatch_projection_cta_tiles;
pub(crate) use with_projection_cta_tiles;

pub use body::{
    nvfp4_projection_cta_kernel_body, nvfp4_projection_cta_kernel_body_at_aligned_row_pair,
    nvfp4_projection_cta_nobias_kernel_body, nvfp4_projection_cta_nobias_kernel_body_at,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned,
    nvfp4_projection_cta_nobias_kernel_body_at_aligned_row_pair,
    nvfp4_projection_cta_relu2_kernel_body,
    nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair,
};
pub use tile::{
    NVFP4_PROJECTION_CTA_A_PACKS, NVFP4_PROJECTION_CTA_A_SCALES, NVFP4_PROJECTION_CTA_B_PACKS,
    NVFP4_PROJECTION_CTA_B_SCALES, NVFP4_PROJECTION_CTA_K, NVFP4_PROJECTION_CTA_M,
    NVFP4_PROJECTION_CTA_N, NVFP4_PROJECTION_CTA_THREADS, Nvfp4ProjectionCtaTile,
    projection_cta_grid_dim, projection_cta_row_pair_grid_dim, projection_cta_row_pair_tile_count,
    projection_cta_shape_aligned,
};
