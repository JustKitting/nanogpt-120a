use crate::mma::projection_cta::store::{store_relu2_accumulator, store_relu2_accumulator_aligned};

projection_cta_biased_body_fns!(
    nvfp4_projection_cta_relu2_kernel_body,
    nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair,
    store_relu2_accumulator,
    store_relu2_accumulator_aligned,
    extra: [
        bias_bytes: &[u8],
        bias_scales: &[u8],
        pre_activation: &mut cuda_device::DisjointSlice<'_, f32>,
        out: &mut cuda_device::DisjointSlice<'_, f32>,
    ],
    store_args: [bias_bytes, bias_scales, pre_activation, out],
);
