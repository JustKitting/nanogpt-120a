cta_add_matmul_body_fn!(
    cta_matmul_add_f32_rhs_transposed_base_body,
    rhs,
    super::cta_stage_f32::stage_tiles_f32_rhs_transposed
);
