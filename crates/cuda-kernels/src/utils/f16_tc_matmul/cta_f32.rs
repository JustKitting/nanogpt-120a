cta_bt_matmul_body_fn!(
    cta_matmul_f32_body,
    f32,
    f32,
    super::cta_stage_f32::stage_tiles_f32_b_t,
    super::cta_stage_f32::stage_tiles_f32_b_t_aligned
);
