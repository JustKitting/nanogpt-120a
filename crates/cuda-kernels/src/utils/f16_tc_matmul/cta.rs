cta_bt_matmul_body_fn!(
    cta_matmul_body,
    u16,
    u16,
    super::cta_stage::stage_tiles,
    super::cta_stage::stage_tiles_aligned
);

cta_bt_matmul_lower_body_fn!(
    cta_matmul_lower_body,
    u16,
    u16,
    super::cta_stage::stage_tiles,
    super::cta_stage::stage_tiles_aligned
);
