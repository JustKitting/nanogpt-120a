macro_rules! pack_padded_transpose_chunk {
    (
        chunk_amax,
        input: $input_fn:ident($bytes:expr, $scales:expr, $source_global_scales:expr),
        chunk: $chunk:expr,
        output: [$out_fp4:expr, $out_scales:expr, $out_global_scales:expr, $out_chunk_amax:expr],
        dims: [$source_rows:expr, $source_cols:expr, $dst_row_len:expr],
        scale: [$global_scale:expr, $scale_override:expr, $scale_seed:expr],
        sign_seed: $sign_seed:expr $(,)?
    ) => {{
        let lane = warp::lane_id();
        let chunk_base = $chunk * HADAMARD_DIM;
        let input = $input_fn(
            $bytes,
            $scales,
            $source_global_scales,
            chunk_base,
            lane,
            $source_rows,
            $source_cols,
            $dst_row_len,
            $sign_seed,
        );
        ms_eden_pack_chunk(
            input,
            $out_fp4,
            $out_scales,
            $out_global_scales,
            $out_chunk_amax,
            $chunk,
            $dst_row_len,
            $global_scale,
            $scale_override,
            $scale_seed,
        );
    }};
    (
        no_chunk_amax,
        input: $input_fn:ident($bytes:expr, $scales:expr, $source_global_scales:expr),
        chunk: $chunk:expr,
        output: [$out_fp4:expr, $out_scales:expr, $out_global_scales:expr],
        dims: [$source_rows:expr, $source_cols:expr, $dst_row_len:expr],
        scale: [$global_scale:expr, $scale_override:expr, $scale_seed:expr],
        sign_seed: $sign_seed:expr $(,)?
    ) => {{
        let lane = warp::lane_id();
        let chunk_base = $chunk * HADAMARD_DIM;
        let input = $input_fn(
            $bytes,
            $scales,
            $source_global_scales,
            chunk_base,
            lane,
            $source_rows,
            $source_cols,
            $dst_row_len,
            $sign_seed,
        );
        ms_eden_pack_chunk_no_chunk_amax(
            input,
            $out_fp4,
            $out_scales,
            $out_global_scales,
            $chunk,
            $dst_row_len,
            $global_scale,
            $scale_override,
            $scale_seed,
        );
    }};
}

pub(super) use pack_padded_transpose_chunk;
