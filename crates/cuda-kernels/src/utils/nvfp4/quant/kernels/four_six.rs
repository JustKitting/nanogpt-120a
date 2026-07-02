use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::convert::cvt_rn_satfinite_e2m1x2_f32;

#[path = "four_six/helpers.rs"]
pub(crate) mod helpers;

#[cuda_module]
pub(crate) mod module {
    use super::helpers::*;
    use super::*;

    #[derive(Clone, Copy)]
    struct FourSixGroup {
        group: usize,
        base: usize,
        lane: usize,
        mask: u32,
        leader: u32,
    }

    struct FourSixOutputs<'a> {
        fp4: DisjointSlice<'a, u8>,
        scales: DisjointSlice<'a, u8>,
        global_scale: DisjointSlice<'a, f32>,
    }

    #[inline(always)]
    fn four_six_group_ctx() -> FourSixGroup {
        let (lane, mask, leader) = four_six_lane();
        let group = four_six_block_group();
        FourSixGroup {
            group,
            base: group * GROUP_SIZE,
            lane,
            mask,
            leader,
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        row_len: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let row_len = row_len as usize;
            let scalar_scale = row_len == 0;
            let scale_row_len = if scalar_scale { usize::MAX } else { row_len };
            let row = group_ctx.base / scale_row_len;
            let writes_global_scale = if scalar_scale {
                group_ctx.group == 0
            } else {
                group_ctx.base == row * scale_row_len
            };
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group(
                x,
                amax,
                out,
                group_ctx,
                row,
                writes_global_scale,
                scale_override,
            );
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_rowwise_pow2_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        row_shift: u32,
        row_mask: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();
        let row = (group_ctx.base as u32 >> row_shift) as usize;
        let writes_global_scale = (group_ctx.base as u32 & row_mask) == 0;
        let out = FourSixOutputs {
            fp4: out_fp4,
            scales: out_scales,
            global_scale: out_global_scale,
        };
        pack_four_six_group(
            x,
            amax,
            out,
            group_ctx,
            row,
            writes_global_scale,
            scale_override,
        );
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_padded_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        padded_cols: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let base = group_ctx.base as u32;
            let value = padded_value(x, base + group_ctx.lane as u32, rows, cols, padded_cols);
            let pair = group_ctx.lane as u32 * 2;
            let hi = padded_value(x, base + pair, rows, cols, padded_cols);
            let lo = padded_value(x, base + pair + 1, rows, cols, padded_cols);
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                amax[0],
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
                value,
                hi,
                lo,
            );
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_exact_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group(
                x,
                amax,
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
            );
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_padded_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        source_rows: u32,
        source_cols: u32,
        padded_cols: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let base = group_ctx.base as u32;
            let value = transposed_padded_value(
                x,
                base + group_ctx.lane as u32,
                source_rows,
                source_cols,
                padded_cols,
            );
            let pair = group_ctx.lane as u32 * 2;
            let hi = transposed_padded_value(x, base + pair, source_rows, source_cols, padded_cols);
            let lo =
                transposed_padded_value(x, base + pair + 1, source_rows, source_cols, padded_cols);
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                amax[0],
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
                value,
                hi,
                lo,
            );
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_exact_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        source_rows: u32,
        source_cols: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let base = group_ctx.base as u32;
            let value =
                transposed_exact_value(x, base + group_ctx.lane as u32, source_rows, source_cols);
            let pair = group_ctx.lane as u32 * 2;
            let hi = transposed_exact_value(x, base + pair, source_rows, source_cols);
            let lo = transposed_exact_value(x, base + pair + 1, source_rows, source_cols);
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                amax[0],
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
                value,
                hi,
                lo,
            );
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_exact_pow2_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        source_rows_shift: u32,
        source_rows_mask: u32,
        source_cols: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let base = group_ctx.base as u32;
            let value = transposed_exact_value_pow2(
                x,
                base + group_ctx.lane as u32,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let pair = group_ctx.lane as u32 * 2;
            let hi = transposed_exact_value_pow2(
                x,
                base + pair,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let lo = transposed_exact_value_pow2(
                x,
                base + pair + 1,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                amax[0],
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
                value,
                hi,
                lo,
            );
        }
    }

    fn pack_four_six_group(
        x: &[f32],
        amax: &[f32],
        out: FourSixOutputs<'_>,
        group_ctx: FourSixGroup,
        row: usize,
        writes_global_scale: bool,
        scale_override: f32,
    ) {
        let value = x[group_ctx.base + group_ctx.lane];
        let pair = group_ctx.lane * 2;
        let (hi, lo) = if group_ctx.lane < GROUP_SIZE / 2 {
            (x[group_ctx.base + pair], x[group_ctx.base + pair + 1])
        } else {
            (0.0, 0.0)
        };
        pack_four_six_group_values(
            amax[row],
            out,
            group_ctx,
            row,
            writes_global_scale,
            scale_override,
            value,
            hi,
            lo,
        );
    }

    #[expect(clippy::too_many_arguments, reason = "device pack path is explicit")]
    fn pack_four_six_group_values(
        tensor_amax: f32,
        mut out: FourSixOutputs<'_>,
        group_ctx: FourSixGroup,
        scale_row: usize,
        writes_global_scale: bool,
        scale_override: f32,
        value: f32,
        hi: f32,
        lo: f32,
    ) {
        let global_scale = four_six_global_scale(tensor_amax, scale_override);
        let (scale_bits, inv_scale) = four_six_group_scale(
            value,
            global_scale,
            scale_override,
            group_ctx.mask,
            group_ctx.leader,
            group_ctx.lane,
        );
        unsafe {
            if writes_global_scale && group_ctx.lane == 0 {
                *out.global_scale.get_unchecked_mut(scale_row) = global_scale;
            }
            if group_ctx.lane == 0 {
                *out.scales.get_unchecked_mut(group_ctx.group) = scale_bits;
            }
            if group_ctx.lane < GROUP_SIZE / 2 {
                *out.fp4
                    .get_unchecked_mut(group_ctx.base / 2 + group_ctx.lane) =
                    cvt_rn_satfinite_e2m1x2_f32(lo * inv_scale, hi * inv_scale);
            }
        }
    }

    fn padded_value(x: &[f32], output_index: u32, rows: u32, cols: u32, padded_cols: u32) -> f32 {
        let row = output_index / padded_cols;
        let col = output_index - row * padded_cols;
        if row < rows && col < cols {
            x[(row * cols + col) as usize]
        } else {
            0.0
        }
    }

    fn transposed_padded_value(
        x: &[f32],
        output_index: u32,
        source_rows: u32,
        source_cols: u32,
        padded_cols: u32,
    ) -> f32 {
        let row = output_index / padded_cols;
        let col = output_index - row * padded_cols;
        if row < source_cols && col < source_rows {
            x[(col * source_cols + row) as usize]
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn transposed_exact_value(
        x: &[f32],
        output_index: u32,
        source_rows: u32,
        source_cols: u32,
    ) -> f32 {
        let row = output_index / source_rows;
        let col = output_index - row * source_rows;
        x[(col * source_cols + row) as usize]
    }

    #[inline(always)]
    fn transposed_exact_value_pow2(
        x: &[f32],
        output_index: u32,
        source_rows_shift: u32,
        source_rows_mask: u32,
        source_cols: u32,
    ) -> f32 {
        let row = output_index >> source_rows_shift;
        let col = output_index & source_rows_mask;
        x[(col * source_cols + row) as usize]
    }
}
