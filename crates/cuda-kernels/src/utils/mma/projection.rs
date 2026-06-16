use cuda_core::DeviceCopy;
use cuda_device::{DisjointSlice, thread};

use crate::float_ptx::{fma_f32, max_f32};
use crate::mma::mma_m16n8k64_scale4x_ue4m3;
use crate::nvfp4::nvfp4_value;

pub const NVFP4_PROJECTION_THREADS_PER_BLOCK: u32 = 32;
pub const NVFP4_PROJECTION_M: u32 = 16;
pub const NVFP4_PROJECTION_N: u32 = 8;
pub const NVFP4_PROJECTION_ACTIVATION_NONE: u32 = 0;
pub const NVFP4_PROJECTION_ACTIVATION_RELU2: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Nvfp4ProjectionParams {
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub weight_global_scale: f32,
    pub bias_global_scale: f32,
    pub residual_add: u32,
    pub activation: u32,
}

unsafe impl DeviceCopy for Nvfp4ProjectionParams {}

#[derive(Clone, Copy)]
pub struct Nvfp4ProjectionTile {
    pub tile_row: u32,
    pub tile_col: u32,
    pub group: u32,
    pub thread_in_group: u32,
}

pub fn projection_grid_dim(token_count: u32, output_dim: u32) -> (u32, u32, u32) {
    (
        output_dim.div_ceil(NVFP4_PROJECTION_N),
        token_count.div_ceil(NVFP4_PROJECTION_M),
        1,
    )
}

macro_rules! load_fragments {
    ($len:expr, $loader:ident, $($arg:expr),+ $(,)?) => {{
        let mut fragments = [0_u32; $len];
        let mut register = 0;
        while register < $len {
            fragments[register as usize] = $loader($($arg,)* register);
            register += 1;
        }
        fragments
    }};
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn nvfp4_projection_kernel_body(
    input_bytes: &[u8],
    input_scales: &[u8],
    input_global_scales: &[f32],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    out: &mut DisjointSlice<'_, f32>,
    params: Nvfp4ProjectionParams,
) {
    let lane = thread::threadIdx_x();
    if lane >= NVFP4_PROJECTION_THREADS_PER_BLOCK {
        return;
    }

    let tile_col = thread::blockIdx_x() * NVFP4_PROJECTION_N;
    let tile_row = thread::blockIdx_y() * NVFP4_PROJECTION_M;
    let group = lane >> 2;
    let thread_in_group = lane & 0x3;
    let tile = Nvfp4ProjectionTile {
        tile_row,
        tile_col,
        group,
        thread_in_group,
    };
    let acc = nvfp4_projection_accumulate_tile(
        input_bytes,
        input_scales,
        weight_bytes,
        weight_scales,
        tile,
        &params,
    );

    store_accumulator(
        acc,
        group,
        thread_in_group,
        StoreAccumulatorArgs {
            input_global_scales,
            bias_bytes,
            bias_scales,
            tile_row,
            tile_col,
            params: &params,
        },
        out,
    );
}

#[inline(always)]
pub fn nvfp4_projection_accumulate_tile(
    input_bytes: &[u8],
    input_scales: &[u8],
    weight_bytes: &[u8],
    weight_scales: &[u8],
    tile: Nvfp4ProjectionTile,
    params: &Nvfp4ProjectionParams,
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    let mut k_base = 0;

    while k_base < params.input_dim {
        let a = load_fragments!(
            4,
            load_a_fragment,
            input_bytes,
            tile.tile_row,
            k_base,
            tile.group,
            tile.thread_in_group,
            params,
        );
        let b = load_fragments!(
            2,
            load_b_fragment,
            weight_bytes,
            tile.tile_col,
            k_base,
            tile.group,
            tile.thread_in_group,
            params,
        );
        let scale_a = load_a_scale4(
            input_scales,
            tile.tile_row,
            k_base,
            tile.group,
            tile.thread_in_group,
            params,
        );
        let scale_b = load_b_scale4(weight_scales, tile.tile_col, k_base, tile.group, params);

        mma_m16n8k64_scale4x_ue4m3(a, b, &mut acc, scale_a, scale_b);
        k_base += MMA_K;
    }

    acc
}

const MMA_K: u32 = 64;
const SCALE_GROUP: u32 = 16;
const E4M3_ONE_PACKED4: u32 = 0x3838_3838;

#[inline(always)]
fn load_a_fragment(
    input_bytes: &[u8],
    tile_row: u32,
    k_base: u32,
    group: u32,
    thread_in_group: u32,
    params: &Nvfp4ProjectionParams,
    register: u32,
) -> u32 {
    let row = tile_row + group + if register & 1 == 0 { 0 } else { 8 };
    let col = k_base + thread_in_group * 8 + if register < 2 { 0 } else { 32 };

    if row < params.token_count && col + 7 < params.input_dim {
        load_packed8(
            input_bytes,
            row as usize * params.input_dim as usize + col as usize,
        )
    } else {
        0
    }
}

#[inline(always)]
fn load_b_fragment(
    weight_bytes: &[u8],
    tile_col: u32,
    k_base: u32,
    group: u32,
    thread_in_group: u32,
    params: &Nvfp4ProjectionParams,
    register: u32,
) -> u32 {
    let col = tile_col + group;
    let row = k_base + thread_in_group * 8 + if register == 0 { 0 } else { 32 };

    if col < params.output_dim && row + 7 < params.input_dim {
        load_packed8(
            weight_bytes,
            col as usize * params.input_dim as usize + row as usize,
        )
    } else {
        0
    }
}

#[inline(always)]
fn load_a_scale4(
    input_scales: &[u8],
    tile_row: u32,
    k_base: u32,
    group: u32,
    thread_in_group: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let row = tile_row + group + if thread_in_group == 1 { 8 } else { 0 };
    if row < params.token_count {
        let scale_base = (row * params.input_dim + k_base) / SCALE_GROUP;
        load_scale4(input_scales, scale_base as usize)
    } else {
        E4M3_ONE_PACKED4
    }
}

#[inline(always)]
fn load_b_scale4(
    weight_scales: &[u8],
    tile_col: u32,
    k_base: u32,
    group: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let col = tile_col + group;
    if col < params.output_dim {
        let scales_per_col = params.input_dim / SCALE_GROUP;
        let scale_base = col * scales_per_col + k_base / SCALE_GROUP;
        load_scale4(weight_scales, scale_base as usize)
    } else {
        E4M3_ONE_PACKED4
    }
}

#[inline(always)]
fn load_packed8(bytes: &[u8], element_base: usize) -> u32 {
    let byte_base = element_base / 2;
    if byte_base + 3 < bytes.len() {
        (bytes[byte_base] as u32)
            | ((bytes[byte_base + 1] as u32) << 8)
            | ((bytes[byte_base + 2] as u32) << 16)
            | ((bytes[byte_base + 3] as u32) << 24)
    } else {
        0
    }
}

#[inline(always)]
fn load_scale4(scales: &[u8], scale_base: usize) -> u32 {
    if scale_base + 3 < scales.len() {
        (scales[scale_base] as u32)
            | ((scales[scale_base + 1] as u32) << 8)
            | ((scales[scale_base + 2] as u32) << 16)
            | ((scales[scale_base + 3] as u32) << 24)
    } else {
        E4M3_ONE_PACKED4
    }
}

struct StoreAccumulatorArgs<'a> {
    input_global_scales: &'a [f32],
    bias_bytes: &'a [u8],
    bias_scales: &'a [u8],
    tile_row: u32,
    tile_col: u32,
    params: &'a Nvfp4ProjectionParams,
}

#[inline(always)]
fn store_accumulator(
    acc: [f32; 4],
    group: u32,
    thread_in_group: u32,
    args: StoreAccumulatorArgs<'_>,
    out: &mut DisjointSlice<'_, f32>,
) {
    let mut i = 0;
    while i < 4 {
        let row = args.tile_row + group + if i < 2 { 0 } else { 8 };
        let col = args.tile_col + thread_in_group * 2 + (i & 1);

        if row < args.params.token_count && col < args.params.output_dim {
            let global_scale =
                args.input_global_scales[row as usize] * args.params.weight_global_scale;
            let bias = nvfp4_value(
                args.bias_bytes,
                args.bias_scales,
                args.params.bias_global_scale,
                col as usize,
            );
            let value = apply_activation(
                fma_f32(acc[i as usize], global_scale, bias),
                args.params.activation,
            );
            let index = row as usize * args.params.output_dim as usize + col as usize;

            unsafe {
                let value = if args.params.residual_add == 0 {
                    value
                } else {
                    *out.get_unchecked_mut(index) + value
                };
                *out.get_unchecked_mut(index) = value;
            }
        }

        i += 1;
    }
}

#[inline(always)]
fn apply_activation(value: f32, activation: u32) -> f32 {
    if activation == NVFP4_PROJECTION_ACTIVATION_RELU2 {
        let relu = max_f32(value, 0.0);
        relu * relu
    } else {
        value
    }
}
