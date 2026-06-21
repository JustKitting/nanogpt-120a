use cuda_device::{DisjointSlice, SharedArray, thread, warp};

pub const LINEAR_BIAS_THREADS_PER_BLOCK: u32 = 256;

const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = LINEAR_BIAS_THREADS_PER_BLOCK / WARP_SIZE;
const COLS_PER_BLOCK: u32 = WARP_SIZE;
const ROW_UNROLL: u32 = 4;
const ROW_STRIDE: u32 = WARPS_PER_BLOCK;
const UNROLLED_ROW_STRIDE: u32 = ROW_STRIDE * ROW_UNROLL;

pub fn grid_dim(output_dim: u32) -> u32 {
    output_dim.div_ceil(COLS_PER_BLOCK)
}

pub fn linear_bias_grad_body(
    e: &[f32],
    dbias: &mut DisjointSlice<f32>,
    token_count: u32,
    output_dim: u32,
    local_sums: &mut SharedArray<f32, { LINEAR_BIAS_THREADS_PER_BLOCK as usize }>,
) {
    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_in_block = tid / WARP_SIZE;
    let col = thread::blockIdx_x() * COLS_PER_BLOCK + lane;
    let mut local = 0.0f32;

    if col < output_dim {
        let mut row = warp_in_block;

        macro_rules! accumulate_row {
            ($row:expr) => {{
                let row = $row;
                local += e[row as usize * output_dim as usize + col as usize];
            }};
        }

        while row + ROW_STRIDE * 3 < token_count {
            accumulate_row!(row);
            accumulate_row!(row + ROW_STRIDE);
            accumulate_row!(row + ROW_STRIDE * 2);
            accumulate_row!(row + ROW_STRIDE * 3);
            row += UNROLLED_ROW_STRIDE;
        }

        while row < token_count {
            accumulate_row!(row);
            row += ROW_STRIDE;
        }
    }

    local_sums[tid as usize] = local;
    thread::sync_threads();

    if warp_in_block == 0 && col < output_dim {
        let sum = local_sums[lane as usize]
            + local_sums[(lane + WARP_SIZE) as usize]
            + local_sums[(lane + WARP_SIZE * 2) as usize]
            + local_sums[(lane + WARP_SIZE * 3) as usize]
            + local_sums[(lane + WARP_SIZE * 4) as usize]
            + local_sums[(lane + WARP_SIZE * 5) as usize]
            + local_sums[(lane + WARP_SIZE * 6) as usize]
            + local_sums[(lane + WARP_SIZE * 7) as usize];
        unsafe {
            *dbias.get_unchecked_mut(col as usize) = sum;
        }
    }
}
