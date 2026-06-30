use cuda_device::thread;

pub const NVFP4_PROJECTION_CTA_M: u32 = 32;
pub const NVFP4_PROJECTION_CTA_N: u32 = 64;
pub const NVFP4_PROJECTION_CTA_K: u32 = 128;
pub const NVFP4_PROJECTION_CTA_K_ATOMS: u32 = NVFP4_PROJECTION_CTA_K / 64;
pub const NVFP4_PROJECTION_CTA_THREADS: u32 = 512;
pub const NVFP4_PROJECTION_CTA_PACKS_PER_ROW: u32 = NVFP4_PROJECTION_CTA_K / 8;
pub const NVFP4_PROJECTION_CTA_A_PACKS: usize =
    (NVFP4_PROJECTION_CTA_M * NVFP4_PROJECTION_CTA_PACKS_PER_ROW) as usize;
pub const NVFP4_PROJECTION_CTA_B_PACKS: usize =
    (NVFP4_PROJECTION_CTA_N * NVFP4_PROJECTION_CTA_PACKS_PER_ROW) as usize;
pub const NVFP4_PROJECTION_CTA_A_SCALES: usize =
    (NVFP4_PROJECTION_CTA_M * NVFP4_PROJECTION_CTA_K_ATOMS) as usize;
pub const NVFP4_PROJECTION_CTA_B_SCALES: usize =
    (NVFP4_PROJECTION_CTA_N * NVFP4_PROJECTION_CTA_K_ATOMS) as usize;

#[derive(Clone, Copy)]
pub struct Nvfp4ProjectionCtaTile {
    pub row_base: u32,
    pub col_base: u32,
    pub warp_m: u32,
    pub warp_n: u32,
    pub group: u32,
    pub thread_in_group: u32,
}

impl Nvfp4ProjectionCtaTile {
    pub fn new(thread_id: u32) -> Self {
        Self::from_grid_tile(thread::blockIdx_x(), thread::blockIdx_y(), thread_id)
    }

    pub fn from_grid_tile(tile_col: u32, tile_row: u32, thread_id: u32) -> Self {
        let lane = thread_id & 31;
        let warp = thread_id >> 5;
        Self {
            row_base: tile_row * NVFP4_PROJECTION_CTA_M,
            col_base: tile_col * NVFP4_PROJECTION_CTA_N,
            warp_m: warp >> 3,
            warp_n: warp & 7,
            group: lane >> 2,
            thread_in_group: lane & 3,
        }
    }

    pub fn row_pair(thread_id: u32) -> (Self, Self) {
        let tile_col = thread::blockIdx_x();
        let tile_row_pair = thread::blockIdx_y();
        (
            Self::from_grid_tile(tile_col, tile_row_pair * 2, thread_id),
            Self::from_grid_tile(tile_col, tile_row_pair * 2 + 1, thread_id),
        )
    }

    pub fn packed_row_pair(
        tile_index: u32,
        grid_col_mask: u32,
        grid_col_shift: u32,
        thread_id: u32,
    ) -> (Self, Self) {
        let tile_col = tile_index & grid_col_mask;
        let tile_row_pair = tile_index >> grid_col_shift;
        (
            Self::from_grid_tile(tile_col, tile_row_pair * 2, thread_id),
            Self::from_grid_tile(tile_col, tile_row_pair * 2 + 1, thread_id),
        )
    }

    #[inline(always)]
    pub fn mma_row_base(self) -> u32 {
        self.row_base + self.warp_m * 16
    }

    #[inline(always)]
    pub fn mma_col_base(self) -> u32 {
        self.col_base + self.warp_n * 8
    }
}

pub fn projection_cta_grid_dim(token_count: u32, output_dim: u32) -> (u32, u32, u32) {
    (
        output_dim.div_ceil(NVFP4_PROJECTION_CTA_N),
        token_count.div_ceil(NVFP4_PROJECTION_CTA_M),
        1,
    )
}

pub fn projection_cta_row_pair_grid_dim(token_count: u32, output_dim: u32) -> (u32, u32, u32) {
    let grid = projection_cta_grid_dim(token_count, output_dim);
    (grid.0, grid.1.div_ceil(2), 1)
}

pub fn projection_cta_row_pair_tile_count(token_count: u32, output_dim: u32) -> u32 {
    let grid = projection_cta_row_pair_grid_dim(token_count, output_dim);
    grid.0 * grid.1
}

pub fn projection_cta_shape_aligned(token_count: u32, input_dim: u32, output_dim: u32) -> bool {
    token_count % NVFP4_PROJECTION_CTA_M == 0
        && input_dim % NVFP4_PROJECTION_CTA_K == 0
        && output_dim % NVFP4_PROJECTION_CTA_N == 0
}
