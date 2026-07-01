use cuda_core::DeviceCopy;

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

impl Nvfp4ProjectionParams {
    #[inline(always)]
    pub fn new(token_count: u32, input_dim: u32, output_dim: u32) -> Self {
        Self {
            token_count,
            input_dim,
            output_dim,
            weight_global_scale: 1.0,
            bias_global_scale: 1.0,
            residual_add: 0,
            activation: NVFP4_PROJECTION_ACTIVATION_NONE,
        }
    }

    #[inline(always)]
    pub fn with_residual_add(self, residual_add: u32) -> Self {
        Self {
            residual_add,
            ..self
        }
    }

    #[inline(always)]
    pub fn with_global_scales(self, weight: f32, bias: f32) -> Self {
        Self {
            weight_global_scale: weight,
            bias_global_scale: bias,
            ..self
        }
    }
}

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
