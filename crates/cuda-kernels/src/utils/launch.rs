use cuda_core::LaunchConfig;

#[inline]
pub(crate) fn launch_config(grid_dim: (u32, u32, u32), threads_per_block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim,
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[inline]
pub(crate) fn grid_x_config(grid_x: u32, threads_per_block: u32) -> LaunchConfig {
    launch_config((grid_x, 1, 1), threads_per_block)
}

#[inline]
pub(crate) fn linear_config(element_count: u32, threads_per_block: u32) -> LaunchConfig {
    grid_x_config(element_count.div_ceil(threads_per_block), threads_per_block)
}
