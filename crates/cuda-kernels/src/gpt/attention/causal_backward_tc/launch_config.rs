use cuda_core::LaunchConfig;

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::launch::{launch_config, linear_config as linear_launch_config};

const SOFTMAX_D_THREADS_PER_BLOCK: u32 = 64;

pub(super) fn linear_config(element_count: u32) -> LaunchConfig {
    linear_launch_config(element_count, TC_BACKWARD_THREADS_PER_BLOCK)
}

pub(super) fn attention_config(seq_len: u32, head_count: u32, batch_size: u32) -> LaunchConfig {
    launch_config(
        (seq_len, head_count, batch_size),
        SOFTMAX_D_THREADS_PER_BLOCK,
    )
}
