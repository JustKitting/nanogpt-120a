use cuda_core::LaunchConfig;

use crate::launch::launch_config;

const SOFTMAX_D_THREADS_PER_BLOCK: u32 = 64;

pub(super) fn attention_config(seq_len: u32, head_count: u32, batch_size: u32) -> LaunchConfig {
    launch_config(
        (seq_len, head_count, batch_size),
        SOFTMAX_D_THREADS_PER_BLOCK,
    )
}
