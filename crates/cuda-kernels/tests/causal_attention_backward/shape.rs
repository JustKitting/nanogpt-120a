pub const TOKEN_COUNT: usize = 5;
pub const HEADS: usize = 2;
pub const HEAD_DIM: usize = 8;
pub const EMBEDDING: usize = HEADS * HEAD_DIM;
pub const QKV_DIM: usize = 3 * EMBEDDING;

pub fn qkv_index(token: usize, head: usize, dim: usize, offset: usize) -> usize {
    token * QKV_DIM + offset + head * HEAD_DIM + dim
}

pub fn hidden_index(token: usize, head: usize, dim: usize) -> usize {
    token * EMBEDDING + head * HEAD_DIM + dim
}

pub fn log_sum_exp_index(token: usize, head: usize) -> usize {
    head * TOKEN_COUNT + token
}
