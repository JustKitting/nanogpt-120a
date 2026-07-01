#[path = "body/pair.rs"]
mod pair;
#[path = "body/position.rs"]
mod position;
#[path = "body/rotate.rs"]
mod rotate;

use cuda_device::DisjointSlice;

use super::ApplyRopeParams;
use pair::{read_pair, store_pair_f16};
use position::rope_position;
use rotate::rotate_section;

pub(super) fn apply_rope_body(mut qkv: DisjointSlice<f32>, params: ApplyRopeParams) {
    let Some((batch, token, head, dim)) = rope_position(&params) else {
        return;
    };
    rotate_section(&mut qkv, batch, token, head, dim, 0, &params);
    rotate_section(
        &mut qkv,
        batch,
        token,
        head,
        dim,
        params.embedding_dim,
        &params,
    );
}

pub(super) fn apply_rope_save_f16_body(
    mut qkv: DisjointSlice<f32>,
    mut qkv_f16: DisjointSlice<u16>,
    params: ApplyRopeParams,
) {
    let Some((batch, token, head, dim)) = rope_position(&params) else {
        return;
    };
    let q = rotate_section(&mut qkv, batch, token, head, dim, 0, &params);
    let k = rotate_section(
        &mut qkv,
        batch,
        token,
        head,
        dim,
        params.embedding_dim,
        &params,
    );
    let v = read_pair(
        &mut qkv,
        batch,
        token,
        head,
        dim,
        params.embedding_dim * 2,
        &params,
    );

    store_pair_f16(&mut qkv_f16, q);
    store_pair_f16(&mut qkv_f16, k);
    store_pair_f16(&mut qkv_f16, v);
}
