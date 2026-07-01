use cuda_device::DisjointSlice;

use super::KdaIntraCtx;

pub(super) fn reverse_prefix_dg(
    k_a_to_dg: &mut DisjointSlice<f32>,
    ctx: KdaIntraCtx<'_>,
    tid: u32,
) {
    let dim = tid;
    if dim < ctx.head_dim {
        let mut acc = 0.0;
        let mut token = ctx.end;
        while token > ctx.start {
            token -= 1;
            let compact = ctx.compact(token, dim);
            acc += unsafe { *k_a_to_dg.get_unchecked_mut(compact) };
            unsafe {
                *k_a_to_dg.get_unchecked_mut(compact) = acc;
            }
        }
    }
}
