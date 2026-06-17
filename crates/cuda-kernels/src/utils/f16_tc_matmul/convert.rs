use cuda_device::{DisjointSlice, ptx_asm, thread};

use super::kernels::F16_THREADS_PER_BLOCK;

pub(super) fn fp32_to_f16_body(src: &[f32], mut dst: DisjointSlice<u16>, element_count: u32) {
    let index = thread::blockIdx_x() * F16_THREADS_PER_BLOCK + thread::threadIdx_x();
    if index < element_count {
        unsafe {
            *dst.get_unchecked_mut(index as usize) = cvt_rn_f16_f32(src[index as usize]);
        }
    }
}

#[inline(always)]
fn cvt_rn_f16_f32(value: f32) -> u16 {
    let half: u16;
    unsafe {
        ptx_asm!(
            "cvt.rn.f16.f32 %0, %1;",
            out("=h") half,
            in("f") value,
            options(register_only),
        );
    }
    half
}
