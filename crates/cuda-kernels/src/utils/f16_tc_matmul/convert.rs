use cuda_device::{DisjointSlice, convert::cvt_f16x2_f32, ptx_asm, thread};

use super::kernels::F16_THREADS_PER_BLOCK;

pub(super) fn fp32_to_f16_body(src: &[f32], mut dst: DisjointSlice<u16>, element_count: u32) {
    let pair = (thread::blockIdx_x() * F16_THREADS_PER_BLOCK + thread::threadIdx_x()) * 2;
    if pair + 1 < element_count {
        let packed = cvt_f16x2_f32(src[pair as usize], src[pair as usize + 1]);
        unsafe {
            *dst.get_unchecked_mut(pair as usize) = (packed & 0xffff) as u16;
            *dst.get_unchecked_mut(pair as usize + 1) = (packed >> 16) as u16;
        }
    } else if pair < element_count {
        unsafe {
            *dst.get_unchecked_mut(pair as usize) = cvt_rn_f16_f32(src[pair as usize]);
        }
    }
}

#[inline(always)]
pub(crate) fn cvt_rn_f16_f32(value: f32) -> u16 {
    (cvt_f16x2_f32(value, 0.0) & 0xffff) as u16
}

#[inline(always)]
pub(crate) fn cvt_f32_f16(bits: u16) -> f32 {
    let value: f32;
    unsafe {
        ptx_asm!(
            "cvt.f32.f16 %0, %1;",
            out("=f") value,
            in("h") bits,
            options(register_only),
        );
    }
    value
}
