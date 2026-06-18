use cuda_device::DisjointSlice;

use super::super::super::threads::MATRIX_THREADS_PER_BLOCK;

#[inline(always)]
pub(super) fn frobenius_local_sum(x: &[f32], tid: u32, len: u32) -> f32 {
    let stride = MATRIX_THREADS_PER_BLOCK;
    let step = stride * 4;
    let mut local = 0.0;
    let mut index = tid;

    while index + 3 * stride < len {
        let v0 = x[index as usize];
        let v1 = x[(index + stride) as usize];
        let v2 = x[(index + 2 * stride) as usize];
        let v3 = x[(index + 3 * stride) as usize];
        local += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        index += step;
    }

    while index < len {
        let value = x[index as usize];
        local += value * value;
        index += stride;
    }

    local
}

#[inline(always)]
pub(super) fn frobenius_local_sum_disjoint(x: &mut DisjointSlice<f32>, tid: u32, len: u32) -> f32 {
    let stride = MATRIX_THREADS_PER_BLOCK;
    let step = stride * 4;
    let mut local = 0.0;
    let mut index = tid;

    while index + 3 * stride < len {
        unsafe {
            let v0 = *x.as_mut_ptr().add(index as usize);
            let v1 = *x.as_mut_ptr().add((index + stride) as usize);
            let v2 = *x.as_mut_ptr().add((index + 2 * stride) as usize);
            let v3 = *x.as_mut_ptr().add((index + 3 * stride) as usize);
            local += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        }
        index += step;
    }

    while index < len {
        unsafe {
            let value = *x.as_mut_ptr().add(index as usize);
            local += value * value;
        }
        index += stride;
    }

    local
}

#[inline(always)]
pub(super) fn store_normalized(
    x: &[f32],
    out: &mut DisjointSlice<f32>,
    inv_norm: f32,
    tid: u32,
    len: u32,
) {
    let stride = MATRIX_THREADS_PER_BLOCK;
    let step = stride * 4;
    let mut index = tid;

    while index + 3 * stride < len {
        unsafe {
            *out.get_unchecked_mut(index as usize) = x[index as usize] * inv_norm;
            *out.get_unchecked_mut((index + stride) as usize) =
                x[(index + stride) as usize] * inv_norm;
            *out.get_unchecked_mut((index + 2 * stride) as usize) =
                x[(index + 2 * stride) as usize] * inv_norm;
            *out.get_unchecked_mut((index + 3 * stride) as usize) =
                x[(index + 3 * stride) as usize] * inv_norm;
        }
        index += step;
    }

    while index < len {
        unsafe {
            *out.get_unchecked_mut(index as usize) = x[index as usize] * inv_norm;
        }
        index += stride;
    }
}

#[inline(always)]
pub(super) fn scale_normalized(x: &mut DisjointSlice<f32>, inv_norm: f32, tid: u32, len: u32) {
    let stride = MATRIX_THREADS_PER_BLOCK;
    let step = stride * 4;
    let mut index = tid;

    while index + 3 * stride < len {
        unsafe {
            *x.get_unchecked_mut(index as usize) *= inv_norm;
            *x.get_unchecked_mut((index + stride) as usize) *= inv_norm;
            *x.get_unchecked_mut((index + 2 * stride) as usize) *= inv_norm;
            *x.get_unchecked_mut((index + 3 * stride) as usize) *= inv_norm;
        }
        index += step;
    }

    while index < len {
        unsafe {
            *x.get_unchecked_mut(index as usize) *= inv_norm;
        }
        index += stride;
    }
}
