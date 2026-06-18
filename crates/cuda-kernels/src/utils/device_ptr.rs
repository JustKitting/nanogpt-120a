#[inline(always)]
pub(crate) fn read_f32(ptr: *const f32, index: u32) -> f32 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
pub(crate) fn write_f32(ptr: *mut f32, index: u32, value: f32) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}
