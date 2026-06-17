use cuda_device::ptx_asm;

#[inline(always)]
pub unsafe fn atomic_add_f32(ptr: *mut f32, value: f32) {
    unsafe {
        ptx_asm!(
            "{ .reg .f32 old; atom.global.add.f32 old, [%0], %1; }",
            in("l") ptr as u64,
            in("f") value,
        );
    }
}
