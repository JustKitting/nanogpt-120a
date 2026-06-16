use cuda_device::ptx_asm;

#[inline(always)]
pub fn e2m1_value(bits: u8) -> f32 {
    let value: f32;
    let packed = bits as u16;

    unsafe {
        ptx_asm!(
            "{ .reg .b8 e2; .reg .b32 h2; .reg .b16 lo; cvt.u8.u16 e2, %1; cvt.rn.f16x2.e2m1x2 h2, e2; cvt.u16.u32 lo, h2; cvt.f32.f16 %0, lo; }",
            out("=f") value,
            in("h") packed,
            options(register_only),
        );
    }
    value
}

#[inline(always)]
pub fn e4m3_value(bits: u16) -> f32 {
    let value: f32;

    unsafe {
        ptx_asm!(
            "{ .reg .b32 h2; .reg .b16 lo; cvt.rn.f16x2.e4m3x2 h2, %1; cvt.u16.u32 lo, h2; cvt.f32.f16 %0, lo; }",
            out("=f") value,
            in("h") bits,
            options(register_only),
        );
    }
    value
}
