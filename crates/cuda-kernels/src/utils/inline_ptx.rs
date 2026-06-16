use cuda_device::ptx_asm;

#[inline(always)]
pub fn sqrt_f32(x: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!(
            "sqrt.rn.f32 %0, %1;",
            out("=f") y,
            in("f") x,
            options(register_only),
        );
    }
    y
}

#[inline(always)]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!(
            "fma.rn.f32 %0, %1, %2, %3;",
            out("=f") y,
            in("f") a,
            in("f") b,
            in("f") c,
            options(register_only),
        );
    }
    y
}

#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!(
            "abs.f32 %0, %1;",
            out("=f") y,
            in("f") x,
            options(register_only),
        );
    }
    y
}

#[inline(always)]
pub fn max_f32(a: f32, b: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!(
            "max.f32 %0, %1, %2;",
            out("=f") y,
            in("f") a,
            in("f") b,
            options(register_only),
        );
    }
    y
}
