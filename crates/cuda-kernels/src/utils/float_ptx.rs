use cuda_device::ptx_asm;

const POSITIVE_DENOM_EPS: f32 = 1.0e-20;

#[inline(always)]
pub fn safe_positive_denom(x: f32) -> f32 {
    x + POSITIVE_DENOM_EPS
}

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
pub fn exp_f32(x: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!(
            "ex2.approx.ftz.f32 %0, %1;",
            out("=f") y,
            in("f") x * core::f32::consts::LOG2_E,
            options(register_only),
        );
    }
    y
}

#[inline(always)]
pub fn ln_f32(x: f32) -> f32 {
    const LN_2: f32 = core::f32::consts::LN_2;
    let y: f32;
    unsafe {
        ptx_asm!(
            "lg2.approx.ftz.f32 %0, %1;",
            out("=f") y,
            in("f") x,
            options(register_only),
        );
    }
    y * LN_2
}

#[inline(always)]
pub fn sincos_f32(x: f32) -> (f32, f32) {
    let x = reduce_angle_f32(x);
    (sin_reduced_f32(x), cos_reduced_f32(x))
}

#[inline(always)]
fn sin_reduced_f32(x: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!(
            "sin.approx.ftz.f32 %0, %1;",
            out("=f") y,
            in("f") x,
            options(register_only),
        );
    }
    y
}

#[inline(always)]
fn cos_reduced_f32(x: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!(
            "cos.approx.ftz.f32 %0, %1;",
            out("=f") y,
            in("f") x,
            options(register_only),
        );
    }
    y
}

#[inline(always)]
fn reduce_angle_f32(x: f32) -> f32 {
    const INV_TAU: f32 = 0.159_154_94;
    const NEG_TAU: f32 = -6.283_185_5;
    let y: f32;
    unsafe {
        ptx_asm!(
            "{ .reg .u32 n; .reg .f32 scaled, nearest; mul.rn.f32 scaled, %1, %2; cvt.rni.u32.f32 n, scaled; cvt.rn.f32.u32 nearest, n; fma.rn.f32 %0, nearest, %3, %1; }",
            out("=f") y,
            in("f") x,
            in("f") INV_TAU,
            in("f") NEG_TAU,
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
