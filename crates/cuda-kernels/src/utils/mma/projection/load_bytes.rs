use cuda_device::ptx_asm;

pub(crate) const E4M3_ONE_PACKED4: u32 = 0x3838_3838;

#[inline(always)]
pub(crate) fn load_packed8(bytes: &[u8], element_base: usize) -> u32 {
    load_u32_or(bytes, element_base / 2, 0)
}

#[inline(always)]
pub(crate) fn load_scale4(scales: &[u8], scale_base: usize) -> u32 {
    load_u32_or(scales, scale_base, E4M3_ONE_PACKED4)
}

#[inline(always)]
pub(crate) fn load_packed8_aligned(bytes: &[u8], element_base: usize) -> u32 {
    load_u32_aligned(bytes, element_base / 2)
}

#[inline(always)]
pub(crate) fn load_scale4_aligned(scales: &[u8], scale_base: usize) -> u32 {
    load_u32_aligned(scales, scale_base)
}

#[inline(always)]
fn load_u32_or(bytes: &[u8], byte_base: usize, fallback: u32) -> u32 {
    if byte_base + 3 < bytes.len() {
        (bytes[byte_base] as u32)
            | ((bytes[byte_base + 1] as u32) << 8)
            | ((bytes[byte_base + 2] as u32) << 16)
            | ((bytes[byte_base + 3] as u32) << 24)
    } else {
        fallback
    }
}

#[inline(always)]
fn load_u32_aligned(bytes: &[u8], byte_base: usize) -> u32 {
    let value: u32;
    let ptr = unsafe { bytes.as_ptr().add(byte_base) };
    unsafe {
        ptx_asm!(
            "ld.global.u32 %0, [%1];",
            out("=r") value,
            in("l") ptr as u64,
            options(register_only),
        );
    }
    value
}
