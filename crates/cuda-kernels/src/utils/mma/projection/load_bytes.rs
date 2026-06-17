pub(super) const E4M3_ONE_PACKED4: u32 = 0x3838_3838;

#[inline(always)]
pub(super) fn load_packed8(bytes: &[u8], element_base: usize) -> u32 {
    let byte_base = element_base / 2;
    if byte_base + 3 < bytes.len() {
        (bytes[byte_base] as u32)
            | ((bytes[byte_base + 1] as u32) << 8)
            | ((bytes[byte_base + 2] as u32) << 16)
            | ((bytes[byte_base + 3] as u32) << 24)
    } else {
        0
    }
}

#[inline(always)]
pub(super) fn load_scale4(scales: &[u8], scale_base: usize) -> u32 {
    if scale_base + 3 < scales.len() {
        (scales[scale_base] as u32)
            | ((scales[scale_base + 1] as u32) << 8)
            | ((scales[scale_base + 2] as u32) << 16)
            | ((scales[scale_base + 3] as u32) << 24)
    } else {
        E4M3_ONE_PACKED4
    }
}
