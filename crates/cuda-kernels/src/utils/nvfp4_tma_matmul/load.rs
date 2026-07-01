pub const E4M3_ONE_PACKED4: u32 = 0x3838_3838;

/// Load 16 NVFP4 elements (8 bytes) as a packed u64.
/// Two e2m1 values per byte, little-endian.
#[inline(always)]
pub fn load_packed16(bytes: &[u8], element_base: usize) -> u64 {
    let byte_base = element_base / 2;
    if byte_base + 7 < bytes.len() {
        let lo = (bytes[byte_base] as u64)
            | ((bytes[byte_base + 1] as u64) << 8)
            | ((bytes[byte_base + 2] as u64) << 16)
            | ((bytes[byte_base + 3] as u64) << 24);
        let hi = (bytes[byte_base + 4] as u64)
            | ((bytes[byte_base + 5] as u64) << 8)
            | ((bytes[byte_base + 6] as u64) << 16)
            | ((bytes[byte_base + 7] as u64) << 24);
        lo | (hi << 32)
    } else {
        0
    }
}

/// Load 8 NVFP4 elements (4 bytes) as a packed u32.
/// Two e2m1 values per byte, little-endian.
#[inline(always)]
pub fn load_packed8(bytes: &[u8], element_base: usize) -> u32 {
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

/// Load a 4-element e4m3 scale pack as u32.
#[inline(always)]
pub fn load_scale4(scales: &[u8], scale_base: usize) -> u32 {
    if scale_base + 3 < scales.len() {
        (scales[scale_base] as u32)
            | ((scales[scale_base + 1] as u32) << 8)
            | ((scales[scale_base + 2] as u32) << 16)
            | ((scales[scale_base + 3] as u32) << 24)
    } else {
        E4M3_ONE_PACKED4
    }
}
