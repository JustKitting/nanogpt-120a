pub fn set_e2m1_one(bytes: &mut [u8], element: usize) {
    let byte = &mut bytes[element / 2];
    if element & 1 == 0 {
        *byte = (*byte & 0xf0) | 0x2;
    } else {
        *byte = (*byte & 0x0f) | 0x20;
    }
}

pub fn repeating_identity_bytes(byte_len: usize, cols: usize, row_len: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; byte_len];
    for col in 0..cols {
        set_e2m1_one(&mut bytes, col * row_len + col % row_len);
    }
    bytes
}
