pub fn set_e2m1_one(bytes: &mut [u8], element: usize) {
    let byte = &mut bytes[element / 2];
    if element & 1 == 0 {
        *byte = (*byte & 0xf0) | 0x2;
    } else {
        *byte = (*byte & 0x0f) | 0x20;
    }
}
