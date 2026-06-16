use std::collections::HashMap;

pub(crate) fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xa1..=0xac);
    bs.extend(0xae..=0xff);

    let mut cs = bs.iter().map(|byte| *byte as u32).collect::<Vec<_>>();
    let mut n = 0u32;
    for byte in 0u8..=255 {
        if !bs.contains(&byte) {
            bs.push(byte);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut byte_encoder = ['\0'; 256];
    let mut byte_decoder = HashMap::new();
    for (byte, codepoint) in bs.into_iter().zip(cs) {
        let ch = char::from_u32(codepoint).expect("GPT-2 byte unicode codepoint is valid");
        byte_encoder[byte as usize] = ch;
        byte_decoder.insert(ch, byte);
    }

    (byte_encoder, byte_decoder)
}
