#[derive(Clone, Debug)]
pub struct Gpt2Rng {
    state: u64,
}

impl Gpt2Rng {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub(crate) fn next_u8(&mut self) -> u8 {
        self.next_u64() as u8
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

pub(crate) type InitRng = Gpt2Rng;
