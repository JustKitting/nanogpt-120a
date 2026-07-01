use std::fmt;

#[derive(Clone)]
pub struct FixedBytes<const N: usize>(Box<[u8; N]>);

impl<const N: usize> FixedBytes<N> {
    pub fn zeroed() -> Self {
        let bytes = Box::<[u8]>::new_zeroed_slice(N);
        // SAFETY: every zeroed byte pattern is valid for u8.
        let bytes = unsafe { bytes.assume_init() };
        let bytes = bytes
            .try_into()
            .unwrap_or_else(|_| unreachable!("boxed slice length must match const size"));
        Self(bytes)
    }
}

impl<const N: usize> AsRef<[u8]> for FixedBytes<N> {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl<const N: usize> AsMut<[u8]> for FixedBytes<N> {
    fn as_mut(&mut self) -> &mut [u8] {
        self.0.as_mut()
    }
}

impl<const N: usize> fmt::Debug for FixedBytes<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedBytes")
            .field("len", &N)
            .finish_non_exhaustive()
    }
}
