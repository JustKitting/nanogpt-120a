use std::fmt;
use std::marker::PhantomData;

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

impl<const N: usize> Clone for FixedBytes<N> {
    fn clone(&self) -> Self {
        let mut bytes = Self::zeroed();
        bytes.as_mut().copy_from_slice(self.as_ref());
        bytes
    }
}

impl<const N: usize> fmt::Debug for FixedBytes<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedBytes")
            .field("len", &N)
            .finish_non_exhaustive()
    }
}

pub trait Nvfp4Shape {
    const ROWS: usize;
    const COLS: usize;
    const BYTE_LEN: usize;
    const SCALE_LEN: usize;

    type Bytes: AsRef<[u8]> + AsMut<[u8]> + Clone + fmt::Debug;
    type Scales: AsRef<[u8]> + AsMut<[u8]> + Clone + fmt::Debug;

    fn zero_bytes() -> Self::Bytes;
    fn zero_scales() -> Self::Scales;
}

pub struct Nvfp4Tensor<S: Nvfp4Shape> {
    pub bytes: S::Bytes,
    pub scales: S::Scales,
    pub global_scale: f32,
    shape: PhantomData<S>,
}

impl<S: Nvfp4Shape> Nvfp4Tensor<S> {
    pub const ROWS: usize = S::ROWS;
    pub const COLS: usize = S::COLS;
    pub const LEN: usize = S::ROWS * S::COLS;
    pub const BYTE_LEN: usize = S::BYTE_LEN;
    pub const SCALE_LEN: usize = S::SCALE_LEN;

    pub fn new(bytes: S::Bytes, scales: S::Scales, global_scale: f32) -> Self {
        Self {
            bytes,
            scales,
            global_scale,
            shape: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        Self::LEN
    }

    pub fn is_empty(&self) -> bool {
        Self::LEN == 0
    }
}

impl<S: Nvfp4Shape> Clone for Nvfp4Tensor<S> {
    fn clone(&self) -> Self {
        Self::new(self.bytes.clone(), self.scales.clone(), self.global_scale)
    }
}

impl<S: Nvfp4Shape> fmt::Debug for Nvfp4Tensor<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Nvfp4Tensor")
            .field("rows", &S::ROWS)
            .field("cols", &S::COLS)
            .field("byte_len", &S::BYTE_LEN)
            .field("scale_len", &S::SCALE_LEN)
            .field("global_scale", &self.global_scale)
            .finish_non_exhaustive()
    }
}
