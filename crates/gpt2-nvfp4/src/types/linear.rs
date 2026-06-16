use crate::{Nvfp4Shape, Nvfp4Tensor};

#[derive(Clone, Debug)]
pub struct LinearWeights<W: Nvfp4Shape, B: Nvfp4Shape> {
    pub weight: Nvfp4Tensor<W>,
    pub bias: Nvfp4Tensor<B>,
}
