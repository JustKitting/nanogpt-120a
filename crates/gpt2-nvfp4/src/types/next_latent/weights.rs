use crate::Nvfp4Tensor;
use crate::random::InitRng;
use crate::types::{
    HiddenVectorShape, LinearWeights, NextLatHiddenShape, NextLatInputShape, NextLatOutWeightShape,
    NextLatProjectionWeightShape, NextLatTransitionWeightShape, Nvfp4ShapeInit,
};

#[derive(Clone, Debug)]
pub struct NextLatWeights {
    pub norm_weight: Nvfp4Tensor<NextLatInputShape>,
    pub norm_bias: Nvfp4Tensor<NextLatInputShape>,
    pub input_projection: LinearWeights<NextLatProjectionWeightShape, NextLatHiddenShape>,
    pub transition: LinearWeights<NextLatTransitionWeightShape, NextLatHiddenShape>,
    pub output_projection: LinearWeights<NextLatOutWeightShape, HiddenVectorShape>,
}

impl NextLatWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        Self {
            norm_weight: NextLatInputShape::one_tensor(),
            norm_bias: NextLatInputShape::zero_tensor(),
            input_projection: LinearWeights::init(rng),
            transition: LinearWeights::init(rng),
            output_projection: LinearWeights::init(rng),
        }
    }
}
