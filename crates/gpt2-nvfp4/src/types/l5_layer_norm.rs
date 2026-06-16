use cuda_core::DriverError;

use super::{HiddenStateDevice, HiddenVectorShape, LayerNormTensor, Nvfp4ShapeInit};

pub struct LayerNormForwardArgs<'a> {
    pub hidden: HiddenStateDevice<'a>,
}

#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    pub weight: LayerNormTensor,
    pub bias: LayerNormTensor,
}

impl LayerNormWeights {
    pub(crate) fn init() -> Self {
        Self {
            weight: HiddenVectorShape::one_tensor(),
            bias: HiddenVectorShape::zero_tensor(),
        }
    }

    pub fn input_from_block<'a>(hidden: HiddenStateDevice<'a>) -> LayerNormForwardArgs<'a> {
        LayerNormForwardArgs { hidden }
    }

    pub fn forward<'a>(
        &self,
        args: LayerNormForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        Ok(args.hidden)
    }
}
