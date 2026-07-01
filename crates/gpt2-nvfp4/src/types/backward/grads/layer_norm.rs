use cuda_core::DeviceBuffer;

pub struct LayerNormGrads<'a> {
    pub d_residual: &'a mut DeviceBuffer<f32>,
    pub d_normalized: &'a mut DeviceBuffer<f32>,
    pub d_weight: &'a mut DeviceBuffer<f32>,
    pub d_bias: &'a mut DeviceBuffer<f32>,
}

impl<'a> LayerNormGrads<'a> {
    pub fn reborrow(&mut self) -> LayerNormGrads<'_> {
        LayerNormGrads {
            d_residual: &mut *self.d_residual,
            d_normalized: &mut *self.d_normalized,
            d_weight: &mut *self.d_weight,
            d_bias: &mut *self.d_bias,
        }
    }

    pub fn reborrow_with_residual<'b>(
        &'b mut self,
        d_residual: &'b mut DeviceBuffer<f32>,
    ) -> LayerNormGrads<'b> {
        LayerNormGrads {
            d_residual,
            d_normalized: &mut *self.d_normalized,
            d_weight: &mut *self.d_weight,
            d_bias: &mut *self.d_bias,
        }
    }
}
