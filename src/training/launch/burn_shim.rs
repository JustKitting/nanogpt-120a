use burn::module::{EmptyRecord, Module, Param};
use burn::optim::{GradientsParams, LearningRate, MultiGradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::train::{InferenceStep, LearningComponentsMarker, TrainOutput, TrainStep};

use super::data_loader::CudaValidationInput;
use super::metrics::{CudaTrainOutput, CudaValidOutput};

pub(in crate::training) type BurnInnerBackend = burn::backend::NdArray;
pub(in crate::training) type BurnBackend = burn::backend::Autodiff<BurnInnerBackend>;
pub(in crate::training) type CudaBurnModel = CudaBurnModule<BurnBackend>;
pub(in crate::training) type CudaLearningComponents =
    LearningComponentsMarker<BurnBackend, LearningRate, CudaBurnModel, CudaNoopOptimizer>;
pub(super) type CudaTrainInput = Result<super::super::data::TokenWindowBatch, String>;
pub(super) type CudaValidInput = Result<CudaValidationInput, String>;

#[derive(Module, Debug)]
pub(in crate::training) struct CudaBurnModule<B: Backend> {
    marker: Param<Tensor<B, 1>>,
}

impl<B: Backend> CudaBurnModule<B> {
    pub(super) fn new(device: &B::Device) -> Self {
        Self {
            marker: Param::from_data([0.0_f32], device),
        }
    }
}

impl Default for CudaBurnModule<BurnBackend> {
    fn default() -> Self {
        Self::new(&Default::default())
    }
}

impl TrainStep for CudaBurnModule<BurnBackend> {
    type Input = CudaTrainInput;
    type Output = CudaTrainOutput;

    fn step(&self, _item: Self::Input) -> TrainOutput<Self::Output> {
        panic!("CudaBurnModel::step must not be called; CudaTrainingStrategy owns training")
    }
}

impl InferenceStep for CudaBurnModule<BurnInnerBackend> {
    type Input = CudaValidInput;
    type Output = CudaValidOutput;

    fn step(&self, _item: Self::Input) -> Self::Output {
        panic!("CudaBurnModel::step must not be called; CudaTrainingStrategy owns validation")
    }
}

#[derive(Clone, Debug, Default)]
pub(in crate::training) struct CudaNoopOptimizer;

impl Optimizer<CudaBurnModel, BurnBackend> for CudaNoopOptimizer {
    type Record = EmptyRecord;

    fn step(
        &mut self,
        _lr: LearningRate,
        module: CudaBurnModel,
        _grads: GradientsParams,
    ) -> CudaBurnModel {
        module
    }

    fn step_multi(
        &mut self,
        _lr: LearningRate,
        module: CudaBurnModel,
        _grads: MultiGradientsParams,
    ) -> CudaBurnModel {
        module
    }

    fn to_record(&self) -> Self::Record {
        EmptyRecord::new()
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}
