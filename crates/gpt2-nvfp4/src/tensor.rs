#[derive(Clone, Debug)]
pub struct Nvfp4Tensor {
    pub bytes: Vec<u8>,
    pub scales: Vec<u8>,
    pub global_scale: f32,
    pub shape: Vec<usize>,
}

impl Nvfp4Tensor {
    pub fn new(shape: Vec<usize>, bytes: Vec<u8>, scales: Vec<u8>, global_scale: f32) -> Self {
        Self {
            bytes,
            scales,
            global_scale,
            shape,
        }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
