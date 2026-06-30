#[path = "backward/grads.rs"]
mod grads;
#[path = "backward/saved.rs"]
mod saved;

pub use grads::{BlockBackwardGrads, Gpt2BackwardGrads, LayerNormGrads};
pub use saved::{BlockForwardSaved, Gpt2ForwardSaved, LayerNormSaved};

pub struct Gpt2BackwardContext<'a> {
    pub saved: Gpt2ForwardSaved<'a>,
    pub grads: Gpt2BackwardGrads<'a>,
}
