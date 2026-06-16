use std::error::Error;

mod byte_unicode;
mod tokenizer;
mod vocab;

pub type AppResult<T> = Result<T, Box<dyn Error>>;

pub use tokenizer::Gpt2Bpe;
