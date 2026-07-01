use std::error::Error;
use std::io;
use std::path::Path;

use tokenizers::Tokenizer;

pub type AppResult<T> = Result<T, Box<dyn Error>>;

pub const TOKENIZER_NAME: &str = "llama2";
pub const VOCAB_SIZE: usize = 32_000;
pub const BOS_TOKEN: u32 = 1;
pub const EOS_TOKEN: u32 = 2;

pub struct Llama2Tokenizer {
    tokenizer: Tokenizer,
}

impl Llama2Tokenizer {
    pub fn from_default_assets() -> AppResult<Self> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/tokenizers/llama2");
        Self::from_file(root.join("tokenizer.json"))
    }

    pub fn from_file(path: impl AsRef<Path>) -> AppResult<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(tokenizer_error)?;
        if tokenizer.get_vocab_size(false) != VOCAB_SIZE {
            return Err(format!(
                "Llama 2 tokenizer vocab size is {}, expected {VOCAB_SIZE}",
                tokenizer.get_vocab_size(false)
            )
            .into());
        }
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> AppResult<Vec<u32>> {
        self.encode_with_special_tokens(text, true)
    }

    pub fn encode_ordinary(&self, text: &str) -> AppResult<Vec<u32>> {
        self.encode_with_special_tokens(text, false)
    }

    fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> AppResult<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(tokenizer_error)?
            .get_ids()
            .to_vec())
    }

    pub fn bos_token(&self) -> u32 {
        BOS_TOKEN
    }

    pub fn eos_token(&self) -> u32 {
        EOS_TOKEN
    }

    pub fn decode(&self, ids: &[u32]) -> AppResult<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(tokenizer_error)
            .map_err(Into::into)
    }
}

fn tokenizer_error(error: Box<dyn Error + Send + Sync>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_assets_encode_like_llama2() -> AppResult<()> {
        let tokenizer = Llama2Tokenizer::from_default_assets()?;
        let text = "First Citizen:\nBefore we proceed any further, hear me speak.";
        assert_eq!(
            tokenizer.encode_ordinary(text)?,
            [
                3824, 21353, 19642, 29901, 13, 18743, 591, 8469, 738, 4340, 29892, 8293, 592, 7726,
                29889,
            ]
        );
        assert_eq!(tokenizer.encode(text)?[0], BOS_TOKEN);
        assert_eq!(tokenizer.bos_token(), BOS_TOKEN);
        assert_eq!(tokenizer.eos_token(), EOS_TOKEN);
        Ok(())
    }
}
