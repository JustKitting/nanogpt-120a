use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;

use fancy_regex::Regex;

pub type AppResult<T> = Result<T, Box<dyn Error>>;

const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const EOT: &str = "<|endoftext|>";
const EOT_TOKEN: u32 = 50256;

pub struct Gpt2Bpe {
    encoder: HashMap<String, u32>,
    decoder: Vec<String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    pattern: Regex,
    cache: RefCell<HashMap<String, Vec<u32>>>,
}

impl Gpt2Bpe {
    pub fn from_default_assets() -> AppResult<Self> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/tokenizers/gpt2");
        Self::from_files(root.join("encoder.json"), root.join("vocab.bpe"))
    }

    pub fn from_files(
        encoder_json_path: impl AsRef<Path>,
        vocab_bpe_path: impl AsRef<Path>,
    ) -> AppResult<Self> {
        let encoder_json = fs::read_to_string(encoder_json_path)?;
        let encoder: HashMap<String, u32> = serde_json::from_str(&encoder_json)?;
        let decoder = build_decoder(&encoder)?;
        let bpe_ranks = load_bpe_ranks(vocab_bpe_path)?;
        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            pattern: Regex::new(GPT2_PATTERN)?,
            cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn encode(&self, text: &str) -> AppResult<Vec<u32>> {
        let mut ids = Vec::new();
        let mut rest = text;

        while let Some(index) = rest.find(EOT) {
            self.encode_ordinary_into(&rest[..index], &mut ids)?;
            ids.push(EOT_TOKEN);
            rest = &rest[index + EOT.len()..];
        }

        self.encode_ordinary_into(rest, &mut ids)?;
        Ok(ids)
    }

    pub fn encode_ordinary(&self, text: &str) -> AppResult<Vec<u32>> {
        let mut ids = Vec::new();
        self.encode_ordinary_into(text, &mut ids)?;
        Ok(ids)
    }

    pub fn eot_token(&self) -> u32 {
        EOT_TOKEN
    }

    pub fn decode(&self, ids: &[u32]) -> AppResult<String> {
        let mut bytes = Vec::new();

        for &id in ids {
            if id == EOT_TOKEN {
                bytes.extend_from_slice(EOT.as_bytes());
                continue;
            }

            let token = self.decoder.get(id as usize).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("token id {id} is outside GPT-2 vocab"),
                )
            })?;

            for ch in token.chars() {
                let byte = self.byte_decoder.get(&ch).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("token contains non GPT-2 byte-mapped char {ch:?}"),
                    )
                })?;
                bytes.push(*byte);
            }
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn encode_ordinary_into(&self, text: &str, ids: &mut Vec<u32>) -> AppResult<()> {
        for piece in self.pattern.find_iter(text) {
            let piece = piece?;
            let token = self.byte_encode(piece.as_str());
            ids.extend(self.bpe(&token)?);
        }
        Ok(())
    }

    fn byte_encode(&self, text: &str) -> String {
        text.as_bytes()
            .iter()
            .map(|byte| self.byte_encoder[*byte as usize])
            .collect()
    }

    fn bpe(&self, token: &str) -> AppResult<Vec<u32>> {
        if let Some(ids) = self.cache.borrow().get(token) {
            return Ok(ids.clone());
        }

        let mut word = token.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();

        while word.len() > 1 {
            let mut best_index = None;
            let mut best_rank = usize::MAX;

            for index in 0..word.len() - 1 {
                let pair = (word[index].clone(), word[index + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = Some(index);
                    }
                }
            }

            let Some(index) = best_index else {
                break;
            };

            let merged = format!("{}{}", word[index], word[index + 1]);
            word[index] = merged;
            word.remove(index + 1);
        }

        let mut ids = Vec::with_capacity(word.len());
        for piece in word {
            let id = self.encoder.get(&piece).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("BPE piece {piece:?} is missing from encoder.json"),
                )
            })?;
            ids.push(*id);
        }

        self.cache
            .borrow_mut()
            .insert(token.to_owned(), ids.clone());
        Ok(ids)
    }
}

pub fn run_default() -> AppResult<()> {
    let tokenizer = Gpt2Bpe::from_default_assets()?;
    let text = "Hello, world!<|endoftext|>";
    let ids = tokenizer.encode(text)?;
    let decoded = tokenizer.decode(&ids)?;

    println!("gpt2_bpe ids={ids:?} decoded={decoded:?}");
    Ok(())
}

fn load_bpe_ranks(path: impl AsRef<Path>) -> AppResult<HashMap<(String, String), usize>> {
    let vocab = fs::read_to_string(path)?;
    let mut ranks = HashMap::new();

    for (line_index, line) in vocab.lines().enumerate() {
        if line_index == 0 && line.starts_with("#version:") {
            continue;
        }

        let Some((left, right)) = line.split_once(' ') else {
            continue;
        };
        ranks.insert((left.to_owned(), right.to_owned()), ranks.len());
    }

    Ok(ranks)
}

fn build_decoder(encoder: &HashMap<String, u32>) -> AppResult<Vec<String>> {
    let max_id = encoder.values().copied().max().unwrap_or(0) as usize;
    let mut decoder = vec![String::new(); max_id + 1];

    for (token, id) in encoder {
        decoder[*id as usize] = token.clone();
    }

    Ok(decoder)
}

fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xa1..=0xac);
    bs.extend(0xae..=0xff);

    let mut cs = bs.iter().map(|byte| *byte as u32).collect::<Vec<_>>();
    let mut n = 0u32;
    for byte in 0u8..=255 {
        if !bs.contains(&byte) {
            bs.push(byte);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut byte_encoder = ['\0'; 256];
    let mut byte_decoder = HashMap::new();
    for (byte, codepoint) in bs.into_iter().zip(cs.into_iter()) {
        let ch = char::from_u32(codepoint).expect("GPT-2 byte unicode codepoint is valid");
        byte_encoder[byte as usize] = ch;
        byte_decoder.insert(ch, byte);
    }

    (byte_encoder, byte_decoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_known_gpt2_text() {
        let tokenizer = Gpt2Bpe::from_default_assets().unwrap();
        assert_eq!(
            tokenizer.encode("Hello, world!").unwrap(),
            vec![15496, 11, 995, 0]
        );
    }

    #[test]
    fn roundtrips_text_and_eot() {
        let tokenizer = Gpt2Bpe::from_default_assets().unwrap();
        let text = "Hello, world!<|endoftext|> tabs\tand unicode: Δ";
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
        assert!(ids.contains(&EOT_TOKEN));
    }
}
