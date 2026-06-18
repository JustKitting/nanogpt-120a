use gpt2_bpe::Gpt2Bpe;

use super::AppResult;
use super::shards::ShardWriter;

pub fn tokenize_doc(text: &str, tokenizer: &Gpt2Bpe, writer: &mut ShardWriter) -> AppResult<()> {
    let mut ids = Vec::with_capacity(1 + text.len() / 4);
    ids.push(tokenizer.eot_token());
    ids.extend(tokenizer.encode_ordinary(text)?);

    for id in ids {
        let token = u16::try_from(id)?;
        writer.push(token)?;
    }

    Ok(())
}
