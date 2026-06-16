use std::fs::File;
use std::path::Path;

use arrow_array::{Array, LargeStringArray, StringArray, StringArrayType};
use gpt2_bpe::Gpt2Bpe;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::AppResult;
use super::shards::ShardWriter;
use super::tokenize::tokenize_doc;

pub fn tokenize_parquet_file(
    path: &Path,
    tokenizer: &Gpt2Bpe,
    writer: &mut ShardWriter,
    docs_seen: &mut usize,
) -> AppResult<()> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(1024)
        .build()?;

    for batch in reader {
        let batch = batch?;
        let text_column = batch
            .column_by_name("text")
            .ok_or("FineWeb parquet batch missing text column")?;

        if let Some(array) = text_column.as_any().downcast_ref::<StringArray>() {
            tokenize_text_array(array, tokenizer, writer, docs_seen)?;
        } else if let Some(array) = text_column.as_any().downcast_ref::<LargeStringArray>() {
            tokenize_text_array(array, tokenizer, writer, docs_seen)?;
        } else {
            return Err("FineWeb text column is not utf8 or large_utf8".into());
        }
    }

    Ok(())
}

fn tokenize_text_array<'a>(
    array: impl StringArrayType<'a>,
    tokenizer: &Gpt2Bpe,
    writer: &mut ShardWriter,
    docs_seen: &mut usize,
) -> AppResult<()> {
    for text in array.iter().flatten() {
        tokenize_doc(text, tokenizer, writer)?;
        *docs_seen += 1;
    }
    Ok(())
}
