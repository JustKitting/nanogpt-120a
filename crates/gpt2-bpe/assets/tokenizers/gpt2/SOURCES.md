# GPT-2 Tokenizer Files

Downloaded from OpenAI's public GPT-2 tokenizer artifacts:

- `encoder.json`
  - source: `https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json`
  - sha256: `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783`

- `vocab.bpe`
  - source: `https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe`
  - sha256: `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`

These hashes match the GPT-2 constructor in OpenAI `tiktoken`:

`https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py`
