use std::{fs, io, path::Path};

use crate::AppResult;

pub(super) fn read_u16_tokens(path: &Path) -> AppResult<Vec<u16>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(format!("{} has odd byte length", path.display()).into());
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect())
}

pub(super) fn write_u16_tokens(path: &Path, tokens: &[u16]) -> AppResult<()> {
    let bytes = tokens
        .iter()
        .flat_map(|&token| token.to_ne_bytes())
        .collect::<Vec<_>>();
    fs::write(path, bytes).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("failed to write {}: {err}", path.display()),
        )
        .into()
    })
}
