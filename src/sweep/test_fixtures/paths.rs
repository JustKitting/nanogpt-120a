use std::path::PathBuf;

pub(in crate::sweep) fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "{}-{}-{name}",
        std::process::id(),
        std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
    ))
}
