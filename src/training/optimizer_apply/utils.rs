use std::time::Instant;

pub(super) fn timed_ms<E>(f: impl FnOnce() -> Result<(), E>) -> Result<f64, E> {
    let start = Instant::now();
    f()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}
