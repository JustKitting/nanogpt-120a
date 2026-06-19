#[derive(Clone, Copy, Debug, Default)]
pub struct RunResult {
    pub val_loss: Option<f64>,
    pub completed_steps: Option<usize>,
    pub saw_nan: bool,
}

impl RunResult {
    pub fn update(&mut self, line: &str) {
        if line.starts_with("step=") {
            self.completed_steps = field(line, "step=")
                .and_then(parse_usize)
                .map(|step| step + 1);
        }
        if line.contains("loss=NaN") || line.contains("finite=false") {
            self.saw_nan = true;
        }
        if line.starts_with("stopped_by_wall_clock=true") {
            self.completed_steps = field(line, "completed_steps=").and_then(parse_usize);
        }
        if line.starts_with("heldout_eval ") {
            self.val_loss = field(line, "val_loss=").and_then(parse_f64);
            self.completed_steps = field(line, "completed_steps=").and_then(parse_usize);
        }
    }
}

fn field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let start = line.find(prefix)? + prefix.len();
    let value = &line[start..];
    Some(value.split_whitespace().next().unwrap_or(value))
}

fn parse_f64(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn parse_usize(value: &str) -> Option<usize> {
    value.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::RunResult;

    #[test]
    fn detects_nan_step() {
        let mut result = RunResult::default();
        result.update("step=1250 loss=NaN finite=false");
        assert!(result.saw_nan);
        assert_eq!(result.completed_steps, Some(1251));
        assert_eq!(result.val_loss, None);
    }
}
