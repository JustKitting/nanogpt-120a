#[derive(Clone, Copy, Debug, Default)]
pub struct RunResult {
    pub val_loss: Option<f64>,
    pub completed_steps: Option<usize>,
    pub last_step: Option<usize>,
    pub last_elapsed_s: Option<f64>,
    pub last_train_loss: Option<f64>,
    pub saw_nan: bool,
}

impl RunResult {
    pub fn update(&mut self, line: &str) {
        if line.starts_with("step=") {
            if let Some(step) = field(line, "step=").and_then(parse_usize) {
                self.last_step = Some(step);
                self.completed_steps = Some(step + 1);
            }
            self.last_elapsed_s = field(line, "elapsed_s=").and_then(parse_f64);
            self.last_train_loss = field(line, "loss=").and_then(parse_f64);
        }
        if line.starts_with("TrainingProgress ") {
            if let Some(step) = debug_field(line, "iteration: Some(").and_then(parse_usize) {
                self.last_step = Some(step);
            }
            if let Some(steps) = debug_field(line, "global_progress: Progress { items_processed: ")
                .and_then(parse_usize)
            {
                self.completed_steps = Some(steps);
            }
        }
        if line.contains("loss=NaN") || line.contains("finite=false") {
            self.saw_nan = true;
        }
        if line.starts_with("stopped_by_wall_clock=true") {
            self.completed_steps = field(line, "completed_steps=").and_then(parse_usize);
            self.last_elapsed_s = field(line, "elapsed_s=").and_then(parse_f64);
        }
        if line.starts_with("heldout_eval ") {
            self.val_loss = field(line, "val_loss=").and_then(parse_f64);
            self.completed_steps = field(line, "completed_steps=").and_then(parse_usize);
            self.last_elapsed_s = field(line, "train_elapsed_s=").and_then(parse_f64);
        }
    }
}

pub(super) fn field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let start = line.find(prefix)? + prefix.len();
    let value = &line[start..];
    Some(value.split_whitespace().next().unwrap_or(value))
}

fn debug_field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let start = line.find(prefix)? + prefix.len();
    let value = &line[start..];
    Some(value.split([',', ')', '}']).next().unwrap_or(value).trim())
}

pub(super) fn parse_f64(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|value| value.is_finite())
}

pub(super) fn parse_usize(value: &str) -> Option<usize> {
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

    #[test]
    fn parses_final_heldout_elapsed_time() {
        let mut result = RunResult::default();
        result.update(
            "heldout_eval split=val val_loss=4.125000 train_elapsed_s=900.250 completed_steps=4096",
        );

        assert_eq!(result.val_loss, Some(4.125));
        assert_eq!(result.completed_steps, Some(4096));
        assert_eq!(result.last_elapsed_s, Some(900.25));
    }

    #[test]
    fn parses_burn_cli_progress() {
        let mut result = RunResult::default();
        result.update(
            "TrainingProgress { progress: Some(Progress { items_processed: 11, items_total: 100 }), global_progress: Progress { items_processed: 11, items_total: 100 }, iteration: Some(10) }",
        );

        assert_eq!(result.last_step, Some(10));
        assert_eq!(result.completed_steps, Some(11));
    }
}
