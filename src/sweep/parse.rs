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
            if let Some(step) = usize_field(line, "step=") {
                self.last_step = Some(step);
                self.completed_steps = Some(step + 1);
            }
            self.last_elapsed_s = f64_field(line, "elapsed_s=");
            self.last_train_loss = f64_field(line, "loss=");
        }
        if line.starts_with("TrainingProgress ") {
            if let Some(step) = debug_usize(line, "iteration: Some(") {
                self.last_step = Some(step);
            }
            if let Some(steps) = debug_usize(line, "global_progress: Progress { items_processed: ")
            {
                self.completed_steps = Some(steps);
            }
        }
        self.saw_nan |= line.contains("loss=NaN") || line.contains("finite=false");
        if line.starts_with("stopped_by_wall_clock=true") {
            self.completed_steps = usize_field(line, "completed_steps=");
            self.last_elapsed_s = f64_field(line, "elapsed_s=");
        }
        if line.starts_with("heldout_eval ") {
            self.val_loss = f64_field(line, "val_loss=");
            self.completed_steps = usize_field(line, "completed_steps=");
            self.last_elapsed_s = f64_field(line, "train_elapsed_s=");
        }
    }
}

fn field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let value = &line[line.find(prefix)? + prefix.len()..];
    Some(value.split_whitespace().next().unwrap_or(value))
}

fn debug_field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let value = &line[line.find(prefix)? + prefix.len()..];
    Some(value.split([',', ')', '}']).next().unwrap_or(value).trim())
}

pub(super) fn f64_field(line: &str, prefix: &str) -> Option<f64> {
    field(line, prefix)?
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

pub(super) fn usize_field(line: &str, prefix: &str) -> Option<usize> {
    field(line, prefix)?.parse().ok()
}

fn debug_usize(line: &str, prefix: &str) -> Option<usize> {
    debug_field(line, prefix)?.parse().ok()
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
