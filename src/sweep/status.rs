use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::Path,
};

use time::OffsetDateTime;

use super::{candidate::Candidate, fmt, parse::RunResult};

pub fn record(
    sweep_dir: &Path,
    trial_dir: &Path,
    trial_index: usize,
    candidate: &Candidate,
    event: &str,
    result: &RunResult,
) -> io::Result<()> {
    let record = StatusRecord::new(trial_dir, trial_index, candidate, event, result);
    for path in [sweep_dir.join("status.env"), trial_dir.join("status.env")] {
        write_status(&path, &record)?;
    }
    append_event(sweep_dir, &record)
}

struct StatusRecord<'a> {
    updated_at_utc: String, event: &'a str, trial_index: usize, trial_key: String, trial_dir: String,
    completed_steps: String, last_step: String, last_elapsed_s: String,
    last_train_loss: String, val_loss: String, saw_nan: bool,
}

impl<'a> StatusRecord<'a> {
    fn new(trial_dir: &Path, trial_index: usize, candidate: &Candidate, event: &'a str, result: &RunResult) -> Self {
        Self {
            updated_at_utc: timestamp(), event, trial_index, trial_key: candidate.key(), trial_dir: trial_dir.display().to_string(),
            completed_steps: fmt::optional_usize(result.completed_steps),
            last_step: fmt::optional_usize(result.last_step),
            last_elapsed_s: fmt::optional_f64_6(result.last_elapsed_s),
            last_train_loss: fmt::optional_f64_6(result.last_train_loss),
            val_loss: fmt::optional_f64_6(result.val_loss),
            saw_nan: result.saw_nan,
        }
    }
}

fn write_status(path: &Path, record: &StatusRecord<'_>) -> io::Result<()> {
    let text = format!(
        "UPDATED_AT_UTC={}\nEVENT={}\nTRIAL_INDEX={}\nTRIAL_KEY={}\nTRIAL_DIR={}\nCOMPLETED_STEPS={}\nLAST_STEP={}\nLAST_ELAPSED_S={}\nLAST_TRAIN_LOSS={}\nVAL_LOSS={}\nSAW_NAN={}\n",
        record.updated_at_utc, record.event, record.trial_index, record.trial_key, record.trial_dir,
        record.completed_steps, record.last_step, record.last_elapsed_s, record.last_train_loss,
        record.val_loss, record.saw_nan,
    );
    fs::write(path, text)
}

fn append_event(sweep_dir: &Path, record: &StatusRecord<'_>) -> io::Result<()> {
    let path = sweep_dir.join("events.tsv");
    let write_header = !path.exists();
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if write_header {
        writeln!(
            file,
            "updated_at_utc\tevent\ttrial_index\ttrial_key\ttrial_dir\tcompleted_steps\tlast_step\tlast_elapsed_s\tlast_train_loss\tval_loss\tsaw_nan"
        )?;
    }
    writeln!(
        file,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        record.updated_at_utc, record.event, record.trial_index, record.trial_key, record.trial_dir,
        record.completed_steps, record.last_step, record.last_elapsed_s, record.last_train_loss,
        record.val_loss, record.saw_nan,
    )
}

fn timestamp() -> String {
    let now = OffsetDateTime::now_utc();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}
