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
    write_status(
        &sweep_dir.join("status.env"),
        trial_dir,
        trial_index,
        candidate,
        event,
        result,
    )?;
    write_status(
        &trial_dir.join("status.env"),
        trial_dir,
        trial_index,
        candidate,
        event,
        result,
    )?;
    append_event(sweep_dir, trial_dir, trial_index, candidate, event, result)
}

fn write_status(
    path: &Path,
    trial_dir: &Path,
    trial_index: usize,
    candidate: &Candidate,
    event: &str,
    result: &RunResult,
) -> io::Result<()> {
    let text = format!(
        "UPDATED_AT_UTC={}\nEVENT={event}\nTRIAL_INDEX={trial_index}\nTRIAL_KEY={}\nTRIAL_DIR={}\nCOMPLETED_STEPS={}\nLAST_STEP={}\nLAST_ELAPSED_S={}\nLAST_TRAIN_LOSS={}\nVAL_LOSS={}\nSAW_NAN={}\n",
        timestamp(),
        candidate.key(),
        trial_dir.display(),
        fmt::optional_usize(result.completed_steps),
        fmt::optional_usize(result.last_step),
        fmt::optional_f64_6(result.last_elapsed_s),
        fmt::optional_f64_6(result.last_train_loss),
        fmt::optional_f64_6(result.val_loss),
        result.saw_nan,
    );
    fs::write(path, text)
}

fn append_event(
    sweep_dir: &Path,
    trial_dir: &Path,
    trial_index: usize,
    candidate: &Candidate,
    event: &str,
    result: &RunResult,
) -> io::Result<()> {
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
        timestamp(),
        event,
        trial_index,
        candidate.key(),
        trial_dir.display(),
        fmt::optional_usize(result.completed_steps),
        fmt::optional_usize(result.last_step),
        fmt::optional_f64_6(result.last_elapsed_s),
        fmt::optional_f64_6(result.last_train_loss),
        fmt::optional_f64_6(result.val_loss),
        result.saw_nan,
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
