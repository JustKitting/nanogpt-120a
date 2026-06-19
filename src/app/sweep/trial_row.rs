use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use super::{candidate::Candidate, history::Trial};

pub fn read_trials(path: &Path) -> Vec<Trial> {
    fs::read_to_string(path)
        .ok()
        .map(|text| text.lines().skip(1).filter_map(parse_trial).collect())
        .unwrap_or_default()
}

pub fn append(path: &Path, trial: &Trial) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let new_file = !path.exists();
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if new_file {
        writeln!(file, "{}", header())?;
    }
    writeln!(file, "{}", format_trial(trial))
}

fn header() -> &'static str {
    "status\tval_loss\tcompleted_steps\tbatch_size\tn_layer\tn_embd\tn_head\taurora_phases\taurora_blocks\tlr_scale\tadam_lr_scale\twarmup_steps\tstart_ratio\tamuse_beta1\tamuse_rho\tlog_path"
}

fn format_trial(trial: &Trial) -> String {
    let c = &trial.candidate;
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}",
        trial.status,
        trial.val_loss.map(fmt).unwrap_or_else(|| "NaN".to_string()),
        trial
            .completed_steps
            .map(|v| v.to_string())
            .unwrap_or_default(),
        c.batch_size,
        c.n_layer,
        c.n_embd,
        c.n_head,
        c.aurora_phases,
        c.aurora_blocks,
        c.lr_scale,
        c.adam_lr_scale,
        c.warmup_steps,
        c.start_ratio,
        c.amuse_beta1,
        c.amuse_rho,
        trial.log_path.display()
    )
}

fn parse_trial(line: &str) -> Option<Trial> {
    let p = line.split('\t').collect::<Vec<_>>();
    if p.len() != 16 {
        return None;
    }
    Some(Trial {
        status: p[0].to_string(),
        val_loss: parse_loss(p[1]),
        completed_steps: p[2].parse().ok(),
        candidate: parse_candidate(&p)?,
        log_path: PathBuf::from(p[15]),
    })
}

fn parse_candidate(p: &[&str]) -> Option<Candidate> {
    Some(Candidate {
        batch_size: p[3].parse().ok()?,
        n_layer: p[4].parse().ok()?,
        n_embd: p[5].parse().ok()?,
        n_head: p[6].parse().ok()?,
        aurora_phases: p[7].parse().ok()?,
        aurora_blocks: p[8].parse().ok()?,
        lr_scale: p[9].parse().ok()?,
        adam_lr_scale: p[10].parse().ok()?,
        warmup_steps: p[11].parse().ok()?,
        start_ratio: p[12].parse().ok()?,
        amuse_beta1: p[13].parse().ok()?,
        amuse_rho: p[14].parse().ok()?,
    })
}

fn parse_loss(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn fmt(value: f64) -> String {
    format!("{value:.6}")
}
