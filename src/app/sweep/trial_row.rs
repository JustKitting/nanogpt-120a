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
    "status\tval_loss\tcompleted_steps\tbatch_size\tn_layer\tn_embd\tn_head\taurora_phases\taurora_blocks\tlr_scale\tadam_lr_scale\twarmup_steps\tstart_ratio\tamuse_beta1\tamuse_rho\tlog_path\telapsed_s"
}

fn format_trial(trial: &Trial) -> String {
    let c = &trial.candidate;
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}",
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
        trial.log_path.display(),
        trial.elapsed_s.map(fmt).unwrap_or_default()
    )
}

fn parse_trial(line: &str) -> Option<Trial> {
    let p = line.split('\t').collect::<Vec<_>>();
    if p.len() != 16 && p.len() != 17 {
        return None;
    }
    Some(Trial {
        status: p[0].to_string(),
        val_loss: parse_loss(p[1]),
        completed_steps: p[2].parse().ok(),
        candidate: parse_candidate(&p)?,
        log_path: PathBuf::from(p[15]),
        elapsed_s: p.get(16).and_then(|value| parse_loss(value)),
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{format_trial, parse_trial};
    use crate::sweep::{candidate::Candidate, history::Trial};

    #[test]
    fn roundtrips_elapsed_time_in_new_rows() {
        let trial = Trial {
            status: "success".to_string(),
            val_loss: Some(4.25),
            completed_steps: Some(512),
            candidate: candidate(),
            log_path: PathBuf::from("target/train.log"),
            elapsed_s: Some(123.5),
        };

        let parsed = parse_trial(&format_trial(&trial)).unwrap();
        assert_eq!(parsed.elapsed_s, Some(123.5));
        assert_eq!(parsed.completed_steps, Some(512));
        assert_eq!(parsed.candidate.key(), trial.candidate.key());
    }

    #[test]
    fn parses_old_rows_without_elapsed_time() {
        let parsed = parse_trial(
            "success\t4.250000\t512\t8\t4\t1024\t16\t4\t80\t1.000000\t1.000000\t20\t0.100000\t0.400000\t0.800000\ttarget/train.log",
        )
        .unwrap();

        assert_eq!(parsed.elapsed_s, None);
        assert_eq!(parsed.completed_steps, Some(512));
        assert_eq!(parsed.candidate.batch_size, 8);
    }

    fn candidate() -> Candidate {
        Candidate {
            batch_size: 8,
            n_layer: 4,
            n_embd: 1024,
            n_head: 16,
            aurora_phases: 4,
            aurora_blocks: 80,
            lr_scale: 1.0,
            adam_lr_scale: 1.0,
            warmup_steps: 20,
            start_ratio: 0.1,
            amuse_beta1: 0.4,
            amuse_rho: 0.8,
        }
    }
}
