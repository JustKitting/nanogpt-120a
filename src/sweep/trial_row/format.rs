use super::super::history::Trial;

pub(super) fn header() -> &'static str {
    "status\tval_loss\tcompleted_steps\tbatch_size\tn_layer\tn_embd\tn_head\taurora_phases\taurora_blocks\tlr_scale\tadam_lr_scale\tnextlat_lr_scale\twarmup_steps\tstart_ratio\tamuse_beta1\tamuse_rho\tlog_path\telapsed_s\tscreen_val_loss\tscreen_completed_steps\tscreen_elapsed_s\tscreen_reason"
}

pub(super) fn format_trial(trial: &Trial) -> String {
    let c = &trial.candidate;
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}",
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
        c.nextlat_lr_scale,
        c.warmup_steps,
        c.start_ratio,
        c.amuse_beta1,
        c.amuse_rho,
        trial.log_path.display(),
        trial.elapsed_s.map(fmt).unwrap_or_default(),
        trial.screen_val_loss.map(fmt).unwrap_or_default(),
        trial
            .screen_completed_steps
            .map(|value| value.to_string())
            .unwrap_or_default(),
        trial.screen_elapsed_s.map(fmt).unwrap_or_default(),
        trial.screen_reason.as_deref().unwrap_or_default()
    )
}

fn fmt(value: f64) -> String {
    format!("{value:.6}")
}
