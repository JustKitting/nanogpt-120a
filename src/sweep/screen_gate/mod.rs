use std::{fs, io, path::Path};

use super::{analysis::CandidateScore, parse::RunResult};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct Decision {
    pub pass: bool,
    pub reason: &'static str,
    pub screen_loss: Option<f64>,
    pub baseline_loss: Option<f64>,
    pub expected_quality: Option<f64>,
    pub survival_prior: Option<f64>,
    pub completed_steps: Option<usize>,
}

pub fn decide(
    result: &RunResult,
    baseline_loss: Option<f64>,
    score: Option<&CandidateScore>,
) -> Decision {
    if result.saw_nan {
        return decision(false, "nan", result, baseline_loss, score);
    }
    let Some(screen_loss) = result.val_loss else {
        return decision(false, "missing_val_loss", result, baseline_loss, score);
    };
    if baseline_loss.is_none() {
        return decision(true, "no_baseline", result, baseline_loss, score);
    }
    if baseline_loss.is_some_and(|baseline| screen_loss <= baseline) {
        return decision(true, "screen_loss_improved", result, baseline_loss, score);
    }
    decision(false, "screen_loss_worse", result, baseline_loss, score)
}

pub fn write(path: &Path, decision: &Decision) -> io::Result<()> {
    fs::write(
        path,
        format!(
            "PASS={}\nREASON={}\nSCREEN_LOSS={}\nBASELINE_SCREEN_LOSS={}\nEXPECTED_QUALITY={}\nSURVIVAL_PRIOR={}\nCOMPLETED_STEPS={}\n",
            decision.pass,
            decision.reason,
            fmt_f64(decision.screen_loss),
            fmt_f64(decision.baseline_loss),
            fmt_f64(decision.expected_quality),
            fmt_f64(decision.survival_prior),
            decision
                .completed_steps
                .map(|v| v.to_string())
                .unwrap_or_default(),
        ),
    )
}

fn decision(
    pass: bool,
    reason: &'static str,
    result: &RunResult,
    baseline_loss: Option<f64>,
    score: Option<&CandidateScore>,
) -> Decision {
    Decision {
        pass,
        reason,
        screen_loss: result.val_loss,
        baseline_loss,
        expected_quality: score.map(|score| score.expected_quality),
        survival_prior: score.map(|score| score.survival_prior),
        completed_steps: result.completed_steps,
    }
}

fn fmt_f64(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}
