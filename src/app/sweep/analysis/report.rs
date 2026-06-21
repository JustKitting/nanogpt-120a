use std::{fs, io, path::Path};

use super::{super::config::SweepConfig, SweepAnalysis, beliefs};

pub fn write(sweep_dir: &Path, analysis: &SweepAnalysis, config: &SweepConfig) -> io::Result<()> {
    fs::write(sweep_dir.join("analysis_summary.md"), summary(analysis))?;
    fs::write(
        sweep_dir.join("analysis_effects.tsv"),
        effects_tsv(analysis),
    )?;
    fs::write(
        sweep_dir.join("analysis_interactions.tsv"),
        interactions_tsv(analysis),
    )?;
    fs::write(
        sweep_dir.join("analysis_beliefs.tsv"),
        beliefs::tsv(analysis, config),
    )
}

fn summary(analysis: &SweepAnalysis) -> String {
    let mut text = String::new();
    text.push_str("# Sweep Statistical Analysis\n\n");
    text.push_str(&format!("trial_count={}\n\n", analysis.trial_count));
    if let Some(prior) = analysis.stability_prior {
        text.push_str(&format!(
            "stability_prior_n={} stability_prior_positive={:.3} stability_prior_posterior_mean={:.6}\n\n",
            prior.n, prior.positive, prior.posterior_mean
        ));
    }
    for response in &analysis.models {
        response_summary(&mut text, response);
    }
    text
}

fn response_summary(text: &mut String, response: &super::ResponseModel) {
    text.push_str(&format!("## {}\n\n", response.name));
    text.push_str(&format!(
        "n={} residual_std={:.6} best_value={:.6} best_z={:.6}\n\n",
        response.model.n,
        response.model.residual_std,
        response.model.best_value,
        response.model.best_standard_score
    ));
    text.push_str("| factor | coefficient | stderr | t | p_positive |\n");
    text.push_str("|---|---:|---:|---:|---:|\n");
    for effect in response.model.effects.iter().take(12) {
        text.push_str(&format!(
            "| {} | {:.6} | {:.6} | {:.3} | {:.3} |\n",
            effect.name, effect.coefficient, effect.stderr, effect.t, effect.p_positive
        ));
    }
    text.push('\n');
    interaction_summary(text, response);
}

fn interaction_summary(text: &mut String, response: &super::ResponseModel) {
    let interactions = response
        .model
        .effects
        .iter()
        .filter(|effect| effect.name.contains('*'))
        .take(12)
        .collect::<Vec<_>>();
    if interactions.is_empty() {
        return;
    }

    text.push_str("Top pairwise standardized product effects:\n\n");
    text.push_str("| interaction | coefficient | stderr | t | p_positive |\n");
    text.push_str("|---|---:|---:|---:|---:|\n");
    for effect in interactions {
        text.push_str(&format!(
            "| {} | {:.6} | {:.6} | {:.3} | {:.3} |\n",
            effect.name, effect.coefficient, effect.stderr, effect.t, effect.p_positive
        ));
    }
    text.push('\n');
}

fn effects_tsv(analysis: &SweepAnalysis) -> String {
    let mut text = String::from("response\tn\tfactor\tcoefficient\tstderr\tt\tp_positive\n");
    for response in &analysis.models {
        for effect in &response.model.effects {
            text.push_str(&format!(
                "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\n",
                response.name,
                response.model.n,
                effect.name,
                effect.coefficient,
                effect.stderr,
                effect.t,
                effect.p_positive
            ));
        }
    }
    text
}

fn interactions_tsv(analysis: &SweepAnalysis) -> String {
    let mut text = String::from("response\tn\tinteraction\tcoefficient\tstderr\tt\tp_positive\n");
    for response in &analysis.models {
        for effect in response
            .model
            .effects
            .iter()
            .filter(|effect| effect.name.contains('*'))
        {
            text.push_str(&format!(
                "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\n",
                response.name,
                response.model.n,
                effect.name,
                effect.coefficient,
                effect.stderr,
                effect.t,
                effect.p_positive
            ));
        }
    }
    text
}
