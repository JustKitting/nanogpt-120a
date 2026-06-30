use super::super::{ResponseModel, SweepAnalysis, regression::Effect};

pub(super) fn text(analysis: &SweepAnalysis) -> String {
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

fn response_summary(text: &mut String, response: &ResponseModel) {
    text.push_str(&format!("## {}\n\n", response.name));
    text.push_str(&format!(
        "n={} residual_std={:.6} best_value={:.6} best_z={:.6}\n\n",
        response.model.n,
        response.model.residual_std,
        response.model.best_value,
        response.model.best_standard_score
    ));
    effect_markdown_table(text, "factor", response.model.effects.iter().take(12));
    text.push('\n');
    interaction_summary(text, response);
}

fn interaction_summary(text: &mut String, response: &ResponseModel) {
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
    effect_markdown_table(text, "interaction", interactions.into_iter());
    text.push('\n');
}

fn effect_markdown_table<'a>(
    text: &mut String,
    column: &str,
    effects: impl Iterator<Item = &'a Effect>,
) {
    text.push_str(&format!(
        "| {column} | coefficient | stderr | t | p_positive |\n"
    ));
    text.push_str("|---|---:|---:|---:|---:|\n");
    for effect in effects {
        text.push_str(&format!(
            "| {} | {:.6} | {:.6} | {:.3} | {:.3} |\n",
            effect.name, effect.coefficient, effect.stderr, effect.t, effect.p_positive
        ));
    }
}
