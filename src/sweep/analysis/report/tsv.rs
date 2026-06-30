use super::super::{SweepAnalysis, regression::Effect};

pub(super) fn effects(analysis: &SweepAnalysis) -> String {
    filtered_effects(analysis, "factor", |_| true)
}

pub(super) fn interactions(analysis: &SweepAnalysis) -> String {
    filtered_effects(analysis, "interaction", |effect| effect.name.contains('*'))
}

fn filtered_effects(
    analysis: &SweepAnalysis,
    value_column: &str,
    include: impl Fn(&Effect) -> bool,
) -> String {
    let mut text = format!("response\tn\t{value_column}\tcoefficient\tstderr\tt\tp_positive\n");
    for response in &analysis.models {
        for effect in response
            .model
            .effects
            .iter()
            .filter(|effect| include(effect))
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
