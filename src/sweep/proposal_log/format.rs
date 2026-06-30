use super::super::{
    analysis::{CandidateScore, Prediction},
    optimizer::{Proposal, ScoredCandidate},
};

mod sources;
#[cfg(test)]
mod tests;

pub(super) use sources::sources_tsv;

pub(super) fn selected_text(proposal: &Proposal) -> String {
    let (source, selected) = selected_score(proposal);
    format!(
        "candidate={}\nreason={}\nsource={}\nscore={:.6}\nexpected_quality={:.6}\nsurvival_prior={:.6}\nprobability_improvement={:.6}\nexpected_improvement={:.6}\nuncertainty={:.6}\nexploration={:.6}\nquality={}\nstability={}\n",
        proposal.candidate.key(),
        proposal.reason,
        source,
        selected.score,
        selected.expected_quality,
        selected.survival_prior,
        selected.probability_improvement,
        selected.expected_improvement,
        selected.uncertainty,
        selected.exploration,
        fmt_prediction(selected.predicted_quality),
        fmt_prediction(selected.predicted_stability)
    )
}

pub(super) fn ranked_tsv(proposal: &Proposal) -> String {
    let mut text = String::from(
        "rank\tselected\tsource\tcandidate\tscore\texpected_quality\tsurvival_prior\tprobability_improvement\texpected_improvement\tuncertainty\texploration\tquality_value\tquality_z\tquality_uncertainty\tstability_value\tstability_z\tstability_uncertainty\n",
    );
    for (rank, scored) in proposal.ranked.iter().enumerate() {
        push_ranked_row(
            &mut text,
            rank,
            scored,
            scored.candidate.key() == proposal.candidate.key(),
        );
    }
    text
}

fn push_ranked_row(text: &mut String, rank: usize, scored: &ScoredCandidate, selected: bool) {
    let quality = scored.score.predicted_quality;
    let stability = scored.score.predicted_stability;
    text.push_str(&format!(
        "{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        rank,
        selected,
        scored.source,
        scored.candidate.key(),
        scored.score.score,
        scored.score.expected_quality,
        scored.score.survival_prior,
        scored.score.probability_improvement,
        scored.score.expected_improvement,
        scored.score.uncertainty,
        scored.score.exploration,
        value(quality),
        standard_score(quality),
        uncertainty(quality),
        value(stability),
        standard_score(stability),
        uncertainty(stability)
    ));
}

fn selected_score(proposal: &Proposal) -> (&'static str, &CandidateScore) {
    proposal
        .ranked
        .iter()
        .find(|scored| scored.candidate.key() == proposal.candidate.key())
        .map(|scored| (scored.source, &scored.score))
        .unwrap_or((proposal.ranked[0].source, &proposal.ranked[0].score))
}

fn fmt_prediction(value: Option<Prediction>) -> String {
    value
        .map(|prediction| {
            format!(
                "value={:.6},z={:.6},uncertainty={:.6}",
                prediction.value, prediction.standard_score, prediction.uncertainty
            )
        })
        .unwrap_or_else(|| "n/a".to_string())
}

fn value(prediction: Option<Prediction>) -> String {
    prediction_field(prediction, |prediction| prediction.value)
}

fn standard_score(prediction: Option<Prediction>) -> String {
    prediction_field(prediction, |prediction| prediction.standard_score)
}

fn uncertainty(prediction: Option<Prediction>) -> String {
    prediction_field(prediction, |prediction| prediction.uncertainty)
}

fn prediction_field(
    prediction: Option<Prediction>,
    value: impl FnOnce(Prediction) -> f64,
) -> String {
    prediction
        .map(|prediction| format!("{:.8}", value(prediction)))
        .unwrap_or_default()
}
