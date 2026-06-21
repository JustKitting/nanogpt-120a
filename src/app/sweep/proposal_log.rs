use std::{fs, path::Path};

use super::{
    analysis::{CandidateScore, Prediction},
    optimizer::{Proposal, ScoredCandidate},
};

pub fn write(
    sweep_dir: &Path,
    index: usize,
    proposal: &Proposal,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(
        sweep_dir.join(format!("candidate_{index:04}_score.txt")),
        selected_text(proposal),
    )?;
    fs::write(
        sweep_dir.join(format!("candidate_{index:04}_ranked.tsv")),
        ranked_tsv(proposal),
    )?;
    Ok(())
}

fn selected_text(proposal: &Proposal) -> String {
    let (source, selected) = selected_score(proposal);
    format!(
        "candidate={}\nreason={}\nsource={}\nscore={:.6}\nexpected_quality={:.6}\nexpected_speed={:.6}\nsurvival_prior={:.6}\nprobability_improvement={:.6}\nexpected_improvement={:.6}\nuncertainty={:.6}\nexploration={:.6}\nquality={}\nspeed={}\nstability={}\n",
        proposal.candidate.key(),
        proposal.reason,
        source,
        selected.score,
        selected.expected_quality,
        selected.expected_speed,
        selected.survival_prior,
        selected.probability_improvement,
        selected.expected_improvement,
        selected.uncertainty,
        selected.exploration,
        fmt_prediction(selected.predicted_quality),
        fmt_prediction(selected.predicted_speed),
        fmt_prediction(selected.predicted_stability)
    )
}

fn ranked_tsv(proposal: &Proposal) -> String {
    let mut text = String::from(
        "rank\tselected\tsource\tcandidate\tscore\texpected_quality\texpected_speed\tsurvival_prior\tprobability_improvement\texpected_improvement\tuncertainty\texploration\tquality_value\tquality_z\tquality_uncertainty\tspeed_value\tspeed_z\tspeed_uncertainty\tstability_value\tstability_z\tstability_uncertainty\n",
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
    let speed = scored.score.predicted_speed;
    let stability = scored.score.predicted_stability;
    text.push_str(&format!(
        "{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        rank,
        selected,
        scored.source,
        scored.candidate.key(),
        scored.score.score,
        scored.score.expected_quality,
        scored.score.expected_speed,
        scored.score.survival_prior,
        scored.score.probability_improvement,
        scored.score.expected_improvement,
        scored.score.uncertainty,
        scored.score.exploration,
        value(quality),
        standard_score(quality),
        uncertainty(quality),
        value(speed),
        standard_score(speed),
        uncertainty(speed),
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
    prediction
        .map(|prediction| format!("{:.8}", prediction.value))
        .unwrap_or_default()
}

fn standard_score(prediction: Option<Prediction>) -> String {
    prediction
        .map(|prediction| format!("{:.8}", prediction.standard_score))
        .unwrap_or_default()
}

fn uncertainty(prediction: Option<Prediction>) -> String {
    prediction
        .map(|prediction| format!("{:.8}", prediction.uncertainty))
        .unwrap_or_default()
}
