use std::{collections::BTreeMap, fs, path::Path};

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
    fs::write(
        sweep_dir.join(format!("candidate_{index:04}_sources.tsv")),
        sources_tsv(proposal),
    )?;
    Ok(())
}

fn selected_text(proposal: &Proposal) -> String {
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

fn ranked_tsv(proposal: &Proposal) -> String {
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

#[derive(Clone, Debug)]
struct SourceSummary<'a> {
    count: usize,
    selected: bool,
    best_rank: usize,
    best: &'a CandidateScore,
}

fn sources_tsv(proposal: &Proposal) -> String {
    let mut summaries = BTreeMap::<&'static str, SourceSummary<'_>>::new();
    for (rank, scored) in proposal.ranked.iter().enumerate() {
        let selected = scored.candidate.key() == proposal.candidate.key();
        summaries
            .entry(scored.source)
            .and_modify(|summary| {
                summary.count += 1;
                summary.selected |= selected;
            })
            .or_insert(SourceSummary {
                count: 1,
                selected,
                best_rank: rank,
                best: &scored.score,
            });
    }

    let mut text = String::from(
        "source\tcount\tselected\tbest_rank\tbest_score\tbest_expected_quality\tbest_survival_prior\tbest_probability_improvement\tbest_expected_improvement\tbest_uncertainty\n",
    );
    for (source, summary) in summaries {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\n",
            source,
            summary.count,
            summary.selected,
            summary.best_rank,
            summary.best.score,
            summary.best.expected_quality,
            summary.best.survival_prior,
            summary.best.probability_improvement,
            summary.best.expected_improvement,
            summary.best.uncertainty
        ));
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

#[cfg(test)]
mod tests {
    use super::sources_tsv;
    use crate::sweep::{
        analysis::CandidateScore,
        candidate::Candidate,
        optimizer::{Proposal, ScoredCandidate},
    };

    #[test]
    fn source_summary_counts_ranked_candidate_sources() {
        let proposal = Proposal {
            candidate: candidate(16),
            reason: "model",
            ranked: vec![
                scored("guided", candidate(16), 2.0),
                scored("variance", candidate(8), 1.0),
                scored("guided", candidate(4), 0.5),
            ],
        };
        let text = sources_tsv(&proposal);

        assert!(text.contains("source\tcount\tselected\tbest_rank"));
        assert!(text.contains("guided\t2\ttrue\t0\t2.00000000"));
        assert!(text.contains("variance\t1\tfalse\t1\t1.00000000"));
    }

    fn scored(source: &'static str, candidate: Candidate, score: f64) -> ScoredCandidate {
        ScoredCandidate {
            candidate,
            source,
            score: CandidateScore {
                score,
                expected_quality: score,
                survival_prior: 1.0,
                probability_improvement: 0.0,
                expected_improvement: 0.0,
                uncertainty: 0.0,
                exploration: 0.0,
                predicted_quality: None,
                predicted_stability: None,
            },
        }
    }

    fn candidate(batch_size: usize) -> Candidate {
        Candidate {
            batch_size,
            n_layer: 4,
            n_embd: 1024,
            n_head: 16,
            aurora_phases: 4,
            aurora_blocks: 80,
            lr_scale: 1.0,
            adam_lr_scale: 1.0,
            nextlat_lr_scale: 1.0,
            warmup_steps: 20,
            start_ratio: 0.1,
            amuse_beta1: 0.4,
            amuse_rho: 0.8,
        }
    }
}
