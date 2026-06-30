use std::collections::BTreeMap;

use super::super::super::{analysis::CandidateScore, optimizer::Proposal};

#[derive(Clone, Debug)]
struct SourceSummary<'a> {
    count: usize,
    selected: bool,
    best_rank: usize,
    best: &'a CandidateScore,
}

pub(in crate::sweep::proposal_log) fn sources_tsv(proposal: &Proposal) -> String {
    let mut summaries = BTreeMap::<&'static str, SourceSummary<'_>>::new();
    let selected_key = proposal.candidate.key();
    for (rank, scored) in proposal.ranked.iter().enumerate() {
        let selected = scored.candidate.key() == selected_key;
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
