use super::sources_tsv;
use crate::sweep::{
    analysis::CandidateScore,
    optimizer::{Proposal, ScoredCandidate},
    test_fixtures::basic_candidate,
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

fn scored(
    source: &'static str,
    candidate: crate::sweep::candidate::Candidate,
    score: f64,
) -> ScoredCandidate {
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

fn candidate(batch_size: usize) -> crate::sweep::candidate::Candidate {
    basic_candidate(batch_size, 4)
}
