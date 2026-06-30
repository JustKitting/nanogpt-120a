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
