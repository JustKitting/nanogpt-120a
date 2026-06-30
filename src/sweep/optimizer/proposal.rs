use crate::sweep::{
    analysis::{self, CandidateScore, SweepAnalysis},
    candidate::Candidate,
    config::SweepConfig,
    proposal_pool::PooledCandidate,
};

#[derive(Clone, Debug)]
pub struct Proposal {
    pub candidate: Candidate,
    pub reason: &'static str,
    pub ranked: Vec<ScoredCandidate>,
}

#[derive(Clone, Debug)]
pub struct ScoredCandidate {
    pub candidate: Candidate,
    pub source: &'static str,
    pub score: CandidateScore,
}

impl Proposal {
    pub fn selected_scored(&self) -> Option<&ScoredCandidate> {
        let selected_key = self.candidate.key();
        self.ranked
            .iter()
            .find(|scored| scored.candidate.key() == selected_key)
            .or_else(|| self.ranked.first())
    }

    pub(super) fn single(
        reason: &'static str,
        candidate: Candidate,
        analysis: &SweepAnalysis,
        config: &SweepConfig,
    ) -> Self {
        let score = analysis::score_candidate(analysis, config, &candidate);
        Self {
            candidate: candidate.clone(),
            reason,
            ranked: vec![ScoredCandidate {
                candidate,
                source: reason,
                score,
            }],
        }
    }
}

impl ScoredCandidate {
    pub(super) fn from_pooled(
        pooled: PooledCandidate,
        analysis: &SweepAnalysis,
        config: &SweepConfig,
    ) -> Self {
        let score = analysis::score_candidate(analysis, config, &pooled.candidate);
        Self {
            candidate: pooled.candidate,
            source: pooled.source,
            score,
        }
    }
}
