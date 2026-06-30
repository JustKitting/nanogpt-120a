use super::super::fixtures::{candidate, config, trial};
use super::propose;

#[test]
fn model_proposal_records_sorted_ranked_candidates() {
    let trials = [
        trial("success", Some(5.0), candidate(4, 4, 0.8)),
        trial("success", Some(4.0), candidate(8, 4, 1.0)),
        trial("success", Some(3.5), candidate(16, 8, 1.2)),
        trial("rejected_screen", None, candidate(4, 8, 2.0)),
    ];
    let config = config(0, 8);
    let proposal = propose(&trials, &config, None);

    assert_eq!(proposal.reason, "model");
    assert_eq!(proposal.ranked.len(), config.candidate_samples);
    assert!(
        proposal
            .ranked
            .iter()
            .any(|ranked| ranked.candidate.key() == proposal.candidate.key())
    );
    assert!(
        proposal
            .ranked
            .windows(2)
            .all(|pair| { pair[0].score.score >= pair[1].score.score })
    );
}
