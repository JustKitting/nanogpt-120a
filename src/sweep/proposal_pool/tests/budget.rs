use crate::sweep::{analysis, test_fixtures::success_trial as trial};

use super::{config, wide_candidate};

#[test]
fn source_budget_keeps_guided_off_without_response_model() {
    let config = config();
    let analysis = analysis::analyze(&[], &config);
    let budget = super::super::source_budget(40, &analysis, &config);

    assert_eq!(budget.guided, 0);
    assert_eq!(budget.local, 0);
    assert!(budget.variance > 0);
    assert!(budget.coverage > 0);
    assert!(budget.random > 0);
    assert_eq!(budget.total(), 40);
}

#[test]
fn source_budget_moves_toward_guided_when_model_matures() {
    let config = config();
    let empty = analysis::analyze(&[], &config);
    let mature_trials = (0..64)
        .map(|i| trial(wide_candidate(i), 64.0 - i as f64))
        .collect::<Vec<_>>();
    let mature = analysis::analyze(&mature_trials, &config);

    let empty_budget = super::super::source_budget(40, &empty, &config);
    let mature_budget = super::super::source_budget(40, &mature, &config);

    assert!(mature_budget.guided > empty_budget.guided);
    assert!(mature_budget.local > empty_budget.local);
    assert!(mature_budget.guided >= mature_budget.coverage);
}
