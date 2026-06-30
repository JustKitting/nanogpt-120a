use crate::sweep::{parse::RunResult, status};

use super::fixtures::{measured_candidate, temp_path};

#[test]
fn records_sweep_owned_trial_status_and_events() {
    let sweep_dir = temp_path("sweep-status");
    let trial_dir = sweep_dir.join("trial_0000");
    std::fs::create_dir_all(&trial_dir).unwrap();
    let candidate = measured_candidate();
    let result = RunResult {
        val_loss: Some(4.2),
        completed_steps: Some(128),
        last_step: Some(127),
        last_elapsed_s: Some(90.5),
        last_train_loss: Some(4.8),
        saw_nan: false,
    };

    status::record(&sweep_dir, &trial_dir, 0, &candidate, "success", &result).unwrap();

    let root_status = std::fs::read_to_string(sweep_dir.join("status.env")).unwrap();
    let trial_status = std::fs::read_to_string(trial_dir.join("status.env")).unwrap();
    let events = std::fs::read_to_string(sweep_dir.join("events.tsv")).unwrap();
    assert!(root_status.contains("EVENT=success"));
    assert!(root_status.contains("VAL_LOSS=4.200000"));
    assert!(trial_status.contains("COMPLETED_STEPS=128"));
    assert!(events.contains("success\t0\tb8_l2_d1024_h16_p2_c80_lr1.0140"));
    let _ = std::fs::remove_dir_all(sweep_dir);
}
