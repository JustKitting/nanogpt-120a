use super::super::MAX_ITERATIONS;
use super::schedule_name;

pub(super) const SAFETY_VALUES: [f32; 9] = [1.0, 1.01, 1.02, 1.03, 1.04, 1.045, 1.05, 1.06, 1.08];

const LATE_SAFETY_VALUES: [f32; 5] = [1.03, 1.04, 1.045, 1.05, 1.06];
const RAMPED_SAFETY_VALUES: [f32; 4] = [1.04, 1.045, 1.05, 1.06];

pub(super) fn seed_safety_schedules() -> Vec<[f32; MAX_ITERATIONS]> {
    let mut schedules = SAFETY_VALUES.map(|value| [value; MAX_ITERATIONS]).to_vec();
    schedules.extend(
        (0..MAX_ITERATIONS).flat_map(|start| {
            LATE_SAFETY_VALUES.map(move |value| late_safety_schedule(start, value))
        }),
    );
    for high in RAMPED_SAFETY_VALUES {
        let mut schedule = [1.0; MAX_ITERATIONS];
        for (iter, slot) in schedule.iter_mut().enumerate() {
            let t = iter as f32 / (MAX_ITERATIONS - 1) as f32;
            *slot = 1.0 + (high - 1.0) * t;
        }
        schedules.push(schedule);
    }
    schedules.sort_by_key(schedule_name);
    schedules.dedup();
    schedules
}

pub(super) fn corrected_schedule_candidates(
    best_raw: [f32; MAX_ITERATIONS],
) -> Vec<[f32; MAX_ITERATIONS]> {
    let mut schedules = vec![
        best_raw,
        [1.01; MAX_ITERATIONS],
        [1.03; MAX_ITERATIONS],
        [1.05; MAX_ITERATIONS],
    ];
    schedules.extend((2..=5).map(|start| late_safety_schedule(start, 1.03)));
    schedules.sort_by_key(schedule_name);
    schedules.dedup();
    schedules
}

fn late_safety_schedule(start: usize, value: f32) -> [f32; MAX_ITERATIONS] {
    let mut schedule = [1.0; MAX_ITERATIONS];
    schedule[start..].fill(value);
    schedule
}
