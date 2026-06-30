use super::{SourceBudget, SourceWeights};

pub(super) fn normalized_budget(target: usize, weights: SourceWeights) -> SourceBudget {
    let raw = [
        weights.guided,
        weights.local,
        weights.factorial,
        weights.variance,
        weights.coverage,
        weights.random,
    ];
    let total = raw.iter().sum::<f64>();
    if total <= 0.0 {
        return SourceBudget {
            random: target,
            ..SourceBudget::default()
        };
    }

    let mut counts = [0usize; 6];
    let mut remainders = [(0usize, 0.0); 6];
    for (index, weight) in raw.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        let exact = target as f64 * *weight / total;
        counts[index] = exact.floor() as usize;
        if counts[index] == 0 {
            counts[index] = 1;
        }
        remainders[index] = (index, exact - exact.floor());
    }

    while counts.iter().sum::<usize>() > target {
        let Some(index) = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .min_by(|a, b| remainders[a.0].1.total_cmp(&remainders[b.0].1))
            .map(|(index, _)| index)
        else {
            break;
        };
        counts[index] -= 1;
    }
    while counts.iter().sum::<usize>() < target {
        remainders.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (index, _) in remainders {
            if counts.iter().sum::<usize>() >= target {
                break;
            }
            if raw[index] > 0.0 {
                counts[index] += 1;
            }
        }
    }

    SourceBudget {
        guided: counts[0],
        local: counts[1],
        factorial: counts[2],
        variance: counts[3],
        coverage: counts[4],
        random: counts[5],
    }
}
