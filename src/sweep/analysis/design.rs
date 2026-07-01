use super::super::features::{FEATURE_COUNT, FEATURE_NAMES};
use super::stats::{EPS, mean_stddev};

#[derive(Clone, Copy, Debug)]
pub enum Term {
    Main(usize),
    Pair(usize, usize),
}

#[derive(Clone, Debug)]
pub struct BaseStats {
    means: [f64; FEATURE_COUNT],
    stds: [f64; FEATURE_COUNT],
}

pub fn terms() -> Vec<Term> {
    let mut terms = (0..FEATURE_COUNT).map(Term::Main).collect::<Vec<_>>();
    for left in 0..FEATURE_COUNT {
        for right in left + 1..FEATURE_COUNT {
            terms.push(Term::Pair(left, right));
        }
    }
    terms
}

pub fn term_name(term: Term) -> String {
    match term {
        Term::Main(index) => FEATURE_NAMES[index].to_string(),
        Term::Pair(left, right) => {
            format!("{}*{}", FEATURE_NAMES[left], FEATURE_NAMES[right])
        }
    }
}

pub fn base_stats(rows: &[[f64; FEATURE_COUNT]]) -> BaseStats {
    let stats = std::array::from_fn(|index| {
        let values = rows.iter().map(|row| row[index]).collect::<Vec<_>>();
        mean_stddev(&values)
    });
    BaseStats {
        means: stats.map(|(mean, _)| mean),
        stds: stats.map(|(_, stddev)| stddev),
    }
}

pub fn values_from_base(
    base: &[f64; FEATURE_COUNT],
    stats: &BaseStats,
    terms: &[Term],
) -> Vec<f64> {
    let normalized: [f64; FEATURE_COUNT] = std::array::from_fn(|index| {
        if stats.stds[index] > EPS {
            (base[index] - stats.means[index]) / stats.stds[index]
        } else {
            0.0
        }
    });
    terms
        .iter()
        .map(|term| match *term {
            Term::Main(index) => normalized[index],
            Term::Pair(left, right) => normalized[left] * normalized[right],
        })
        .collect()
}
