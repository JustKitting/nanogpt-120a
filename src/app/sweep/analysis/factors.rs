use super::super::candidate::Candidate;

pub const FEATURE_COUNT: usize = 12;

pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "batch_size",
    "n_layer",
    "n_embd",
    "aurora_phases",
    "aurora_blocks",
    "ln_lr_scale",
    "ln_adam_lr_scale",
    "ln_nextlat_lr_scale",
    "ln_warmup_steps",
    "start_ratio",
    "amuse_beta1",
    "amuse_rho",
];

pub fn candidate_features(candidate: &Candidate) -> [f64; FEATURE_COUNT] {
    [
        candidate.batch_size as f64,
        candidate.n_layer as f64,
        candidate.n_embd as f64 / 1024.0,
        candidate.aurora_phases as f64,
        candidate.aurora_blocks as f64 / 80.0,
        candidate.lr_scale.ln(),
        candidate.adam_lr_scale.ln(),
        candidate.nextlat_lr_scale.ln(),
        (candidate.warmup_steps as f64).ln(),
        candidate.start_ratio,
        candidate.amuse_beta1,
        candidate.amuse_rho,
    ]
}
