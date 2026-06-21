use super::{candidate_space, rng::SweepRng};

pub const MIN_N_LAYER: usize = 4;
pub(super) use candidate_space::valid_aurora_phases;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub batch_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub aurora_phases: usize,
    pub aurora_blocks: usize,
    pub lr_scale: f64,
    pub adam_lr_scale: f64,
    pub warmup_steps: usize,
    pub start_ratio: f64,
    pub amuse_beta1: f64,
    pub amuse_rho: f64,
}

impl Candidate {
    pub fn random(rng: &mut SweepRng) -> Self {
        candidate_space::random(rng)
    }

    pub fn with_min_layers(&self) -> Self {
        if self.n_layer >= MIN_N_LAYER {
            return self.clone();
        }

        let n_layer = MIN_N_LAYER;
        let phases = candidate_space::valid_aurora_phases(n_layer * 4, self.aurora_blocks);
        let aurora_phases = phases
            .iter()
            .copied()
            .find(|phase| *phase >= self.aurora_phases)
            .or_else(|| phases.first().copied())
            .unwrap_or(self.aurora_phases);
        Self {
            n_layer,
            aurora_phases,
            ..self.clone()
        }
    }

    pub fn key(&self) -> String {
        format!(
            "b{}_l{}_d{}_h{}_p{}_c{}_lr{:.4}_alr{:.4}_w{}_s{:.2}_b{:.2}_r{:.2}",
            self.batch_size,
            self.n_layer,
            self.n_embd,
            self.n_head,
            self.aurora_phases,
            self.aurora_blocks,
            self.lr_scale,
            self.adam_lr_scale,
            self.warmup_steps,
            self.start_ratio,
            self.amuse_beta1,
            self.amuse_rho
        )
    }

    pub fn build_env(&self) -> Vec<(&'static str, String)> {
        vec![
            ("GPT2_BATCH_SIZE", self.batch_size.to_string()),
            ("GPT2_N_LAYER", self.n_layer.to_string()),
            ("GPT2_N_EMBD", self.n_embd.to_string()),
            ("GPT2_N_HEAD", self.n_head.to_string()),
            ("AURORA_MATRIX_PHASES", self.aurora_phases.to_string()),
            ("AURORA_COOPERATIVE_BLOCKS", self.aurora_blocks.to_string()),
        ]
    }

    pub fn run_env(&self) -> Vec<(&'static str, String)> {
        vec![
            ("TRAIN_LR_SCALE", format!("{:.6}", self.lr_scale)),
            ("TRAIN_ADAM_LR_SCALE", format!("{:.6}", self.adam_lr_scale)),
            ("TRAIN_LR_WARMUP_STEPS", self.warmup_steps.to_string()),
            ("TRAIN_LR_START_RATIO", format!("{:.6}", self.start_ratio)),
            ("TRAIN_AMUSE_BETA1", format!("{:.6}", self.amuse_beta1)),
            ("TRAIN_AMUSE_RHO", format!("{:.6}", self.amuse_rho)),
        ]
    }
}
