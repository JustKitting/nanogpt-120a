use super::rng::SweepRng;

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
        let (n_embd, n_head) = rng.choose(&[(1024, 16), (1536, 12), (2048, 16)]);
        let n_layer = rng.choose(&[2, 4, 8]);
        let slots = n_layer * 4;
        let aurora_blocks = rng.choose(&[120, 160, 180]);
        let phases = [4, 8, 16]
            .into_iter()
            .filter(|phase| {
                slots % phase == 0 && cooperative_blocks(slots, *phase, aurora_blocks) <= 360
            })
            .collect::<Vec<_>>();
        Self {
            batch_size: rng.choose(&[4, 8, 16]),
            n_layer,
            n_embd,
            n_head,
            aurora_phases: rng.choose(&phases),
            aurora_blocks,
            lr_scale: rng.log_uniform(0.5, 2.5),
            adam_lr_scale: rng.log_uniform(0.5, 2.5),
            warmup_steps: rng.choose(&[5, 20, 50, 100]),
            start_ratio: rng.choose(&[0.0, 0.05, 0.1, 0.2]),
            amuse_beta1: rng.choose(&[0.2, 0.4, 0.6]),
            amuse_rho: rng.choose(&[0.5, 0.8, 1.0]),
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

fn cooperative_blocks(slots: usize, phases: usize, blocks: usize) -> usize {
    blocks * (slots / phases)
}
