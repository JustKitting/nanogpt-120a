# GPT-2 NVFP4 Training Optimization Notes

This note tracks throughput goals, measured baselines, experiments already tried,
and the next optimization work. Keep this file factual: add measured commands and
results before changing the plan.

## Goal

Train a GPT-2-small-shaped model with local Rust/CUDA kernels, targeting
NVFP4-heavy training where tensor-core matmuls dominate runtime.

External reference target:

- llm.c / Karpathy GPT-2 124M FineWeb run: 10B tokens in roughly 4-24 hours on
  one strong single GPU equivalent, with higher-end 8x A100 runs around 90
  minutes.

Primary optimization target:

- Lowest held-out validation loss after a fixed wall-clock training budget.
- Training loss, fixed-step loss, tokens/s, and isolated kernel timings are
  diagnostics only. They do not prove an optimization unless the held-out
  validation loss at the same wall-clock budget improves or is preserved.
- Use the validation split endpoint line:

```text
heldout_eval split=val val_loss=... train_elapsed_s=... completed_steps=...
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Make the SYNTH target runnable without full snapshot prep.
status: implementation cleanup
decision:
  Keep the same PleIAs/SYNTH dataset, sorted parquet order, tokenizer, and shard
  naming, but do not snapshot-download/tokenize the whole dataset before
  training can start. The prep path now lists parquet files, downloads one at a
  time, and stops inside the parquet row loop once the default validation shard
  and first train shard have been written.
reason:
  The current trainer reads the first synth_llama2_train shard and the
  synth_llama2_val shard. Preparing every parquet file first was wasted work and
  blocked fixed-time validation experiments on SYNTH.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo test -p synth-prep: pass
  cargo oxide build --arch sm_120a: pass
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Move the default training target to PleIAs/SYNTH.
status: implementation cleanup
decision:
  Keep SYNTH as the default training dataset and use the shard split produced
  by synth-prep directly. Training reads the first synth_llama2_train shard.
  Held-out validation reads synth_llama2_val_000000 instead of reserving the
  tail of the train shard.
source:
  Hugging Face dataset PleIAs/SYNTH is parquet text data with train split and
  query, synthetic_reasoning, and synthetic_answer columns.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Same-tokenizer width increase from d1536 to d2048 at L2.
status: rejected, slower and worse held-out validation
decision:
  Revert to the L2 d1536/head12 candidate. The d2048/head16 candidate has
  more capacity and head_dim=128, but the slower step rate loses at the fixed
  five-minute validation budget.
changes:
  Tested GPT2_N_EMBD=2048, GPT2_N_HEAD=16, GPT2_N_LAYER=2,
  AURORA_MATRIX_PHASES=8. Tokenizer, dataset, sequence length, MLP ratio,
  batch size, and optimizer math stayed the same.
result:
  Current L2 d1536/head12 reference:
    target/fixed_time_val_wide1536_l2_b4_phase8_300s_20260619T042826Z.log
    completed_steps=1013
    heldout_eval split=val val_loss=4.772994 train_elapsed_s=300.497
  L2 d2048/head16 candidate:
    target/fixed_time_val_d2048_l2_h16_b4_300s_20260619T044705Z.log
    completed_steps=496
    heldout_eval split=val val_loss=4.953667 train_elapsed_s=300.684
quality:
  The d2048 run was finite=true and nonzero=true at every logged step, but it
  did not beat held-out validation loss.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  d2048 300-second direct GPU run: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Same-tokenizer width cut from d1536 to d1024 at L2.
status: rejected, faster but worse held-out validation
decision:
  Revert to the L2 d1536/head12 candidate. The d1024/head16 candidate is
  faster and Tensor-Core friendly with head_dim=64, but it loses too much
  capacity for the current fixed five-minute validation target.
changes:
  Tested GPT2_N_EMBD=1024, GPT2_N_HEAD=16, GPT2_N_LAYER=2,
  AURORA_MATRIX_PHASES=8. Tokenizer, dataset, sequence length, MLP ratio,
  batch size, and optimizer math stayed the same.
result:
  Current L2 d1536/head12 reference:
    target/fixed_time_val_wide1536_l2_b4_phase8_300s_20260619T042826Z.log
    completed_steps=1013
    heldout_eval split=val val_loss=4.772994 train_elapsed_s=300.497
  L2 d1024/head16 candidate:
    target/fixed_time_val_d1024_l2_h16_b4_300s_20260619T044108Z.log
    completed_steps=2156
    heldout_eval split=val val_loss=4.973414 train_elapsed_s=300.248
quality:
  The d1024 run was finite=true and nonzero=true at every logged step, but it
  did not beat held-out validation loss.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  d1024 300-second direct GPU run: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Same-tokenizer architecture cut from 2 layers to 1 layer.
status: rejected, faster but worse held-out validation
decision:
  Revert to the L2 candidate. L1 completed many more steps in the same
  wall-clock budget, but validation loss was worse, so capacity is now winning
  over step count at this boundary.
changes:
  Tested GPT2_N_LAYER=1 with AURORA_MATRIX_PHASES=4. Tokenizer, dataset,
  sequence length, width, MLP size, batch size, and optimizer math stayed the
  same.
result:
  Current L2 B4 reference:
    target/fixed_time_val_wide1536_l2_b4_phase8_300s_20260619T042826Z.log
    completed_steps=1013
    heldout_eval split=val val_loss=4.772994 train_elapsed_s=300.497
  L1 B4 candidate:
    target/fixed_time_val_wide1536_l1_b4_phase4_300s_20260619T043429Z.log
    completed_steps=1798
    heldout_eval split=val val_loss=4.853128 train_elapsed_s=300.214
quality:
  The L1 run was finite=true and nonzero=true at every logged step, but it did
  not beat held-out validation loss at the fixed wall-clock budget.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  L1 300-second direct GPU run: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Same-tokenizer architecture cut from 4 layers to 2 layers.
status: success, keep L2 for current five-minute validation target
decision:
  Keep GPT2_N_LAYER=2 and AURORA_MATRIX_PHASES=8 for the current Shakespeare
  five-minute optimization loop. This preserves tokenizer, dataset, sequence
  length, width, MLP size, batch size, and optimizer math.
motivation:
  The L4 profile still had Aurora as the dominant kernel:
    target/nsys/current_l4_b4_20_20260619T042709Z_kernel_sum.csv
    aurora_mega_update_cooperative_kernel 7.958s total over 20 steps,
    397.908ms avg, 71.7% of GPU kernel time.
  Reducing depth again halves the number of Aurora-updated matrix weights.
  L2 has 8 Aurora matrix slots, so the phase count must be 8 rather than 16.
result:
  Previous L4 B4:
    target/fixed_time_val_wide1536_l4_b4_300s_20260619T042125Z.log
    completed_steps=540
    heldout_eval split=val val_loss=5.098733 train_elapsed_s=300.616
  Initial L2 run with AURORA_MATRIX_PHASES=16 failed before training:
    target/fixed_time_val_wide1536_l2_b4_300s_20260619T042759Z.log
    assertion left=8 right=0 in aurora_mega assert.
  L2 B4 with AURORA_MATRIX_PHASES=8:
    target/fixed_time_val_wide1536_l2_b4_phase8_300s_20260619T042826Z.log
    completed_steps=1013
    heldout_eval split=val val_loss=4.772994 train_elapsed_s=300.497
quality:
  The run was finite=true and nonzero=true at every logged step.
  This is a held-out validation improvement at the same wall-clock budget.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  L2 300-second direct GPU run: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Same-tokenizer architecture cut from 8 layers to 4 layers.
status: success, keep L4 for current five-minute validation target
decision:
  Keep GPT2_N_LAYER=4 for the current Shakespeare five-minute optimization
  loop. This does not change tokenizer, dataset, sequence length, width, MLP
  size, batch size, or optimizer. It reduces the number of transformer blocks
  and therefore the number of Aurora-updated matrix weights.
motivation:
  Current B4 L8 nsys showed Aurora dominating GPU kernel time:
    target/nsys/current_b4_20_20260619T040750Z_kernel_sum.csv
    aurora_mega_update_cooperative_kernel 10.401s total over 20 steps,
    520.032ms avg, 65.0% of GPU kernel time.
  Layer count directly multiplies QKV, c_proj, MLP up, and MLP down matrix
  updates, so reducing depth is an architecture-level way to improve useful
  optimizer steps per fixed wall-clock budget while preserving the tokenizer.
result:
  Previous L8 B4 default:
    target/fixed_time_val_wide1536_l8_b4_default_lr1p5_300s_20260619T034639Z.log
    completed_steps=373
    heldout_eval split=val val_loss=5.219253 train_elapsed_s=301.241
  New L4 B4 fixed 300-second run:
    target/fixed_time_val_wide1536_l4_b4_300s_20260619T042125Z.log
    completed_steps=540
    heldout_eval split=val val_loss=5.098733 train_elapsed_s=300.616
quality:
  The run was finite=true and nonzero=true at every logged step.
  This is a real held-out validation improvement at the same wall-clock budget,
  not a throughput-only result.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  L4 300-second direct GPU run: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Current B4 nsys profile and rejected Aurora geometry tests.
status: no code change kept
decision:
  Do not change Aurora phase geometry from the current 180 blocks and 16
  phases. The attempted lower-barrier schedules made the dominant optimizer
  kernel slower.
current_profile:
  Current B4 20-step nsys:
    target/nsys/current_b4_20_20260619T040750Z.nsys-rep
    target/nsys/current_b4_20_20260619T040750Z_kernel_sum.csv
    aurora_mega_update_cooperative_kernel 10.401s total, 520.032ms avg,
    65.0% of GPU kernel time.
  Next largest kernels in that profile:
    linear_backward_projection_cta_device_scale_kernel 1.243s total.
    fp32_to_nvfp4_ms_eden_device_scale_kernel 0.955s total.
    causal_attention_kernel 0.577s total.
rejected:
  AURORA_MATRIX_PHASES=8 with 180 blocks failed launch:
    target/aurora_phase8_b4_20_20260619T040928Z.log
    DriverError(720, "too many blocks in cooperative launch")
  AURORA_MATRIX_PHASES=8 with 90 blocks launched but was slower:
    target/nsys/aurora_phase8_blocks90_b4_20_20260619T040953Z_kernel_sum.csv
    aurora_mega_update_cooperative_kernel 12.017s total, 600.849ms avg.
  AURORA_ACTIVE_MATRICES=3 with 120 blocks required an inactive-barrier path
  for partial final phases and passed the serial Aurora recurrence tests, but
  was slower:
    target/nsys/aurora_active3_blocks120_b4_20_20260619T041708Z_kernel_sum.csv
    aurora_mega_update_cooperative_kernel 11.859s total, 592.963ms avg.
  AURORA_ACTIVE_MATRICES=2 with 120 blocks was slower:
    target/nsys/aurora_active2_blocks120_b4_20_20260619T041757Z_kernel_sum.csv
    aurora_mega_update_cooperative_kernel 14.636s total, 731.778ms avg.
notes:
  The failed active-matrix support code was reverted. The next useful speed
  work should target algorithmic kernel cost, not more Aurora launch geometry.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Fixed 5-minute LR-scale comparison for wide Llama 2 B4.
status: TRAIN_LR_SCALE=1.5 wins among tested default, 1.5, and 2.0.
decision:
  Set the default shared LR multiplier to 1.5. It improved held-out validation
  loss at the same 300-second budget without reducing completed steps.
changes:
  Tested B4 wide shape with TRAIN_LR_SCALE=1.5 and TRAIN_LR_SCALE=2.0.
  The code default is now 1.5 while TRAIN_LR_SCALE remains an explicit override.
result:
  Default scale 1.0:
    target/fixed_time_val_wide1536_l8_b4_300s_20260619T025909Z.log
    completed_steps=372
    heldout_eval split=val val_loss=5.335806 train_elapsed_s=301.131
  TRAIN_LR_SCALE=1.5:
    target/fixed_time_val_wide1536_l8_b4_lr1p5_300s_20260619T033539Z.log
    completed_steps=373
    heldout_eval split=val val_loss=5.220257 train_elapsed_s=300.787
  TRAIN_LR_SCALE=2.0:
    target/fixed_time_val_wide1536_l8_b4_lr2p0_300s_20260619T034049Z.log
    completed_steps=372
    heldout_eval split=val val_loss=5.264327 train_elapsed_s=300.986
  New default scale 1.5, no TRAIN_LR_SCALE override:
    target/fixed_time_val_wide1536_l8_b4_default_lr1p5_300s_20260619T034639Z.log
    completed_steps=373
    heldout_eval split=val val_loss=5.219253 train_elapsed_s=301.241
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  All fixed-time GPU runs completed with finite=true and nonzero=true in the
  logged steps. The final run used no TRAIN_LR_SCALE override and reported
  adam_lr=6e-5 and aurora_lr=3e-5 at step 0, confirming the default path.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Fixed 5-minute validation-loss comparison for wide Llama 2 B3.
status: valid run, worse than B4
decision:
  Do not keep B3 as the current candidate. B4 remains best among B2, B3, B4,
  and B8 on the fixed 300-second held-out validation metric.
changes:
  GPT2_BATCH_SIZE changed from 2 to 3 for the experiment. Chunked validation
  remained enabled so the same four held-out windows were evaluated.
result:
  B3 300-second run:
    target/fixed_time_val_wide1536_l8_b3_300s_20260619T032748Z.log
    completed_steps=407
    heldout_eval split=val val_loss=5.371977 train_elapsed_s=300.955
comparison:
  B4 fixed 300-second val_loss=5.335806.
  B3 fixed 300-second val_loss=5.371977.
  B2 fixed 300-second val_loss=5.394615.
  B8 fixed 300-second val_loss=5.438506.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  B3 300-second direct GPU run: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Fixed 5-minute validation-loss comparison for wide Llama 2 B2.
status: valid run after chunked validation fix, worse than B4
decision:
  Do not keep B2 as the current candidate. It beats B8 on this run but does not
  beat B4 on held-out validation loss at the same 300-second budget.
changes:
  Validation now evaluates the fixed four-window held-out slice in chunks no
  larger than GPT2_BATCH_SIZE. This avoids overflowing fixed GPU buffers when
  the training batch size is below the held-out window count.
  GPT2_BATCH_SIZE changed from 4 to 2 for the experiment.
result:
  Initial B2 run trained to the wall-clock stop but failed final validation:
    target/fixed_time_val_wide1536_l8_b2_300s_20260619T031557Z.log
    DriverError(700, "an illegal memory access was encountered")
  B2 rerun after chunked validation:
    target/fixed_time_val_wide1536_l8_b2_300s_chunkedval_20260619T032212Z.log
    completed_steps=450
    heldout_eval split=val val_loss=5.394615 train_elapsed_s=300.692
comparison:
  B4 fixed 300-second val_loss=5.335806.
  B8 fixed 300-second val_loss=5.438506.
  B2 lands between B4 and B8, so B4 remains best among tested batch sizes.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  B2 300-second direct GPU run with chunked validation: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Fixed 5-minute validation-loss comparison for wide Llama 2 B4 vs B8.
status: B4 wins this fixed-wall-clock validation run
decision:
  Use validation loss at fixed wall-clock as the ranking signal. Token
  throughput is only diagnostic. For this 300-second Shakespeare check, B4 is
  the better current candidate despite lower token throughput.
changes:
  Training now defaults to TRAIN_MAX_SECONDS=300 with TRAIN_STEPS acting only
  as a large safety cap when unset.
  The run directory label uses the time budget instead of a step count.
  Final held-out validation evaluation always runs and prints heldout_eval.
  Validation uses a fixed four-window held-out batch independent of training
  batch size, so B4 and B8 compare against the same validation slice.
result:
  B4 300-second run:
    target/fixed_time_val_wide1536_l8_b4_300s_20260619T025909Z.log
    completed_steps=372
    heldout_eval split=val val_loss=5.335806 train_elapsed_s=301.131
  B8 300-second run:
    target/fixed_time_val_wide1536_l8_b8_300s_20260619T030431Z.log
    completed_steps=275
    heldout_eval split=val val_loss=5.438506 train_elapsed_s=301.753
comparison:
  B4 beat B8 by 0.102700 validation loss at the same wall-clock budget.
  The previous fixed-step comparison was misleading for ranking because it used
  train loss and did not normalize by wall-clock time.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  B4 300-second direct GPU run: pass.
  B8 300-second direct GPU run: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Drop wide Llama 2 model batch size from B8 to B4.
status: valid sustained runs, stable but lower throughput
decision:
  Do not call B4 a win yet. It is stable and gets close to the B8 loss at the
  same wall-clock time, but B8 still has better token throughput and slightly
  better final fixed-wall-clock loss in this check.
changes:
  GPT2_BATCH_SIZE changed from 8 to 4 while keeping d_model=1536, mlp=6144,
  layers=8, heads=12, and AURORA_COOPERATIVE_BLOCKS=180.
result:
  B4 100-step Shakespeare run:
    target/llama2_wide1536_l8_b4_aurora180_100steps_20260619T025109Z.log
    step 0 loss=10.448714, step 99 loss=5.905748, elapsed_s=80.111,
    finite=true and nonzero=true throughout logged steps.
  B4 equal-wall-clock run against the B8 100-step time:
    target/llama2_wide1536_l8_b4_aurora180_135steps_equalwall_20260619T025245Z.log
    step 0 loss=10.448714, step 134 loss=5.742118, elapsed_s=108.950,
    finite=true and nonzero=true throughout logged steps.
comparison:
  B8 wide run with Aurora 180:
    step 99 loss=5.686881, elapsed_s=108.926, tokens=819,200,
    throughput=7,521 tokens/s.
  B4 100-step run:
    tokens=409,600, throughput=5,113 tokens/s.
  B4 equal-wall-clock run:
    tokens=552,960, throughput=5,075 tokens/s.
  B4 is not unstable, but it does not beat B8 on this fixed wall-clock check.
  Its lower batch gives more optimizer steps per minute, but it gives up too
  much token throughput in the current kernels.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  B4 100-step direct GPU training run: pass.
  B4 135-step equal-wall-clock direct GPU training run: pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Widen model to d_model=1536, mlp=6144, layers=8, heads=12.
status: valid sustained run, better fixed-step loss, still slower wall-clock
decision:
  Keep as an active candidate only if equal-wall-clock validation improves.
  The shape learns faster per step on Shakespeare. The first launch sizing
  pass narrowed the wall-clock gap, but it is still slower than the d_model=768,
  layers=12 baseline at fixed step count.
changes:
  GPT2_N_EMBD changed from 768 to 1536.
  GPT2_MLP remains 4 * d_model, so it changed from 3072 to 6144.
  GPT2_N_LAYER changed from 12 to 8.
  GPT2_N_HEAD stayed 12, giving head_dim=128. This widens the linear layers
  without doubling causal-attention square buffers via 24 heads.
  AURORA_COOPERATIVE_BLOCKS changed from 120 to 180 after profiling the wide
  model's Aurora cooperative launch.
result:
  Initial B8 100-step Shakespeare run:
    target/llama2_wide1536_l8_b8_100steps_20260619T023718Z.log
    step 0 loss=10.456142, step 99 loss=5.690090, elapsed_s=129.361,
    finite=true and nonzero=true throughout logged steps.
  After Aurora cooperative block sizing to 180:
    target/llama2_wide1536_l8_b8_aurora180_100steps_20260619T024448Z.log
    step 0 loss=10.456142, step 99 loss=5.686881, elapsed_s=108.926,
    finite=true and nonzero=true throughout logged steps.
profiling:
  Wide B8 20-step nsys, AURORA_COOPERATIVE_BLOCKS=120:
    target/nsys_wide1536_l8_b8_20_20260619T024112Z.sqlite
    aurora_mega_update_cooperative_kernel avg=729.840785ms.
  Wide B8 20-step nsys, AURORA_COOPERATIVE_BLOCKS=160:
    target/nsys_wide1536_l8_b8_aurora160_20_20260619T024254Z.sqlite
    aurora_mega_update_cooperative_kernel avg=542.993246ms.
  Wide B8 20-step nsys, AURORA_COOPERATIVE_BLOCKS=180:
    target/nsys_wide1536_l8_b8_aurora180_20_20260619T024339Z.sqlite
    aurora_mega_update_cooperative_kernel avg=521.137870ms.
  AURORA_COOPERATIVE_BLOCKS=192 failed at runtime with:
    DriverError(720, "too many blocks in cooperative launch").
comparison:
  d_model=768, layers=12, Llama 2 tokenizer baseline:
    step 99 loss=6.439928, elapsed_s=50.288.
  Wider shape with the 180-block Aurora launch improves fixed-step loss by
  0.753047 at step 99, but still takes 2.17x longer for the same number of
  steps. This must be judged by held-out loss at fixed wall-clock time, not
  fixed step count.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  B8 100-step direct GPU training run: pass.
notes:
  loss_sync_ms rose to roughly 396ms in the logged wide run. That may be a
  synchronization/reporting artifact, but it is real wall-clock cost in the
  current training loop and needs profiling before calling the wider shape a
  speed win. The first useful optimization target was Aurora launch sizing, not
  abandoning the wide shape.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Switch active training tokenizer from GPT-2 BPE to Llama 2 32k BPE.
status: success, sustained runtime check passed
decision:
  Keep as the next baseline. This reduces the tied embedding/lm-head/logits
  path from vocab 50,257 to vocab 32,000 without adding a hidden GPT-2 fallback.
changes:
  Added llama2-tokenizer crate with vendored NousResearch Llama 2 tokenizer
  artifacts.
  GPT2_VOCAB_SIZE now derives from llama2_tokenizer::VOCAB_SIZE = 32,000.
  SYNTH prep and Shakespeare prep use Llama2Tokenizer directly.
  Shard names changed to synth_llama2_* and shakespeare_llama2_* so old GPT-2
  token shards are not silently reused.
  Generation now encodes/decodes with Llama 2 and pads generation windows with
  the Llama 2 EOS token.
result:
  Local Shakespeare Llama 2 shard was generated:
    data/shakespeare/shards/shakespeare_llama2_train_000000.bin
    tokens=368,634.
  B8 logits per step dropped from the GPT-2-tokenizer shape's 411,705,344 to
  262,144,000.
  B8 100-step Shakespeare run:
    target/llama2_tokenizer_b8_100steps_20260619T023033Z.log
    step 0 loss=10.394513, step 99 loss=6.439928, elapsed_s=50.217,
    finite=true and nonzero=true throughout logged steps.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo test -p llama2-tokenizer: pass
  cargo oxide build --arch sm_120a: pass
  B8 100-step direct GPU training run: pass.
notes:
  The old GPT-2 BPE crate was removed from the workspace and source tree. The
  active app, SYNTH prep, generation, and model-vocab path use Llama 2 only.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Rename Aurora mega scratch contract from scaled to polar_next.
status: success, behavior-preserving fusion-audit cleanup
decision:
  Keep. The buffer is not a standalone scaled matrix output in the fused
  Aurora path; it is the Polar Express ping-pong/next buffer. The rename makes
  the remaining irreducible global scratch state explicit when auditing future
  in-place memory reductions.
changes:
  Renamed AuroraMegaUpdateArgs.scaled, AuroraScratchBuffers.scaled, and the
  fused kernel parameter to polar_next.
  No optimizer math, launch geometry, schedule, or batch-size behavior changed.
evidence:
  rust_kernels_cuda.ptx has one visible Aurora entry:
    aurora_mega_update_cooperative_kernel.
  Helper Aurora symbols are emitted as device functions beneath that entry,
  not separate host launches.
  target/aurora_polar_next_rename_b8_100steps_20260619T020807Z.log
quality:
  B8 100-step Shakespeare run remained finite and nonzero throughout logged
  steps. Loss went from 10.808983 at step 0 to 6.406268 at step 99 in 53.166s,
  matching the prior B8 baseline shape.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass
  B8 100-step direct GPU training run: pass
notes:
  This does not remove a grid::sync boundary. It only makes the remaining
  scratch ownership accurate before deeper in-place fusion work.
```

Throughput target:

- Reach at least 100k tokens/s sustained without hurting fixed-budget held-out
  validation loss.
- This makes a 10B-token run day-scale instead of week-scale.

Quality canary before SYNTH:

- Shakespeare first. A roughly 3-minute run should reach a held-out validation
  loss and generated sample quality in the same qualitative class as the
  Karpathy baby-GPT Shakespeare example.
- For this canary, generated text quality matters alongside held-out validation
  loss. Kernel timing alone is not evidence that the learner is healthy.
- Do not move optimization focus to SYNTH until this canary produces legible
  Shakespeare-style text under the wall-clock budget.

## Current Aurora Mega-Kernel Dependency Map

The active Aurora optimizer route is one cooperative launch:

```text
aurora_mega_update_cooperative_kernel
```

Current per-matrix body:

1. `momentum_orient`
   - Reads FP32 gradients and momentum.
   - Writes updated momentum and oriented Nesterov matrix.
   - Requires a grid sync before Polar consumes the complete oriented matrix.

2. Optional rectangular `row_balance`
   - Runs only when the oriented matrix is rectangular.
   - Writes row-balanced `scaled` scratch and per-block Frobenius chunks.
   - Requires a grid sync before global normalization consumes all chunks.

3. Initial Polar normalization
   - Reduces Frobenius chunks into one inverse norm.
   - Writes initial Polar `x` buffer.
   - Requires a grid sync before the first Polar TC stage consumes the full
     normalized matrix.

4. Polar Express iterations
   - `G = X X^T`: TC matmul. The current path computes only upper CTA tiles and
     mirrors off-diagonal stores.
   - `AX = G X`: TC matmul.
   - `X_next = a X + b AX + c G AX`: TC-backed polynomial update.
   - These three stages have true data dependencies and currently require
     grid-wide visibility between stages.

5. `update_master_chunks`
   - Applies Aurora update to FP32 master `z`/`x` weights.
   - Produces per-block amax chunks for the following quantization.
   - Requires a grid sync before global scale reduction consumes all chunks.

6. `reduce_global_scale`
   - Reduces per-block amax chunks and writes one global NVFP4 scale.
   - Requires a grid sync before per-group NVFP4 encoding reads that scale.

7. `encode_four_six`
   - Re-encodes FP32 master `x` into NVFP4 bytes/local scales using 4/6
     selection.

Current judgement:

- The old multi-launch Aurora path has been replaced by the cooperative mega
  kernel.
- The remaining high-cost irreducible-looking region is the dependent Polar
  Express TC chain.
- Resource-sensitive attempts that add mapping/control code inside the
  cooperative launch can fail real training launch with DriverError(720), even
  when small optimizer tests pass.
- Experiments that change optimizer semantics, such as reducing Polar Express
  iteration count, are not kernel optimizations. They can be recorded as
  diagnostics, but they are invalid unless the explicit experiment is optimizer
  quality tuning under the held-out wall-clock objective.

## Current Baseline

```text
date: 2026-06-18
commit: uncommitted
experiment: Shrink Aurora polar_ax scratch to square-phase size.
status: kept as scratch cleanup; tiny favorable profile move
decision:
  Keep the separate max_ax_len contract. Rectangular Aurora phases already use
  scaled scratch as AX, so polar_ax is only consumed by the square c_proj phase.
  It does not need max_matrix scratch capacity.
changes:
  Added max_ax_len to AuroraMegaUpdateArgs and the cooperative launch contract.
  polar_ax offsets now use max_ax_len.
  Training scratch allocates polar_ax as GPT2_N_EMBD x GPT2_N_EMBD instead of
  GPT2_MLP x GPT2_N_EMBD.
result:
  Training behavior is unchanged. The kernel profile moved slightly in the
  favorable direction, but this should be treated primarily as scratch-memory
  cleanup.
performance:
  Remove-amax profile: aurora_mega_update_cooperative_kernel 111.582ms avg.
  Shrink-polar-ax profile: aurora_mega_update_cooperative_kernel 111.521ms avg.
evidence:
  target/nsys_aurora_remove_amax_scratch_b8_20_20260618T230914Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_shrink_polar_ax_b8_20_20260618T231344Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/aurora_shrink_polar_ax_100steps_20260618T231243Z.log
  target/runs/20260618_231244Z_shakespeare_100steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438527
  20-step nsys GPU training profile: pass
notes:
  This reduces live scratch memory for the fused Aurora route. It does not
  change the Polar Express TC dependency chain.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Remove Aurora mega tensor-amax scratch/output from the public
  kernel contract.
status: kept as contract cleanup; no speedup
decision:
  Keep the removal. Training does not consume this amax output; the actual
  quantized weight state is carried by bytes, local scales, and global scale.
  Tests now validate the global scale directly instead of reading the removed
  scratch output.
changes:
  Removed amax from AuroraMegaUpdateArgs, the cooperative kernel signature,
  launcher assertions, training scratch allocation, and Aurora optimizer tests.
  reduce_global_scale now writes only out_global_scale.
result:
  Training behavior is unchanged. Profile is effectively flat and slightly
  worse in this run, so this should not be counted as a performance win.
performance:
  Encode-scale-once profile: aurora_mega_update_cooperative_kernel 111.440ms
  avg.
  Remove-amax profile: aurora_mega_update_cooperative_kernel 111.582ms avg.
evidence:
  target/nsys_aurora_encode_scale_once_b8_20_20260618T230459Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_remove_amax_scratch_b8_20_20260618T230914Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/aurora_remove_amax_scratch_100steps_20260618T230813Z.log
  target/runs/20260618_230813Z_shakespeare_100steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438527
  20-step nsys GPU training profile: pass
notes:
  This reduces host/device contract surface and scratch allocation. It is not a
  kernel-time optimization.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Load Aurora encode global scale once per thread instead of once per
  quantization group.
status: kept; tiny speedup
decision:
  Keep the cleanup. reduce_global_scale publishes one global scale before the
  encode phase, so each thread can load it once before walking its assigned
  groups.
changes:
  Moved the out_global_scale load out of encode_four_six's group loop.
result:
  Training behavior is unchanged. The profile shows a very small Aurora
  mega-kernel improvement.
performance:
  Symmetric Gram profile: aurora_mega_update_cooperative_kernel 111.557ms avg.
  Encode-scale-once profile: aurora_mega_update_cooperative_kernel 111.440ms
  avg.
evidence:
  target/nsys_aurora_symmetric_gram_b8_20_20260618T225411Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_encode_scale_once_b8_20_20260618T230459Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/aurora_encode_scale_once_100steps_20260618T230359Z.log
  target/runs/20260618_230359Z_shakespeare_100steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438527
  20-step nsys GPU training profile: pass
notes:
  This is not a major fusion win. It is a low-risk cleanup in the existing
  quantization tail.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Write rectangular row-balanced values directly into transposed
  Polar x and normalize x in place.
status: rejected; real cooperative training launch fails
decision:
  Keep the existing rectangular row_balance -> scaled scratch ->
  normalize_source_to_x_from_chunks path. The direct-to-x route is a reasonable
  dataflow idea, but it changes the cooperative kernel enough that the real
  training launch fails.
attempt:
  Added a prebalanced-x Polar entry point.
  Row-balance wrote row-normalized values directly into polar_x in transposed
  layout.
  A new in-place global scaling pass normalized polar_x from the existing
  Frobenius chunks.
result:
  The path compiled and passed the small GPU optimizer tests, but the real
  100-step training launch failed before step 0 with DriverError(720, "too many
  blocks in cooperative launch").
evidence:
  target/aurora_rowbalance_to_x_100steps_20260618T230112Z.log
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: failed at launch
notes:
  This reinforces that the current 32-block cooperative launch is near a device
  resource boundary. Dataflow changes that add code/register pressure can fail
  the real launch even when smaller optimizer tests pass.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Compact upper-triangle CTA scheduling for the symmetric Polar Gram
  stage.
status: rejected; real cooperative training launch fails
decision:
  Keep the branch-skip symmetric scheduler. It computes only upper-triangle
  Gram tiles but still iterates over the square tile index space. Attempts to
  compact the tile index space changed device resource usage enough to fail the
  real cooperative training launch.
attempt:
  First tried inverse-square-root mapping from compact triangular index to
  tile_row/tile_col. Then tried a small integer row-width walk to avoid sqrt.
result:
  Both variants built and passed the small GPU optimizer tests, but failed the
  real 100-step training launch before step 0 with DriverError(720, "too many
  blocks in cooperative launch").
evidence:
  target/aurora_compact_triangular_gram_100steps_20260618T225656Z.log
  target/aurora_compact_triangular_integer_100steps_20260618T225733Z.log
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: failed at launch for both
  compact mapping variants
notes:
  This is a useful negative result: the real cooperative launch constraint is
  sensitive to device code shape, and the small optimizer tests are not enough
  evidence for launch viability.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Compute only upper CTA tiles for Polar Express Gram matrix and
  mirror off-diagonal stores.
status: kept; meaningful Aurora speedup
decision:
  Keep the symmetric Gram path. The first Polar stage computes
  G = X X^T, so off-diagonal CTA tiles were being computed twice. The new path
  computes only tile_col >= tile_row and writes the mirrored tile for
  off-diagonal results.
changes:
  Added run_symmetric_tiles for the Gram stage only.
  Added store_plain_transposed for mirrored off-diagonal Gram stores.
  Left the later G X and polynomial update stages on the existing full tile
  path because they are not symmetric.
result:
  Training behavior remains stable. The Aurora mega kernel drops by roughly
  9.56ms average versus the restored phase-contract profile.
performance:
  Rectangular AX alias + phase-contract profile:
  aurora_mega_update_cooperative_kernel 121.116ms avg.
  Symmetric Gram profile:
  aurora_mega_update_cooperative_kernel 111.557ms avg.
evidence:
  target/nsys_aurora_rect_alias_phase_contract_b8_20_20260618T225035Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_symmetric_gram_b8_20_20260618T225411Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/aurora_symmetric_gram_100steps_20260618T225310Z.log
  target/runs/20260618_225311Z_shakespeare_100steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438527
  20-step nsys GPU training profile: pass
notes:
  This is a real algebra-aware reduction of redundant TC work inside the
  cooperative mega kernel, not just launch fusion.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Test 16 cooperative blocks per matrix with 2 matrix phases
  (24 active matrices) instead of 32 blocks with 4 phases (12 active matrices).
status: rejected; cooperative launch fails
decision:
  Keep the 32-block, 4-phase layout. The 16x24 layout builds and passes the
  small GPU optimizer tests after scratch sizing is derived from phase count,
  but the real training launch fails on this GPU.
attempt:
  Shared AURORA_MATRIX_PHASES between host launcher and device kernel.
  Derived Aurora scratch active slots from total_matrix_slots / phases.
  Set AURORA_COOPERATIVE_BLOCKS = 16 and AURORA_MATRIX_PHASES = 2.
result:
  The training launch returns DriverError(720, "too many blocks in cooperative
  launch") before step 0 completes.
evidence:
  target/aurora_16blocks_2phase_100steps_20260618T224848Z.log
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: failed at launch with
  DriverError(720, "too many blocks in cooperative launch")
notes:
  Total block count alone is not the whole launch constraint here. More active
  matrices in the y dimension is not a viable path on this machine.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Alias rectangular Aurora Polar AX scratch onto dead row-balance
  scratch.
status: kept; small speedup and reduced scratch pressure
decision:
  Keep the rectangular-only alias. After row_balance and Polar normalization
  consume scaled_ptr as the balanced source, rectangular slots no longer need
  that buffer. Reusing it for the later AX temporary removes one live scratch
  region without changing the Polar dependency graph.
changes:
  run_polar_step passes scaled_ptr as polar_ax only for rectangular matrices.
  Square matrices still keep polar_ax separate because scaled_ptr is used as the
  Polar ping-pong target there.
result:
  Training behavior is unchanged. The profile shows a small but measurable
  Aurora mega-kernel improvement.
performance:
  Square in-place Polar-x profile: aurora_mega_update_cooperative_kernel
  121.219ms avg.
  Rectangular AX alias profile: aurora_mega_update_cooperative_kernel
  120.878ms avg.
evidence:
  target/nsys_aurora_square_inplace_polar_x_b8_20_20260618T223923Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_rect_alias_polar_ax_b8_20_20260618T224506Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/aurora_rect_alias_polar_ax_100steps_20260618T224402Z.log
  target/runs/20260618_224402Z_shakespeare_100steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438509
  20-step nsys GPU training profile: pass
notes:
  This is still scratch/dataflow cleanup inside the mega kernel, not a deeper
  Polar Express algebra fusion.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Reuse oriented as the initial Polar buffer for square Aurora
  matrices.
status: kept; tiny speedup and valid scratch-write reduction
decision:
  Keep the square-only in-place normalization path. Rectangular matrices still
  use the separate polar_x buffer because their normalization transposes the
  balanced source, so source/destination aliasing would corrupt reads.
changes:
  run_polar_step now passes oriented_ptr as x_ptr for square matrices.
  Rectangular matrices keep using polar_x as a separate destination.
result:
  Training behavior is unchanged. The profile moves only slightly, but the data
  flow is more accurate: square slots do not write the initial normalized
  Polar source to a separate scratch buffer.
performance:
  No-post-polar-sync profile: aurora_mega_update_cooperative_kernel
  121.232ms avg.
  Square in-place Polar-x profile: aurora_mega_update_cooperative_kernel
  121.219ms avg.
evidence:
  target/nsys_aurora_no_post_polar_sync_b8_20_20260618T223452Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_square_inplace_polar_x_b8_20_20260618T223923Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/aurora_square_inplace_polar_x_100steps_20260618T223820Z.log
  target/runs/20260618_223821Z_shakespeare_100steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438527
  20-step nsys GPU training profile: pass
notes:
  This is a narrow in-place memory cleanup, not a major throughput win.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Remove redundant body-level grid sync after Polar Express returns.
status: kept as sync-graph cleanup; performance flat
decision:
  Keep the removal. polar_express_from_source_ptr always returns after a
  grid-wide sync, so the caller-side sync before update_master_chunks was
  duplicate. Removing it makes the mega-kernel dependency graph more accurate.
changes:
  Removed the extra grid::sync after run_polar_step in aurora_matrix_update_body.
  Added a local comment documenting that Polar Express returns after a grid-wide
  sync.
result:
  Correctness and training behavior are unchanged. Profile is effectively flat.
performance:
  Balance/norm fused profile: aurora_mega_update_cooperative_kernel
  121.221ms avg.
  No-post-polar-sync profile: aurora_mega_update_cooperative_kernel
  121.232ms avg.
evidence:
  target/nsys_aurora_balancenorm_fused_b8_20_20260618T222550Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_no_post_polar_sync_b8_20_20260618T223452Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/aurora_remove_post_polar_sync_100steps_20260618T223350Z.log
  target/runs/20260618_223350Z_shakespeare_100steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438710
  20-step nsys GPU training profile: pass
notes:
  This is not a speed win. It is still useful because it removes one false
  boundary from the mega-kernel reasoning model.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Try fusing square-matrix momentum/orient with initial Polar Express
  Frobenius norm chunk generation.
status: rejected; reverted
decision:
  Do not keep this path. It is mathematically valid, but it adds reduction work
  during momentum/orient and did not improve the measured mega-kernel time.
attempt:
  For square matrices that do not need row balancing, momentum_orient computed
  the same per-block Frobenius chunks consumed by Polar normalization. This
  avoided a later full scan of the oriented square source.
result:
  Correctness and 100-step training were fine, but the profile was slightly
  worse than the kept balance/norm fusion baseline.
performance:
  Balance/norm fused profile: aurora_mega_update_cooperative_kernel
  121.221ms avg.
  Momentum/norm fused profile: aurora_mega_update_cooperative_kernel
  121.250ms avg.
evidence:
  target/nsys_aurora_balancenorm_fused_b8_20_20260618T222550Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_momentumnorm_fused_b8_20_20260618T223025Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/aurora_momentumnorm_fused_100steps_20260618T222924Z.log
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438710
  20-step nsys GPU training profile: pass
notes:
  This establishes that not every mathematically valid pass fusion is useful.
  The square-path normalization scan was not the dominant cost, and folding its
  reduction into momentum/orient did not pay for itself.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Fuse rectangular row-balance with initial Polar Express Frobenius
  norm chunk generation.
status: success; small speedup
decision:
  Keep the precomputed balanced-source chunk path. It removes a redundant full
  scan of the row-balanced matrix for rectangular Aurora matrices.
changes:
  row_balance now writes per-block Frobenius chunks for the balanced source.
  Polar normalization can consume those chunks directly instead of rescanning
  the balanced matrix to produce the same chunks.
result:
  Aurora math is unchanged. The normalized source still uses the same global
  Frobenius scale; the scale chunks are just produced by the row-balance pass.
performance:
  Prior 32-block profile: aurora_mega_update_cooperative_kernel 121.682ms avg.
  Fused balance/norm profile: aurora_mega_update_cooperative_kernel
  121.221ms avg.
evidence:
  target/nsys_aurora_32blocks_b8_20_20260618T221949Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_balancenorm_fused_b8_20_20260618T222550Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/runs/20260618_222449Z_shakespeare_100steps/loss.png
  target/aurora_balancenorm_fused_100steps_20260618T222449Z.log
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438710
  20-step nsys GPU training profile: pass
notes:
  This is a valid in-kernel memory-pass fusion before the Polar polynomial. It
  does not remove the irreducible-looking G = X X^T -> GX -> G(GX) dependency
  chain inside each Polar Express iteration.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Increase Aurora mega cooperative CTAs per matrix and make launch
  geometry use the shared AURORA_COOPERATIVE_BLOCKS contract.
status: success; keep 32 blocks per matrix
decision:
  Keep AURORA_COOPERATIVE_BLOCKS = 32. This is the best valid measured point.
  48 and 64 blocks per matrix fail cooperative launch on this GPU.
changes:
  Removed the stale local 8-block launch constant from the Aurora mega launcher.
  Routed launch geometry and scratch sizing through one shared
  AURORA_COOPERATIVE_BLOCKS contract.
result:
  32 blocks per matrix reduces serial tile walking inside the same mega kernel.
  It does not change Aurora math or remove the remaining Polar Express internal
  grid-sync boundaries.
performance:
  8 blocks:  aurora_mega_update_cooperative_kernel 339.969ms average.
  16 blocks: aurora_mega_update_cooperative_kernel 191.242ms average.
  32 blocks: aurora_mega_update_cooperative_kernel 121.682ms average.
  48 blocks: rejected, DriverError(720, "too many blocks in cooperative launch").
  64 blocks: rejected, DriverError(720, "too many blocks in cooperative launch").
evidence:
  target/nsys_reduce_util_aurora_b8_20_20260618T221147Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_16blocks_b8_20_20260618T221751Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_aurora_32blocks_b8_20_20260618T221949Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/runs/20260618_221848Z_shakespeare_100steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run at 32 blocks: pass, final loss
  6.438818
  20-step nsys GPU training profile at 32 blocks: pass
notes:
  This is scheduling/parallelism inside the mega kernel, not a deeper Polar
  Express algebra fusion. The remaining expensive body still materializes
  G = X X^T and GX = G X between internal cooperative syncs.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Move Aurora block reduction wrappers onto shared util reduction
  helpers.
status: cleanup only; performance unchanged
decision: keep the shared helper route because it removes duplicated reduction
  bodies without changing the Aurora route
changes:
  Added shared-array block sum/max helpers in utils/block_reduce.rs.
  Replaced Aurora-private duplicated block_sum/block_max implementations with
  wrappers around the shared helpers.
result:
  Checks and GPU optimizer tests still pass.
  100-step Shakespeare run still reaches final loss 6.438445.
performance:
  Fresh nsys: aurora_mega_update_cooperative_kernel 20 launches over 20 steps,
  6.799386408s total, 339.969ms average, 45.7% of kernel time.
  This is flat versus the prior mega-only profile at 339.461ms average.
evidence:
  target/runs/20260618_220955Z_shakespeare_100steps/loss.png
  target/nsys_reduce_util_aurora_b8_20_20260618T221147Z_kernel_sum_cuda_gpu_kern_sum.csv
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438445
  20-step nsys GPU training profile: pass
notes:
  The next speed target is still the body of aurora_mega_update_cooperative_kernel,
  not reduction ownership cleanup.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Remove the old standalone Aurora launcher/kernel path and make
  tests exercise only the mega Aurora update route.
status: success for route cleanup; performance unchanged
decision: keep only aurora_mega_update_cooperative_kernel for Aurora optimizer
  updates; the remaining work is reducing the cost of that kernel itself
changes:
  Removed the old standalone Aurora launcher/device path and stale fused-update
  argument type.
  Rewrote Aurora optimizer tests to call the mega update route directly.
  Split the Aurora test fixture into small files so the tested path is easier
  to audit.
result:
  Kernel profile now shows only aurora_mega_update_cooperative_kernel under
  Aurora-specific work.
performance:
  Mega-only nsys: aurora_mega_update_cooperative_kernel 20 launches over
  20 steps, 6.789227634s total, 339.461ms average, 45.9% of kernel time.
evidence:
  target/runs/20260618_220415Z_shakespeare_100steps/loss.png
  target/mega_only_aurora_100steps_20260618T220415Z.log
  target/nsys_mega_only_aurora_b8_20_20260618T220545Z_kernel_sum_cuda_gpu_kern_sum.csv
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438581
  20-step nsys GPU training profile: pass
notes:
  This removed confusing dead routes. It did not make Aurora faster because the
  actual mega kernel body still does the same Polar Express and writeback work.
```

Last measured AMUSE 2k Shakespeare run:

```text
batch_size = 8
seq_len = 1024
tokens_per_step = 8192
late-step time with logged loss sync ~= 0.72 s
late-step time without loss sync ~= 0.54 s
throughput excluding loss sync ~= 15k tokens/s
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Try replacing Polar Express left Gram with right Gram, then keep
  only the valid minimum-Gram scratch tightening.
status: mixed; right-Gram rewrite rejected, scratch tightening kept
decision: keep the original source-transposed left-Gram TC staging; keep
  smaller polar_gram scratch sizing and rectangular recurrence tests
attempt:
  The hypothesis was that Aurora was computing a large rows x rows Gram and
  could switch to X^T X. Inspection and tests showed this was wrong: the
  existing Polar Express path already transposes the source when rows > cols,
  so the left Gram is already on the smaller side.
result:
  The right-Gram rewrite failed rectangular GPU recurrence tests:
  tall 64x32 max update error was about 3.565e-3 before correcting scale.
  wide 32x64 max update error was about 2.620e-2, proving the rewrite changed
  the rectangular path.
  The rewrite was removed.
  Added tall and wide rectangular Aurora recurrence tests so this path is now
  covered.
  Tightened polar_gram scratch sizing from max dimension to min/oriented Gram
  dimension for the training model; this is a memory-footprint cleanup, not a
  runtime win.
performance:
  Min-Gram scratch profile: aurora_mega_update_cooperative_kernel 20 launches
  over 20 steps, 6.790835668s total, 339.542ms average.
  This is effectively flat versus prior block-amax-fused profile at 339.813ms.
evidence:
  target/runs/20260618_215344Z_shakespeare_100steps/loss.png
  target/nsys_mega_min_gram_scratch_b8_20_20260618T215509Z_kernel_sum_cuda_gpu_kern_sum.csv
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438581
  20-step nsys GPU training profile: pass
notes:
  The current Polar Express TC staging is already using the smaller Gram side.
  The remaining expensive boundary is the three dependent TC products per polar
  iteration. Those dependencies are algorithmic for a normal cooperative kernel:
  X X^T must be globally visible before G X, and G X before G(G X).
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Fuse Aurora matrix optimizer into one cooperative mega launch and
  remove reducible row-scale and chunk-amax scratch handoffs.
status: success for launch fusion; performance mostly unchanged
decision: keep as the active Aurora route; remaining work is Polar Express
  algorithm/kernel-shape optimization, not another host-launch cleanup pass
baseline:
  Previous grouped Aurora route used 4 Aurora launches per training step.
  The best prior grouped profile was about 80 launches over 20 steps with
  aurora_grouped_update_cooperative_kernel around 6.775s total, or about
  338.8ms per training step.
changes:
  Added aurora_mega_update_cooperative_kernel as one cooperative launch per
  training step.
  Structured the launch as 4 internal same-shape phases, each processing the
  12 transformer-block matrices in parallel along grid.y.
  Removed the unused grouped Aurora launcher/device branch after mega wiring.
  Fused row balance so each block scales its row immediately after the row-norm
  block reduction; this removed the Aurora row_scale buffer and one grid sync.
  Replaced per-chunk update amax scratch with per-block amax in polar_chunks;
  this removed Aurora chunk_amax scratch and one quantization reduction pass.
result:
  Naive all-48-slots-concurrent mega launch was rejected: it failed to reach
  step 10 after several minutes and was killed.
  Phase-based mega launch reached final loss 6.438581 on 100 Shakespeare steps.
  Row-scale-fused run reached final loss 6.434855 on 100 Shakespeare steps.
  Block-amax-fused run reached final loss 6.438581 on 100 Shakespeare steps.
performance:
  Phase-based mega nsys: aurora_mega_update_cooperative_kernel 20 launches over
  20 steps, 6.821945095s total, 341.097ms average.
  Row-scale-fused nsys: 20 launches, 6.794577591s total, 339.729ms average.
  Block-amax-fused nsys: 20 launches, 6.796254085s total, 339.813ms average.
  These cleanups reduced scratch and synchronization but did not materially
  reduce wall time.
evidence:
  target/runs/20260618_212527Z_shakespeare_100steps/loss.png
  target/runs/20260618_213400Z_shakespeare_100steps/loss.png
  target/runs/20260618_213958Z_shakespeare_100steps/loss.png
  target/nsys_mega_phase_aurora_b8_20_20260618T212650Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_mega_balance_fused_b8_20_20260618T213523Z_kernel_sum_cuda_gpu_kern_sum.csv
  target/nsys_mega_block_amax_fused_b8_20_20260618T214121Z_kernel_sum_cuda_gpu_kern_sum.csv
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass
  100-step Shakespeare direct GPU training run after each fused change: pass
  20-step nsys GPU training profile after each fused change: pass
notes:
  The current one-launch Aurora path still has unavoidable global barriers
  between momentum/orient, polar, update, and quantization because each stage
  consumes complete matrix-scale or tensor-scale results from the prior stage.
  Further meaningful speedup requires changing the Polar Express TC stage shape
  or algorithmic staging. Simple launch-count or scalar scratch cleanup is no
  longer the dominant lever.
```

10B-token ETA at current throughput:

```text
10,000,000,000 / 15,000 tokens/s ~= 7.7 days
```

Reference target comparison:

```text
24 hour target ~= 116k tokens/s
4 hour target  ~= 694k tokens/s
```

## Training Quality Snapshots

### Pre-AMUSE 2k Shakespeare

```text
final train loss = 2.639826
final val loss   = 4.240733
final adam_lr    = 2e-5
final aurora_lr  = 1e-5
```

### AMUSE 2k Shakespeare

```text
final train loss = 2.989065
final val loss   = 4.409284
final adam_lr    = 2e-4
final aurora_lr  = 1e-4
```

AMUSE did learn, but did not beat the older decayed run in this short 2k-step
Shakespeare test.

## Tried

- Full forward path through token embedding, layer norm, attention, MLP, final
  norm, tied LM head.
- Full backward chain wired through linear, attention, layer norm, MLP, residual,
  and token embedding gradient paths.
- Quartet/MS-EDEN quantization in backward GEMM operand paths.
- NVFP4 CTA projection rewrite for `linear_backward_projection_device_scale`.
- BF16/FP16 tensor-core attention-backward experiments; FP16 internal attention
  backward looked much closer numerically than NVFP4/NVFP8 for that path.
- Aurora/Muon-style matrix optimizer path with tensor-core helper kernels.
- Schedule-free AMUSE-style state:
  - materialize `Y = (1 - beta) Z + beta X` before forward/backward
  - update fast sequence `Z`
  - average inference sequence `X`
  - save/eval/generate from `X`
- Removed synthetic random data fallback from training.
- Shakespeare smoke and 2k experiments.
- Default dataset switched from FineWeb to PleIAs/SYNTH.
- Run artifacts now go to datetime-labelled run folders.
- Aurora rectangular power-balancing reduced from two passes to one pass:
  - 20-step Shakespeare smoke kept loss behavior comparable.
  - Aurora polar TC helper launch counts dropped from 20160 to 11520 over 20
    steps.

## Current Bottlenecks

From recent logs and profiling:

- Optimizer is still expensive, roughly 140 ms/step after one-pass Aurora
  rectangular power balancing.
- Aurora is roughly 115 ms/step after one-pass rectangular power balancing.
- Backward enqueue is roughly 240 ms/step.
- Loss sync is roughly 184 ms on logged steps, but this can be amortized by
  larger `TRAIN_LOG_INTERVAL`.
- Forward is small, roughly 1 ms/step, so optimizing forward alone is not the
  next priority.

Likely root causes:

- Too many helper kernels in optimizer/materialization/update paths.
- Too many global memory round trips for quantize, transpose, norm, polar,
  materialize, and writeback.
- Some backward matmul-heavy paths still need better CTA-level tiling and data
  reuse.
- Batch size is likely too small to amortize fixed per-step overhead.
- Tensor-core use is not yet dominant across the full training graph.

## Not Yet Tried

- Full batch-size sweep after AMUSE:
  - batch 8, 16, 32, and higher if memory allows
  - track tokens/s, memory use, loss behavior, and GPU power
- nsys profile on current AMUSE + SYNTH path.
- ncu SOL and instruction mix on the current top kernels.
- Fusion of schedule-free materialization with quantization/writeback where
  possible.
- Further fusion or removal of Aurora helper kernels.
- Replacing Aurora polar helper path with fewer, larger TC kernels.
- Larger CTA-tiled rewrites for remaining backward matmul-heavy paths.
- Capturing CUDA graph or equivalent launch-overhead reduction.
- Multi-step loss reduction on GPU so logging does not require copying the full
  per-token loss buffer.
- Longer SYNTH training curve and qualitative generation check.

## Near-Term Plan

1. Verify SYNTH shard creation on a limited slice or first parquet file.
2. Run a short SYNTH smoke and confirm loss is finite.
3. Profile the current AMUSE SYNTH training step with nsys.
4. Batch-size sweep for throughput and memory:
   - start at current batch 8
   - test 16 and 32
   - keep log interval large enough that loss sync does not dominate
5. Use nsys top kernels to pick one bottleneck at a time.
6. Attack optimizer/materialization overhead first unless profiling says
   backward matmul has overtaken it.
7. Re-run 2k-step smoke after each major kernel change and record:
   - train loss
   - validation loss
   - tokens/s
   - top kernel timings
   - generated sample path

## Experiment Log Template

```text
date:
commit:
experiment:
status: success | failure | inconclusive
decision:
baseline:
result:
quality:
performance:
evidence:
verification:
notes:
```

## Experiment Log

```text
date: 2026-06-18
commit: uncommitted
experiment: Add wall-clock-budgeted benchmark harness with held-out validation
  endpoint.
status: success
decision:
  Keep the harness. The optimization target is held-out validation loss after a
  fixed wall-clock training budget, so runs using TRAIN_MAX_SECONDS now emit a
  final heldout_eval line against the validation split.
baseline:
  Previous runs optimized and reported mostly fixed-step training loss plus
  nsys kernel timings. Those are diagnostics, not the actual target.
result:
  TRAIN_MAX_SECONDS stops after a completed training step and then evaluates the
  validation split without applying another update.
quality:
  This is measurement infrastructure only. It does not change model math.
performance:
  The endpoint eval is only run for TRAIN_MAX_SECONDS runs, so ordinary nsys
  kernel profiles are not polluted unless explicitly using the wall-clock
  benchmark harness.
evidence:
  target/wallclock_heldout_smoke_20260618T234406Z.log
  target/runs/20260618_234406Z_shakespeare_1000steps/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  CUDA_DEVICE_INDEX=0 TRAIN_DATASET=shakespeare TRAIN_STEPS=1000 TRAIN_MAX_SECONDS=5 TRAIN_LOG_INTERVAL=20 ./target/release/rust-kernels: pass
  heldout_eval split=val val_loss=10.778596 train_elapsed_s=5.208 eval_elapsed_s=0.159 completed_steps=10
notes:
  Future optimizer/kernel experiments should compare heldout_eval val_loss at
  the same TRAIN_MAX_SECONDS budget. Fixed-step loss and isolated kernel timing
  can explain a result but should not be treated as the result.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Reduce Aurora Polar Express iteration count from 5 to 1.
status: invalid optimization; rejected
decision:
  Rejected. Do not reduce Polar Express iterations as a performance
  optimization: it changes the optimizer approximation rather than improving the
  kernel implementation. Restore POLAR_ITERATIONS=5.
baseline:
  POLAR_ITERATIONS=5 profile: aurora_mega_update_cooperative_kernel 111.521ms
  average over 20 launches.
  100-step Shakespeare final loss: 6.438527.
result:
  POLAR_ITERATIONS=1 profile: aurora_mega_update_cooperative_kernel 25.882ms
  average over 20 launches.
  100-step Shakespeare final loss: 6.442367.
quality:
  Not acceptable as optimization evidence. The 100-step loss is worse than the
  5-iteration baseline, and speed alone is not a valid reason to alter optimizer
  semantics.
performance:
  Aurora mega kernel improved by about 76.8% on the 20-step nsys profile, but
  this is only a diagnostic speed boundary, not an acceptable optimization.
evidence:
  target/aurora_polar_iter1_100steps_20260618T232933Z.log
  target/runs/20260618_232933Z_shakespeare_100steps/loss.png
  target/nsys_aurora_polar_iter1_b8_20_20260618T233025Z_kernel_sum_cuda_gpu_kern_sum.csv
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.442367
  20-step nsys GPU profile: pass
notes:
  This should not be retried as a speed path. Real optimization work must keep
  the Aurora algorithm semantics intact and reduce kernel/data movement cost.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Reduce Aurora Polar Express iteration count from 5 to 2.
status: invalid optimization; rejected
decision:
  Rejected. Do not reduce Polar Express iterations as a performance
  optimization: it changes the optimizer approximation rather than improving the
  kernel implementation. Restore POLAR_ITERATIONS=5.
baseline:
  POLAR_ITERATIONS=5 profile: aurora_mega_update_cooperative_kernel 111.521ms
  average over 20 launches.
  100-step Shakespeare final loss: 6.438527.
result:
  POLAR_ITERATIONS=2 profile: aurora_mega_update_cooperative_kernel 47.438ms
  average over 20 launches.
  100-step Shakespeare final loss: 6.441639.
quality:
  Not acceptable as optimization evidence. The final loss is worse than the
  5-iteration baseline and speed alone is not a valid reason to alter optimizer
  semantics.
performance:
  Aurora mega kernel improved by about 57.5% on the 20-step nsys profile, but
  this is only a diagnostic speed boundary, not an acceptable optimization.
evidence:
  target/aurora_polar_iter2_100steps_20260618T232743Z.log
  target/runs/20260618_232743Z_shakespeare_100steps/loss.png
  target/nsys_aurora_polar_iter2_b8_20_20260618T232840Z_kernel_sum_cuda_gpu_kern_sum.csv
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.441639
  20-step nsys GPU profile: pass
notes:
  This should not be retried as a speed path. Real optimization work must keep
  the Aurora algorithm semantics intact and reduce kernel/data movement cost.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Reduce Aurora Polar Express iteration count from 5 to 3.
status: invalid optimization; rejected
decision:
  Rejected. Do not reduce Polar Express iterations as a performance
  optimization: it changes the optimizer approximation rather than improving the
  kernel implementation. Restore POLAR_ITERATIONS=5.
baseline:
  POLAR_ITERATIONS=5 profile: aurora_mega_update_cooperative_kernel 111.521ms
  average over 20 launches.
  100-step Shakespeare final loss: 6.438527.
result:
  POLAR_ITERATIONS=3 profile: aurora_mega_update_cooperative_kernel 68.825ms
  average over 20 launches.
  100-step Shakespeare final loss: 6.438921.
quality:
  Not acceptable as optimization evidence. Even when short-run loss is close,
  changing iteration count changes optimizer semantics and does not satisfy the
  kernel optimization goal.
performance:
  Aurora mega kernel improved by about 38.3% on the 20-step nsys profile, but
  this is only a diagnostic speed boundary, not an acceptable optimization.
evidence:
  target/aurora_polar_iter3_100steps_20260618T232548Z.log
  target/runs/20260618_232548Z_shakespeare_100steps/loss.png
  target/nsys_aurora_polar_iter3_b8_20_20260618T232645Z_kernel_sum_cuda_gpu_kern_sum.csv
verification:
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 5 tests
  100-step Shakespeare direct GPU training run: pass, final loss 6.438921
  20-step nsys GPU profile: pass
notes:
  This should not be retried as a speed path. Real optimization work must keep
  the Aurora algorithm semantics intact and reduce kernel/data movement cost.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Hoist Aurora direct/transposed branches out of per-element momentum
  orientation and master update paths.
status: rejected; real cooperative training launch fails
decision: reverted
baseline:
  Shrink-polar-ax stable profile had aurora_mega_update_cooperative_kernel
  111.521ms average, and the 100-step Shakespeare check reached final loss
  6.438527.
result:
  cargo check, sm_120a build, and ignored GPU optimizer tests passed.
  The real 100-step Shakespeare run failed before step 0 with
  DriverError(720, "too many blocks in cooperative launch").
quality:
  No quality result. The real launch failed.
performance:
  Rejected before profiling. The split direct/transposed functions increased
  cooperative kernel resource pressure enough to exceed the launch limit.
evidence:
  target/aurora_branch_hoist_100steps_20260618T232210Z.log
  target/aurora_after_branch_hoist_revert_100steps_20260618T232333Z.log
verification:
  After revert: cargo check --workspace --tests passed.
  After revert: cargo oxide build --arch sm_120a passed.
  After revert: 100-step Shakespeare run passed with final loss 6.438527.
notes:
  Do not retry this as a branch micro-optimization inside the mega kernel
  without first lowering cooperative kernel resource usage elsewhere.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Pre-convert the left operand of Aurora B @ X to FP16 before the
  in-place TC matmul.
status: failure
decision: reverted
baseline:
  PP_ITERATIONS=2 profile had optimizer_ms around 293ms at step 19.
  f16_cta_tc_matmul_add_f32_in_place_kernel took 2.870s over 20160 launches.
  fp32_to_f16_kernel took 384ms over 22560 launches.
result:
  optimizer_ms increased to around 306ms at step 19.
  replacement f16_cta_tc_matmul_add_f16_a_f32_in_place_kernel still took 2.870s
  over 20160 launches.
  fp32_to_f16_kernel increased to 416ms over 42720 launches.
quality:
  No quality benefit measured; this was a kernel performance experiment only.
performance:
  Failed. The extra conversion launches/cost outweighed any staging benefit.
evidence:
  target/nsys/prepared_a_direct_route_20.nsys-rep
verification:
  CUDA ignored kernel tests passed before profiling.
notes:
  This route should stay rejected unless the conversion can be fused into an
  existing producer or reused across multiple matmuls without extra launches.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: Reduce rectangular Aurora power balancing from two passes to one
  pass with PP_ITERATIONS 2 -> 1.
status: success, preliminary
decision: keep for now; needs 2k-step quality check before committing as safe
baseline:
  PP_ITERATIONS=2 profile had optimizer_ms around 293ms at step 19.
  Aurora was around 282ms at step 19.
  Each Aurora polar TC helper launched 20160 times over 20 steps.
result:
  PP_ITERATIONS=1 profile had optimizer_ms around 140ms at step 19.
  Aurora was around 115ms at step 19.
  Aurora polar TC helper launches dropped to 11520 over 20 steps.
quality:
  20-step Shakespeare loss stayed comparable: final_train_loss 9.131195 direct,
  9.127897 under nsys.
performance:
  20-step direct wall throughput was 12917 tokens/s including startup and two
  loss syncs.
  Top profiled kernels shifted to causal attention, MS-EDEN quantization,
  Aurora in-place TC matmul, and linear backward projection.
evidence:
  target/nsys/aurora_pp1_20.nsys-rep
  target/runs/aurora_pp1_20/loss.png
verification:
  cargo check --workspace --tests: pass
  cargo build --release: pass
  20-step direct GPU training run: pass
  20-step nsys GPU training run: pass
notes:
  Baseline PP_ITERATIONS=2 profile had optimizer_ms around 293ms at step 19
  and 20160 launches for each Aurora polar TC helper over 20 steps. PP1 profile
  had optimizer_ms around 140ms at step 19 and 11520 launches for those helpers.
```

```text
date: 2026-06-18
commit: uncommitted
experiment: 100-step quality smoke for PP_ITERATIONS=1.
status: success, short-run only
decision: keep as preliminary evidence; still requires 2k-step Shakespeare run
baseline:
  Prior 100-step grouped-Aurora experiment reached final loss around 6.440577.
result:
  100-step Shakespeare final_train_loss was 6.438200.
quality:
  Loss dropped from 10.808949 at step 0 to 6.438200 at step 99.
performance:
  Wall throughput was 13203 tokens/s including startup and two loss syncs.
evidence:
  target/runs/aurora_pp1_100/loss.png
verification:
  cargo test --workspace: pass
  100-step direct GPU training run: pass
notes:
  Short learning check stayed healthy; this does not replace a 2k-step quality
  check.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: AMUSE optimizer correctness pass against arXiv 2605.22432.
status: success, short-run only
decision: keep the math corrections; still needs 2k-step Shakespeare quality
  check and profiling after commit.
changes:
  Removed rectangular row balancing from the live matrix update path because
  AMUSE uses Muon-style momentum orthogonalization directly.
  Switched the matrix update scale to the AdamW-aux AMUSE scale:
  0.2 * sqrt(max(rows, cols)).
  Reduced cooperative Aurora grid x-blocks from 32 to 8 so the grouped
  12-matrix training launch is valid on the local GPU.
result:
  100-step Shakespeare final_train_loss was 6.408423.
quality:
  Loss dropped from 10.808983 at step 0 to 6.408423 at step 99.
performance:
  Wall throughput was about 11466 tokens/s including startup and every-10-step
  loss syncs.
evidence:
  target/runs/20260619_004717Z_shakespeare_100steps/loss.png
  target/amuse_correctness_100steps_20260619T004717Z.log
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass
  100-step direct GPU training run: pass
notes:
  The first optimizer GPU test run failed with a SIGSEGV because the root PTX
  file was stale after the kernel signature changed. Regenerating PTX fixed
  that ABI mismatch before the passing run.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Batch-size sweep after fused Aurora scheduler tuning.
status: success, short-run plus 100-step B32 check
decision: keep GPT2_BATCH_SIZE=32 for now; it improves throughput and the
  100-step Shakespeare check stays healthy.
baseline:
  B8 with 24 blocks per active matrix and 4 active matrices reached about
  12.2k tok/s on a 20-step direct run. Its fused Aurora kernel took about
  278.2ms per step in nsys.
result:
  B16 with 32 blocks per active matrix reached about 14.7k tok/s on a 20-step
  direct run. Its fused Aurora kernel took about 273.5ms per step in nsys.
  B32 with 32 blocks per active matrix reached about 16.5k tok/s on a 20-step
  direct run and about 16.2k tok/s on a 100-step run with every-10-step loss
  sync.
quality:
  B32 100-step Shakespeare loss dropped from 10.820921 at step 0 to 6.394851
  at step 99. finite=true and nonzero=true throughout logged steps.
performance:
  B32 nsys showed batch-dependent kernels moving ahead of Aurora:
  causal_attention_kernel 7.991s, fp32_to_nvfp4_ms_eden_device_scale_kernel
  6.403s, aurora_mega_update_cooperative_kernel 5.482s over 20 steps.
  Aurora remains a one-launch-per-step cooperative kernel.
evidence:
  target/b16_aurora32x3_20steps_20260619T005608Z.log
  target/b32_aurora32x3_20steps_20260619T005858Z.log
  target/b32_aurora32x3_100steps_20260619T010134Z.log
  target/nsys/b16_aurora32x3_20.nsys-rep
  target/nsys/b32_aurora32x3_20.nsys-rep
  target/runs/20260619_010134Z_shakespeare_100steps/loss.png
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass
  B32 100-step direct GPU training run: pass
notes:
  Larger batch mostly amortizes the fixed-cost fused Aurora update. B64 was
  not tested in this pass; it should be treated as a separate memory/perf
  experiment because logits and attention scratch scale directly with batch.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Remove row-balance-era normalization branch from fused Aurora
  Polar path.
status: success, cleanup only
decision: keep; it removes a dead branch and parameter from the current AMUSE
  fused path without changing the measured training curve.
changes:
  Removed the norm_chunks_ready parameter from polar_express_from_source_ptr.
  Made normalize_source_to_x_from_chunks private to normalize.rs because it is
  now only the second half of normalize_source_to_x.
quality:
  B32 100-step Shakespeare loss dropped from 10.820921 at step 0 to 6.394860
  at step 99. finite=true and nonzero=true throughout logged steps.
performance:
  B32 100-step wall throughput was about 16.2k tok/s including every-10-step
  loss sync.
evidence:
  target/b32_fused_norm_cleanup_100steps_20260619T010737Z.log
  target/runs/20260619_010737Z_shakespeare_100steps/loss.png
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass
  B32 100-step direct GPU training run: pass
notes:
  This does not remove a required full-grid sync. The remaining normalization
  barriers still guard real dependencies: source norm reduction, global inverse
  norm publication, and normalized matrix availability before TC tiles.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Fuse-audit cleanup for Aurora mega kernel helper ownership.
status: success, cleanup plus sustained verification
decision: keep; Aurora remains one cooperative kernel launch per training step,
  and the cleanup removes local duplicate helpers without changing behavior.
changes:
  Removed Aurora-local block reduction wrapper modules; fused Aurora and Polar
  now call the shared block_reduce utilities directly.
  Moved duplicate raw f32 pointer read/write helpers into utils/device_ptr.rs.
  Verified that generated PTX exposes only one Aurora entry kernel:
  aurora_mega_update_cooperative_kernel. The remaining Aurora symbols in PTX
  are device functions, not separate launches.
quality:
  B32 100-step Shakespeare loss dropped from 10.820921 at step 0 to 6.395337
  at step 99. finite=true and nonzero=true throughout logged steps.
performance:
  B32 100-step wall throughput was about 16.3k tok/s including every-10-step
  loss sync.
  Current nsys B32 20-step profile shows aurora_mega_update_cooperative_kernel
  as 20 launches over 20 steps, 5.484s total, 274.200ms average.
  Top B32 kernels remain causal_attention_kernel 7.896s,
  fp32_to_nvfp4_ms_eden_device_scale_kernel 6.396s,
  aurora_mega_update_cooperative_kernel 5.484s, and
  linear_backward_projection_cta_device_scale_kernel 4.779s over 20 steps.
fusion_boundaries:
  The current fused kernel still requires grid-wide visibility at these
  points:
  momentum/orient writes before Polar norm reads the oriented source;
  per-block norm sums before block 0 computes and publishes inverse norm;
  inverse norm publication before normalized matrix writes complete;
  normalized matrix availability before the first TC Gram stage;
  each Polar Express dependent stage Gram -> AX -> next-X;
  master update block amax writes before global scale reduction;
  global scale publication before four/six NVFP4 encoding.
  These are not host-launch boundaries anymore; they are cooperative-kernel
  grid::sync boundaries inside the mega kernel.
evidence:
  target/b32_aurora_helper_cleanup_100steps_20260619T011707Z.log
  target/runs/20260619_011707Z_shakespeare_100steps/loss.png
  target/aurora_helper_cleanup_b32_20_20260619T012038Z.log
  target/nsys/aurora_helper_cleanup_b32_20_20260619T012038Z.nsys-rep
  target/nsys/aurora_helper_cleanup_b32_20_20260619T012038Z_kernel_sum.csv_cuda_gpu_kern_sum.csv
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass
  B32 100-step direct GPU training run: pass
  B32 20-step nsys profile: pass
notes:
  Focused ncu SOL capture for aurora_mega_update_cooperative_kernel was
  attempted but blocked by ERR_NVGPUCTRPERM for this user, so no ncu SOL
  claims are recorded from this pass.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Increase fused Aurora cooperative CTAs from NCU launch
  underutilization evidence.
status: success
decision: keep AURORA_COOPERATIVE_BLOCKS=120 with AURORA_MATRIX_PHASES=16.
  This gives 120 blocks per active matrix and 360 cooperative CTAs total for
  the current 3-active-matrix mega launch.
baseline:
  App-replay NCU on the 32-block schedule reported:
  Grid Size 96 on 188 SMs, Waves Per SM 0.26, Achieved Occupancy 16.67%,
  Compute (SM) Throughput 9.26%, Memory Throughput 21.46%, DRAM Throughput
  1.22%, Pipe Tensor Cycles Active 1.10%, Duration 299.63ms.
  NCU explicitly flagged the grid as too small to fill the device.
changes:
  Increased AURORA_COOPERATIVE_BLOCKS from 32 to 64, then to 120. No optimizer
  math changed; only the number of CTAs participating in the existing fused
  cooperative kernel and its reductions changed.
result:
  32 blocks per active matrix: aurora_mega_update_cooperative_kernel
  274.200ms average over 20 launches.
  64 blocks per active matrix: aurora_mega_update_cooperative_kernel
  173.055ms average over 20 launches.
  120 blocks per active matrix: aurora_mega_update_cooperative_kernel
  113.807ms average over 20 launches.
quality:
  B32 100-step Shakespeare loss with 120 blocks dropped from 10.820921 at
  step 0 to 6.396184 at step 99. finite=true and nonzero=true throughout
  logged steps.
performance:
  B32 100-step wall throughput with 120 blocks was about 17.4k tok/s including
  every-10-step loss sync, versus about 16.3k tok/s for the previous 32-block
  helper-cleanup run.
  In the 120-block nsys profile, Aurora is no longer a top-three kernel:
  causal_attention_kernel 7.939s, fp32_to_nvfp4_ms_eden_device_scale_kernel
  6.412s, linear_backward_projection_cta_device_scale_kernel 4.785s,
  aurora_mega_update_cooperative_kernel 2.276s over 20 steps.
evidence:
  target/ncu_aurora_mega_basic_appreplay_b32_20260619T012919Z.txt
  target/ncu_aurora_mega_basic_appreplay_b32_20260619T012919Z.ncu-rep
  target/nsys/aurora64_b32_20_20260619T013109Z_kernel_sum.csv_cuda_gpu_kern_sum.csv
  target/nsys/aurora120_b32_20_20260619T013241Z_kernel_sum.csv_cuda_gpu_kern_sum.csv
  target/aurora120_b32_100steps_20260619T013350Z.log
  target/runs/20260619_013350Z_shakespeare_100steps/loss.png
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass
  B32 20-step nsys profile with 64 blocks: pass
  B32 20-step nsys profile with 120 blocks: pass
  B32 100-step direct GPU training run with 120 blocks: pass
notes:
  This is a launch-geometry/reduction-participation change, not a kernel split
  or algorithm change. The remaining fusion bottlenecks are now outside Aurora
  in the current nsys profile, but Aurora still has internal grid::sync
  boundaries for true data dependencies documented in the previous entry.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Correct batch-size comparison target to equal wall-clock loss.
status: partial, enough to reject B32 as a throughput-only default
decision:
  Do not treat B32 as better from tokens/s alone. The current code is set back
  to GPT2_BATCH_SIZE=8. The previous B32 throughput result is only a diagnostic
  throughput fact, not an optimization decision.
target:
  Loss must be compared at a fixed wall-clock budget. Held-out validation loss
  is the real target; train loss at fixed wall-clock is only a lower-quality
  proxy when a prior run did not emit heldout_eval.
result:
  Existing B32 Aurora-120 reference:
    target/aurora120_b32_100steps_20260619T013350Z.log
    elapsed_s=188.223, step=99, train loss=6.396184, loss_ema=8.333464,
    batch_size=32, tokens_per_step=32768.
    No heldout_eval line was emitted.
  New B8 fixed-wall run:
    target/aurora120_b8_188s_20260619T015120Z.log
    stopped_by_wall_clock elapsed_s=188.291, completed_steps=349.
    last logged train loss near endpoint: step=340 elapsed_s=184.056
    loss=5.991461, loss_ema=6.248581, batch_size=8,
    tokens_per_step=8192.
    heldout_eval split=val val_loss=6.048583 train_elapsed_s=188.504
    eval_elapsed_s=0.164 completed_steps=349.
quality:
  On the available equal-wall train-loss comparison, B8 is better than B32
  despite lower raw tokens/s. Because the B32 reference did not emit heldout
  eval, this pass does not prove B8 vs B32 on held-out validation loss.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  B8 fixed-wall 188s run: pass, finite=true and nonzero=true throughout logged
  steps.
notes:
  The aborted TRAIN_STEPS=100000 fixed-wall attempt exposed a scheduler bug:
  TRAIN_STEPS changed the default LR warmup. That coupling was removed; default
  warmup is now fixed unless TRAIN_LR_WARMUP_STEPS is explicitly set.
```
