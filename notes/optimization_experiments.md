# GPT-2 NVFP4 Training Optimization Notes

This note tracks throughput goals, measured baselines, experiments already tried,
and the next optimization work. Keep this file factual: add measured commands and
results before changing the plan.

Decision rules live in [optimization_rules.md](optimization_rules.md). Apply
those rules when accepting, rejecting, or comparing experiments in this file.

## Goal

Train a GPT-2-small-shaped model with local Rust/CUDA kernels, targeting
NVFP4-heavy training where tensor-core matmuls dominate runtime.

External reference target:

- llm.c / Karpathy GPT-2 124M FineWeb run: 10B tokens in roughly 4-24 hours on
  one strong single GPU equivalent, with higher-end 8x A100 runs around 90
  minutes.

Primary optimization target:

- Lowest held-out validation loss after a fixed 15-minute wall-clock training
  budget.
- Training loss, fixed-step loss, tokens/s, and isolated kernel timings are
  diagnostics only. They do not prove an optimization unless the held-out
  validation loss at the same wall-clock budget improves or is preserved.
- Use the validation split endpoint line:

```text
heldout_eval split=val val_loss=... train_elapsed_s=... completed_steps=...
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted after 20-step screen
experiment: Attention-only FP16 CTA matmul tile M128/N32/K16.
status: rejected_screen
change:
  Routed only causal-attention-backward FP32-to-FP16 TC matmuls through a
  separate CTA tile with M=128, N=32, K=16. Aurora and other square FP16 TC
  callers stayed on the existing M64/N64 tile.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test f16_tc_matmul_tiled -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/attention_m128n32_b16_l4d1024_20_20260622T161719Z.run.log
    val_loss=9.063751, train_elapsed_s=5.874, completed_steps=20.
measured_effect:
  Against the accepted descriptor profile
  target/nsys/aurora_slot_descriptor_b16_l4d1024_20_20260622T153325Z.run.log,
  total profiled train time moved from 5.915s to 5.874s, but the intended
  attention matmul target regressed. Estimated attention matmul time moved from
  about 690.0ms to 720.8ms over 20 steps:
    f32_input: 308.7ms estimated old attention share -> 334.3ms new.
    a_transposed_rhs: 254.4ms old -> 265.8ms new.
    rhs: 127.0ms estimated old attention share -> 120.7ms new.
  The total run improvement came from unrelated profiler variance in Aurora,
  projection, and LM-head kernels, not from the candidate attention tile.
decision:
  Reject before the 100-step and 900-second gates. The attention-specific
  M128/N32 tile made the target matmul path slower overall, so the code was
  reverted and only this note is kept.
```
```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Use unchecked byte/scale loads in aligned NVFP4 projection CTA staging.
status: rejected_screen
change:
  Added temporary unchecked helpers for aligned packed E2M1 and UE4M3 scale
  loads, then routed projection_cta aligned staging through them. The guarded
  helpers remained on edge-handling paths.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/projection_unchecked_loads_b16_l4d1024_20_20260622T144858Z.run.log
    val_loss=9.063751, train_elapsed_s=6.074, completed_steps=20.
measured_effect:
  Against the fresh current profile
  target/nsys/current_clean_b16_l4d1024_20_20260622T143653Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  1237.737ms to 1270.698ms over 400 calls, lm_head_kernel moved from
  210.261ms to 217.341ms over 21 calls, and profiled train time moved from
  5.968s to 6.074s.
decision:
  Reject before the 100-step and 900-second gates. Removing the explicit guard
  helpers did not reduce the hot aligned projection path; it regressed the
  short nsys screen. Code was reverted.
```
```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Compute cross-entropy dlogits row amax directly from target
  probability.
status: rejected_screen
change:
  Replaced the third per-row max-reduction in cross_entropy_kernel with the
  identity max_i |p_i - y_i| = 1 - p_target. The dense dlogits write and loss
  computation were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/loss_direct_amax_b16_l4d1024_20_20260622T144158Z.run.log
    val_loss=9.063751, train_elapsed_s=5.972, completed_steps=20.
measured_effect:
  Against the clean profile
  target/nsys/current_clean_b16_l4d1024_20_20260622T143653Z.run.log,
  cross_entropy_kernel moved from 58.225311ms to 58.241029ms over 21 calls.
  Full profiled training time moved from 5.968s to 5.972s with identical
  20-step validation loss.
decision:
  Reject before the 100-step and 900-second gates. Although the identity is
  correct, removing the reduction did not improve the generated kernel in the
  current profile. Code was reverted.
```
```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Force Aurora cooperative kernel launch bounds to 256 threads and
  4 blocks per SM.
status: rejected_screen
change:
  Added cuda-oxide #[launch_bounds(256, 4)] to
  aurora_mega_update_cooperative_kernel. The generated PTX contained
  .maxntid 256, 1, 1 and .minnctapersm 4, so the compiler hint did land.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  20-step nsys:
    target/nsys/aurora_launch_bounds_256x4_b16_l4d1024_20_20260622T143827Z.run.log
    val_loss=9.063751, train_elapsed_s=6.021, completed_steps=20.
measured_effect:
  Against the fresh clean profile
  target/nsys/current_clean_b16_l4d1024_20_20260622T143653Z.run.log, Aurora
  regressed from 1914.892196ms to 1958.861395ms over 20 calls. Full profiled
  training time moved from 5.968s to 6.021s with identical 20-step validation
  loss. Linear backward projection also moved slightly worse, from
  1237.737052ms to 1240.959636ms.
decision:
  Reject before the 100-step and 900-second gates. Forcing the occupancy hint
  did not improve useful work and made the dominant Aurora kernel slower.
  Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Save layer-norm residual tape as f16.
status: accepted_current_nextlat
change:
  Changed saved layer-norm residual tape storage from FP32 to f16 for GPT block
  and final layer-norm saved activations. The saved mean and inv_std remain
  FP32. GPT layer-norm backward widens the saved f16 residual when computing
  xhat. NextLat backward keeps using explicit FP32 layer-norm backward wrappers
  for its live concat buffer, so this change targets the GPT training tape only.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test layer_norm_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test layer_norm_backward_params -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_synth_ln_residual_f16_tape_20_20260622T032048Z.run.log
    val_loss=9.152641, train_elapsed_s=6.296, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_ln_residual_f16_tape_synth_100_20260622T032135Z.log
    val_loss=6.534838, train_elapsed_s=31.508, completed_steps=100.
  900-second held-out gate:
    target/nextlat_ln_residual_f16_tape_synth_900_20260622T032223Z.log
    val_loss=3.801502, train_elapsed_s=900.034, completed_steps=2789.
measured_effect:
  Against the prior residual_after_attention tape-removal profile, total D2D
  copy traffic over 20 profiled steps moved from 1781 copies / 18.962ms to
  1592 copies / 4.052ms. The shared fp32_to_f16_kernel moved from 40.892ms to
  51.740ms over the same 20-step profile because it now also saves layer-norm
  residual tape. Profiled 20-step train time moved from 6.333s to 6.296s.
  Against the active NextLat baseline, validation loss moved from 3.795963 to
  3.801502 (+0.146%) and completed steps moved from 2785 to 2789.
decision:
  Accept for the active NextLat branch under the kernel/runtime noise-band rule:
  validation stayed within +/-1% and completed step count increased. Update
  notes/sweep_baseline.env to this active-NextLat baseline.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Remove unused saved residual_after_attention tape.
status: accepted_current_nextlat
change:
  Removed the saved FP32 residual_after_attention forward-tape tensor from the
  block tape. Backward already uses ln_2.saved.residual for layer-norm backward
  and d_residual_after_attention as a gradient buffer; the saved forward copy
  was not consumed.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_synth_no_residual_after_attention_tape_20_20260622T025126Z.run.log
    val_loss=9.154136, train_elapsed_s=6.333, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_no_residual_after_attention_tape_synth_100_20260622T025426Z.log
    val_loss=6.538291, train_elapsed_s=31.553, completed_steps=100.
  900-second held-out gate:
    target/nextlat_no_residual_after_attention_tape_synth_900_20260622T025511Z.log
    val_loss=3.795963, train_elapsed_s=900.001, completed_steps=2785.
measured_effect:
  Against the prior qkv-f16-tape profile, total D2D copy traffic over 20
  profiled steps moved from 1865 copies / 26.198ms to 1781 copies / 18.962ms.
  The 64MiB D2D copy bucket moved from 273 copies to 189 copies, matching one
  removed hidden-state tape copy per block. Against the prior active NextLat
  baseline, validation loss moved from 3.792330 to 3.795963 (+0.096%) and
  completed steps moved from 2775 to 2785.
decision:
  Accept for the active NextLat branch under the kernel/runtime noise-band rule:
  validation stayed within +/-1% and completed step count increased. Update
  notes/sweep_baseline.env to this active-NextLat baseline; pre-NextLat
  validation results are not protected baselines for this branch.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Save QKV activation tape as f16.
status: accepted_current_nextlat
change:
  Changed the saved block qkv tape from FP32 to f16. The live forward QKV
  scratch remains FP32 for the forward attention path; only the saved backward
  tape is f16. Attention backward gather now widens saved f16 Q/K/V values into
  the existing FP32 scratch buffers before the TC-backed attention-backward
  matmuls.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_synth_qkv_f16_tape_20_20260622T023144Z.run.log
    val_loss=9.154136, train_elapsed_s=6.341, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_qkv_f16_tape_synth_100_20260622T023216Z.log
    val_loss=6.535542, train_elapsed_s=31.746, completed_steps=100.
  900-second held-out gate:
    target/nextlat_qkv_f16_tape_synth_900_20260622T023301Z.log
    val_loss=3.792330, train_elapsed_s=900.102, completed_steps=2775.
measured_effect:
  Against the prior attention_out-f16-tape profile, the 201326592-byte D2D
  copy bucket was removed. Total D2D copy traffic over 20 profiled steps moved
  from 1949 copies / 47.050ms to 1865 copies / 26.198ms. The shared
  fp32_to_f16_kernel now covers qkv, attention_out, and mlp_up tape saves and
  moved from 26.366ms to 40.903ms over 20 steps. gather_qkv_dout_kernel moved
  from 26.751ms to 22.092ms despite widening f16 qkv values. Against the prior
  900-second attention_out-f16-tape gate, validation loss moved from 3.811845
  to 3.792330 and completed steps moved from 2770 to 2775.
decision:
  Promote for the current NextLat branch. Held-out validation loss improved
  and completed step count increased under the fixed 900-second SYNTH budget.
  This is an accepted active-NextLat baseline candidate; pre-NextLat validation
  results are not protected baselines for this branch.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Save attention output tape as f16.
status: accepted_current_nextlat
change:
  Changed the saved block attention_out tape from FP32 to f16. Attention
  backward only uses this saved tensor in the softmax-d dot product, so the
  upstream d_out and the rest of the attention backward path remain FP32.
  Added softmax_d_f16_kernel, removed the unused FP32 softmax_d_kernel, and
  exposed the existing generic fp32_to_f16 utility through F16TcMatmulModule so
  both attention_out and mlp_up tape saves use the same converter.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_synth_attention_out_f16_tape_20_20260622T021132Z.run.log
    val_loss=9.154136, train_elapsed_s=6.343, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_attention_out_f16_tape_synth_100_20260622T021204Z.log
    val_loss=6.534555, train_elapsed_s=31.752, completed_steps=100.
  900-second held-out gate:
    target/nextlat_attention_out_f16_tape_synth_900_20260622T021253Z.log
    val_loss=3.811845, train_elapsed_s=900.189, completed_steps=2770.
measured_effect:
  Against the prior mlp_up-f16-tape profile, 64MiB D2D copies moved from 357
  to 273 over 20 profiled steps. Total D2D copy traffic moved from 2033
  copies / 55.030ms to 1949 copies / 47.050ms. The generic fp32_to_f16_kernel
  now covers both f16 tape saves and took 26.366ms over 20 steps. The new
  softmax_d_f16_kernel took 9.372ms over 20 steps. Against the prior
  900-second mlp_up-f16-tape gate, validation loss moved from 3.813107 to
  3.811845 while completed steps moved from 2773 to 2770.
decision:
  Promote for the current NextLat branch because held-out validation loss
  improved under the same fixed 900-second SYNTH budget. This is an accepted
  active-NextLat baseline candidate; pre-NextLat validation results are not
  protected baselines for this branch.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Save MLP relu^2 pre-activation tape as f16.
status: accepted_current_nextlat
change:
  Changed the saved block mlp_up pre-activation tape from FP32 to f16. The
  live forward scratch remains FP32; only the saved tape used by relu^2
  backward is stored as f16. Added an f16 save kernel and a relu2 backward
  variant that widens f16 to FP32 before applying d_pre = d_out * 2 * relu(x).
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_synth_mlp_up_f16_tape_20_20260622T014751Z.run.log
    val_loss=9.149878, train_elapsed_s=6.338, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_mlp_up_f16_tape_synth_100_20260622T014823Z.log
    val_loss=6.538740, train_elapsed_s=31.727, completed_steps=100.
  900-second held-out gate:
    target/nextlat_mlp_up_f16_tape_synth_900_20260622T014923Z.log
    val_loss=3.813107, train_elapsed_s=900.293, completed_steps=2773.
measured_effect:
  Against the prior no-mlp-relu2-tape profile, the 268MiB D2D copy bucket was
  removed. Total D2D copy traffic over 20 profiled steps moved from 2117
  copies / 83.979ms to 2033 copies / 55.030ms. New f16 save work appeared as
  save_pre_activation_f16_kernel at 20.587ms over 20 steps. relu2 backward
  moved from the FP32 relu2_backward_kernel at 40.397ms to
  relu2_backward_f16_kernel at 33.824ms. Against the prior 900-second
  no-mlp-relu2-tape gate, validation loss moved from 3.818655 to 3.813107 and
  completed steps moved from 2770 to 2773.
decision:
  Promote for the current NextLat branch. Held-out validation improved against
  the immediate predecessor and completed step count increased. This is an
  accepted active-NextLat baseline candidate; pre-NextLat validation results are
  not protected baselines for this branch.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Remove unused saved MLP relu2 FP32 tape.
status: accepted_current_nextlat
change:
  Removed the saved FP32 mlp_relu2 forward-tape tensor. MLP backward uses the
  saved mlp_up pre-activation for the relu^2 derivative and the saved
  mlp_down_input_nvfp4 tensor for the down-projection backward pass, so the
  separate post-activation FP32 tape copy was not consumed.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_synth_no_mlp_relu2_tape_20_20260622T012359Z.run.log
    val_loss=9.155092, train_elapsed_s=6.357, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_no_mlp_relu2_tape_synth_100_20260622T012605Z.log
    val_loss=6.528619, train_elapsed_s=31.737, completed_steps=100.
  900-second held-out gate:
    target/nextlat_no_mlp_relu2_tape_synth_900_20260622T012650Z.log
    val_loss=3.818655, train_elapsed_s=900.297, completed_steps=2770.
measured_effect:
  Against the prior tape-min profile, 268MiB D2D copies moved from 168 to 84
  over 20 profiled steps. Total D2D copy time moved from 114.530ms to
  83.979ms, while top kernel timings were essentially unchanged. Against the
  prior 900-second tape-min gate, validation loss moved from 3.810689 to
  3.818655 (+0.209%) and completed steps moved from 2757 to 2770.
decision:
  Promote for the current NextLat branch under the active +/-1% noise-band
  rule: validation loss stayed inside the noise band and completed step count
  increased. This is an accepted active-NextLat baseline candidate; pre-NextLat
  validation results are not protected baselines for this branch.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add confidence-weighted factor beliefs.
status: accepted_tooling_dry_run
change:
  Added analysis_beliefs.tsv as an aggregate per-factor statistical belief file.
  Each factor records confidence-weighted direction, confidence, variance,
  average positive probability, and evidence count. Guided proposal generation
  now consumes these aggregate beliefs instead of summing raw coefficients
  directly from per-response effects, so the same confidence/variance summary
  is both reported and used for knob movement.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo test --bin sweep: pass.
  dry-run analysis:
    target/sweeps/dryrun_factor_beliefs_20260621T173410Z
    generated analysis_beliefs.tsv and ranked acquisition files.
observed_effect:
  analysis_beliefs.tsv reported aggregate directions such as
  ln_adam_lr_scale=0.29632548, aurora_blocks=-0.24345067,
  ln_lr_scale=0.23559764, and n_embd=-0.17306380. The selected candidate
  remained source=guided with score=3.95386689 and uncertainty=51.36550742.
  A focused unit test verifies that factor_beliefs produces positive direction,
  positive confidence, and nonnegative variance for a constructed positive
  batch_size signal.
decision:
  Accept as sweep infrastructure. This makes confidence and variance explicit
  per knob and ties guided proposal movement to that reported statistical
  summary.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add explicit variance-directed sweep candidates.
status: accepted_tooling_dry_run
change:
  The proposal pool now has three recorded sources: guided, variance, and
  random. Variance candidates are selected from a larger random search batch by
  highest model uncertainty before the final interaction-aware scorer ranks the
  whole candidate set. Ranked candidate TSVs now include a source column, and
  candidate_NNNN_score.txt records the selected candidate source.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo test --bin sweep: pass.
  dry-run analysis:
    target/sweeps/dryrun_variance_source_20260621T173018Z
    candidate_0000_ranked.tsv contained guided, variance, and random rows.
observed_effect:
  candidate_0000_ranked.tsv source counts were 11 guided, 11 variance, and
  10 random candidates plus the header. The selected rank-0 candidate was from
  source=guided with score=3.95386689 and uncertainty=51.36550742. Variance
  rows explicitly preserved high-uncertainty alternatives, e.g. rank 2 from
  source=variance had uncertainty=59.51639535.
decision:
  Accept as sweep infrastructure. This directly addresses the requirement to
  place some runs in regions that test and reduce model variance, while still
  letting the final acquisition score choose the best candidate.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add statistic-guided sweep candidate pool.
status: accepted_tooling_dry_run
change:
  Proposal generation now mixes random candidates with guided candidates derived
  from the fitted multivariable model's weighted main-effect directions. The
  first guided candidate follows the inferred direction deterministically; later
  guided candidates add jitter, and the remaining pool stays random. The
  interaction-aware scorer still ranks the whole combined pool, so interactions,
  uncertainty, speed, quality, and stability continue to determine the final
  selected proposal.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo test --bin sweep: pass.
  dry-run analysis:
    target/sweeps/dryrun_guided_pool_20260621T172640Z
    candidate_0000_ranked.tsv contained 32 candidates plus header.
observed_effect:
  The selected rank-0 candidate was
  b16_l4_d1024_h16_p8_c160_lr1.9121_alr0.6693_w50_s0.20_b0.40_r0.50
  with score=4.89670338, uncertainty=27.99442934, and
  exploration=3.36710372. A focused unit test verifies that when full_quality
  says higher batch_size and n_layer are better, the guided pool's first
  candidate moves to batch_size=16 and n_layer=8.
decision:
  Accept as sweep infrastructure. This changes the sweep from random candidate
  generation plus statistical ranking to mixed random/statistically guided
  generation plus statistical ranking.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Persist ranked sweep acquisition candidates.
status: accepted_tooling_dry_run
change:
  The optimizer now returns a proposal containing the selected candidate, the
  acquisition reason, and the ranked scored sample set used to make the choice.
  Each sweep trial writes candidate_NNNN_score.txt plus
  candidate_NNNN_ranked.tsv. The ranked TSV includes score, uncertainty,
  exploration, and predicted quality/speed/stability value, z score, and
  uncertainty for every sampled candidate.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo test --bin sweep: pass.
  dry-run analysis:
    target/sweeps/dryrun_ranked_acquisition_20260621T172207Z
    candidate_0000_score.txt recorded reason=model and the selected candidate.
    candidate_0000_ranked.tsv contained 32 ranked samples plus header.
observed_effect:
  The selected rank-0 candidate was
  b16_l8_d1024_h16_p8_c90_lr1.4222_alr0.7757_w20_s0.10_b0.20_r0.50
  with score=4.00244483, uncertainty=49.27732722, and
  exploration=3.91755422. Lower-ranked rows preserve the alternative scores, so
  the sweep decision is auditable without rerunning the sampler.
decision:
  Accept as sweep infrastructure. This makes the automatic acquisition decision
  inspectable and keeps variance/uncertainty evidence attached to each trial.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Use pairwise factorial terms in sweep acquisition.
status: accepted_tooling_dry_run
change:
  Replaced the report-only interaction probe with one ridge-regularized design
  matrix containing standardized main effects and standardized pairwise product
  terms. Candidate scoring now uses that interaction-aware model directly, so
  interaction effects influence the next proposed run instead of only appearing
  in analysis files. Exploration still uses prediction uncertainty, but applies
  ln(1 + uncertainty) so sparse high-dimensional uncertainty does not dominate
  the quality/stability/speed terms.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo test --bin sweep: pass.
  dry-run analysis:
    target/sweeps/dryrun_interaction_acquisition_20260621T171734Z
    generated interaction-aware analysis_summary.md,
    analysis_interactions.tsv, and candidate_0000_score.txt.
observed_effect:
  The dry run reports interaction terms as first-class fitted effects. Example:
  stability top term became n_embd*ln_warmup_steps, and candidate scoring
  reported score=4.002445, uncertainty=49.277327, exploration=3.917554 for
  b16_l8_d1024_h16_p8_c90_lr1.4222_alr0.7757_w20_s0.10_b0.20_r0.50.
  A focused unit test verifies that if held-out quality is only recoverable from
  batch_size*n_layer, the scorer gives the interaction-positive candidate a
  higher predicted quality score than the crossed candidate.
decision:
  Accept as sweep infrastructure. This moves the implementation closer to the
  factorial-design goal by using interaction terms in candidate acquisition, not
  just reporting them.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Replace heuristic sweep proposal scoring with multivariable statistical analysis.
status: accepted_tooling_dry_run
change:
  Added per-iteration sweep analysis files for main effects, pairwise
  interactions, prediction uncertainty, quality, speed, and stability. Candidate
  proposals now score sampled candidates with the fitted multivariable surrogate
  instead of a manual good/bad KDE split. Screen rejects and failed runs remain
  evidence, so the next proposal is informed by prior failures automatically.
  This is not a full Gaussian-process Bayesian optimizer; it is a ridge
  regularized multivariable surrogate with uncertainty and interaction
  reporting.
verification:
  cargo fmt --all: pass.
  cargo check --all-targets: pass.
  cargo test --bin sweep: pass.
  dry-run analysis:
    target/sweeps/dryrun_stat_analysis_20260621T171039Z
    generated analysis_summary.md, analysis_effects.tsv,
    analysis_interactions.tsv, and candidate_0000_score.txt.
observed_effect:
  Existing history produced fitted response models for screen_quality,
  screen_tokens_per_s, full_quality, full_tokens_per_s, and stability.
  Example dry-run top effects included n_embd as a negative driver for
  full_tokens_per_s and n_layer as a negative driver for stability. The same
  run recorded pairwise effects such as n_layer*aurora_phases for full_quality
  and stability.
decision:
  Accept as sweep infrastructure. This does not promote any training
  hyperparameter result by itself.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add 500-step screen gate to Bayesian hyperparameter sweep.
status: accepted_infra_no_baseline_change
change:
  Sweep trials now run a step-capped screen before the full 900-second gate.
  The sweep first builds and screens the current baseline candidate for the same
  500-step horizon, then rejects candidate trials whose 500-step held-out
  validation loss does not beat that screen baseline. Screen rejects are stored
  with status=rejected_screen and no comparable 900-second val_loss.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass.
  cargo check --all-targets: pass.
run:
  target/sweeps/hyperparam_screen500_900_20260621T143115Z
  screen baseline:
    val_loss=5.450167, completed_steps=500, train_elapsed_s=88.667.
  trials:
    0 rejected_screen val_loss=5.550757, d=2048.
    1 rejected_screen val_loss=5.855838, d=2048.
    2 rejected_screen val_loss=5.891170, d=1024.
    3 rejected_screen val_loss=5.940176, d=1024.
    4 rejected_screen val_loss=5.687976, d=1024.
    5 rejected_screen val_loss=5.727386, d=1024.
    6 rejected_screen val_loss=5.654946, d=2048.
    7 rejected_screen val_loss=5.569425, d=1024.
result:
  No candidate passed the screen, so no full 900-second gate ran and
  notes/sweep_baseline.env stayed at val_loss=3.954098, completed_steps=5028.
decision:
  Keep the screen gate as sweep infrastructure. Do not promote any
  hyperparameter candidate from this sweep.
follow_up:
  Screen-rejected trials now feed the proposal scorer as bad evidence without
  becoming comparable 900-second validation rows. Dry-run proposal check:
  target/sweeps/dryrun_screen_penalty_20260621T151027Z proposed d1024 phase-16
  candidates first, then d1536/L8 exploration; it did not immediately repeat
  the same rejected d2048 screen candidates.
  A later capped sweep showed d1536/h12 candidates fail deterministically at
  crates/cuda-kernels/src/gpt/linear_backward.rs:253 with
  assertion failed: dinput_grid.0.is_power_of_two(). Candidate generation now
  excludes d1536/h12 until that kernel shape is supported.
  The capped run target/sweeps/hyperparam_screen500_cap180_900_20260621T153515Z
  used screen_max_seconds=180. Baseline screen was val_loss=5.446744 at 500
  steps. All eight candidates were rejected_screen; no full 900-second gate ran.
  d2048 candidates that previously could consume a full screen gate were stopped
  at 90-170 completed steps when the 180-second screen cap expired.
  Dry-run target/sweeps/dryrun_no_d1536_20260621T155257Z confirmed the next
  candidate set no longer includes d1536.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Hoist Aurora update decay and scaled learning-rate invariants.
status: rejected_pre_gate
change:
  Moved decay = 1.0 - learning_rate * weight_decay and
  update_scale = learning_rate * 0.2 * sqrt(max(rows, cols)) out of the
  per-element Aurora update_one path and passed the precomputed values through
  update_four_amax. Optimizer semantics were intended to remain unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aurora_update_invariant_hoist_l4_b8_20_20260621T103114Z.run.log
    target/nsys/aurora_update_invariant_hoist_l4_b8_20_20260621T103114Z_stats.txt
    val_loss=8.503360, train_elapsed_s=3.413, completed_steps=20.
measured_effect:
  Against the accepted MS-EDEN reciprocal profile
  target/nsys/ms_eden_inv_scale_l4_b8_20_20260621T093759Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1.362570839s to
  1.383341516s over 20 calls. The intended top kernel regressed.
decision:
  Reject before the 900-second gate and revert the code.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Hoist cross-entropy dlogits grad_scale reciprocal.
status: rejected_pre_gate
change:
  Moved grad_scale = 1.0 / token_count out of the dlogits vocab loop in
  cross_entropy_kernel. Loss math, dlogits layout, block size, and launch
  shape were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/cross_entropy_grad_scale_hoist_l4_b8_20_20260621T102747Z.run.log
    target/nsys/cross_entropy_grad_scale_hoist_l4_b8_20_20260621T102747Z_stats.txt
    val_loss=8.505094, train_elapsed_s=3.416, completed_steps=20.
measured_effect:
  Against the accepted MS-EDEN reciprocal profile
  target/nsys/ms_eden_inv_scale_l4_b8_20_20260621T093759Z.run.log,
  cross_entropy_kernel moved from 55.345090ms to 55.366645ms over 21 calls.
  The explicit hoist did not improve generated code for the current profile.
decision:
  Reject before the 900-second gate and revert the code.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Host-routed aligned f32-input FP16 CTA matmul kernel.
status: rejected_pre_gate
change:
  Added f16_cta_tc_matmul_f32_aligned_kernel and routed aligned
  batched_matmul_f32_input calls to it from the host. The candidate removed
  the per-K-tile aligned/generic branch from the hot f32-input FP16 CTA
  matmul body while preserving the same staging, MMA, and store math.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul_tiled -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/f16_f32_host_aligned_l4_b8_20_20260621T102555Z.run.log
    target/nsys/f16_f32_host_aligned_l4_b8_20_20260621T102555Z_stats.txt
    val_loss=8.505094, train_elapsed_s=3.421, completed_steps=20.
measured_effect:
  Against the accepted MS-EDEN reciprocal profile
  target/nsys/ms_eden_inv_scale_l4_b8_20_20260621T093759Z.run.log,
  the f32-input FP16 CTA matmul path regressed from
  f16_cta_tc_matmul_f32_kernel=217.291413ms over 244 calls to
  f16_cta_tc_matmul_f32_aligned_kernel=233.626397ms over 244 calls.
  The short run wall-clock was effectively unchanged, but the targeted hot
  kernel got slower.
decision:
  Reject before the 900-second gate and revert the code. Do not retry this
  host-routed aligned split without a different staging/store implementation.
```

```text
date: 2026-06-21
commit: historical uncommitted candidate, reverted before note
experiment: Eight-row unroll for linear_bias_grad_kernel.
status: rejected_profile
change:
  Increased the linear-bias gradient row unroll from 4 rows/thread to 8 rows/thread.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass after rerun with completed PTX build.
  20-step nsys screen:
    target/nsys/linear_bias_unroll8_l4_b8_20_20260621T054112Z.run.log
    val_loss=8.505538, train_elapsed_s=3.591, completed_steps=20.
measured_effect:
  Compared with the promoted MS-EDEN barrier baseline profile
  target/nsys/ms_eden_no_pack_barrier_l4_b8_20_20260621T045551Z.run.log,
  linear_bias_grad_kernel moved only from 41.008457ms to 40.993777ms over
  20 profiled steps. Total profiled train time regressed from 3.588s to
  3.591s, and linear_backward_projection_pair_cta_device_scale_kernel also
  regressed from 621.499531ms to 622.905559ms.
decision:
  Reject before the 100-step and 900-second gates. The target-kernel gain was
  too small and the short profiled wall-clock regressed. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Early inactive return in attention_prob_ds_kernel.
status: rejected_profile
change:
  Rewrote the backward attention probability/dS kernel to return immediately
  for inactive causal-mask cells and to compute the log-sum-exp index once for
  active cells.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/attention_prob_early_return_l4_b8_20_20260621T060045Z.run.log
    val_loss=8.505538, train_elapsed_s=3.589, completed_steps=20.
measured_effect:
  Compared with the promoted MS-EDEN barrier baseline profile
  target/nsys/ms_eden_no_pack_barrier_l4_b8_20_20260621T045551Z.run.log,
  attention_prob_ds_kernel moved from about 90.620ms to 90.675ms over 20
  profiled steps. Total profiled train time also did not improve, moving from
  3.588s to 3.589s.
decision:
  Reject before the 100-step and 900-second gates. The change did not improve
  the target kernel or the short profiled wall-clock. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted diagnostic
experiment: NVFP4/RHT Polar Express estimator for Aurora.
status: diagnostic
change:
  Added an ignored GPU comparator that runs the existing NVFP4 MS-EDEN tensor
  core matmul helper inside the Polar Express recurrence and compares the
  resulting update direction against the current FP16-leaf Polar reference.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer nvfp4_rht_polar_estimator_reports_update_error -- --ignored --nocapture: pass.
measured_effect:
  Single first-iteration products were directionally close:
    first_gram cosine=0.99911314 rel_l2=4.24877554e-2
    first_ax_from_expected_gram cosine=0.99855441 rel_l2=5.37767597e-2
    first_aax_from_expected_inputs cosine=0.99920607 rel_l2=4.01885584e-2
  Full NVFP4/RHT Polar was unstable:
    iterations=1 cosine=0.99815118 rel_l2=6.08224683e-2
    iterations=2 cosine=0.96626502 rel_l2=2.65107155e-1
    iterations=3 cosine=0.53992063 rel_l2=2.01288104e0
    iterations=4 cosine=0.15201239 rel_l2=1.26888269e3
    iterations=5 cosine=0.15815794 rel_l2=1.21603512e18
  A one-iteration FP4 prefix followed by FP16 remained close:
    hybrid_fp4_prefix=1 cosine=0.99875975 rel_l2=5.08906357e-2
  Prefix lengths 2 and 3 became non-finite after the later recurrence.
decision:
  Do not replace all Aurora Polar GEMMs with NVFP4/RHT. The only viable
  follow-up candidate from this diagnostic is a first-iteration-only NVFP4
  Polar prefix, followed by the existing FP16 Polar leaf path.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Compact upper-triangle tile enumeration inside Aurora Polar
  symmetric Gram stage.
status: rejected_pre_gate
change:
  Replaced full square tile enumeration plus lower-triangle skip with direct
  upper-triangle tile enumeration inside run_symmetric_tiles. Cooperative launch
  dimensions and optimizer math were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  100-step SYNTH screen:
    target/aurora_tri_compact_l4_b8_100_20260620T183208Z.log
    val_loss=6.546818, train_elapsed_s=19.318, completed_steps=100.
  20-step nsys screen:
    target/nsys/aurora_tri_compact_l4_b8_20_20260620T183242Z.run.log
    train_elapsed_s=3.811, completed_steps=20.
measured_effect:
  The intended Aurora kernel got slower. aurora_mega_update_cooperative_kernel
  increased from about 1.367s to about 1.385s over 20 profiled steps versus
  target/nsys/ce_dlogits_amax_l4_b8_20_20260620T181000Z.run.log. Total
  profiled train time regressed from 3.794s to 3.811s. The 100-step screen was
  effectively flat in runtime and slightly worse in validation loss.
decision:
  Reject before the 900-second gate. Code was reverted to the promoted
  scheduler.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Reuse cross-entropy dlogits row amax for final-head MS-EDEN
  quantization.
status: accepted
change:
  The cross-entropy kernel now records per-row absolute max while it writes
  dlogits. The final LM-head linear backward path uses those row maxima to
  derive the MS-EDEN global scale, avoiding a separate full dlogits amax scan.
  Progress logs now label the loss copy/sync timer as loss_host_wait_ms,
  because it measures host wait around queued CUDA work, not standalone GPU
  kernel time.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture: pass.
  100-step SYNTH screen:
    target/ce_dlogits_amax_l4_b8_100_20260620T180927Z.log
    val_loss=6.543247, train_elapsed_s=19.220, completed_steps=100.
  Default-log 100-step check:
    target/ce_dlogits_amax_default_log_l4_b8_100_20260620T181035Z.log
    val_loss=6.545713, train_elapsed_s=19.323, completed_steps=100.
  20-step nsys screen:
    target/nsys/ce_dlogits_amax_l4_b8_20_20260620T181000Z.run.log
    train_elapsed_s=3.794, completed_steps=20.
  900-second held-out gate:
    target/ce_dlogits_amax_defaultlog_l4_b8_900_20260620T181310Z.log
    val_loss=4.021274, train_elapsed_s=900.162, completed_steps=4558.
measured_effect:
  Against the promoted paired-linear-backward 900-second baseline
  target/paired_linear_backward_l4_b8_900_20260620T170432Z.log, held-out
  validation improved from 4.054840 to 4.021274 and completed steps increased
  from 4539 to 4558. The 20-step nsys screen showed tensor_chunk_amax_f32 work
  dropping from about 19.81ms to about 6.37ms over 20 steps, with no meaningful
  increase in cross_entropy_kernel time.
decision:
  Promote. This is a same-math runtime optimization that passed the 900-second
  held-out validation gate.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Defer optimizer NVFP4 writeback until eval/save/generate boundaries.
status: rejected_pre_gate
change:
  Tested skipping end-of-step NVFP4 requantization for Adam and Aurora optimizer
  updates, then materializing weights before held-out eval. The first version
  used schedule-free interpolation with beta=1.0 at eval; the second version
  used direct x_master requantization to better match the previous eval state.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer --
  --ignored --nocapture: pass, 7 tests.
baseline:
  Current promoted 100-step reference:
    target/reusable_batch_l4_b8_100_20260620T122059Z.log
    val_loss=6.545963, train_elapsed_s=19.350, completed_steps=100.
pre_gate_screens:
  target/defer_optimizer_writeback_l4_b8_100_20260620T152803Z.log
    beta=1.0 eval interpolation
    val_loss=6.549026, train_elapsed_s=19.289, completed_steps=100.
  target/defer_optimizer_writeback_direct_eval_l4_b8_100_20260620T153039Z.log
    direct x_master eval requantization
    val_loss=6.546259, train_elapsed_s=19.305, completed_steps=100.
  target/defer_optimizer_writeback_direct_eval_l4_b8_100_log250_20260620T153113Z.log
    direct x_master eval requantization, TRAIN_LOG_INTERVAL=250
    val_loss=6.549611, train_elapsed_s=19.417, completed_steps=100.
measured_effect:
  Skipping intermediate writeback reduced local optimizer timing, but the
  objective-facing screens did not improve held-out validation. The best screen
  was about 0.045s faster over 100 steps but still worsened validation by
  0.000296.
decision:
  Reject and revert. Do not spend a 900-second gate unless a future version
  proves bit-equivalent eval/training behavior or shows a stronger pre-gate
  validation result.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Replace per-step token upload event sync with a 1024-slot batch ring.
status: rejected_pre_gate
change:
  Replaced the single reusable token/target device buffer and pinned host
  staging buffer with 1024 independent slots. Each slot synchronized only when
  reused, preserving async pinned-copy safety while removing the per-step event
  wait from the upload path.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  cargo build --release: pass.
baseline:
  Current promoted 100-step reference:
    target/reusable_batch_l4_b8_100_20260620T122059Z.log
    val_loss=6.545963, train_elapsed_s=19.350, completed_steps=100.
pre_gate_screens:
  target/batch_ring1024_l4_b8_100_20260620T151813Z.log
    TRAIN_LOG_INTERVAL=25
    val_loss=6.548292, train_elapsed_s=19.482, completed_steps=100.
  target/batch_ring1024_l4_b8_100_log250_20260620T151851Z.log
    TRAIN_LOG_INTERVAL=250
    val_loss=6.550458, train_elapsed_s=19.573, completed_steps=100.
measured_effect:
  Both screens worsened held-out validation loss and runtime. The sparse
  logging screen was also slower, so the ring did not improve the target path
  despite reducing the intended per-step upload synchronization.
decision:
  Reject and revert. Do not spend a 900-second gate on this form of host batch
  staging.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Unroll Aurora momentum orientation loop four-wide.
status: rejected_after_900_second_gate
change:
  Replaced the per-thread scalar momentum/orientation stride loop in the fused
  Aurora mega update with a four-wide unrolled elementwise loop. The optimizer
  math and state layout were intended to be unchanged.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  cargo build --release: pass.
pre_gate_screen:
  target/aurora_momentum_unroll4_l4_b8_100_20260620T150045Z.log
  val_loss=6.545153, train_elapsed_s=19.348, completed_steps=100.
baseline:
  Current promoted 100-step reference:
    target/reusable_batch_l4_b8_100_20260620T122059Z.log
    val_loss=6.545963, train_elapsed_s=19.350, completed_steps=100.
  Current promoted 900-second baseline:
    target/reusable_batch_l4_b8_900_20260620T122227Z.log
    val_loss=4.023637, completed_steps=4522.
measured_result:
  target/aurora_momentum_unroll4_l4_b8_900_20260620T150122Z.log
  val_loss=4.070844, train_elapsed_s=900.019, completed_steps=4520.
measured_effect:
  The 100-step pre-gate looked slightly better, but the full fixed-wall
  held-out validation gate regressed by 0.047207 and completed two fewer steps.
decision:
  Reject and revert. Do not use the 100-step improvement as promotion evidence.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Increase current L4 Aurora cooperative blocks from 90 to 120.
status: rejected_pre_gate
change:
  Rebuilt the current promoted L4/B8/d1024 baseline with
  AURORA_COOPERATIVE_BLOCKS=120 and AURORA_MATRIX_PHASES=4. This preserves the
  same four active matrices per Aurora phase and only changes cooperative CTA
  participation per matrix.
verification:
  AURORA_COOPERATIVE_BLOCKS=120 AURORA_MATRIX_PHASES=4 cargo build --release:
  pass.
  100-step SYNTH screen:
    target/aurora_blocks120_p4_l4_b8_100_20260620T144633Z.log
    val_loss=6.547503, train_elapsed_s=19.282, completed_steps=100.
baseline:
  Current promoted pre-gate reference:
    target/reusable_batch_l4_b8_100_20260620T122059Z.log
    val_loss=6.545963, train_elapsed_s=19.350, completed_steps=100.
measured_effect:
  The candidate was about 0.068s faster over 100 steps, but held-out
  validation loss worsened by 0.001540. The optimization target is validation
  loss over fixed wall-clock, so the speed-only gain is not enough to justify a
  900-second gate.
decision:
  Do not promote and do not spend a 900-second gate. Restore the default
  promoted baseline geometry, AURORA_COOPERATIVE_BLOCKS=90 and
  AURORA_MATRIX_PHASES=4.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Source-backed loop-state transformer with loop_count=2.
status: never_properly_tested
change:
  Implemented the requested looped-transformer shape as a loop state rather
  than physical block repetition: z0 starts from token embeddings, the second
  pass enters the same physical block stack as embedding + z1, forward tape was
  shaped per loop, and backward ran loop 2 normally then loop 1 with shared
  parameter-gradient accumulation into the same physical weights.
verification:
  cargo fmt --all --check: pass.
  GPT2_LOOP_COUNT=2 cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 GPT2_LOOP_COUNT=2 cargo test -p gpt2-nvfp4 --test
  block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 GPT2_LOOP_COUNT=2 cargo test -p gpt2-nvfp4 --test
  l3_mlp -- --ignored --nocapture: pass.
  One-step SYNTH launch:
    target/loop_state2_l4_b8_1_20260620T143715Z.log
    val_loss=10.362991, finite=true, completed_steps=1.
  100-step SYNTH screen:
    target/loop_state2_l4_b8_100_20260620T143756Z.log
    val_loss=8.954576, train_elapsed_s=28.413, completed_steps=100.
measured_effect:
  The source-backed loop_count=2 candidate was much worse than the promoted
  baseline 100-step screen target/reusable_batch_l4_b8_100_20260620T122059Z.log,
  which had val_loss=6.545963 and train_elapsed_s=19.350. It also slowed the
  pre-gate by about 9.063s.
decision:
  Do not promote and do not spend a 900-second gate. Code was reverted to the
  promoted baseline. This result should be treated as evidence that the local
  implementation was likely wrong or incomplete. It is not a valid test of the
  looped-transformer architecture.
```

```text
date: 2026-06-20
commit: this commit
experiment: Fuse row amax and four/six rowwise activation quantization.
status: rejected_pre_gate
change:
  Added a fused rowwise derived-amax four/six quantization kernel for
  attention-output and MLP-activation requantization. The candidate preserved
  the old amax scratch side effect after the attention test caught the missing
  write.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l2_attention --
  --ignored --nocapture: pass after restoring the amax side effect.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored
  --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored
  --nocapture: pass.
pre_gate_screen:
  target/fused_rowwise_amax_quant_l4_b8_100_20260620T141740Z.log
  val_loss=6.549544, train_elapsed_s=19.305, completed_steps=100.
baseline:
  target/reusable_batch_l4_b8_100_20260620T122059Z.log
  val_loss=6.545963, train_elapsed_s=19.350, completed_steps=100.
measured_effect:
  Runtime improved by about 0.045 seconds over 100 steps, but held-out
  validation loss moved in the wrong direction by 0.003581.
decision:
  Do not run the 900-second gate. Code was reverted to the promoted baseline.
```

```text
date: 2026-06-20
commit: this commit
experiment: Direct MLP FP32 activation writes into forward tape.
status: rejected_pre_gate
change:
  During training, routed the MLP up-projection pre-activation and relu2 output
  directly into the block forward tape buffers instead of writing to shared
  scratch and copying both 134 MB tensors into tape afterward.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored
  --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward
  -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored
  --nocapture: pass.
pre_gate_screen:
  target/mlp_direct_tape_l4_b8_100_20260620T141011Z.log
  val_loss=6.548380, train_elapsed_s=19.199, completed_steps=100.
baseline:
  target/reusable_batch_l4_b8_100_20260620T122059Z.log
  val_loss=6.545963, train_elapsed_s=19.350, completed_steps=100.
measured_effect:
  Runtime improved by about 0.151 seconds over 100 steps, but held-out
  validation loss moved in the wrong direction by 0.002417.
decision:
  Do not run the 900-second gate. Code was reverted to the promoted baseline.
```

```text
date: 2026-06-20
commit: this commit
experiment: Source-faithful looped transformer boundary with loop_count=2.
status: never_properly_tested
change:
  Tested a source-style z = F(x + z) loop boundary for GPT blocks with
  GPT2_LOOP_COUNT=2. The implementation kept weights physical-layer sized,
  saved activation tape per logical pass, added the saved embedding residual at
  the second loop boundary, propagated loop-boundary gradients into both the
  previous loop state and the embedding gradient, and folded tied logical
  parameter gradients into the physical block before clipping/optimizer update.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  residual_backward -- --ignored --nocapture: pass, 2 tests.
  Default loop_count=1 forward test:
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored
  --nocapture: pass.
launch_check:
  target/loop_source_l4_b8_1_20260620T135518Z.log
  finite=true, val_loss=10.344292, completed_steps=1.
pre_gate_screen:
  target/loop_source_l4_b8_100_20260620T135526Z.log
  val_loss=7.301424, train_elapsed_s=28.490, completed_steps=100.
baseline:
  target/reusable_batch_l4_b8_100_20260620T122059Z.log
  val_loss=6.545963, train_elapsed_s=19.350, completed_steps=100.
measured_effect:
  The source-faithful loop_count=2 candidate was much worse on held-out
  validation at 100 steps and was about 9.140 seconds slower over the same
  step count.
decision:
  Do not run the 900-second gate. Code was reverted to the promoted baseline.
  Loop count 2 is the right source-backed starting count for this architecture
  family, but this run should be treated as an invalid/incomplete local
  implementation, not as a proper architecture test.
```

```text
date: 2026-06-20
commit: this commit
experiment: AMUSE beta from future averaging coefficient.
status: measured, rejected after 900-second gate; code reverted
implementation:
  Tested replacing the current README closed-form beta_t with the current
  official implementation shape from github.com/kjeiun/amuse:
  compute c_{t+1} from the same schedule-free averaging weight used for x,
  remember c_warmup, then set beta_t from
  c_{t+1}(1-c_warmup)/(c_warmup(1-c_{t+1})).
  This changed only host-side schedule-free materialization beta; optimizer
  grouping, LR scales, batch size, model shape, and kernels stayed unchanged.
correctness_checks:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
pre_gate_screen:
  target/amuse_ckp_beta_l4_b8_100_20260620T113954Z.log
  val_loss=6.577282, train_elapsed_s=19.420, completed_steps=100.
baseline:
  target/grad_clip_l4_b8_900_20260620T101626Z.log
  val_loss=4.044528, completed_steps=4520.
measured_result:
  target/amuse_ckp_beta_l4_b8_900_20260620T114040Z.log
  stopped_by_wall_clock=true, val_loss=4.152211, completed_steps=4522.
measured_effect:
  Held-out validation loss worsened by 0.107683 while completing only two more
  steps in the same 900-second wall-clock budget.
runtime_effect:
  Runtime was effectively unchanged versus the accepted baseline.
stability:
  Finite for the full 900-second run.
decision:
  Reject and revert. Although this matches the current official AMUSE code shape
  more closely than the README closed form, it is worse for this repo's fixed
  wall-clock held-out objective.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: NVFP4 projection CTA 64x64 tile.
status: measured, rejected after 900-second gate; code reverted
implementation:
  Tested widening the shared NVFP4 projection CTA tile from 32x32 to 64x64.
  Each CTA kept 8 warps but each warp computed four N-subtiles, reusing staged
  A fragments across more output columns. This touched lm-head, QKV/MLP
  projection, attention projection, and linear backward projection users of the
  shared projection CTA body.
correctness_checks:
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head
  -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward
  -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward
  -- --ignored --nocapture: pass.
pre_gate_screen:
  target/projection_cta64_split_l4_b8_100_20260620T111351Z.log
  train_elapsed_s=18.830, val_loss=6.548421, completed_steps=100.
diagnostic_profile:
  target/nsys_projection_cta64_l4_b8_20_20260620T111055Z_stats.txt
  linear_backward_projection_cta_device_scale_kernel improved from 768.198ms
  to 662.972ms over 20 profiled steps.
  lm_head_kernel improved from 141.203ms to 117.436ms over 20 profiled steps.
baseline:
  target/grad_clip_l4_b8_900_20260620T101626Z.log
  val_loss=4.044528, completed_steps=4520.
measured_result:
  target/projection_cta64_l4_b8_900_20260620T111425Z.log
  val_loss=4.047818, completed_steps=4674.
measured_effect:
  Completed 154 more steps in the same wall-clock budget, but held-out
  validation loss worsened by 0.003290.
runtime_effect:
  The projection kernels were faster, but the fixed-wall-clock objective did
  not improve.
stability:
  Finite for the full 900-second run.
decision:
  Reject and revert. More steps and faster projection kernels do not qualify
  when held-out validation is worse than the current baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Aurora phase/block geometry p2/c45.
status: measured, rejected before profile and 900-second gate
implementation:
  Built with AURORA_MATRIX_PHASES=2 and AURORA_COOPERATIVE_BLOCKS=45.
  This keeps the same total L4 cooperative scheduling capacity as the baseline
  p4/c90 build, but halves per-kernel phase loops and blocks per matrix.
baseline:
  target/grad_clip_l4_b8_100_20260620T101549Z.log
  train_elapsed_s=19.436, val_loss=6.548097.
measured_result:
  target/aurora_p2_c45_l4_b8_100_20260620T110342Z.log
  train_elapsed_s=21.095, val_loss=6.546229, completed_steps=100.
measured_effect:
  100-step held-out validation was effectively unchanged, but runtime worsened
  by 1.659 seconds over 100 steps.
runtime_effect:
  Slower than baseline on the objective-facing pre-gate screen.
stability:
  Finite for 100 steps.
decision:
  Reject. Do not profile or run the 900-second gate for this geometry because
  the pre-gate runtime filter regressed.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Add non-constant Aurora rectangular recurrence test.
status: test-only guardrail; no training-path change
implementation:
  Added a five-iteration wide rectangular Aurora GPU test using non-constant
  gradients and a CPU reference for the Polar Express recurrence with f16-rounded
  matmul inputs. This covers a vector-valued update instead of the previous
  constant-gradient scalar recurrence.
evidence:
  cargo check --workspace --tests: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer
  aurora_mega_update_matches -- --ignored --nocapture: pass, 4 tests.
limitation:
  This guard did not catch the rejected wide-orientation rewrite when tested
  locally, so it must not be cited as proof for that specific failure mode. It is
  still useful coverage for future Aurora math changes because it verifies a
  non-constant five-iteration rectangular update path.
decision:
  Keep as correctness coverage only. It does not improve held-out validation
  loss and does not make any candidate promotable.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Remove wide-matrix double transpose inside Aurora momentum orientation.
status: measured, rejected; code reverted
target:
  Test whether QKV and MLP-up Aurora updates can avoid writing Nesterov momentum
  transposed and then transposing it back during Polar normalization.
implementation_tested:
  momentum_orient wrote all matrices in original row-major order, and
  run_polar_step treated the scratch source as untransposed. Tall matrices still
  transposed inside Polar normalization through the existing rows > cols path.
verification_before_screen:
  cargo check --workspace --tests: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer
  aurora_mega_update_matches -- --ignored --nocapture: pass, 3 tests.
measured_result:
  target/aurora_no_wide_double_transpose_100_20260620T104824Z.log
  SYNTH, B8/L4/d1024/h16, current baseline hyperparameters.
  step 25 loss=NaN finite=false nonzero=false.
  heldout_eval split=val val_loss=NaN completed_steps=100.
decision:
  Revert the code candidate. The focused constant-gradient Aurora tests were too
  weak to prove training-path equivalence, and the sustained 100-step screen
  failed quickly. Do not retry this exact orientation rewrite without a stronger
  non-constant rectangular Aurora recurrence test and a clear derivation of the
  orientation expected by the update path.
```

```text
date: 2026-06-20
commit: this commit
experiment: Global GPU gradient clipping before optimizer update.
status: measured, promoted
implementation:
  Added a parameter-gradient pointer table built at buffer initialization.
  Each optimizer step accumulates global gradient squared norm on GPU, computes
  clip scale for max norm 1.0 on GPU, and scales all parameter-gradient buffers
  in place before Adam/Aurora update kernels consume them. This clips LM head,
  layer norm weights/biases, linear weights, and linear biases together.
correctness_checks:
  cargo check --workspace --tests
  cargo oxide build --arch sm_120a
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer \
    global_clip_scales_all_gradient_buffers_together -- --ignored --nocapture
  100-step SYNTH screen: target/grad_clip_l4_b8_100_20260620T101549Z.log
baseline:
  target/no_backward_clear_l4_b8_900_20260620T090607Z.log
  heldout_val_loss=4.069893
  completed_steps=4535
measured_result:
  target/grad_clip_l4_b8_900_20260620T101626Z.log
  heldout_val_loss=4.044528
  completed_steps=4520
  stopped_by_wall_clock=true
measured_effect:
  Held-out validation loss improved by 0.025365.
runtime_effect:
  Completed 15 fewer optimizer steps in the same 900-second budget.
stability:
  Finite for the full 900-second run.
decision:
  Promote GPU global gradient clipping because it improves the fixed-wall-clock
  held-out validation objective despite the small step-count decrease.
```

```text
date: 2026-06-20
commit: note only
experiment: Retain per-step TokenBatch device buffers until explicit sync.
status: screened, rejected before 900-second gate
implementation:
  Tested holding per-step token/target DeviceBuffers past the training step and
  clearing them only after explicit sync/log-retainer points. This was intended
  to remove the hidden cuMemFree synchronization from every step without
  changing sample order or training math.
diagnostic_profile:
  Before: target/nsys_grad_clip_l4_b8_20_20260620T103258Z_stats.txt
    cuMemFree_v2=3726.943ms over 769 calls.
  Candidate: target/nsys_retained_batches_l4_b8_20_20260620T103657Z_stats.txt
    cuMemFree_v2=36.684ms over 769 calls.
measured_effect:
  API-level cuMemFree time dropped sharply, but the wait shifted into launch
  and explicit stream synchronization while GPU work remained the limiter.
screen:
  Baseline 100-step screen:
    target/grad_clip_l4_b8_100_20260620T101549Z.log
    train_elapsed_s=19.436, val_loss=6.548097.
  Candidate 100-step screen:
    target/retained_batches_l4_b8_100_20260620T103735Z.log
    train_elapsed_s=19.482, val_loss=6.547900.
decision:
  Revert the code candidate. It cleaned up API attribution but did not improve
  objective-facing wall-clock progress enough to justify the 900-second gate.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Local two-loop four-physical-block shortcut.
status: measured, rejected
implementation:
  Tested GPT2_LOOP_COUNT=2 as four trainable physical blocks reused for eight
  logical passes. Forward tape and backward activation gradients were logical
  pass sized. Parameter gradients from pass i and i + GPT2_N_LAYER were folded
  into the matching physical block before optimizer update. This was not a
  source-faithful looped-transformer implementation.
baseline:
  target/attention_backward_no_transpose_l4_b8_900_20260620T081345Z.log
  heldout_val_loss=4.077696
  completed_steps=4483
measured_result:
  target/loop2_l4_b8_900_20260620T084124Z.log
  heldout_val_loss=4.200931
  completed_steps=3047
  stopped_by_wall_clock=true
measured_effect:
  Held-out validation loss worsened by 0.123235.
runtime_effect:
  Completed 1436 fewer optimizer steps in the same 900-second budget.
stability:
  Finite for the full 900-second run.
decision:
  Do not promote this local shortcut. The paper-backed language-model loop
  count is two, but this implementation was arbitrary relative to the source
  architecture and loses the fixed-wall-clock validation target versus the
  current L4 baseline.
```

```text
date: 2026-06-20
commit: this commit
experiment: Remove materialized attention-backward P and dS transposes.
status: measured, promoted
implementation:
  Added an f32-input TC matmul variant that reads the left operand as a
  logical transpose from row-major source while reading the RHS row-major.
  Used it for dK = dS^T @ Q and dV = P^T @ dO, then deleted the old
  transpose_matrix_kernel path and the unused q_t/k_t/d_out_t/p_t/ds_t
  attention scratch buffers.
baseline:
  target/attention_tc_forward_l4_b8_900_20260620T075050Z.log
  heldout_val_loss=4.104358
  completed_steps=4331
measured_result:
  target/attention_backward_no_transpose_l4_b8_900_20260620T081345Z.log
  heldout_val_loss=4.077696
  completed_steps=4483
  stopped_by_wall_clock=true
measured_effect:
  Completed 152 more steps in the same 900-second budget.
  Held-out validation loss improved by 0.026662.
stability:
  Finite for the full 900-second run.
decision:
  Promote this candidate as the current baseline.
```

```text
date: 2026-06-20
commit: this commit
experiment: Forward causal attention through TC matmul.
status: measured, promoted
implementation:
  Replaced scalar forward causal attention with a compact TC path:
  gather Q/K/V, compute QK^T through f16 TC matmul, apply causal softmax
  and log-sum-exp, compute P@V through f16 TC matmul, then scatter back to
  hidden layout. Also removed three small attention-backward transpose uses by
  adding a row-major RHS f32 TC matmul variant.
baseline:
  target/aurora_stage_unroll_l4_900s_20260620T062251Z.log
  heldout_val_loss=4.133352
  completed_steps=3756
measured_result:
  target/attention_tc_forward_l4_b8_900_20260620T075050Z.log
  heldout_val_loss=4.104358
  completed_steps=4331
  stopped_by_wall_clock=true
measured_effect:
  Completed 575 more steps in the same 900-second budget.
  Held-out validation loss improved by 0.028994.
stability:
  Finite for the full 900-second run.
decision:
  Promote this candidate as the current baseline.
```

```text
date: 2026-06-20
commit: this commit
experiment: Matched-compute two-loop transformer block tying.
status: measured, not promoted
implementation:
  GPT2_LOOP_COUNT=2 on the current L4/B8/d1024 baseline. This gives two
  unique physical blocks applied as four logical passes in order 0,1,0,1.
  Logical activation tapes stay per pass. Parameter gradients from the second
  loop are accumulated into the matching physical block gradient buffers before
  optimizer update. Residual projection init uses loop-aware scaling.
baseline:
  target/aurora_stage_unroll_l4_900s_20260620T062251Z.log
  heldout_val_loss=4.133352
  completed_steps=3756
measured_result:
  target/loop_count2_l4_900s_20260620T071755Z.log
  heldout_val_loss=4.226233
  completed_steps=4102
  stopped_by_wall_clock=true
measured_effect:
  Completed 346 more steps than baseline in the same 900-second budget.
  Held-out validation loss was worse by 0.092881.
stability:
  Finite for the full 900-second run.
decision:
  Do not promote this two-loop tied-block architecture. It improved step count
  but failed the optimization target: lower held-out validation loss over fixed
  wall-clock.
```

```text
date: 2026-06-20
commit: committed
experiment: Unroll fixed Aurora Polar Express shared-memory staging loads.
status: promoted at fixed-wall validation
hypothesis:
  Aurora's Polar Express TC matmul staging path loaded CTA_A_ELEMS and
  CTA_B_ELEMS with two fixed-count while loops. With CTA_ELEMS=1024 and
  CTA_THREADS=256, each thread performs exactly four staging offsets for A and
  four for B. Explicitly unrolling those loads should reduce loop overhead in
  the active optimizer path without changing optimizer math.
implementation:
  Replaced the two staging while loops in
  crates/cuda-kernels/src/gpt/optimizer/aurora/polar/fused/stage.rs with four
  explicit stage_a calls and four explicit stage_b calls.
measured_effect:
  100-step sanity:
    target/aurora_stage_unroll_l4_100step_20260620T062209Z.log
    train_elapsed_s=23.429, val_loss=6.605456.
stability_effect:
  The run stayed finite/nonzero. The Aurora optimizer GPU recurrence tests
  passed after regenerating sm_120a PTX.
validation_result:
  target/aurora_stage_unroll_l4_900s_20260620T062251Z.log
  stopped_by_wall_clock=true elapsed_s=900.090 completed_steps=3756.
  heldout_eval split=val val_loss=4.133352 train_elapsed_s=900.329
  completed_steps=3756.
comparison:
  Previous measured baseline:
    target/square_cta_tape_l4_900s_20260620T055702Z.log
    val_loss=4.136663, completed_steps=3692.
  The staging unroll completed 64 more steps and improved validation loss by
  0.003311 under the same 900-second wall-clock gate.
decision:
  Promote this as the current baseline. Continue looking for active-path
  square/TC optimizer improvements before changing model architecture.
```

```text
date: 2026-06-20
commit: committed
experiment: Audit attempted c_proj tape CTA route and retain best repeated 900-second baseline.
status: baseline updated, optimization claim invalidated
hypothesis:
  The regular attention c_proj path used the CTA projection body, but the
  tape-producing c_proj path still used the old one-warp 16x8 projection body.
  Moving that active training path to the CTA accumulator should preserve math
  while improving the fixed-wall validation outcome.
implementation:
  Added CTA residual/tape store support and ran validation, then audited call
  sites before continuing. CProjTapeArgs and c_proj_tape had zero callers in
  the forward path. The actual backward tape stores the quantized c_proj input
  through RowwiseNvfp4Tape::save before c_proj runs; the dead c_proj_tape API
  stored projection output instead. The dead API and residual/tape CTA helpers
  were removed.
measured_effect:
  100-step sanity:
    target/square_cta_tape_l4_100step_20260620T055627Z.log
    train_elapsed_s=23.798, val_loss=6.606071.
stability_effect:
  Focused ignored GPU tests passed for l2_attention, l3_mlp, forward,
  lm_head, and linear_backward_projection_cta after regenerating sm_120a PTX.
  The 100-step and 900-second runs stayed finite/nonzero.
validation_result:
  target/square_cta_tape_l4_900s_20260620T055702Z.log
  stopped_by_wall_clock=true elapsed_s=900.014 completed_steps=3692.
  heldout_eval split=val val_loss=4.136663 train_elapsed_s=900.257
  completed_steps=3692.
comparison:
  Previous CTA-forward baseline:
    target/square_cta_forward_l4_900s_20260620T052934Z.log
    val_loss=4.158232, completed_steps=3693.
  The attempted c_proj_tape change did not affect the active training path, so
  the lower validation number is a repeated measurement of the same CTA-forward
  path, not proof of this attempted optimization.
decision:
  Do not promote c_proj_tape as a separate optimization. Retain the lower
  validation run as the current measured baseline because the optimization
  target is best held-out validation loss over the fixed wall-clock budget.
  Future code changes must beat target/square_cta_tape_l4_900s_20260620T055702Z.log.
```

```text
date: 2026-06-20
commit: committed
experiment: Use 32x32 CTA NVFP4 projection tiles for forward attention and MLP projections.
status: promoted at fixed-wall validation
hypothesis:
  The forward QKV, attention c_proj, MLP up, and MLP down projections were still
  using the one-warp 16x8 projection body. Reusing the existing 32x32 CTA
  projection tile for these square-friendly forward projections should improve
  wall-clock step count without changing the model math.
implementation:
  Added CTA projection bodies for affine and ReLU2 stores and routed regular
  attention/MLP forward projection launches through the CTA config. The
  tape-specific attention residual projection path stays on the old warp body.
measured_effect:
  100-step sanity:
    baseline restored L4: target/l4_floor_restored_100step_20260620T052029Z.log
      train_elapsed_s=25.318, val_loss=6.603828.
    CTA forward: target/square_cta_forward_l4_100step_20260620T052842Z.log
      train_elapsed_s=23.786, val_loss=6.606071.
stability_effect:
  Focused ignored GPU tests passed for l2_attention, l3_mlp, forward,
  lm_head, and linear_backward_projection_cta. The 100-step and 900-second
  runs stayed finite/nonzero.
validation_result:
  target/square_cta_forward_l4_900s_20260620T052934Z.log
  stopped_by_wall_clock=true elapsed_s=900.047 completed_steps=3693.
  heldout_eval split=val val_loss=4.158232 train_elapsed_s=900.290
  completed_steps=3693.
comparison:
  Previous L4 floor baseline:
    target/l4_min_length_candidate_900s_20260620T031205Z.log
    val_loss=4.181291, completed_steps=3467.
  CTA forward completed 226 more steps and improved validation loss by
  0.023059 under the same 900-second wall-clock gate.
decision:
  Promote this as the current L4 square/uniform baseline. Continue optimizing
  square-friendly TC paths; do not reintroduce non-uniform layer widths without
  square grouped projection kernels.
```

```text
date: 2026-06-20
commit: committed
experiment: Back off failed non-uniform/active-width work to the L4 floor.
status: rollback to last stable L4-minimum patch
measured_effect:
  The later active-width/non-uniform branch produced NaN during the short L4
  sanity path, so it is not a candidate for the 900-second validation gate.
stability_effect:
  Restore the codebase to the L4-minimum baseline state: GPT2_N_LAYER defaults
  to 4, build.rs rejects layer counts below 4, and the sweep machinery ignores
  sub-L4 history/candidates.
runtime_effect:
  No new runtime claim. This rollback removes the failed branch rather than
  promoting a speed change.
validation_result:
  Current baseline remains target/l4_min_length_candidate_900s_20260620T031205Z.log
  with val_loss=4.181291 and completed_steps=3467.
sustained_check:
  target/l4_floor_restored_100step_20260620T052029Z.log
  completed 100 steps, finite=true and nonzero=true at step 99.
  heldout_eval split=val val_loss=6.603828 train_elapsed_s=25.318
  completed_steps=100.
next:
  Continue optimization from the square/uniform L4 baseline. Do not reintroduce
  non-uniform layer widths until the square grouped projection kernels exist.
```

```text
date: 2026-06-20
commit: pending
experiment: Fresh post-AMUSE-beta coupled sweep, trial 2.
status: promoted at fixed-wall validation
hypothesis:
  After correcting the AMUSE beta_t formula, the old sweep history was no
  longer valid for proposing candidates. A clean coupled sweep seeded only with
  the post-formula baseline could find a better schedule/runtime point without
  mixing stale pre-formula results.
implementation:
  Used a temporary target-local seed file containing only the promoted
  post-AMUSE-beta baseline. Ran a coupled sweep over model shape, Aurora
  cooperative launch shape, and optimizer schedule fields. The sweep was
  stopped after trial 3 had started, but trial 2 had already completed a full
  fixed-wall validation and was automatically promoted by the sweep harness.
sweep:
  target/sweeps/post_amuse_beta_900_20260620T020520Z
completed_trials:
  trial_0000:
    key=b8_l8_d1536_h12_p16_c80_lr0.5814_alr2.2599_w5_s0.00_b0.60_r1.00
    val_loss=5.801490, completed_steps=636, rejected.
  trial_0001:
    key=b8_l4_d1024_h16_p16_c80_lr0.9708_alr1.7773_w20_s0.10_b0.20_r0.80
    val_loss=4.728477, completed_steps=1979, rejected.
  trial_0002:
    key=b8_l2_d1024_h16_p2_c90_lr2.2089_alr1.4141_w20_s0.20_b0.60_r0.50
    val_loss=3.995972, completed_steps=5962, promoted.
validation_result:
  target/sweeps/post_amuse_beta_900_20260620T020520Z/trial_0002/train.log
  stopped_by_wall_clock=true elapsed_s=900.060 completed_steps=5962.
  heldout_eval split=val val_loss=3.995972 train_elapsed_s=900.210
  completed_steps=5962.
comparison:
  Previous promoted baseline:
    target/amuse_beta_formula_b8_l2d1024_900s_20260620T014632Z.log
    val_loss=4.030268, completed_steps=5674.
  Trial 2 completed 288 more steps and improved validation loss by 0.034296.
decision:
  Keep the sweep promotion. The new baseline is b8_l2_d1024_h16 with
  AURORA_MATRIX_PHASES=2, AURORA_COOPERATIVE_BLOCKS=90,
  TRAIN_LR_SCALE=2.208900, TRAIN_ADAM_LR_SCALE=1.414137,
  TRAIN_LR_WARMUP_STEPS=20, TRAIN_LR_START_RATIO=0.200000,
  TRAIN_AMUSE_BETA1=0.600000, TRAIN_AMUSE_RHO=0.500000.
```

```text
date: 2026-06-20
commit: pending
experiment: Replace derived AMUSE beta schedule with the published AMUSE formula.
status: promoted at fixed-wall validation
hypothesis:
  The previous beta_t implementation derived beta from the schedule-free
  averaging coefficient. The AMUSE paper/repo define the post-warmup schedule
  directly as beta_t = 1 - ((T0 - 1) / (t - 1))^rho * (1 - beta1). Using the
  published gradient-evaluation interpolation should improve validation loss.
implementation:
  Changed only schedule_free_beta. Warmup behavior and all default hyperparams
  were left unchanged.
source:
  https://github.com/kjeiun/amuse
  README Method Overview lines define Y_t = (1 - beta_t) Z_t + beta_t X_t and
  beta_t = beta1 during warmup, then
  1 - ((T0 - 1) / (t - 1))^rho * (1 - beta1).
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
sustained_check:
  target/amuse_beta_formula_100step_synth_20260620T014604Z.log
  heldout_eval split=val val_loss=6.723605 at 100 steps; finite and nonzero.
validation_result:
  target/amuse_beta_formula_b8_l2d1024_900s_20260620T014632Z.log
  stopped_by_wall_clock=true elapsed_s=900.072 completed_steps=5674.
  heldout_eval split=val val_loss=4.030268 train_elapsed_s=900.230
  completed_steps=5674.
comparison:
  Previous promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The candidate completed 9 fewer steps but validation loss improved by
  0.099685, so it is a promotion under the held-out objective.
decision:
  Keep the formula change and update notes/sweep_baseline.env to the new
  baseline. Future candidates must compare against val_loss=4.030268 at the
  same 900-second wall-clock gate.
```

```text
date: 2026-06-20
commit: not committed
experiment: BF16 Polar Express tensor-core fragments in Aurora.
status: rejected at fixed-wall validation
hypothesis:
  Match official AMUSE more closely by running Aurora's Polar Express
  orthogonalization matmuls with BF16 TC fragments instead of FP16 fragments.
  The official AMUSE reference casts the matrix update to bfloat16 before the
  Newton-Schulz / Polar Express iteration.
implementation:
  Added a temporary mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 wrapper
  and a cvt.rn.bf16.f32 converter. Routed only the Aurora Polar Express staging
  and tile compute path through BF16. The general attention-backward f16 TC
  helper path was unchanged.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  The generated PTX contained both mma.sync ... bf16.bf16 and cvt.rn.bf16.f32.
  CUDA_DEVICE_INDEX=0 cargo test --test optimizer -- --ignored --nocapture:
  pass after updating the Aurora scalar test reference to BF16 operand rounding.
sustained_check:
  target/bf16_polar_100step_synth_20260620T011946Z.log
  heldout_eval split=val val_loss=6.886864 at 100 steps; finite and nonzero.
validation_result:
  target/bf16_polar_b8_l2d1024_900s_20260620T012011Z.log
  stopped_by_wall_clock=true elapsed_s=900.044 completed_steps=5680.
  heldout_eval split=val val_loss=4.162605 train_elapsed_s=900.202
  completed_steps=5680.
comparison:
  Current promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The candidate completed 3 fewer steps and validation loss regressed by
  0.032652, so it is not a promotion.
decision:
  Revert the code. Matching AMUSE's BF16 Polar precision did not improve the
  current 15-minute held-out validation objective.
```

```text
date: 2026-06-20
commit: not committed
experiment: AMUSE-style second-moment-only Adam fallback for non-matrix tensors.
status: rejected at fixed-wall validation
hypothesis:
  Match the official AMUSE non-Muon fallback more closely by removing the
  Adam first-moment buffer from non-matrix tensors and using the current
  gradient divided by the bias-corrected second-moment RMS. Aurora matrix
  weights were unchanged.
implementation:
  Temporarily removed first_moment, beta1, and beta1_correction from the
  AdamW update args, kernel, optimizer state, diagnostics, and optimizer test.
  The Adam fallback updated z_master with grad / sqrt(v_hat) and then used the
  existing schedule-free x_master average plus NVFP4 requantization path.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test --test optimizer -- --ignored --nocapture:
  pass when run from crates/cuda-kernels so the generated PTX path resolved.
sustained_check:
  target/sf_adamw_aux_100step_synth_20260620T005503Z.log
  heldout_eval split=val val_loss=6.606052 at 100 steps; finite and nonzero.
validation_result:
  target/sf_adamw_aux_b8_l2d1024_900s_20260620T005537Z.log
  stopped_by_wall_clock=true elapsed_s=900.060 completed_steps=5692.
  heldout_eval split=val val_loss=4.140446 train_elapsed_s=900.218
  completed_steps=5692.
comparison:
  Current promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The candidate completed 9 more steps but validation loss regressed by
  0.010493, so it is not a promotion.
decision:
  Revert the code. Official fallback alignment improved the early 100-step
  check, but did not improve the 15-minute held-out validation objective under
  the current baseline.
```

```text
date: 2026-06-20
commit: not committed
experiment: Global parameter-gradient clipping at norm 1.0.
status: rejected at fixed-wall validation
hypothesis:
  Match the llm.c/nanoGPT stability practice of clipping global gradient norm
  to 1.0 before optimizer updates. The implementation clipped optimizer
  parameter gradients after the tied embedding lookup gradient was accumulated,
  so the tied token/LM-head gradient was clipped as one combined gradient.
implementation:
  Added GPU-only gradient clipping with a chunk pointer table over parameter
  gradients: token/LM-head tied gradient, layer-norm weight/bias gradients,
  attention and MLP matrix gradients, and all projection bias gradients.
  Intermediate activation gradients were not included because they are not
  optimizer parameters. The clip path used a device norm accumulator and scaled
  gradients in place; there was no GPU-to-CPU norm readback per step.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer --
  --ignored --nocapture: pass, 7 tests including the new grad-clip primitive.
sustained_check:
  target/grad_clip_global_norm_100step_synth_20260620T002611Z.log
  heldout_eval val_loss=6.928789 at 100 steps; finite and nonzero.
  The clip overhead logged around 0.004-0.009 ms per logged step.
validation_result:
  target/grad_clip_global_norm_b8_l2d1024_900s_20260620T002649Z.log
  stopped_by_wall_clock=true elapsed_s=900.004 completed_steps=5654.
  heldout_eval split=val val_loss=4.178542 train_elapsed_s=900.162
  completed_steps=5654.
comparison:
  Current promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The candidate completed 29 fewer steps and validation loss regressed by
  0.048589, so it is not a promotion.
decision:
  Revert the code. A fixed global clip norm of 1.0 is too restrictive for the
  current AMUSE/Aurora setup. If clipping is revisited, it should be as a
  coupled sweep dimension with LR/optimizer settings, not as a single manual
  knob.
```

```text
date: 2026-06-20
commit: not committed
experiment: Hoist QKV address arithmetic in scalar causal attention.
status: rejected at fixed-wall validation
hypothesis:
  Preserve the existing scalar causal attention algorithm and arithmetic order,
  but precompute batch, head, Q, K, and V base addresses once per block instead
  of recomputing qkv_index through q_value/k_value/v_value helpers inside the
  key loops.
implementation:
  causal_attention_kernel reused qkv_stride, batch_base, head_offset, q_base,
  k_base, and v_base in the existing score and value loops. Added a CPU
  reference forward-attention GPU test for nonzero QKV values.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  causal_attention_log_sum_exp -- --ignored --nocapture: pass, 3 tests.
sustained_check:
  target/causal_index_hoist_100step_synth_20260620T000346Z.log
  heldout_eval val_loss=6.881987 at 100 steps; finite and nonzero.
profile_effect:
  target/nsys/causal_index_hoist_b8_l2d1024_20_20260619T235953Z_kernels_cuda_gpu_kern_sum.csv:
    causal_attention_kernel: 387.022 ms / 42 launches.
  Current promoted baseline profile:
    target/nsys/f16_staged_attention_bwd_cleanup_b8_l2d1024_20_20260619T221654Z_kernels_cuda_gpu_kern_sum.csv:
    causal_attention_kernel: 417.039 ms / 42 launches.
  The candidate reduced forward causal attention time by about 30.0 ms over the
  20-step profile.
validation_result:
  target/causal_index_hoist_b8_l2d1024_900s_20260620T000410Z.log
  stopped_by_wall_clock=true elapsed_s=900.000 completed_steps=5687.
  heldout_eval split=val val_loss=4.162406 train_elapsed_s=900.158
  completed_steps=5687.
comparison:
  Current promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The candidate completed 4 more steps but validation loss regressed by
  0.032453, so it is not a promotion.
decision:
  Revert the code. Even small exact-addressing/codegen changes can move the
  training trajectory; promotion still requires the fixed-wall validation gate.
```

```text
date: 2026-06-19
commit: not committed
experiment: Online-softmax causal attention forward kernel.
status: rejected at fixed-wall validation
hypothesis:
  Replace the scalar forward causal attention kernel's shared score buffer plus
  separate max, denominator, and value rescans with an online softmax recurrence
  over keys. This preserves causal attention semantics, writes the same
  log-sum-exp quantity for backward, and should reduce forward attention time.
implementation:
  causal_attention_kernel kept one block per (batch, head, query), but computed
  running score_max, denominator, and value accumulation as each key score was
  produced. Added a GPU test comparing forward outputs and log_sum_exp against
  a CPU reference.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  causal_attention_log_sum_exp -- --ignored --nocapture: pass, 3 tests.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored
  --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward
  -- --ignored --nocapture: pass.
sustained_check:
  target/online_causal_attention_100step_synth_20260619T234512Z.log
  heldout_eval val_loss=6.884402 at 100 steps; finite and nonzero.
profile_effect:
  target/nsys/online_causal_attention_b8_l2d1024_20_20260619T234529Z_kernels_cuda_gpu_kern_sum.csv:
    causal_attention_kernel: 333.522 ms / 42 launches.
  Current promoted baseline profile:
    target/nsys/f16_staged_attention_bwd_cleanup_b8_l2d1024_20_20260619T221654Z_kernels_cuda_gpu_kern_sum.csv:
    causal_attention_kernel: 417.039 ms / 42 launches.
  The candidate reduced forward causal attention time by about 83.5 ms over the
  20-step profile.
validation_result:
  target/online_causal_attention_b8_l2d1024_900s_20260619T234555Z.log
  stopped_by_wall_clock=true elapsed_s=900.021 completed_steps=5797.
  heldout_eval split=val val_loss=4.139523 train_elapsed_s=900.176
  completed_steps=5797.
comparison:
  Current promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The candidate completed 114 more steps but validation loss regressed by
  0.009570, so it is not a promotion under the fixed-wall objective.
decision:
  Revert the code. Runtime-only improvement is insufficient when held-out
  validation regresses.
```

```text
date: 2026-06-19
commit: not committed
experiment: Disable AdamW weight decay for layer norm vectors and biases.
status: rejected at fixed-wall validation
hypothesis:
  Match nanoGPT/llm.c parameter grouping by applying AdamW decay only to the
  tied token embedding matrix on the Adam path, while leaving layer-norm weights
  and all bias vectors with zero Adam weight decay. Aurora matrix weights were
  unchanged.
implementation:
  Made Adam weight decay explicit at each Adam call site. Token embeddings used
  ADAM_WEIGHT_DECAY=0.005; layer-norm weights, layer-norm biases, QKV biases,
  attention c_proj biases, MLP up biases, and MLP down biases used zero decay.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
sustained_check:
  target/no_decay_vectors_100step_synth_20260619T232536Z.log
  heldout_eval val_loss=6.883940 at 100 steps; finite and nonzero.
validation_result:
  target/no_decay_vectors_b8_l2d1024_900s_20260619T232614Z.log
  stopped_by_wall_clock=true elapsed_s=900.123 completed_steps=5684.
  heldout_eval split=val val_loss=4.163834 train_elapsed_s=900.281
  completed_steps=5684.
comparison:
  Current promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The candidate completed one extra step but validation loss regressed by
  0.033881, so it is not a promotion.
decision:
  Revert the code change. Keep the measured result as a rejected quality
  experiment; do not retry as a single-variable manual tweak.
```

```text
date: 2026-06-19
commit: not committed
experiment: Add fixed-budget cosine LR decay.
status: rejected at fixed-wall validation
target:
  Replace warmup-only LR multipliers with warmup plus cosine decay to
  min_ratio=0.1. The decay endpoint was read from the promoted baseline's
  COMPLETED_STEPS instead of TRAIN_STEPS, so changing a run cap would not alter
  the schedule. Adam and Aurora used the same multiplier, and the
  schedule-free averaging weight used the decayed multiplier as well.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
sustained_check:
  target/cosine_decay_schedule_100step_synth_20260619T230340Z.log
  heldout_eval val_loss=6.882954 at 100 steps; finite and nonzero.
  The progress log confirmed decay was active:
    step 99 adam_lr=3.958472e-4 aurora_lr=1.013410e-4.
validation_result:
  target/cosine_decay_schedule_b8_l2d1024_900s_20260619T230408Z.log
  stopped_by_wall_clock=true elapsed_s=900.101 completed_steps=5677.
  heldout_eval split=val val_loss=4.582866 train_elapsed_s=900.259
  completed_steps=5677.
comparison:
  Current promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The cosine schedule completed 6 fewer steps and validation loss regressed by
  0.452913. It undertrained the current AMUSE/Aurora setup over the 15-minute
  SYNTH budget.
decision:
  Reverted the code. Do not promote or commit this schedule.
```

```text
date: 2026-06-19
commit: not committed
experiment: Add aligned no-bounds CTA projection fast path.
status: rejected at fixed-wall validation
target:
  Remove inner row/column/K bounds checks from the shared NVFP4 CTA projection
  path when token_count, input_dim, and output_dim are exact multiples of the
  CTA M/N/K tile sizes. Route linear backward and LM head through the aligned
  entry points for the current B8 L2 d1024 h16 shape.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head --
  --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored
  --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward
  -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward
  -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward
  -- --ignored --nocapture: pass.
sustained_check:
  target/projection_cta_aligned_100step_synth_20260619T224449Z.log
  heldout_eval val_loss=6.882009 at 100 steps; finite and nonzero.
profile_effect:
  Baseline:
    target/nsys/f16_staged_attention_bwd_cleanup_kernel_sum.csv
    linear_backward_projection_cta_device_scale_kernel: 521.094 ms / 360
    lm_head_kernel: 138.648 ms / 21
  Candidate:
    target/nsys/projection_cta_aligned_b8_l2d1024_20_20260619T224511Z.nsys-rep
    linear_backward_projection_cta_aligned_device_scale_kernel: 471.774 ms / 360
    lm_head_aligned_kernel: 131.170 ms / 21
    total kernel time: 3064.753 ms vs baseline 3117.344 ms
validation_result:
  target/projection_cta_aligned_b8_l2d1024_900s_20260619T224543Z.log
  stopped_by_wall_clock=true elapsed_s=900.153 completed_steps=5781.
  heldout_eval split=val val_loss=4.149405 train_elapsed_s=900.309
  completed_steps=5781.
comparison:
  Current promoted baseline:
    target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
    val_loss=4.129953, completed_steps=5683.
  The candidate completed 98 more steps but validation loss regressed by
  0.019452, so it fails the fixed-wall objective.
decision:
  Reverted the code. Do not promote or commit this change.
```

```text
date: 2026-06-19
commit: this commit
experiment: Stage attention-backward FP32 operands as FP16 inside the TC tile load.
status: validated and promoted
target:
  Remove the separate fp32_to_f16_kernel launches from the attention-backward
  TC matmul path. The TC matmuls still run as f16.f16 with FP32 accumulators,
  but operands are converted while staging each CTA tile into shared memory.
code_change:
  Added a plain f32-input f16 TC matmul kernel using the existing shared-memory
  f32 staging helpers. Routed the attention-backward score and gradient
  matmuls through that entry point and removed attention-only half/padded
  matmul scratch buffers from the app and tests.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test
  causal_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test
  block_attention_backward -- --ignored --nocapture: pass.
sustained_check:
  target/f16_staged_attention_bwd_cleanup_100step_synth_20260619T221626Z.log
  heldout_eval val_loss=6.878532 at 100 steps; finite and nonzero.
profile_effect:
  Baseline:
    target/nsys/rope_preapply_b8_l2d1024_20_20260619T203528Z.nsys-rep
    fp32_to_f16_kernel: 68.082 ms / 400
    f16_cta_tc_matmul_kernel: 155.812 ms / 200
  Candidate:
    target/nsys/f16_staged_attention_bwd_cleanup_b8_l2d1024_20_20260619T221654Z.nsys-rep
    fp32_to_f16_kernel: 0.000 ms / 0
    f16_cta_tc_matmul_f32_kernel: 166.486 ms / 200
  The local attention-backward TC matmul section improved by about 57.4 ms over
  20 steps, but whole-profile kernel time was flat within noise:
    baseline total kernel time: 3113.264 ms
    candidate total kernel time: 3117.344 ms
validation_result:
  target/f16_staged_attention_bwd_b8_l2d1024_900s_20260619T222024Z.log
  stopped_by_wall_clock=true elapsed_s=900.033 completed_steps=5683.
  heldout_eval split=val val_loss=4.129953 train_elapsed_s=900.191
  completed_steps=5683.
comparison:
  Previous promoted baseline:
    target/resid_proj_scaled_init_b8_l2d1024_900s_20260619T215342Z.log
    val_loss=4.143612, completed_steps=5580.
  F32-staged attention backward improves held-out validation loss by 0.013659
  and completes 103 more steps under the same 900-second budget.
decision:
  Keep and promote. notes/sweep_baseline.env, notes/sweep_seed.tsv, and
  notes/sweep_seed_current.tsv were updated to this result.
```

```text
date: 2026-06-19
commit: not committed
experiment: Scale residual projection initialization.
status: validated and promoted
target:
  Apply nanoGPT-style residual projection initialization to the block residual
  projections while keeping the current Llama-2 tokenizer, B8 L2 d1024 h16
  shape, optimizer settings, and kernels unchanged.
code_change:
  Added a scaled NVFP4 smooth initializer and used
  0.02 / sqrt(2 * GPT2_N_LAYER) for attention c_proj and MLP c_proj weights.
  QKV, MLP up, token embeddings, layer norms, and biases keep their previous
  initialization.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
sustained_check:
  target/resid_proj_scaled_init_100step_synth_20260619T215309Z.log
  heldout_eval val_loss=6.876441 at 100 steps; finite and nonzero.
validation_result:
  target/resid_proj_scaled_init_b8_l2d1024_900s_20260619T215342Z.log
  stopped_by_wall_clock=true elapsed_s=900.050 completed_steps=5580.
  heldout_eval split=val val_loss=4.143612 train_elapsed_s=900.211
  completed_steps=5580.
comparison:
  Previous promoted baseline:
    target/rope_preapply_b8_l2d1024_900s_20260619T203854Z.log
    val_loss=4.224687, completed_steps=5577.
  Residual projection scaling improves held-out validation loss by 0.081075
  and completes 3 more steps under the same 900-second budget.
decision:
  Keep and promote. notes/sweep_baseline.env, notes/sweep_seed.tsv, and
  notes/sweep_seed_current.tsv were updated to this result.
```

```text
date: 2026-06-19
commit: not committed
experiment: Fuse Aurora momentum/orientation with Polar Express norm chunks.
status: rejected at profile gate; no meaningful Aurora speedup
target:
  Avoid one full read/reduction pass over each Aurora matrix by accumulating
  the Nesterov/oriented Frobenius norm chunks while writing the oriented matrix,
  then starting Polar Express normalization from those chunks.
code_change_tested:
  Replaced momentum_orient with momentum_orient_sum, removed the separate
  normalize_source_to_x summation entry point, and routed Polar Express through
  a pre-summed-source entry point.
correctness_checks:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer --
  --ignored --nocapture: pass.
sustained_check:
  target/aurora_norm_fused_100step_synth_20260619T215021Z.log
  heldout_eval val_loss=7.386400 at 100 steps; finite and nonzero.
profile_effect:
  Baseline:
    target/nsys/rope_preapply_b8_l2d1024_20_20260619T203528Z.nsys-rep
    aurora_mega_update_cooperative_kernel: 969.237 ms / 20
  Candidate:
    target/nsys/aurora_norm_fused_b8_l2d1024_20_20260619T215048Z.nsys-rep
    aurora_mega_update_cooperative_kernel: 968.412 ms / 20
  Aurora improved by only 0.825 ms over 20 steps, and short wall-clock timing
  was slightly worse.
decision:
  Reverted the code change. Do not run a 900-second validation for this isolated
  change; the profile effect is too small to matter for the fixed-wall
  validation objective.
```

```text
date: 2026-06-19
commit: not committed
experiment: Fuse attention-backward FP32 transpose plus FP16 cast.
status: rejected; fixed-wall validation loss regressed
target:
  Remove the attention-backward path that materialized q_t/k_t/d_out_t/p_t/ds_t
  as FP32 and then converted those transposed buffers to FP16 inside the TC
  matmul launcher.
code_change_tested:
  Added a generic f16_transpose_rows_kernel and prepared-FP16 TC matmul entry
  point. Routed the three attention-backward gradient matmuls through
  transpose-to-FP16 half buffers and removed the old attention FP32 transpose
  scratch fields.
correctness_checks:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test
  causal_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test
  block_attention_backward -- --ignored --nocapture: pass.
sustained_check:
  target/attention_bwd_half_transpose_100step_synth_20260619T212834Z.log
  heldout_eval val_loss=7.383965 at 100 steps; finite and nonzero.
profile_effect:
  Baseline:
    target/nsys/rope_preapply_b8_l2d1024_20_20260619T203528Z.nsys-rep
    transpose_matrix_kernel: 77.314 ms / 200
    fp32_to_f16_kernel: 68.082 ms / 400
  Candidate:
    target/nsys/attention_bwd_half_transpose_b8_l2d1024_20_20260619T212858Z.nsys-rep
    f16_transpose_rows_kernel: 81.862 ms / 200
    fp32_to_f16_kernel: 25.417 ms / 200
  The local transpose/cast path improved by about 38.1 ms over 20 steps, but
  total profiled runtime moved only about one percent.
validation_result:
  target/attention_bwd_half_transpose_b8_l2d1024_900s_20260619T212948Z.log
  stopped_by_wall_clock=true elapsed_s=900.089 completed_steps=5641.
  heldout_eval split=val val_loss=4.231810 train_elapsed_s=900.248
  completed_steps=5641.
comparison:
  Current promoted baseline:
    target/rope_preapply_b8_l2d1024_900s_20260619T203854Z.log
    val_loss=4.224687, completed_steps=5577.
  The candidate completed 64 more steps but validation loss was worse by
  0.007123.
decision:
  Reverted the code change. Do not repeat this as an isolated same-math
  attention-backward launch cleanup; the fixed-wall validation target did not
  improve.
```

```text
date: 2026-06-19
commit: not committed
experiment: Widen NVFP4 projection CTA from 32x32 to 32x64.
status: rejected; fixed-wall validation loss regressed
target:
  Reduce CTA count and improve A-tile reuse for large projection outputs,
  especially linear_backward_projection_cta_device_scale_kernel and
  lm_head_kernel, without changing training math.
code_change_tested:
  crates/cuda-kernels/src/utils/mma/projection_cta/tile.rs changed
  NVFP4_PROJECTION_CTA_N from 32 to 64 and derived a 512-thread, 16-warp
  mapping from N / 8.
correctness_checks:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head --
  --ignored --nocapture: pass.
profile_effect:
  Baseline:
    target/nsys/rope_preapply_b8_l2d1024_20_20260619T203528Z.nsys-rep
    linear_backward_projection_cta_device_scale_kernel: 496.495 ms / 360
    lm_head_kernel: 132.395 ms / 21
  N=64 candidate:
    target/nsys/projection_cta_n64_b8_l2d1024_20_20260619T210050Z.nsys-rep
    linear_backward_projection_cta_device_scale_kernel: 431.362 ms / 360
    lm_head_kernel: 112.262 ms / 21
  Short-profile runtime improved, and the 100-step safety run stayed finite:
    target/projection_cta_n64_100step_synth_20260619T210027Z.log
    heldout_eval val_loss=7.387764 at 100 steps.
validation_result:
  target/projection_cta_n64_b8_l2d1024_900s_20260619T210117Z.log
  stopped_by_wall_clock=true elapsed_s=900.113 completed_steps=5755.
  heldout_eval split=val val_loss=4.240527 train_elapsed_s=900.269
  completed_steps=5755.
comparison:
  Current promoted baseline:
    target/rope_preapply_b8_l2d1024_900s_20260619T203854Z.log
    val_loss=4.224687, completed_steps=5577.
  N=64 completed 178 more steps but validation loss was worse by 0.015840.
decision:
  Revert the tile change. Do not repeat this as a same-math optimization unless
  paired with a substantive projection-kernel redesign that changes the
  validation result, not just isolated kernel time.
```

```text
date: 2026-06-19
commit: 002a9e82
experiment: Pre-apply RoPE to Q/K once after QKV projection.
status: validated and promoted
measured_effect:
  Before:
    target/nsys/direct_all_linear_operands_b8_l2d1024_20_20260619T200422Z.nsys-rep
    causal_attention_kernel: 431.699 ms / 42 launches
  After:
    target/nsys/rope_preapply_b8_l2d1024_20_20260619T203528Z.nsys-rep
    causal_attention_kernel: 397.725 ms / 42 launches
    apply_rope_kernel: 0.789 ms / 42 launches
  Net attention-side reduction: about 33.2 ms over 20 training steps.
stability_effect:
  100-step SYNTH check stayed finite and nonzero:
    target/rope_prequest_100step_synth_20260619T203459Z.log
    heldout_eval val_loss=7.381578 at 100 steps.
validation_result:
  target/rope_preapply_b8_l2d1024_900s_20260619T203854Z.log
  stopped_by_wall_clock=true elapsed_s=900.085 completed_steps=5577.
  heldout_eval split=val val_loss=4.224687 train_elapsed_s=900.245
  completed_steps=5577.
comparison:
  Previous promoted baseline was:
    target/rowwise_direct_input_t_b8_l2d1024_900s_20260619T194141Z.log
    val_loss=4.238420, completed_steps=5503.
  RoPE pre-apply improves held-out validation loss by 0.013733 and completes
  74 more steps under the same 900-second wall-clock budget.
implementation:
  Q and K in the saved qkv activation buffer are now post-RoPE; V is unchanged.
  The forward causal attention kernel reads Q/K directly. The TC backward gather
  also reads Q/K directly, while scatter still inverse-rotates dQ/dK into the
  raw QKV projection gradient coordinates.
next_justified_experiment:
  Fuse the RoPE store into the QKV projection store path if the extra launch
  becomes visible at larger batch/model sizes. Larger remaining bottlenecks are
  still Aurora update, linear backward projection, and LM-head/logits.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Direct MS-EDEN packing for FP32 E^T in linear backward.
status: implemented and profiled; not promoted by itself
target:
  Remove the materialized FP32 transpose of the linear-backward error matrix
  before MS-EDEN quantization. This preserves the Quartet II backward operand
  contract: E is still quantized with derived device scale, and E^T uses the
  same device scale as E.
code_change:
  Added fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel and routed
  LinearBackwardModule::backward_ms_eden to pack E^T directly from row-major E.
  Removed the MLP, attention, and final-head transpose_f32(e -> e_t) calls.
verification:
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass, 2 tests
  cargo check --all-targets: pass
sustained_check:
  target/ms_eden_transpose_100step_synth.log
  CUDA_DEVICE_INDEX=0, SYNTH, 100 steps.
  completed_steps=100, heldout_eval val_loss=7.383308,
  finite=true and nonzero=true through final logged step.
profile:
  target/nsys/ms_eden_transpose_direct_b8_l2d1024_20_20260619T191437Z.nsys-rep
  target/nsys/ms_eden_transpose_direct_b8_l2d1024_20_20260619T191437Z_kernels_cuda_gpu_kern_sum.csv
measured_effect:
  Previous current-code profile
  target/nsys/lm_head_cta_b8_l2d1024_20_20260619T153905Z_stats.txt had:
    fp32_to_nvfp4_ms_eden_device_scale_kernel: 720 launches, 289.640ms
    transpose_f32_kernel: 180 launches, 55.902ms
  New profile has:
    fp32_to_nvfp4_ms_eden_device_scale_kernel: 540 launches, 170.992ms
    fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel: 180 launches, 118.797ms
    transpose_f32_kernel: absent from the kernel summary
  Total MS-EDEN E/E^T packing time is effectively unchanged, and the
  materialized transpose cost is removed. The measured operand-prep improvement
  is about the removed 55.9ms over the 20-step profile.
validation_loss_result:
  target/ms_eden_transpose_b8_l2d1024_900s_20260619T191732Z.log
  heldout_eval val_loss=4.244381, completed_steps=5476.
  Previous promoted baseline was val_loss=4.243804, completed_steps=5376.
  The change improved completed steps but did not improve held-out validation
  loss by itself, so it was not promoted as the objective baseline.
next_justified_experiment:
  Continue the same operand-prep rewrite for saved rowwise NVFP4 activations:
  pack X^T for linear backward directly into MS-EDEN instead of materializing a
  FP32 decoded transpose.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Direct MS-EDEN packing for saved rowwise NVFP4 X^T in linear backward.
status: implemented, profiled, and promoted as current validation baseline
target:
  Remove decode_rowwise_t(saved activation -> FP32 X^T) before MS-EDEN
  quantization in MLP, attention, and final head backward linear paths. The
  new path derives the Quartet/MS-EDEN device global scale from decoded rowwise
  source values on GPU, then packs the transposed source directly into the
  existing LinearBackwardMsEdenScratch input_t_h operand.
code_change:
  Added rowwise_nvfp4_chunk_amax_kernel and
  rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel under the
  MS-EDEN quantization module. LinearBackwardModule::backward_ms_eden now
  accepts LinearBackwardInputTranspose::RowwiseNvfp4 for saved activation
  sources and routes MLP, attention, and final-head input_t operands through
  the direct rowwise source path.
verification:
  cargo fmt && cargo check --all-targets: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass, 2 tests
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass, 2 tests
sustained_check:
  target/rowwise_direct_input_t_100step_synth.log
  CUDA_DEVICE_INDEX=0, SYNTH, 100 steps.
  completed_steps=100, heldout_eval val_loss=7.386125,
  finite=true and nonzero=true through final logged step.
profile:
  target/nsys/rowwise_direct_input_t_b8_l2d1024_20_20260619T194122Z.nsys-rep
  target/nsys/rowwise_direct_input_t_b8_l2d1024_20_20260619T194122Z_kernels.csv
measured_effect:
  The old nvfp4_decode_rowwise_transpose_f32_kernel is absent from the 20-step
  kernel summary. New rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel
  costs 37.243ms over 180 launches. The 100-step training check improved
  train_elapsed_s from the previous direct-E^T check's 16.114s to 15.962s.
validation_loss_result:
  target/rowwise_direct_input_t_b8_l2d1024_900s_20260619T194141Z.log
  heldout_eval val_loss=4.238420, completed_steps=5503.
  Previous promoted baseline was val_loss=4.243804, completed_steps=5376.
  This is a measured improvement in the fixed 900-second held-out validation
  objective, so notes/sweep_baseline.env was updated to this result.
next_justified_experiment:
  Direct W^T packing is implemented as a candidate below. Because it increases
  completed steps but had a worse first seeded validation result, evaluate it
  across repeated deterministic seeds before deciding whether to promote or
  remove it.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Direct MS-EDEN packing for stored NVFP4 W^T in linear backward.
status: implemented and profiled; mixed first validation seed, candidate retained
target:
  Remove decode_weight_t(stored NVFP4 weight -> FP32 W^T) before MS-EDEN
  quantization in MLP, attention, and final-head backward linear paths. The new
  route decodes source values inside the MS-EDEN packing kernel, derives the
  Quartet/MS-EDEN device global scale on GPU, and emits the same packed operand
  consumed by linear_backward_projection_cta_device_scale_kernel.
code_change:
  Added nvfp4_chunk_amax_kernel and
  nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel. LinearBackwardModule
  accepts LinearBackwardWeightTranspose::Nvfp4 and routes stored NVFP4 weights
  through the direct W^T packer. The materialized FP32 path remains available
  for tests through LinearBackwardWeightTranspose::Fp32.
verification:
  cargo fmt && cargo check --all-targets: pass
  cargo oxide build --arch sm_120a: pass
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass, 3 tests
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass, 2 tests
sustained_check:
  target/direct_all_linear_operands_100step_synth.log
  completed_steps=100, heldout_eval val_loss=7.379606,
  finite=true and nonzero=true through final logged step.
  target/direct_all_linear_operands_seed_override_rebuilt_100step_synth.log
  with TRAIN_SEED=0x47505433 completed_steps=100, heldout_eval val_loss=7.401977,
  finite=true and nonzero=true through final logged step.
seed_control:
  TRAIN_SEED is now a runtime override for the model/init/backward RNG seed and
  is recorded in run_info.txt. Verified in
  target/runs/20260619_202503Z_synth_900s/run_info.txt:
    seed=0x47505433
    TRAIN_SEED=0x47505433
profile:
  target/nsys/direct_all_linear_operands_b8_l2d1024_20_20260619T200422Z.nsys-rep
  target/nsys/direct_all_linear_operands_b8_l2d1024_20_20260619T200422Z_kernels.csv
measured_effect:
  nvfp4_decode_transpose_f32_kernel fell to 27 tiny calls in the 20-step
  profile. The new nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel cost
  17.058ms over 180 launches. The 100-step run completed in 15.916s versus
  15.962s for the direct-X^T path.
validation_loss_result:
  target/direct_all_linear_operands_b8_l2d1024_900s_20260619T200449Z.log
  heldout_eval val_loss=4.259685, completed_steps=5515.
  Current promoted baseline is val_loss=4.238420, completed_steps=5503.
interpretation:
  Do not promote from this single seed. Also do not treat it as a hard
  rejection yet: it improves completed steps, and with a fixed random seed a
  single validation sample can be noisy. The correct decision surface is the
  mean held-out validation loss over repeated deterministic seeds at the same
  900-second budget, biased slightly toward faster variants when validation is
  statistically close.
next_justified_experiment:
  Add a repeated-seed validation harness for architecture/kernel variants so
  direct W^T can be compared against the promoted direct-X^T baseline by
  validation mean/variance and completed steps, not one lucky or unlucky seed.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Switch LM head from one-warp projection to CTA NVFP4 projection.
status: kept as a kernel speedup; not a validation-objective win under the existing schedule
target:
  Reuse the existing 32x32 CTA NVFP4 projection body for the tied LM head,
  replacing the older 16x8 one-warp projection path. The math and output layout
  stay the same: row-major FP32 logits from NVFP4 activations and tied NVFP4
  token embedding weights.
code_change:
  crates/cuda-kernels/src/gpt/lm_head.rs now launches with
  projection_cta_grid_dim and NVFP4_PROJECTION_CTA_THREADS, and calls
  nvfp4_projection_cta_nobias_kernel_body with shared packed/scales staging.
short_check:
  target/lm_head_cta_b8_l2d1024_100steps_20260619T153841Z.log
  Shape B8 L2 d1024 h16, Aurora phases=2 blocks=90, current best schedule.
  completed_steps=100, train_elapsed_s=15.545, heldout_eval val_loss=7.285653,
  finite and nonzero.
profile_result:
  target/nsys/lm_head_cta_b8_l2d1024_20_20260619T153905Z.nsys-rep
  lm_head_kernel fell from 322.568587ms over 21 calls in the previous
  no-logits-copy profile to 141.260135ms over 21 calls. The 20-step train loop
  moved from 3.367s to 3.181s.
validation_objective:
  target/lm_head_cta_b8_l2d1024_900s_20260619T153928Z.log
  stopped_by_wall_clock=true elapsed_s=900.113 completed_steps=5655.
  heldout_eval split=val val_loss=4.747964 train_elapsed_s=900.271
  completed_steps=5655.
comparison:
  The previous best remains
  target/mseden_fwht_b8_l2d1024_900s_20260619T145904Z.log with val_loss=4.623164
  and completed_steps=5429. The CTA LM-head path completed 226 more steps in
  the same 900-second budget, but the unchanged step-based optimizer schedule
  ended at worse held-out validation loss.
decision:
  Keep the kernel speedup as an efficiency improvement, but do not count it as
  progress on the validation-loss objective until a coupled schedule/optimizer
  sweep beats 4.623164 under the same 900-second validation gate. Future
  candidates should account for the faster step rate rather than reusing this
  exact schedule blindly.
history:
  notes/sweep_seed.tsv includes the 4.747964 result so future Bayesian/Pareto
  sweeps treat this as measured evidence, not an untested win.
  The default sweep seed was moved to notes/sweep_seed_current.tsv after this
  kernel change because notes/sweep_seed.tsv also contains pre-LM-head-CTA
  measurements, including the same hyperparameter key with a different loss.
  Keeping the default on the current-code seed avoids feeding stale objective
  values into the proposer.
verification:
  cargo fmt --check: pass.
  cargo check -p rust-kernels-cuda --tests: pass.
  cargo check -p gpt2-nvfp4 --tests: pass.
  cargo check --bin rust-kernels: pass.
  GPT2_BATCH_SIZE=8 GPT2_N_LAYER=2 GPT2_N_EMBD=1024 GPT2_N_HEAD=16
  AURORA_MATRIX_PHASES=2 AURORA_COOPERATIVE_BLOCKS=90 cargo oxide build
  --arch sm_120a: pass.
  Matching cargo build --release: pass.
  100-step SYNTH check: pass, finite through heldout_eval.
  900-second SYNTH validation run: pass, finite through heldout_eval.
next_experiment:
  Run a coupled sweep around the faster LM-head path. Do not manually lower
  only one LR knob; vary LR scale, Adam LR scale, warmup/start, AMUSE
  parameters, and possibly Aurora layout together against the 900-second
  held-out validation objective.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Fuse final-head cross-entropy with dlogits transpose.
status: rejected and reverted
target:
  Remove the final-head dlogits transpose launch by having cross-entropy write
  both row-major dlogits and transposed dlogits_t for the LM-head backward
  GEMMs.
code_change_tested:
  Added a cross_entropy_with_transposed_grad kernel that wrote the existing
  row-major dlogits plus dlogits_t[col, row] in the same pass. Final-head
  backward then skipped transpose_dlogits.
correctness_check:
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test loss -- --ignored
  --nocapture: pass while the experiment was applied.
sustained_check:
  target/fused_ce_dlogits_t_b8_l2d1024_100steps_20260619T153440Z.log
  Shape B8 L2 d1024 h16, Aurora phases=2 blocks=90, current best schedule.
  completed_steps=100, train_elapsed_s=16.516, heldout_eval val_loss=7.286378,
  finite and nonzero.
profile_result:
  target/nsys/fused_ce_dlogits_t_b8_l2d1024_20_20260619T153505Z.nsys-rep
  cross_entropy_with_transposed_grad_kernel took 118.193215ms over 20 calls.
  The previous no-logits-copy profile had cross_entropy_kernel at 55.803301ms
  over 21 calls and transpose_f32_kernel at 56.081499ms over 180 calls. After
  fusion, transpose_f32_kernel dropped to 20.626525ms over 160 calls, meaning
  the removed final-head transpose accounted for about 35.45ms over 20 steps.
measured_effect:
  The fused CE kernel added about 62ms of CE time while removing only about
  35ms of final-head transpose time over 20 steps. 20-step train time also
  moved from 3.367s in the previous no-logits-copy profile to 3.384s in the
  fused profile.
decision:
  Reverted. Writing dlogits_t from CE creates scattered stores and is slower
  than keeping the dedicated transpose kernel. Do not retry this exact fusion
  unless the write layout changes.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Remove the duplicate final-logits tape copy.
status: cleanup kept; not a meaningful objective improvement
target:
  Reduce obvious logits-path memory traffic without changing model math,
  optimizer state, data order, sampling, or schedule parameters.
code_change:
  The forward pass already writes final logits into src/training/buffers.rs
  TrainBuffers::logits. Backward now saves a reference to that buffer instead
  of copying the full logits tensor into a second tape-owned logits allocation.
measured_effect:
  nsys 20-step profile after the change:
  target/nsys/no_logits_tape_copy_b8_l2d1024_20_20260619T152748Z.nsys-rep.
  CUDA memcpy Device-to-Device time was 32.047033ms over 1431 operations.
  The prior FWHT profile showed about 61.89ms of D2D memcpy over 20 steps, so
  this removed roughly 30ms of D2D work per 20-step profile.
runtime_effect:
  No material wall-clock improvement. The comparable 100-step run with current
  best schedule finished in 16.498s:
  target/no_logits_tape_copy_best_b8_l2d1024_100steps_20260619T152720Z.log.
  The earlier corrected d1024 100-step check was 16.555s and the FWHT-only
  100-step check was 16.180s, so this is within run-to-run noise.
validation_result:
  100-step held-out val_loss=7.288118, finite, nonzero updates. This matches
  the current 100-step band but is not a new 900-second validation result.
verification:
  cargo fmt --check: pass.
  cargo check -p gpt2-nvfp4 --tests: pass.
  cargo check --bin rust-kernels: pass.
  cargo test --bin sweep: pass.
  GPT2_BATCH_SIZE=8 GPT2_N_LAYER=2 GPT2_N_EMBD=1024 GPT2_N_HEAD=16
  AURORA_MATRIX_PHASES=2 AURORA_COOPERATIVE_BLOCKS=90 cargo oxide build
  --arch sm_120a: pass.
  Matching cargo build --release: pass.
next_experiment:
  Do not spend more time on logits tape copies unless a later profile changes
  the cost picture. Current top kernels remain Aurora mega update, linear
  backward projection CTA, causal attention, LM head, and MS-EDEN quantization.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Add multi-variable Bayesian/Pareto sweep harness.
status: implementation complete; dry-run verified
decision:
  Use this harness for future hyperparameter/shape tuning instead of manual
  single-variable experiments. The sweep treats batch size, LR, Adam LR,
  warmup/start, AMUSE schedule parameters, model depth/width/head count, Aurora
  phase count, and cooperative block count as coupled candidate variables.
measured_effect:
  No training result changed by this implementation. It adds build-time shape
  env generation, run metadata, seeded history, a TPE-style candidate proposer,
  and early trial termination when a training log reports NaN or finite=false.
  NaN trials are recorded as failures and scored as bad regions by the proposer;
  dry-run rows are excluded from proposer scoring. Completed real trials append
  to the sweep-local trials.tsv and automatically sync into the shared measured
  history file, so future candidates see prior results without manual copying.
seeded_history:
  notes/sweep_seed.tsv records the known 900-second L4/B8/LR1.0 success and
  the known L4/B8/LR1.5 and L4/B4/LR1.5 NaN failures so the sweep does not
  rediscover those exact points.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo test --bin sweep: pass
  dry-run sweep wrote coupled candidates and trials.tsv without launching
  training:
    cargo run --bin sweep -- --trials 3 --random-trials 0 --candidate-samples 8 --dry-run --sweep-dir target/sweeps/dry_run_nan_penalty_check
```

```text
date: 2026-06-19
commit: uncommitted
experiment: First seeded coupled 15-minute SYNTH sweep trial.
status: stable, rejected as a quality regression
decision:
  Keep the result as negative evidence for the proposer. Do not treat this
  lower-LR/lower-Aurora-block candidate as an improvement because held-out
  validation loss was worse than the current stable baseline.
candidate:
  GPT2_BATCH_SIZE=8, GPT2_N_LAYER=4, GPT2_N_EMBD=1536, GPT2_N_HEAD=12.
  AURORA_MATRIX_PHASES=8, AURORA_COOPERATIVE_BLOCKS=120.
  TRAIN_LR_SCALE=0.535733, TRAIN_ADAM_LR_SCALE=0.753612,
  TRAIN_LR_WARMUP_STEPS=5, TRAIN_LR_START_RATIO=0.100000,
  TRAIN_AMUSE_BETA1=0.400000, TRAIN_AMUSE_RHO=0.800000.
result:
  target/sweeps/synth_900_20260619T073445Z/trial_0000/train.log
  stopped_by_wall_clock=true elapsed_s=900.017 completed_steps=1322
  heldout_eval split=val val_loss=5.646966 train_elapsed_s=900.698
  completed_steps=1322.
measured_effect:
  The run stayed finite and nonzero, but validation loss regressed versus the
  stable seeded baseline of 5.496781. It processed 10,829,824 tokens in
  900.698 seconds, about 12,024 tokens/s.
verification:
  cargo oxide build --arch sm_120a: pass inside the sweep runner.
  900-second sweep trial: pass, finite through held-out validation.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Seeded coupled SYNTH sweep trial with L2/high-Aurora-LR candidate.
status: rejected, early NaN
decision:
  Keep this result as bad-region evidence for the proposer. Do not rerun this
  candidate without a new coupled hypothesis.
candidate:
  GPT2_BATCH_SIZE=8, GPT2_N_LAYER=2, GPT2_N_EMBD=1536, GPT2_N_HEAD=12.
  AURORA_MATRIX_PHASES=8, AURORA_COOPERATIVE_BLOCKS=180.
  TRAIN_LR_SCALE=2.196803, TRAIN_ADAM_LR_SCALE=0.538864,
  TRAIN_LR_WARMUP_STEPS=50, TRAIN_LR_START_RATIO=0.200000,
  TRAIN_AMUSE_BETA1=0.200000, TRAIN_AMUSE_RHO=0.500000.
result:
  target/sweeps/synth_900_20260619T075109Z/trial_0000/train.log
  step=300 elapsed_s=117.216 loss=NaN finite=false nonzero=false.
  sweep_early_stop=nan_detected.
measured_effect:
  This coupled candidate learned faster early by train-loss snapshots but
  failed numerically before held-out validation. The sweep runner killed the
  child process immediately after detecting NaN and recorded status=nan with
  completed_steps=301.
verification:
  cargo oxide build --arch sm_120a: pass inside the sweep runner.
  Early-NaN sweep trial: pass as failure detection, not as model quality.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Three-trial coupled SYNTH sweep after shared-history sync.
status: best validation loss improved to 4.805441
decision:
  Keep the L4/B8/d1024/h16, Aurora-120, LR-scale-1.844459 candidate as the
  current best 15-minute SYNTH point. Treat the Aurora-160 candidate as stable
  but worse quality despite higher completed steps.
trial_0000:
  status=success, val_loss=5.346710, completed_steps=2755.
  GPT2_BATCH_SIZE=8, GPT2_N_LAYER=4, GPT2_N_EMBD=1024, GPT2_N_HEAD=16.
  AURORA_MATRIX_PHASES=8, AURORA_COOPERATIVE_BLOCKS=120.
  TRAIN_LR_SCALE=1.040728, TRAIN_ADAM_LR_SCALE=0.671306,
  TRAIN_LR_WARMUP_STEPS=5, TRAIN_LR_START_RATIO=0.200000,
  TRAIN_AMUSE_BETA1=0.200000, TRAIN_AMUSE_RHO=0.800000.
trial_0001:
  status=success, val_loss=4.805441, completed_steps=2750.
  GPT2_BATCH_SIZE=8, GPT2_N_LAYER=4, GPT2_N_EMBD=1024, GPT2_N_HEAD=16.
  AURORA_MATRIX_PHASES=8, AURORA_COOPERATIVE_BLOCKS=120.
  TRAIN_LR_SCALE=1.844459, TRAIN_ADAM_LR_SCALE=0.925681,
  TRAIN_LR_WARMUP_STEPS=5, TRAIN_LR_START_RATIO=0.000000,
  TRAIN_AMUSE_BETA1=0.400000, TRAIN_AMUSE_RHO=0.500000.
trial_0002:
  status=success, val_loss=4.895197, completed_steps=2881.
  GPT2_BATCH_SIZE=8, GPT2_N_LAYER=4, GPT2_N_EMBD=1024, GPT2_N_HEAD=16.
  AURORA_MATRIX_PHASES=8, AURORA_COOPERATIVE_BLOCKS=160.
  TRAIN_LR_SCALE=2.116410, TRAIN_ADAM_LR_SCALE=0.503505,
  TRAIN_LR_WARMUP_STEPS=5, TRAIN_LR_START_RATIO=0.000000,
  TRAIN_AMUSE_BETA1=0.400000, TRAIN_AMUSE_RHO=0.500000.
measured_effect:
  The sweep improved held-out validation loss from the previous stable
  5.496781 baseline to 4.805441. Increasing Aurora cooperative blocks from 120
  to 160 in trial_0002 increased completed steps from 2750 to 2881 but worsened
  validation loss from 4.805441 to 4.895197, so throughput alone was not the
  objective win.
evidence:
  target/sweeps/synth_900_multi_20260619T080010Z/trials.tsv
  target/sweeps/synth_900_multi_20260619T080010Z/trial_0000/train.log
  target/sweeps/synth_900_multi_20260619T080010Z/trial_0001/train.log
  target/sweeps/synth_900_multi_20260619T080010Z/trial_0002/train.log
verification:
  All three trials reached heldout_eval with finite validation loss.
  The shared measured-history file notes/sweep_seed.tsv contains all three
  result rows for future proposer runs.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Follow-up coupled SYNTH sweep with d2048 and L2 candidates.
status: no improvement; best validation loss remains 4.805441
decision:
  Keep the previous L4/B8/d1024/h16, Aurora-120, LR-scale-1.844459 candidate
  as the current best. Treat the d2048 high-LR candidate as a bad region, the
  high-Adam-LR L4/d1024 candidate as a validation regression, and the L2
  candidate as throughput-faster but quality-worse.
trial_0000:
  status=nan, completed_steps=451.
  GPT2_BATCH_SIZE=8, GPT2_N_LAYER=4, GPT2_N_EMBD=2048, GPT2_N_HEAD=16.
  AURORA_MATRIX_PHASES=8, AURORA_COOPERATIVE_BLOCKS=120.
  TRAIN_LR_SCALE=2.409066, TRAIN_ADAM_LR_SCALE=0.682491,
  TRAIN_LR_WARMUP_STEPS=20, TRAIN_LR_START_RATIO=0.000000,
  TRAIN_AMUSE_BETA1=0.400000, TRAIN_AMUSE_RHO=0.800000.
  Failed at step=450 with loss=NaN and sweep_early_stop=nan_detected.
trial_0001:
  status=success, val_loss=5.512575, completed_steps=2764.
  GPT2_BATCH_SIZE=8, GPT2_N_LAYER=4, GPT2_N_EMBD=1024, GPT2_N_HEAD=16.
  AURORA_MATRIX_PHASES=8, AURORA_COOPERATIVE_BLOCKS=120.
  TRAIN_LR_SCALE=0.927102, TRAIN_ADAM_LR_SCALE=2.246357,
  TRAIN_LR_WARMUP_STEPS=5, TRAIN_LR_START_RATIO=0.000000,
  TRAIN_AMUSE_BETA1=0.400000, TRAIN_AMUSE_RHO=0.800000.
trial_0002:
  status=success, val_loss=5.132470, completed_steps=4687.
  GPT2_BATCH_SIZE=8, GPT2_N_LAYER=2, GPT2_N_EMBD=1024, GPT2_N_HEAD=16.
  AURORA_MATRIX_PHASES=4, AURORA_COOPERATIVE_BLOCKS=120.
  TRAIN_LR_SCALE=1.359202, TRAIN_ADAM_LR_SCALE=0.511092,
  TRAIN_LR_WARMUP_STEPS=5, TRAIN_LR_START_RATIO=0.000000,
  TRAIN_AMUSE_BETA1=0.600000, TRAIN_AMUSE_RHO=0.800000.
measured_effect:
  None of these candidates improved held-out validation loss over the current
  best 4.805441. The L2 candidate completed far more steps in the same 900s
  budget, but validation loss was worse, so throughput did not translate to
  the objective. The d2048 candidate was both much slower and numerically
  unstable at this coupled setting.
evidence:
  target/sweeps/synth_900_multi_20260619T132443Z/trials.tsv
  target/sweeps/synth_900_multi_20260619T132443Z/trial_0000/train.log
  target/sweeps/synth_900_multi_20260619T132443Z/trial_0001/train.log
  target/sweeps/synth_900_multi_20260619T132443Z/trial_0002/train.log
verification:
  Trial_0000 triggered the NaN early-stop path.
  Trial_0001 and trial_0002 reached heldout_eval with finite validation loss.
  notes/sweep_seed.tsv contains all three result rows for future proposer runs.
```

Measured batch-size effects so far:

- Increasing batch size by itself has worsened fixed-wall-clock validation loss
  in the measured runs. On the 300-second wide-Llama2 comparison, B4 beat B8
  by validation loss 5.335806 vs 5.438506. B3 and B2 were also worse than B4.
- Increasing batch size improved throughput in some short runs and improved
  stability in the L4 900-second SYNTH runs. L4 B4 LR1.5 and L4 B8 LR1.5 both
  reached NaN at logged step 1250, but B8 reached that step later in wall-clock
  time because it processed larger batches. L4 B8 LR1.0 completed the full
  900-second run finite.
- Therefore, "increase batch size" is not a valid standalone optimization
  experiment. Batch size is a stability/throughput lever that must be swept
  together with LR, schedule, shape, and optimizer parameters to measure the
  Pareto surface against held-out validation loss.

Sweep rule:

- Do not tune one hyperparameter at a time by hand. Use a recorded
  multi-variable Bayesian/Pareto sweep over coupled candidates.
- Notes must state measured effects directly. Do not turn a measured loss
  regression into vague language like "may help" or "undetermined."
- Every trial must record build-time shape, runtime optimizer parameters,
  validation loss, stability/finiteness, completed steps, and log path.

```text
date: 2026-06-19
commit: uncommitted
experiment: L4 B8 fresh-SYNTH 15-minute run with LR scale 1.0.
status: stable, but validation loss still high
decision:
  Keep this as the current stable L4/B8 candidate, not as a quality win. The
  measured effect of LR scale 1.0 at B8 was improved stability versus LR scale
  1.5, with worse early training speed. It completed the 900-second validation
  gate finite but still had high validation loss.
changes:
  GPT2_N_LAYER=4, GPT2_BATCH_SIZE=8, GPT2_N_EMBD=1536, GPT2_N_HEAD=12.
  TRAIN_MAX_SECONDS=900. Default LR scale changed from 1.5 to 1.0. Warmup
  remained 5. SYNTH training streamed from the 400M-token fresh train shard set.
result:
  target/synth_l4_b8_lr1_900s_20260619T062251Z.log
  step 1250 loss=5.174037 finite=true nonzero=true elapsed_s=723.342
  stopped_by_wall_clock=true elapsed_s=900.541 completed_steps=1557
  heldout_eval split=val val_loss=5.496781 train_elapsed_s=901.119
comparison:
  L4 B8 default LR reached NaN at logged step 1250.
  L4 B4 default LR reached NaN at logged step 1250.
  LR scale 1.0 reduced update magnitude enough to avoid the known step-1250 NaN
  in this B8 run, but validation loss is still far above the useful-training
  target.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  1-second launch check: pass
  900-second direct GPU run: pass, finite through held-out validation.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: L4 B8 fresh-SYNTH 15-minute run with default LR scale.
status: rejected, NaN before held-out validation
decision:
  Reject as a 900-second candidate. The measured effect of increasing batch
  from B4 to B8 at LR scale 1.5 was better wall-clock stability but not better
  update-count stability: both runs reached NaN at logged step 1250.
changes:
  GPT2_N_LAYER=4, GPT2_BATCH_SIZE=8, GPT2_N_EMBD=1536, GPT2_N_HEAD=12.
  TRAIN_MAX_SECONDS=900. Default LR scale remained 1.5 and warmup remained 5.
  SYNTH training streamed from the 400M-token fresh train shard set.
result:
  target/synth_l4_b8_900s_20260619T060638Z.log
  step 1200 loss=5.246863 finite=true nonzero=true elapsed_s=693.723
  step 1250 loss=NaN finite=false nonzero=false elapsed_s=722.662
  heldout_eval split=val val_loss=NaN train_elapsed_s=901.121
comparison:
  L4 B4 default LR also reached NaN at logged step 1250, at elapsed_s=526.
  B8 processed twice as many tokens per optimizer step and reached the same
  failing update later in wall-clock time, but it did not prevent the same
  update-count failure.
verification:
  cargo fmt --check: pass
  cargo check --workspace --tests: pass
  cargo oxide build --arch sm_120a: pass
  1-second launch check: pass
  900-second direct GPU run: failed numerically with NaN.
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
status: implementation_rejected; concept still open
decision:
  Do not promote either attempted compact-index implementation. This does not
  rule out compact upper-triangle scheduling itself; it only shows that these
  two mappings changed cooperative-kernel resource usage enough to fail the real
  training launch.
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
  This is a useful implementation failure, not an algorithmic rejection. A
  correct follow-up should preserve cooperative launch viability while avoiding
  the square tile-space branch skip, then verify with the full training path.
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

- Multi-variable Bayesian/Pareto sweep after AMUSE:
  - jointly vary batch size, LR scale, Adam LR scale, warmup/start, AMUSE
    schedule parameters, model depth, width, head count, Aurora phase count,
    and cooperative block count
  - target held-out validation loss over the fixed wall-clock budget
  - record stability/finiteness, completed steps, tokens/s, memory use, and GPU
    power as diagnostics
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
4. Run the multi-variable Bayesian/Pareto sweep:
   - include batch size only as a coupled variable, never as a standalone
     increase
   - keep log interval large enough that loss sync does not dominate
   - rank candidates only by 15-minute held-out validation loss
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

```text
date: 2026-06-19
commit: uncommitted
experiment: Three-trial coupled SYNTH sweep after seeded history update.
status: completed, new recorded best
target:
  Held-out validation loss after a fixed 900 second wall-clock budget.
sweep:
  target/sweeps/synth_900_multi_20260619T140614Z
result:
  trial_0000:
    success, val_loss=4.732287, completed_steps=4691.
    shape B8 L2 d1024 h16, Aurora phases=4 blocks=120.
    lr_scale=1.575554, adam_lr_scale=1.271614, warmup=20,
    start_ratio=0.2, amuse_beta1=0.2, amuse_rho=1.0.
  trial_0001:
    failed with NaN at step 400 and stopped early.
    shape B8 L2 d2048 h16, Aurora phases=4 blocks=180.
    lr_scale=1.434873, adam_lr_scale=2.331448, warmup=100,
    start_ratio=0.05, amuse_beta1=0.2, amuse_rho=1.0.
  trial_0002:
    success, val_loss=5.498525, completed_steps=1311.
    shape B8 L2 d2048 h16, Aurora phases=4 blocks=160.
    lr_scale=0.638015, adam_lr_scale=0.526526, warmup=5,
    start_ratio=0.0, amuse_beta1=0.4, amuse_rho=0.5.
validation_objective:
  Best result in this sweep is trial_0000 at val_loss=4.732287.
  Previous best was trial_0001 from
  target/sweeps/synth_900_multi_20260619T080010Z at val_loss=4.805441,
  so this sweep sets a new recorded best by 0.073154 validation loss.
measured_effect:
  L2 d1024 with higher LR and start_ratio=0.2 improved validation loss versus
  the previous L2 d1024 result of 5.132470 and beat the prior L4 d1024 best.
  The result moves the current Pareto front toward faster L2 d1024 candidates.
stability_effect:
  d2048 remains unstable or uncompetitive in this run. The higher Adam LR
  d2048 trial went NaN quickly; the lower LR d2048 trial stayed finite but
  trained far fewer steps and ended with worse validation loss.
runtime_effect:
  The stable d1024 L2 trial completed 4691 steps in 900 seconds.
  The stable d2048 L2 trial completed only 1311 steps in 900 seconds, so the
  larger width is currently too slow for the fixed-wall objective unless kernel
  efficiency improves substantially.
history:
  notes/sweep_seed.tsv contains all three trial rows from this sweep.
next_experiment:
  Continue coupled sweeps around the current Pareto front instead of testing
  d2048 alone: compare L2/L4 d1024 with lr_scale around 1.5-2.1,
  adam_lr_scale around 0.8-1.4, warmup 5-20, start_ratio 0.0-0.2, and
  amuse_beta1/rho around the two successful regions.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Add profiled Aurora phase=2/block=90 layout to sweep space.
status: code change verified; sustained launch check passed
target:
  Reduce optimizer scheduling overhead without changing optimizer math. This is
  a runtime candidate-space fix, not a completed 900-second validation win.
code_change:
  src/app/sweep/candidate.rs now samples Aurora cooperative blocks
  [80, 90, 120, 160, 180] and phases [2, 4, 8, 16], filtered by the existing
  cooperative-block bound. For L2 there are 8 Aurora matrix slots, so
  phases=2 and blocks=90 gives 90 * (8 / 2) = 360 cooperative CTAs and is
  valid. For L4 there are 16 slots, so the same phase=2/block=90 layout would
  require 720 CTAs and remains filtered out.
profile_baseline:
  target/nsys/best_b8_l2d1024_20_20260619T144732Z.nsys-rep
  Shape B8 L2 d1024 h16, Aurora phases=4 blocks=120.
  aurora_mega_update_cooperative_kernel total=1.069122657s over 20 calls,
  avg=53.456ms. Train loop reported about 3.781s for 20 steps.
profile_candidate:
  target/nsys/phase2_blocks90_b8_l2d1024_20_20260619T144852Z.nsys-rep
  Shape B8 L2 d1024 h16, Aurora phases=2 blocks=90.
  aurora_mega_update_cooperative_kernel total=0.782565486s over 20 calls,
  avg=39.128ms. Train loop reported about 3.522s for 20 steps.
measured_effect:
  The profiled candidate reduced the Aurora mega-kernel average by about 26.8%
  and reduced the 20-step train-loop time by about 6.8%. The remaining top
  kernels were fp32_to_nvfp4_ms_eden_device_scale_kernel,
  linear_backward_projection_cta_device_scale_kernel, causal_attention_kernel,
  and lm_head_kernel.
sustained_check:
  target/phase2_blocks90_b8_l2d1024_100steps_20260619T145205Z.log
  CUDA_DEVICE_INDEX=0, SYNTH, 100 steps, same hyperparameters as current best
  L2/d1024 trial: lr_scale=1.575554, adam_lr_scale=1.271614, warmup=20,
  start_ratio=0.2, amuse_beta1=0.2, amuse_rho=1.0.
  completed_steps=100, train_elapsed_s=17.571, heldout_eval val_loss=7.288392.
stability_effect:
  The 100-step run stayed finite=true and nonzero=true through the final step.
  This only verifies that the faster launch layout does not immediately break
  the training path; it is not a substitute for the 900-second validation
  objective.
verification:
  cargo fmt --check: pass
  cargo test --bin sweep: pass
  GPT2_BATCH_SIZE=8 GPT2_N_LAYER=2 GPT2_N_EMBD=1024 GPT2_N_HEAD=16
  AURORA_MATRIX_PHASES=2 AURORA_COOPERATIVE_BLOCKS=90 cargo build --release:
  pass
  100-step SYNTH run: pass, finite through heldout_eval.
next_experiment:
  Do not keep launching 3-trial sweeps. Either run a longer coupled 900-second
  sweep with this expanded search space, or continue code profiling on the next
  top kernels: MS-EDEN quantization, linear backward projection, causal
  attention, and lm_head.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Replace MS-EDEN dense Hadamard loop with shared-memory FWHT.
status: completed, new recorded best 900-second validation loss
target:
  Reduce the fp32_to_nvfp4_ms_eden_device_scale_kernel cost that remained after
  the Aurora phase=2/block=90 scheduler change, then verify against the real
  900-second held-out validation objective.
code_change:
  crates/cuda-kernels/src/utils/nvfp4/quant/kernels/ms_eden.rs now loads one
  signed input per lane and applies a 32-point Walsh-Hadamard transform through
  five shared-memory butterfly stages. The previous path computed each lane's
  rotated value with a dense 32-term loop and 32 input loads. The tested
  out_chunk_amax contract is preserved.
rejected_subchange:
  Removing the per-chunk amax write from the MS-EDEN kernel failed the ignored
  GPU tests because out_chunk_amax is part of the quantizer contract:
  fp32_to_nvfp4_ms_eden_writes_rotated_quantized_outputs and
  linear_backward_ms_eden_quantizes_before_gemms both require finite positive
  chunk amax values. That removal was not kept.
profile_before:
  target/nsys/phase2_blocks90_b8_l2d1024_20_20260619T144852Z.nsys-rep
  fp32_to_nvfp4_ms_eden_device_scale_kernel total=547.410954ms over 720 calls.
  20-step train-loop time was about 3.522s.
profile_after:
  target/nsys/mseden_fwht_b8_l2d1024_20_20260619T145840Z.nsys-rep
  fp32_to_nvfp4_ms_eden_device_scale_kernel total=288.723091ms over 720 calls.
  20-step train-loop time was about 3.245s.
measured_effect:
  The FWHT rewrite reduced the MS-EDEN device-scale kernel time by about 47.3%
  in the 20-step profile and reduced 20-step train-loop time by about 7.9%.
  In the 100-step sustained check, train_elapsed_s improved from 17.571 to
  16.180 on the same B8 L2 d1024 h16 phase=2/block=90 candidate.
stability_effect:
  Focused ignored GPU tests passed after preserving out_chunk_amax:
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test nvfp4_quant --
  --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward --
  --ignored --nocapture: pass.
  The 100-step SYNTH check stayed finite=true and nonzero=true through
  heldout_eval.
validation_objective:
  target/mseden_fwht_b8_l2d1024_900s_20260619T145904Z.log
  Shape B8 L2 d1024 h16, Aurora phases=2 blocks=90.
  lr_scale=1.575554, adam_lr_scale=1.271614, warmup=20,
  start_ratio=0.2, amuse_beta1=0.2, amuse_rho=1.0.
  stopped_by_wall_clock=true elapsed_s=900.116 completed_steps=5429.
  heldout_eval split=val val_loss=4.623164 train_elapsed_s=900.282
  completed_steps=5429.
comparison:
  Previous best recorded 900-second validation point was
  target/sweeps/synth_900_multi_20260619T140614Z/trial_0000/train.log with
  val_loss=4.732287 and completed_steps=4691. This run improves held-out
  validation loss by 0.109123 and increases completed steps by 738 under the
  same 900-second budget.
history:
  notes/sweep_seed.tsv includes this direct validation run so future coupled
  sweeps can score candidates against the current best measured point.
verification:
  cargo fmt --check: pass.
  cargo check -p rust-kernels-cuda --tests: pass.
  cargo test --bin sweep: pass.
  GPT2_BATCH_SIZE=8 GPT2_N_LAYER=2 GPT2_N_EMBD=1024 GPT2_N_HEAD=16
  AURORA_MATRIX_PHASES=2 AURORA_COOPERATIVE_BLOCKS=90 cargo oxide build
  --arch sm_120a: pass.
  900-second SYNTH validation run: pass, finite through heldout_eval.
next_experiment:
  Continue profiling from the new top kernels after FWHT: Aurora mega,
  linear_backward_projection_cta_device_scale_kernel, causal_attention_kernel,
  and lm_head_kernel. Longer coupled sweeps should include the phase=2/block=90
  point as measured history, not as an untested hypothesis.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Fix causal attention launch for head_dim=128 candidates.
status: correctness fix verified; no new 900-second best in this pass
target:
  Keep the optimization loop honest by making wider model candidates compute
  all attention head dimensions. The previous causal attention launch used a
  fixed 64-thread block, so head_dim=128 candidates only wrote dimensions
  0..63 for each head.
code_change:
  crates/cuda-kernels/src/gpt/attention/causal.rs now selects the causal
  attention block size from head_dim, rounded to a warp multiple and capped at
  128 threads. Score max/denom reductions use thread::blockDim_x() and the
  runtime warp count instead of the old fixed 64-thread constants.
invalidated_history:
  Prior 900-second and sweep rows with n_embd/n_head=128 were generated by the
  truncated attention path and are not valid quality evidence. The affected
  rows were removed from notes/sweep_seed.tsv so future Bayesian/Pareto sweeps
  do not score or suppress corrected wider candidates using stale data.
rejected_runtime_experiment:
  A one-phase Aurora layout was tested for the current L2 d1024 shape:
  AURORA_MATRIX_PHASES=1, AURORA_COOPERATIVE_BLOCKS=45.
  target/nsys/phase1_blocks45_b8_l2d1024_20_20260619T151629Z.nsys-rep
  It was worse than the current phase=2/block=90 layout:
  aurora_mega_update_cooperative_kernel increased from 782.292149ms to
  959.618964ms over 20 calls, and 20-step train time increased from about
  3.245s to 3.398s. This layout was not added to the sweep search space.
corrected_wider_check:
  target/attention_dynamic_b8_l2d1536_100steps_20260619T152027Z.log
  Shape B8 L2 d1536 h12, Aurora phases=2 blocks=90, head_dim=128.
  Same optimizer settings as the current d1024 best:
  lr_scale=1.575554, adam_lr_scale=1.271614, warmup=20,
  start_ratio=0.2, amuse_beta1=0.2, amuse_rho=1.0.
  completed_steps=100, train_elapsed_s=34.907, heldout_eval val_loss=7.440040.
comparison:
  The corrected d1024 check after the same attention fix finished 100 steps in
  16.555s with heldout_eval val_loss=7.288638:
  target/attention_dynamic_b8_l2d1024_100steps_20260619T151952Z.log.
  The corrected d1536 shape was finite, but was about 2.1x slower and worse at
  the 100-step held-out check, so it was not promoted to a 900-second direct
  validation run in this pass.
verification:
  cargo fmt --check: pass.
  cargo check -p rust-kernels-cuda --tests: pass.
  cargo test --bin sweep: pass.
  cargo oxide build --arch sm_120a with default d1536/h12 head_dim=128: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test
  causal_attention_log_sum_exp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l2_attention --
  --ignored --nocapture: pass.
  GPT2_BATCH_SIZE=8 GPT2_N_LAYER=2 GPT2_N_EMBD=1024 GPT2_N_HEAD=16
  AURORA_MATRIX_PHASES=2 AURORA_COOPERATIVE_BLOCKS=90 cargo oxide build
  --arch sm_120a: pass.
  100-step d1024 and corrected d1536 SYNTH checks: pass, finite through
  heldout_eval.
next_experiment:
  Future 900-second sweeps may include d1536/d2048 again, but only as corrected
  candidates. The current best measured 900-second point remains B8 L2 d1024
  h16 phase=2/block=90 with val_loss=4.623164.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: nanoGPT practices to test under fixed-wall validation.
status: research note only; no code changes made from this list yet
sources:
  https://github.com/karpathy/nanoGPT/blob/master/train.py
  https://github.com/karpathy/nanoGPT/blob/master/model.py
  https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py
  https://github.com/karpathy/llm.c/discussions/481
observed_nanogpt_practices:
  - Pretraining defaults use bias=false, dropout=0.0, AdamW lr=6e-4,
    beta1=0.9, beta2=0.95, weight_decay=0.1, grad_clip=1.0, warmup,
    cosine decay, and min_lr=lr/10.
  - Small Shakespeare character training uses dropout=0.2, batch=64,
    block_size=256, 6 layers, 6 heads, n_embd=384, lr=1e-3, beta2=0.99,
    warmup=100, and validation averaging every 250 steps.
  - Validation loss is estimated over many batches, not a single held-out
    batch, before checkpoint/best-loss decisions.
  - Parameters are split by dimensionality for decay: 2D matmul/embedding
    tensors get weight decay; bias and layer-norm vectors do not.
  - Residual projection weights named c_proj are initialized with
    std=0.02/sqrt(2*n_layer), while other linear/embedding weights use
    std=0.02.
  - Token embedding and LM-head weights are tied.
  - Vocab is padded to 50304 for GPT-2 training efficiency.
  - Training asynchronously prefetches the next batch before backward.
  - Inference only evaluates the LM head at the final token position.
related_llm_c_practices:
  - GPT-2 reproduction targets about 0.5M tokens per optimizer update, using
    gradient accumulation when a single GPU cannot fit the desired microbatch.
  - The run uses gradient clipping at norm 1.0 and logs gradient norm because
    norm spikes are treated as direct evidence of instability.
  - Activation recomputation is used as a memory-throughput tradeoff: recompute
    GeLU activations to fit a larger batch, then disable recompute only if
    memory allows it without losing throughput.
  - Validation is logged periodically and generation is mostly deferred to the
    end; intermediate samples are not used as the optimization target.
to_test_under_900s_heldout_validation:
  1. Add gradient global-norm clipping as a sweep dimension.
     Coupled search variables: lr_scale, adam_lr_scale, batch_size,
     beta1/beta2 or AMUSE analogs, and grad_clip.
     Measurement: held-out val loss after 900s, NaN/finiteness, completed steps.
  2. Add residual-projection scaled initialization for attention c_proj and MLP
     c_proj weights, using the nanoGPT std=0.02/sqrt(2*n_layer) rule adapted to
     the NVFP4 master-weight initialization path.
     Measurement: 100-step stability gate, then 900s held-out val.
  3. Add parameter-group weight decay semantics.
     2D matrix/embedding weights decay; layer norm vectors and biases do not.
     Measurement: compare against current optimizer update on the same
     Bayesian/coupled sweep budget.
  4. Add dropout as an actual architecture/training sweep variable only if the
     kernels can support it without a CPU path or hidden fallback.
     Measurement: fixed-wall validation on SYNTH, not Shakespeare train loss.
  5. Change held-out validation from the current fixed small slice to a
     multi-window average, preserving deterministic sample selection.
     Measurement: variance of reported val_loss across repeated eval-only runs.
  6. Check whether vocab padding/tokenizer choice can reduce LM-head edge cost.
     Constraint: do not change dataset semantics merely to improve speed; this
     must be evaluated by held-out validation over equal wall-clock.
  7. Implement inference-only final-token LM-head evaluation for generation.
     This is generation speed only and must not affect training loss.
  8. Add async/pipelined batch staging only if it removes observable host-side
     stalls in nsys. This is a throughput test, not a model-quality change.
  9. Test activation recomputation for MLP/relu2 saved activations if it permits
     larger actual batch size without adding more wall-clock than the batch
     increase saves. Measurement must include held-out validation, completed
     steps, and nsys kernel mix.
  10. Add gradient norm reporting and clipping to the sweep contract. The sweep
      should treat NaN/finite=false as failure immediately and should report
      norm spikes as instability evidence, not just final validation loss.
rejected_as_direct_copy:
  nanoGPT's small Shakespeare config is character-level and intentionally
  overfits a tiny dataset. Its dropout=0.2, context=256, and beta2=0.99 are
  useful sweep candidates, but not direct settings for SYNTH tokenizer training.
objective_rule:
  Promote an item only if it improves held-out validation loss at the same
  wall-clock budget or improves throughput without worsening validation in the
  paired fixed-wall test.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Next quantized-throughput target after LM-head CTA.
status: research note only while current sweep is running
current_best_completed:
  Current-code sweep trial_0002 reached heldout_eval val_loss=4.243804 after
  900s and 5376 completed steps:
  target/sweeps/lm_head_cta_current_900_20260619T155927Z/trial_0002/train.log.
profile_basis:
  target/nsys/lm_head_cta_b8_l2d1024_20_20260619T153905Z_stats.txt
  Top GPU kernels after LM-head CTA:
    aurora_mega_update_cooperative_kernel: 795.324697ms, 25.3%
    linear_backward_projection_cta_device_scale_kernel: 527.892098ms, 16.8%
    causal_attention_kernel: 462.040622ms, 14.7%
    fp32_to_nvfp4_ms_eden_device_scale_kernel: 289.639586ms, 9.2%
    lm_head_kernel: 141.260135ms, 4.5%
quantization_position:
  The backward linear path already uses NVFP4 TC projection through
  linear_backward_projection_cta_device_scale_kernel. The remaining overhead is
  not that those GEMMs are FP32; it is operand preparation around them:
  decode_weight_t, transpose_f32, decode_rowwise_t, then MS-EDEN quantization
  of E, W^T, E^T, and X^T before the TC projection.
candidate_experiment:
  Add a fused transpose/decode-to-MS-EDEN operand packer for transposed backward
  operands:
    - FP32 E -> packed rowwise MS-EDEN E^T without materializing FP32 E^T.
    - saved rowwise NVFP4 X -> packed rowwise MS-EDEN X^T without materializing
      FP32 X^T.
    - stored NVFP4 W -> packed MS-EDEN W^T or reuse decoded W^T only if the
      direct packed path is not practical.
  The output contract should remain the existing Nvfp4RowwiseDeviceTensor /
  Nvfp4DeviceScaleMmaWeightTensor consumed by the CTA projection kernel.
implementation_shape:
  Put the fused packers under crates/cuda-kernels/src/utils/nvfp4/quant rather
  than attention/MLP. They are layout-aware Quartet/MS-EDEN operand builders,
  not layer-specific math.
  The existing MS-EDEN body is already row-major over
  (row_count, src_row_len, dst_row_len). The fused variant should share the
  Hadamard/scale/correction logic but replace hadamard_input with source
  loaders:
    - FP32 transpose source: value(row, col) = x[col * row_count + row].
      This replaces transpose_f32_kernel followed by MS-EDEN.
    - rowwise NVFP4 transpose source:
      value(row, col) = nvfp4_rowwise_value(bytes, scales, global_scales,
      src_original_cols, col, row). This replaces
      nvfp4_decode_rowwise_transpose_f32_kernel followed by MS-EDEN.
    - scalar/global NVFP4 transpose source:
      value(row, col) = nvfp4_value(bytes, scales, global_scale[0],
      col * original_cols + row). This can replace
      nvfp4_decode_transpose_f32_kernel followed by MS-EDEN for W^T if the
      source layout matches the weight tensor.
  Add launcher args next to MsEdenDeviceScaleQuantArgs, returning the same
  out_fp4/out_scales/out_global_scales/out_chunk_amax/out_global_scale buffers
  used by LinearBackwardMsEdenScratch today. Then route
  LinearBackwardModule::backward_ms_eden through the fused packers for E^T and
  X^T first, since those remove the obvious FP32 transpose/decode materialized
  buffers without changing the CTA projection call.
expected_effect:
  Reduce transpose_f32_kernel, transpose_matrix_kernel,
  nvfp4_decode_rowwise_transpose_f32_kernel, nvfp4_decode_transpose_f32_kernel,
  and part of fp32_to_nvfp4_ms_eden_device_scale_kernel launch/memory cost.
  This preserves the Quartet II backward quantization contract and increases
  useful NVFP4 TC work per fixed wall-clock.
non_goal:
  Do not replace this with a pure FP16/FP32 backward shortcut. Gradient clipping
  can still be a coupled stability sweep variable, but it is not the main
  quantization-throughput optimization.
verification_plan:
  1. Add focused GPU correctness tests comparing packed fused operands against
     the existing materialize-then-MS-EDEN path for deterministic seeds.
  2. Run the linear backward ignored GPU tests.
  3. Run a 100-step SYNTH or Shakespeare sustained check for finite/nonzero
     behavior.
  4. Profile 20 steps and require reduced operand-prep kernel time.
  5. Promote only after a 900-second held-out validation run is not worse, or
     after the coupled sweep finds a better validation point.
```

```text
date: 2026-06-19
commit: uncommitted
experiment: Make hill-climb baseline promotion automatic.
status: implemented and unit-tested
measured_baseline:
  Source trial:
    target/sweeps/lm_head_cta_current_900_20260619T155927Z/trial_0002/train.log
  Held-out validation loss: 4.243804
  Completed steps in 900s: 5376
  Config:
    GPT2_BATCH_SIZE=8
    GPT2_N_LAYER=2
    GPT2_N_EMBD=1024
    GPT2_N_HEAD=16
    AURORA_MATRIX_PHASES=2
    AURORA_COOPERATIVE_BLOCKS=80
    TRAIN_LR_SCALE=1.014040
    TRAIN_ADAM_LR_SCALE=1.980467
    TRAIN_LR_WARMUP_STEPS=5
    TRAIN_LR_START_RATIO=0.050000
    TRAIN_AMUSE_BETA1=0.200000
    TRAIN_AMUSE_RHO=0.500000
implementation:
  Added notes/sweep_baseline.env as the mutable hill-climb baseline.
  Build scripts read that file when explicit GPT2_* / AURORA_* env vars are
  absent and rerun when the file changes.
  Runtime learning-rate defaults read the same file when TRAIN_* env vars are
  absent.
  The sweep runner loads the baseline, starts a fresh sweep from it when no
  trial history exists, and rewrites the baseline file whenever a successful
  trial beats the current baseline validation loss.
verification:
  cargo test -p rust-kernels --bin sweep -- --nocapture
    result: 6 passed
  dry-run sweep started from:
    b8_l2_d1024_h16_p2_c80_lr1.0140_alr1.9805_w5_s0.05_b0.20_r0.50
  Generated debug build constants:
    GPT2_BATCH_SIZE=8, GPT2_N_LAYER=2, GPT2_N_HEAD=16, GPT2_N_EMBD=1024
    AURORA_COOPERATIVE_BLOCKS=80, AURORA_MATRIX_PHASES=2
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Remove redundant backward buffer clears.
status: promoted
baseline:
  log: target/attention_backward_no_transpose_l4_b8_900_20260620T081345Z.log
  val_loss: 4.077696
  completed_steps: 4483
candidate:
  log: target/no_backward_clear_l4_b8_900_20260620T090607Z.log
  val_loss: 4.069893
  completed_steps: 4535
measured_effect:
  Validation loss improved by 0.007803 over the same 900-second held-out SYNTH
  gate, and completed steps increased by 52.
  The 20-step nsys run showed CUDA memset count dropping from 2560 to 560 and
  CUDA memset time dropping from 57.804744 ms to 8.651551 ms.
stability:
  100-step SYNTH check stayed finite and nonzero.
  900-second run stayed finite and completed normally.
interpretation:
  The cleared buffers were overwritten before use. Removing the clears reduces
  launch/memset overhead without changing optimizer math or data semantics.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Multi-warp MS-EDEN pack kernels.
status: rejected
change:
  Changed MS-EDEN packing kernels from one 32-thread warp per CTA to eight
  warps per CTA, with each warp packing one 32-value Hadamard chunk.
verification:
  cargo check --workspace --tests: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  100-step SYNTH check stayed finite and nonzero:
    target/ms_eden_pack_warp8_100_20260620T092605Z.log
profile_effect:
  20-step nsys log:
    target/nsys/ms_eden_pack_warp8_l4_b8_20_20260620T092633Z.log
  Compared to target/nsys/no_backward_clear_l4_b8_20_20260620T090522Z.log:
    fp32_to_nvfp4_ms_eden_device_scale_kernel:
      162.017935 ms -> 86.824742 ms.
    fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel:
      167.198477 ms -> 119.333682 ms.
    rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel:
      74.137109 ms -> 62.847135 ms.
    nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel:
      24.967339 ms -> 19.067413 ms.
heldout_result:
  Baseline:
    target/no_backward_clear_l4_b8_900_20260620T090607Z.log
    val_loss=4.069893, completed_steps=4535.
  Candidate:
    target/ms_eden_pack_warp8_l4_b8_900_20260620T092657Z.log
    val_loss=4.101368, completed_steps=4697.
measured_effect:
  The candidate increased completed steps by 162 but worsened held-out
  validation loss by 0.031475 over the same 900-second SYNTH gate.
decision:
  Do not promote. The optimization target is held-out validation loss over
  fixed wall-clock, not speed alone. Code reverted to the prior baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: AdamW no-decay policy for layer norm and bias tensors.
status: rejected
source_basis:
  nanoGPT/llm.c-style optimizer grouping decays 2D matrix and embedding
  tensors, but not layer-norm vectors or bias vectors.
change:
  Token embedding AdamW kept ADAM_WEIGHT_DECAY.
  Layer norm weights/biases and linear bias tensors used zero AdamW weight
  decay. Diagnostics were updated to predict Adam deltas with the same decay
  policy.
verification:
  cargo fmt --all --check: pass.
  cargo check --workspace --tests: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH check stayed finite and nonzero:
    target/adam_no_decay_vectors_100_20260620T094811Z.log
    heldout_eval val_loss=6.604518, completed_steps=100.
heldout_result:
  Baseline:
    target/no_backward_clear_l4_b8_900_20260620T090607Z.log
    val_loss=4.069893, completed_steps=4535.
  Candidate:
    target/adam_no_decay_vectors_l4_b8_900_20260620T094843Z.log
    val_loss=4.096563, completed_steps=4534.
measured_effect:
  Held-out validation loss worsened by 0.026670 over the same 900-second SYNTH
  gate, with effectively identical step count.
decision:
  Do not promote. Code reverted to the prior baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Source audit for looped-transformer candidates.
status: research note only
sources:
  https://github.com/Leiay/looped_transformer/blob/main/scripts/models.py
  https://proceedings.iclr.cc/paper_files/paper/2025/file/2676109d49d1eb26d6bc584a8f556305-Paper-Conference.pdf
  https://arxiv.org/abs/2606.18524
source_findings:
  Leiay/looped_transformer implements a separate loop state z initialized from
  zeros or ones, then repeats z = backbone(x + z) or z = backbone(x * z), and
  returns predictions from loop outputs. This is not the same as arbitrary
  repetition of an already-normal GPT residual stack.
  The ICLR 2025 looped-transformer paper defines language-model looping as a
  shared transformer block applied repeatedly after embedding, but reports that
  looped language models are worse on perplexity than iso-flop non-looped
  baselines while being better on reasoning-style downstream tasks.
  The language-model comparison highlights the 12-layer block looped twice
  shape, so a local test should start from loop count 2 rather than an arbitrary
  loop count or arbitrary block repetition.
  The same paper proposes a layer-similarity regularizer as the way to inherit
  looped-model inductive bias without hurting perplexity.
  The residual-scaling paper for tied residual blocks reports that looped/tied
  residual updates need explicit loop-aware scaling. Any future looped
  implementation must include that scaling from the start.
local_evidence:
  Two prior local loop shortcuts and two later source-shaped local attempts
  produced bad objective results:
    target/loop_count2_l4_900s_20260620T071755Z.log:
      val_loss=4.226233, completed_steps=4102.
    target/loop2_l4_b8_900_20260620T084124Z.log:
      val_loss=4.200931, completed_steps=3047.
    target/loop_source_l4_b8_100_20260620T135526Z.log:
      val_loss=7.301424, completed_steps=100.
    target/loop_state2_l4_b8_100_20260620T143756Z.log:
      val_loss=8.954576, completed_steps=100.
  These runs should not be treated as evidence against a real source-faithful
  looped candidate. The state of looped-transformer work in this repo is:
  never properly tested.
decision:
  Do not run another arbitrary looped-forward shortcut. If looped transformer
  is tested again, treat it as a major architecture change and implement the
  source-backed loop count/loop-state/residual-scaling design from the start.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Direct CTA projection staging instead of one-iteration staging loops.
status: rejected_pre_gate
change:
  Replaced the two stage_tiles while loops in projection_cta/stage.rs with
  direct per-thread A/B pack stores, relying on the current 256-pack by
  256-thread CTA contract.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/projection_stage_direct_l4_b8_100_20260620T120355Z.log
    val_loss=6.546502, train_elapsed_s=19.434, completed_steps=100.
measured_effect:
  Runtime was effectively unchanged from the accepted baseline neighborhood.
  This indicates the compiler was already removing most of the one-iteration
  loop overhead or the loop was not material to wall-clock.
decision:
  Do not promote and do not spend a 900-second gate on this candidate. Code was
  reverted to baseline; note kept to prevent repeating the same micro-edit.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Pair-store no-bias CTA projection accumulator rows.
status: rejected_pre_gate
change:
  Rewrote projection_cta/store/nobias.rs to store the two adjacent columns for
  each accumulator row together, with one input-global-scale load per row
  instead of one per element.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/projection_nobias_pair_store_l4_b8_100_20260620T120842Z.log
    val_loss=6.544317, train_elapsed_s=19.428, completed_steps=100.
profile_effect:
  Baseline short profile:
    target/nsys/current_l4_b8_20_20260620T120735Z_stats.txt
    linear_backward_projection_cta_device_scale_kernel=761.753042ms/20.
  Candidate short profile:
    target/nsys/projection_nobias_pair_store_l4_b8_20_20260620T120910Z_stats.txt
    linear_backward_projection_cta_device_scale_kernel=765.991748ms/20.
measured_effect:
  The affected projection kernel got slower by about 4.239ms over 20 steps.
decision:
  Do not promote and do not spend a 900-second gate. Code was reverted to
  baseline; the compiler/current instruction mix favors the original scalar
  stores here.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Explicitly unroll f32-input FP16 TC shared-memory staging loops.
status: rejected_pre_gate
change:
  Replaced the four-iteration staging loops in
  utils/f16_tc_matmul/cta_stage_f32.rs with explicit per-thread staging calls
  for offsets tid, tid+256, tid+512, and tid+768.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/f16_stage_f32_unroll_l4_b8_100_20260620T121135Z.log
    val_loss=6.548014, train_elapsed_s=19.669, completed_steps=100.
measured_effect:
  The 100-step runtime regressed by about 0.23s versus the current baseline
  neighborhood, with no quality upside.
decision:
  Do not promote and do not spend a profile or 900-second gate. Code was
  reverted to baseline; the original loop form is better for this kernel.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Reuse training token/target batch buffers.
status: completed, new recorded best 900-second validation loss
change:
  Main training now allocates one reusable TokenBatch with persistent device
  tokens/targets, pinned host staging buffers, and a CUDA event recorded after
  the stream-ordered HtoD copies. Each step waits only for the prior batch-copy
  event before refilling the pinned host staging area, then enqueues copies into
  the same device buffers.
scope:
  Training hot path only. Evaluation and generation keep their existing
  TokenBatch::from_* allocation path. Model math, data order, optimizer state,
  and validation sample selection are unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/reusable_batch_l4_b8_100_20260620T122059Z.log
    val_loss=6.545963, train_elapsed_s=19.350, completed_steps=100.
  20-step direct screen:
    target/reusable_batch_l4_b8_20_direct_20260620T122209Z.log
    val_loss=8.505538, train_elapsed_s=3.807, completed_steps=20.
  20-step nsys profile:
    target/nsys/reusable_batch_l4_b8_20_20260620T122136Z_stats.txt
profile_effect:
  cuMemFree_v2 dropped from 3.701772145s/769 calls in
  target/nsys/current_l4_b8_20_20260620T120735Z_stats.txt to
  0.036707151s/731 calls in the reusable-batch profile. Top GPU kernel times
  were effectively unchanged to slightly slower, so the value of this change is
  host-side allocator removal, not better kernel code.
heldout_result:
  Previous accepted baseline:
    target/grad_clip_l4_b8_900_20260620T101626Z.log
    val_loss=4.044528, completed_steps=4520.
  Candidate:
    target/reusable_batch_l4_b8_900_20260620T122227Z.log
    val_loss=4.023637, completed_steps=4522.
measured_effect:
  Held-out validation loss improved by 0.020891 over the same 900-second SYNTH
  gate, with two additional completed steps. This clears the promotion rule.
decision:
  Promote and make this the recorded baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Reuse TokenDataLoader training batch Vec.
status: rejected_pre_gate
change:
  Added an internal reusable Vec<u16> to TokenDataLoader and returned borrowed
  batch-token slices instead of allocating a fresh TokenWindowBatch Vec each
  training step.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/reusable_loader_batch_l4_b8_100_20260620T123955Z.log
    val_loss=6.545901, train_elapsed_s=19.442, completed_steps=100.
measured_effect:
  Runtime regressed versus the promoted reusable-device-batch screen
  target/reusable_batch_l4_b8_100_20260620T122059Z.log, which had
  val_loss=6.545963 and train_elapsed_s=19.350.
decision:
  Do not promote and do not spend a 900-second gate. Code was reverted to the
  promoted baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Compile-time lookup-table compact upper-triangle scheduling for the
  symmetric Polar Gram stage.
status: rejected_pre_gate; concept still open
change:
  Replaced the branch-skip square tile walk in run_symmetric_tiles with compact
  upper-triangle lookup tables for the current tile dimensions 16, 48, and 64.
  The mapping avoided runtime triangular-index arithmetic and kept unsupported
  tile dimensions on the old branch-skip path.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass, 8 tests.
  100-step SYNTH screen:
    target/compact_tri_lut_l4_b8_100_20260620T172835Z.log
    val_loss=6.546121, train_elapsed_s=19.711, completed_steps=100.
  20-step nsys:
    target/nsys/compact_tri_lut_l4_b8_20_20260620T172905Z.run.log
measured_effect:
  The candidate launched and remained finite, so it fixed the earlier launch
  failure mode. It still regressed runtime. Against the paired-projection
  baseline profile target/nsys/paired_linear_backward_l4_b8_20_20260620T170414Z.run.log,
  aurora_mega_update_cooperative_kernel increased from 68.380 ms/step to
  72.261 ms/step, and 20-step train_elapsed_s increased from 3.811 to 3.920.
decision:
  Reject before the 900-second gate and revert the code. This rejects the LUT
  implementation, not compact upper-triangle scheduling in general.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Pair the two linear-backward CTA projection launches.
status: accepted_900s
change:
  Replaced the two separate linear_backward_projection_cta_device_scale_kernel
  launches in each linear backward call with one paired CTA kernel. The paired
  kernel maps a linear tile range to dinput or dweight and reuses the existing
  NVFP4 CTA no-bias MMA body for both outputs. Training math, optimizer state,
  data order, and validation sample selection were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  GPU tests:
    cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
    cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  100-step SYNTH screen:
    target/paired_linear_backward_l4_b8_100_20260620T170345Z.log
    val_loss=6.549596, train_elapsed_s=19.285, completed_steps=100.
  20-step nsys:
    target/nsys/paired_linear_backward_l4_b8_20_20260620T170414Z.run.log
    linear backward projection launches dropped from 680 to 340 over 20 steps.
    20-step train_elapsed_s changed from 3.891 to 3.811 versus
    target/nsys/direct_materialize_fused_adam_l4_b8_20_20260620T163922Z.run.log.
  900-second held-out gate:
    target/paired_linear_backward_l4_b8_900_20260620T170432Z.log
    val_loss=4.054840, completed_steps=4539.
measured_effect:
  Compared with baseline target/direct_materialize_fused_adam_l4_b8_900_20260620T163956Z.log,
  validation loss changed from 4.050065 to 4.054840 (+0.118%) while completed
  steps increased from 4529 to 4539.
decision:
  Accept under the current rule: validation loss moved less than 1% and the
  fixed-wall run completed more steps. notes/sweep_baseline.env now points to
  this run as the baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Sparse train-loss logging with existing TRAIN_LOG_INTERVAL.
status: rejected_pre_gate
change:
  Ran the current promoted binary with TRAIN_LOG_INTERVAL=1000000 so the
  100-step screen only synced train loss at step 0 and the final capped step.
verification:
  100-step SYNTH screen:
    target/sparse_log_l4_b8_100_20260620T130641Z.log
    val_loss=6.548050, train_elapsed_s=19.369, completed_steps=100.
measured_effect:
  This did not improve the promoted baseline screen
  target/reusable_batch_l4_b8_100_20260620T122059Z.log, which had
  val_loss=6.545963 and train_elapsed_s=19.350. The loss-sync reduction is too
  small/noisy at this screen length to justify changing the 900-second gate
  protocol.
decision:
  Keep the existing TRAIN_LOG_INTERVAL=250 convention for the 900-second gate.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Direct schedule-free materialization during NVFP4 encode.
status: failed_900s_gate
change:
  Replaced the schedule-free materialization path
  interpolate(z_master, x_master) -> f32 scratch -> tensor amax -> NVFP4 encode
  with direct schedule-free amax and encode kernels that recomputed
  z + beta * (x - z), removing schedule_free_interpolate_kernel from the hot
  path.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/schedule_free_direct_materialize_l4_b8_100_20260620T131220Z.log
    val_loss=6.546241, train_elapsed_s=19.328, completed_steps=100.
  20-step nsys profile:
    target/nsys/schedule_free_direct_materialize_l4_b8_20_20260620T131308Z_stats.txt
    cuLaunchKernel dropped from 15048 to 14028 calls versus
    target/nsys/reusable_batch_l4_b8_20_20260620T122136Z_stats.txt.
heldout_result:
  Baseline:
    target/reusable_batch_l4_b8_900_20260620T122227Z.log
    val_loss=4.023637, completed_steps=4522.
  Candidate:
    target/schedule_free_direct_materialize_l4_b8_900_20260620T131332Z.log
    val_loss=4.067527, completed_steps=4529.
measured_effect:
  The candidate completed 7 more steps in the fixed 900-second run, but
  worsened held-out validation loss by 0.043890. The launch-count reduction did
  not improve the actual objective.
decision:
  Do not promote. Code was reverted to the promoted baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Four-way unroll of Aurora momentum orientation pass.
status: rejected_pre_gate
change:
  Replaced the scalar strided loop in
  crates/cuda-kernels/src/gpt/optimizer/aurora/fused/momentum.rs with four
  explicit strided updates per thread.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/aurora_momentum_unroll_l4_b8_100_20260620T125303Z.log
    val_loss=6.546121, train_elapsed_s=19.515, completed_steps=100.
measured_effect:
  Runtime regressed versus the promoted reusable-device-batch screen
  target/reusable_batch_l4_b8_100_20260620T122059Z.log, which had
  val_loss=6.545963 and train_elapsed_s=19.350.
decision:
  Do not promote and do not spend a 900-second gate. Code was reverted to the
  promoted baseline; the compiler/current memory path favors the scalar loop.
```

```text
date: 2026-06-20
commit: 98d23b5b
experiment: Fuse AdamW update with schedule-free x-average.
status: accepted_900s
change:
  Moved the schedule-free x_master average update into the AdamW FP32 master
  update kernel, removing the separate schedule_free_average launch from the
  Adam optimizer path.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  100-step SYNTH screen:
    target/fused_adam_average_l4_b8_100_20260620T154009Z.log
    val_loss=6.545353, train_elapsed_s=19.326, completed_steps=100.
  900-second SYNTH gate:
    target/fused_adam_average_l4_b8_900_20260620T154046Z.log
    val_loss=4.045264, completed_steps=4528.
baseline:
  Previous accepted baseline:
    target/reusable_batch_l4_b8_900_20260620T122227Z.log
    val_loss=4.023637, completed_steps=4522.
measured_effect:
  The candidate completed 6 more steps in the fixed 900-second run. Validation
  loss was 0.537% worse than the previous baseline, which is inside the current
  +/-1% acceptance band for step-count improvements.
decision:
  Promote as the current baseline under the validation-with-step-count rule.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Direct schedule-free materialization on fused Adam baseline.
status: accepted_900s
change:
  Replaced schedule-free materialization's
  interpolate(z_master, x_master) -> f32 scratch -> amax -> NVFP4 encode path
  with direct schedule-free amax and direct NVFP4 encode kernels over
  z + beta * (x - z). Removed the unused optimizer materialized scratch buffer
  and the dead schedule_free_average kernel left over after Adam average fusion.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  100-step SYNTH screen:
    target/direct_materialize_fused_adam_l4_b8_100_20260620T163848Z.log
    val_loss=6.551194, train_elapsed_s=19.325, completed_steps=100.
  20-step nsys profile:
    target/nsys/direct_materialize_fused_adam_l4_b8_20_20260620T163922Z.run.log
    cuLaunchKernel calls dropped from 14348 to 13328 versus
    target/nsys/current_fused_adam_l4_b8_20_20260620T162020Z.run.log.
heldout_result:
  Baseline:
    target/fused_adam_average_l4_b8_900_20260620T154046Z.log
    val_loss=4.045264, completed_steps=4528.
  Candidate:
    target/direct_materialize_fused_adam_l4_b8_900_20260620T163956Z.log
    val_loss=4.050065, completed_steps=4529.
measured_effect:
  The candidate completed 1 more step in the fixed 900-second run. Validation
  loss was 0.119% worse than the previous baseline, inside the current +/-1%
  acceptance band for step-count improvements.
decision:
  Promote as the current baseline under the validation-with-step-count rule.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Batch linear bias-gradient reductions.
status: rejected_pre_gate
change:
  Tried replacing the 16 per-step linear_bias_grad_kernel launches with a
  pointer-table based batched kernel. The first version used one max-width
  launch for all bias tensors; the second grouped tensors by output width
  (QKV, hidden width, MLP width) to avoid empty blocks.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screens:
    target/batched_bias_grad_l4_b8_100_20260620T125913Z.log
    val_loss=6.547208, train_elapsed_s=19.390, completed_steps=100.
    target/grouped_bias_grad_l4_b8_100_20260620T130125Z.log
    val_loss=6.550293, train_elapsed_s=19.407, completed_steps=100.
measured_effect:
  Both variants failed the pre-gate versus the promoted baseline screen
  target/reusable_batch_l4_b8_100_20260620T122059Z.log, which had
  val_loss=6.545963 and train_elapsed_s=19.350. The grouped version reduced
  reported backward enqueue time but did not improve end-to-end objective-facing
  runtime and changed validation loss in the wrong direction.
decision:
  Do not promote and do not spend a 900-second gate. Code was reverted to the
  promoted baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Two-slot pinned host staging for reusable training batch.
status: rejected_pre_gate
change:
  Replaced the single pinned host token/target staging buffer in
  ReusableTokenBatch with two rotating pinned host slots and one CUDA event per
  slot. Device token/target buffers, stream ordering, data order, optimizer
  math, and validation sample selection were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/double_host_staging_l4_b8_100_20260620T133840Z.log
    val_loss=6.546112, train_elapsed_s=19.329, completed_steps=100.
measured_effect:
  Runtime changed by only about -0.021s versus the promoted reusable-batch
  screen target/reusable_batch_l4_b8_100_20260620T122059Z.log, which had
  val_loss=6.545963 and train_elapsed_s=19.350. Validation loss moved in the
  wrong direction and the runtime effect is too small to justify a 900-second
  gate.
decision:
  Do not promote and do not spend a 900-second gate. Code was reverted to the
  promoted baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Reuse Aurora encode half-warp loads through shuffles.
status: rejected_pre_gate
change:
  In aurora/fused/quant/encode.rs, replaced the second pair of global x loads
  during FP4 pair packing with half-warp shuffle reads from the value each lane
  already loaded for amax/error calculation.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/aurora_encode_shuffle_l4_b8_100_20260620T124305Z.log
    emitted step 0, then failed to reach step 99 after more than 90 seconds.
measured_effect:
  This is a clear runtime regression versus the promoted baseline 100-step
  screen at target/reusable_batch_l4_b8_100_20260620T122059Z.log, which
  completed in 19.350s.
decision:
  Do not promote and do not spend a 900-second gate. Code was reverted to the
  promoted baseline.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Four-way unroll of cross_entropy_kernel vocab scans and dlogit
  stores.
status: rejected_pre_gate
change:
  Unrolled the per-row max pass, denominator pass, and dlogits store pass by
  four columns per participating thread. The mathematical path was unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture: pass.
  100-step SYNTH screen:
    target/ce_unroll_l4_b8_100_20260620T173410Z.log
    val_loss=6.546669, train_elapsed_s=19.323, completed_steps=100.
measured_effect:
  Against the paired linear backward baseline screen
  target/paired_linear_backward_l4_b8_100_20260620T170345Z.log, runtime
  regressed from 19.285s to 19.323s. Validation loss moved from 6.549596 to
  6.546669, which is too small to justify the runtime regression or a
  900-second gate.
decision:
  Reject before profiling and before the 900-second gate. Code was reverted to
  the promoted baseline.
```
```text
date: 2026-06-20
commit: c921cd84 baseline, rejected candidate uncommitted
experiment: Aligned Polar Express CTA load/store path inside Aurora.
status: rejected_pre_gate
change:
  Added aligned Polar Express CTA staging and aligned store variants for the
  current L4/d1024 matrix shapes, moving the alignment branch outside the tile
  loop for the full candidate. The math, optimizer settings, data order, and
  validation split were unchanged.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  load-only 100-step SYNTH screen:
    target/polar_aligned_load_l4_b8_100_20260620T190011Z.log
    val_loss=6.546259, train_elapsed_s=19.107, completed_steps=100.
  full aligned 100-step SYNTH screen:
    target/polar_aligned_full_l4_b8_100_20260620T190357Z.log
    val_loss=6.546829, train_elapsed_s=19.066, completed_steps=100.
  full aligned 20-step nsys:
    target/nsys/polar_aligned_full_l4_b8_20_20260620T190427Z.run.log
    val_loss=8.505538, train_elapsed_s=3.766, completed_steps=20.
measured_effect:
  The load-only version regressed versus the accepted 100-step screen
  target/linear_bwd_aligned_l4_b8_100_20260620T183731Z.log, which had
  train_elapsed_s=19.081. The full aligned version was only 0.015s faster on
  the 100-step screen, but nsys showed no real device-kernel improvement:
  aurora_mega_update_cooperative_kernel was 1.367624575s baseline versus
  1.367621170s candidate over 20 steps, and total profiled train time stayed
  3.766s.
decision:
  Reject before the 900-second gate. Code was reverted to c921cd84; only this
  note remains.
```

```text
date: 2026-06-20
commit: uncommitted
experiment: Aligned CTA path for paired linear-backward projection.
status: accepted
change:
  Added an aligned no-edge variant of the CTA NVFP4 projection body and routed
  the paired linear-backward projection kernel through it for the current
  Tensor-Core-aligned training shape. The host wrapper now asserts the CTA
  alignment contract instead of silently falling back to the edge-checked path.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  100-step SYNTH screen:
    target/linear_bwd_aligned_l4_b8_100_20260620T183731Z.log
    val_loss=6.549100, train_elapsed_s=19.081, completed_steps=100.
  20-step nsys:
    target/nsys/linear_bwd_aligned_l4_b8_20_20260620T183801Z.run.log
    val_loss=8.505538, train_elapsed_s=3.766, completed_steps=20.
  900-second held-out gate:
    target/linear_bwd_aligned_l4_b8_900_20260620T184019Z.log
    val_loss=4.031730, train_elapsed_s=900.033, completed_steps=4587.
measured_effect:
  The profiled linear_backward_projection_pair_cta_device_scale_kernel time
  dropped from 740.275ms to 711.227ms over 20 steps, while total profiled
  train time moved from 3.794s to 3.766s. The 900-second gate completed 29
  more steps than the prior baseline, while validation loss moved from
  4.021274 to 4.031730, a +0.26% change.
decision:
  Promote under the current acceptance rule: validation loss stayed within the
  +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Unroll cross-entropy vocab scans by four.
status: rejected_screen
change:
  Unrolled the max, denominator, and dlogits vocab scans inside
  cross_entropy_kernel by four per thread stride. The visit order and math were
  otherwise unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/cross_entropy_unroll4_l4_b8_20_20260621T071444Z.run.log
    val_loss=8.505538, train_elapsed_s=3.584, completed_steps=20.
measured_effect:
  Against the promoted LM-head aligned baseline
  target/nsys/lm_head_aligned_path_l4_b8_20_20260621T065355Z.run.log,
  cross_entropy_kernel moved from 55.299974ms to 55.286411ms over 20 profiled
  steps. Profiled train time stayed at 3.584s. Neighboring kernels drifted
  slightly worse: linear_backward_projection_pair_cta_device_scale_kernel moved
  from 621.812208ms to 622.345509ms,
  fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel moved from 161.931077ms
  to 162.030945ms, and fp32_to_nvfp4_ms_eden_device_scale_kernel moved from
  160.628750ms to 160.694076ms.
decision:
  Reject before the 900-second gate. The target delta was noise-level, short
  wall-clock did not improve, and adjacent kernel timing drifted slightly
  worse. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Schedule attention probability-gradient with a 3D grid.
status: rejected_screen
change:
  Changed attention_prob_ds_kernel scheduling from a flat element index to
  grid coordinates keyed by key tile, query, and batch-head, removing most
  per-element index recovery divides/mods while keeping the same output layout.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/attention_prob_3d_grid_l4_b8_20_20260621T071943Z.run.log
    val_loss=8.505538, train_elapsed_s=3.579, completed_steps=20.
measured_effect:
  Against the promoted LM-head aligned baseline
  target/nsys/lm_head_aligned_path_l4_b8_20_20260621T065355Z.run.log,
  attention_prob_ds_kernel moved from 90.605765ms to 90.709205ms over 20
  profiled steps. Profiled train time moved from 3.584s to 3.579s, but the
  targeted kernel regressed and the wall-clock delta is within short-profile
  noise.
decision:
  Reject before the 900-second gate. The target kernel regressed, so the small
  short-run wall-clock fluctuation is not enough evidence to continue. Code was
  reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Route aligned affine and relu2 projections through aligned CTA bodies.
status: rejected_screen
change:
  Added aligned CTA affine and relu2 projection bodies and routed QKV/c_proj and
  MLP projections through them when token_count, output_dim, and input_dim were
  tile-aligned. Generic paths stayed available for odd shapes.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l2_attention -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aligned_affine_relu2_l4_b8_20_20260621T072920Z.run.log
    val_loss=8.505538, train_elapsed_s=3.596, completed_steps=20.
measured_effect:
  Against the promoted LM-head aligned baseline
  target/nsys/lm_head_aligned_path_l4_b8_20_20260621T065355Z.run.log,
  mlp_projection_kernel moved from 66.480605ms to 74.798571ms, and
  mlp_projection_relu2_kernel moved from 64.094471ms to 69.181360ms over 20
  profiled steps. attention_projection_kernel moved from 62.614531ms to
  62.415612ms, but the MLP regressions dominated. Profiled train time moved
  from 3.584s to 3.596s.
decision:
  Reject before the 900-second gate. The candidate regressed two hot MLP
  projection kernels and worsened short wall-clock. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Skip the final cooperative grid sync in Aurora mega update.
status: rejected_screen
change:
  Kept the cooperative grid syncs between Aurora matrix phases but skipped the
  terminal sync immediately before aurora_mega_update_cooperative_kernel
  returns.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aurora_skip_final_grid_sync_l4_b8_20_20260621T073447Z.run.log
    val_loss=8.505538, train_elapsed_s=3.590, completed_steps=20.
measured_effect:
  Against the promoted LM-head aligned baseline
  target/nsys/lm_head_aligned_path_l4_b8_20_20260621T065355Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1361.609071ms to
  1370.387667ms over 20 profiled steps. Profiled train time moved from 3.584s
  to 3.590s.
decision:
  Reject before the 900-second gate. The target Aurora kernel regressed and
  short wall-clock regressed. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Aligned f32-input staging and store path for f16 TC QK matmuls.
status: accepted
change:
  Added an aligned path for f16_cta_tc_matmul_f32_kernel when m, n, and k are
  exact CTA tile multiples. The generic edge-checked path remains active for
  non-aligned shapes. The attempted aligned RHS variants were measured
  separately and removed before this gate because they regressed.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l2_attention -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/f16_qk_aligned_stage_l4_b8_20_20260621T043243Z.run.log
    val_loss=8.505538, train_elapsed_s=3.591, completed_steps=20.
  100-step SYNTH screen:
    target/f16_qk_aligned_stage_l4_b8_100_20260621T043300Z.log
    val_loss=6.545244, train_elapsed_s=18.333, completed_steps=100.
  900-second held-out gate:
    target/f16_qk_aligned_stage_l4_b8_900_20260621T043330Z.log
    val_loss=4.012894, train_elapsed_s=900.167, completed_steps=4777.
measured_effect:
  Against the promoted N64 baseline
  target/nsys/projection_cta_n64_l4_b8_20_20260621T032524Z.run.log,
  f16_cta_tc_matmul_f32_kernel moved from 232.576737ms to 218.798849ms
  over 20 profiled steps. Total profiled train time moved from 3.603s to
  3.591s. The 900-second gate completed 11 more steps than the promoted
  N64 baseline while validation loss moved from 4.002766 to 4.012894,
  a +0.25% change.
decision:
  Promote under the current acceptance rule: validation loss stayed within the
  +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: N=64 CTA width for NVFP4 projection matmuls.
status: accepted
change:
  Increased the generic NVFP4 projection CTA tile from 32x32 with 256 threads
  to 32x64 with 512 threads, keeping K=64. The aligned staging path now guards
  A-pack loads because the CTA has more threads than A packs, while every
  thread stages one B pack.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/projection_cta_n64_l4_b8_20_20260621T032524Z.run.log
    val_loss=8.505538, train_elapsed_s=3.603, completed_steps=20.
  100-step SYNTH screen:
    target/projection_cta_n64_l4_b8_100_20260621T032703Z.log
    val_loss=6.546121, train_elapsed_s=18.360, completed_steps=100.
  900-second held-out gate:
    target/projection_cta_n64_l4_b8_900_20260621T032732Z.log
    val_loss=4.002766, train_elapsed_s=900.027, completed_steps=4766.
measured_effect:
  Against the previous promoted baseline
  target/ms_eden_shuffle_rht_l4_b8_900_20260621T030612Z.log, held-out
  validation loss improved from 4.052978 to 4.002766 and completed steps
  increased from 4635 to 4766. The 20-step nsys screen showed
  linear_backward_projection_pair_cta_device_scale_kernel dropping from
  676.216689ms to 621.646763ms, lm_head_kernel dropping from 134.820340ms to
  113.609475ms, and profiled train time dropping from 3.709s to 3.603s.
decision:
  Promote. This passes the fixed-wall objective directly: lower validation
  loss and higher completed step count under the same 900-second budget.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before screen
experiment: Reuse Aurora four-six encoder lane values with half-warp shuffles
  for payload packing.
status: rejected_unverified
change:
  Replaced the second FP32 loads during Aurora in-kernel four-six payload
  packing with half-warp shuffles from the value already loaded for local scale
  selection.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture:
    did not complete. The optimizer test binary remained alive for more than
    three minutes, used 100% of GPU0, and produced no additional output after
    the first four tests. The process was killed manually.
measured_effect:
  No valid performance or validation-loss evidence. The check did not reach a
  100-step screen or a 900-second gate.
decision:
  Reject and revert. Do not treat this as a passed optimizer-path change.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before 100-step screen
experiment: N=128 CTA width for NVFP4 projection matmuls.
status: rejected_screen
change:
  Increased the generic NVFP4 projection CTA tile from 32x64 with 512 threads
  to 32x128 with 1024 threads. The warp map changed to two row groups by
  sixteen eight-column groups.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture:
    passed after temporarily widening the test fixture to 128x128.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/projection_cta_n128_l4_b8_20_20260621T035213Z.run.log
    val_loss=8.505538, train_elapsed_s=3.798, completed_steps=20.
measured_effect:
  Runtime regressed versus the accepted N=64 CTA profile
  target/nsys/projection_cta_n64_l4_b8_20_20260621T032524Z.run.log.
  linear_backward_projection_pair_cta_device_scale_kernel increased from
  621.646763ms to 780.410516ms, lm_head_kernel increased from 113.609475ms to
  139.015822ms, and profiled train time increased from 3.603s to 3.798s.
decision:
  Reject before the 100-step and 900-second gates. Code was reverted to the
  accepted N=64 projection CTA shape.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Inline aligned projection CTA no-bias stores.
status: rejected_screen
change:
  Replaced four store_one_aligned calls with direct row/scale/base computation
  for the two-row, two-column aligned accumulator store pattern.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/proj_store_aligned_l4_b8_20_20260621T032324Z.run.log
    val_loss=8.505538, train_elapsed_s=3.717, completed_steps=20.
measured_effect:
  The target projection kernel regressed from 676.216689ms to 676.719952ms
  over 20 profiled steps versus
  target/nsys/ms_eden_shuffle_rht_l4_b8_20_20260621T030524Z.run.log. Profiled
  train time regressed from 3.709s to 3.717s.
decision:
  Reject before the 900-second gate. Code was reverted to the promoted
  baseline.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Route LM-head through the existing aligned no-edge NVFP4 CTA
  projection body.
status: rejected_pre_gate
change:
  Replaced the LM-head generic edge-checked CTA body with the aligned no-edge
  body and widened the focused LM-head test fixture to aligned dimensions.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/lm_head_aligned_only_l4_b8_20_20260621T035845Z.run.log
    val_loss=8.505538, train_elapsed_s=3.599, completed_steps=20.
  100-step SYNTH screen:
    target/lm_head_aligned_only_l4_b8_100_20260621T035900Z.log
    val_loss=6.546818, train_elapsed_s=18.390, completed_steps=100.
measured_effect:
  The 20-step profile showed only a tiny LM-head improvement versus the
  promoted N=64 baseline profile
  target/nsys/projection_cta_n64_l4_b8_20_20260621T032524Z.run.log:
  lm_head_kernel moved from 113.609475ms to 112.248891ms and profiled train
  time moved from 3.603s to 3.599s. The 100-step screen did not confirm a
  runtime win: train_elapsed_s regressed from 18.360 to 18.390 and validation
  loss moved from 6.546121 to 6.546818.
decision:
  Reject before the 900-second gate. The candidate did not produce a meaningful
  sustained speedup, and code was reverted to the promoted baseline.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted after gate
experiment: Increase cross_entropy_kernel row block from 256 to 512 threads.
status: rejected_gate
change:
  Changed CROSS_ENTROPY_THREADS_PER_BLOCK from 256 to 512 so each token row
  used more warps for the max, denominator, and dlogits vocab scans. The loss
  math, targets, optimizer path, and data order were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/cross_entropy_512_l4_b8_20_20260621T040316Z.run.log
    val_loss=8.502732, train_elapsed_s=3.594, completed_steps=20.
  100-step SYNTH screen:
    target/cross_entropy_512_l4_b8_100_20260621T040329Z.log
    val_loss=6.546054, train_elapsed_s=18.356, completed_steps=100.
  900-second held-out gate:
    target/cross_entropy_512_l4_b8_900_20260621T040404Z.log
    val_loss=4.044516, train_elapsed_s=900.164, completed_steps=4776.
measured_effect:
  The short profile improved the target kernel: cross_entropy_kernel moved from
  55.340644ms to 44.623598ms over 20 profiled steps versus the promoted N=64
  baseline profile target/nsys/projection_cta_n64_l4_b8_20_20260621T032524Z.run.log.
  The 100-step screen was also slightly faster, moving from 18.360s to
  18.356s. The full fixed-wall gate did not pass: completed steps increased
  from 4766 to 4776, but validation loss moved from 4.002766 to 4.044516,
  which is slightly outside the current +1% acceptance band.
decision:
  Reject after the 900-second gate. Code was reverted to the promoted 256-thread
  cross entropy baseline.
```

```text
date: 2026-06-21
commit: generated-artifact-only candidate, rebuilt baseline after screen
experiment: Retest Aurora phase-4 cooperative blocks=120 on the promoted N=64
  projection baseline.
status: rejected_pre_gate
change:
  Rebuilt generated PTX/host artifacts with AURORA_COOPERATIVE_BLOCKS=120 and
  AURORA_MATRIX_PHASES=4. No source files or training hyperparameters changed.
verification:
  AURORA_COOPERATIVE_BLOCKS=120 AURORA_MATRIX_PHASES=4 cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aurora_blocks120_p4_n64_l4_b8_20_20260621T042119Z.run.log
    val_loss=8.503331, train_elapsed_s=3.607, completed_steps=20.
measured_effect:
  Aurora mega-kernel time improved slightly versus the promoted N=64 baseline
  profile target/nsys/projection_cta_n64_l4_b8_20_20260621T032524Z.run.log:
  1.361030472s to 1.354902796s over 20 calls. The overall profile did not
  improve: train_elapsed_s moved from 3.603 to 3.607, and
  linear_backward_projection_pair_cta_device_scale_kernel regressed from
  621.646763ms to 626.128349ms.
decision:
  Reject before the 100-step and 900-second gates. Baseline artifacts were
  rebuilt with the promoted AURORA_COOPERATIVE_BLOCKS=90 and
  AURORA_MATRIX_PHASES=4 settings.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Warp-shuffle MS-EDEN Hadamard rotation.
status: accepted
change:
  Replaced the 32-lane MS-EDEN Hadamard transform shared-memory scratch and
  repeated block synchronizations with warp shuffle butterfly operations. The
  same transformed lane values are used for scale estimation and payload
  packing; the scale/correction math was unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test nvfp4_quant -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_shuffle_rht_l4_b8_20_20260621T030524Z.run.log
    val_loss=8.505538, train_elapsed_s=3.709, completed_steps=20.
  100-step SYNTH screen:
    target/ms_eden_shuffle_rht_l4_b8_100_20260621T030541Z.log
    val_loss=6.549100, train_elapsed_s=18.930, completed_steps=100.
  900-second held-out gate:
    target/ms_eden_shuffle_rht_l4_b8_900_20260621T030612Z.log
    val_loss=4.052978, train_elapsed_s=900.157, completed_steps=4635.
measured_effect:
  Against the promoted CTA staging baseline
  target/nsys/cta_stage_direct_l4_b8_20_20260621T012017Z.run.log, profiled
  train time moved from 3.721s to 3.709s. The direct FP32 MS-EDEN kernel moved
  from 160.804455ms to 160.550770ms over 20 steps, and the FP32-transpose
  MS-EDEN kernel moved from 164.520236ms to 162.279622ms. The 900-second gate
  completed 21 more steps than the previous baseline while validation loss
  moved from 4.047531 to 4.052978, a +0.13% change.
decision:
  Promote under the current acceptance rule: validation loss stayed within the
  +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted after profile screen
experiment: M64/N64 projection CTA shape for NVFP4 projection matmuls.
status: rejected_pre_gate
change:
  Increased NVFP4_PROJECTION_CTA_M from 32 to 64 while keeping
  NVFP4_PROJECTION_CTA_N=64 and NVFP4_PROJECTION_CTA_K=64. This raised CTA
  threads from 512 to 1024 and added an aligned B-pack staging guard needed for
  the larger thread count.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/projection_cta_m64_n64_l4_b8_20_20260621T042316Z.run.log
    val_loss=8.505538, train_elapsed_s=3.656, completed_steps=20.
measured_effect:
  The candidate regressed the projection-heavy kernels versus the promoted
  N64 baseline at target/nsys/projection_cta_n64_l4_b8_20_20260621T032524Z.run.log.
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  621.646763ms to 664.434445ms over 20 profiled steps. lm_head_kernel moved
  from 113.609475ms to 124.122706ms. mlp_projection_kernel moved from
  66.576347ms to 74.347681ms. mlp_projection_relu2_kernel moved from
  64.256203ms to 73.871560ms. attention_projection_kernel moved from
  62.718027ms to 70.668438ms. Total profiled train time regressed from
  3.603s to 3.656s.
decision:
  Reject before the 100-step and 900-second gates. Code was reverted to the
  promoted N64 baseline.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Aligned staging fast path inside the fused Aurora Polar Express
  tile compute loop.
status: rejected_screen
change:
  Added a checked aligned branch that used a separate no-bounds staging helper
  for full CTA tiles in the fused Polar matmul path.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  100-step SYNTH screen:
    target/polar_stage_aligned_l4_b8_100_20260621T014325Z.log
    val_loss=6.542066, train_elapsed_s=19.009, completed_steps=100.
  20-step nsys screen:
    target/nsys/polar_stage_aligned_l4_b8_20_20260621T014505Z.run.log
    val_loss=8.505538, train_elapsed_s=3.730, completed_steps=20.
measured_effect:
  The intended Aurora kernel regressed:
  aurora_mega_update_cooperative_kernel increased from 1.363230788s to
  1.373115164s over 20 profiled steps versus
  target/nsys/cta_stage_direct_l4_b8_20_20260621T012017Z.run.log. The profiled
  train time also regressed from 3.721s to 3.730s.
decision:
  Reject before the 900-second gate. Code was reverted to the promoted
  baseline.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Move Aurora master-update tail guard from per-element updates to a
  chunk-level checked/unchecked split.
status: rejected_pre_gate
change:
  Added an unchecked four-value update path for full 1024-value Aurora update
  chunks, keeping the old checked path only for tail chunks. Current promoted
  matrix lengths are all divisible by 1024, so this was intended to remove an
  unreachable per-element bounds branch.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass after rerun.
  100-step SYNTH screen:
    target/aurora_update_tail_chunk_l4_b8_100_20260621T014025Z.log
    val_loss=6.544455, train_elapsed_s=19.002, completed_steps=100.
  20-step nsys screen:
    target/nsys/aurora_update_tail_chunk_l4_b8_20_20260621T014052Z.run.log
    val_loss=8.505538, train_elapsed_s=3.751, completed_steps=20.
measured_effect:
  The intended Aurora kernel regressed: aurora_mega_update_cooperative_kernel
  increased from 1.363230788s to 1.370852406s over 20 profiled steps versus
  target/nsys/cta_stage_direct_l4_b8_20_20260621T012017Z.run.log. Total
  profiled train time also regressed from 3.721s to 3.751s. The 100-step screen
  was stable but did not offset the profiler regression.
decision:
  Reject before the 900-second gate. Code was reverted to the promoted
  baseline.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted after gate
experiment: AdamW schedule-free updates for all linear matrix weights.
status: rejected_gate
change:
  Added an explicit TRAIN_MATRIX_OPTIMIZER=adam route that replaced Aurora for
  QKV, attention c_proj, MLP up, and MLP down matrix weights while keeping the
  existing AdamW/schedule-free quantized writeback path for embeddings,
  layer norms, and biases.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  cargo build --release: pass.
  100-step SYNTH screen:
    target/adam_matrix_l4_b8_100_20260621T005724Z.log
    val_loss=7.641207, train_elapsed_s=12.510, completed_steps=100.
  900-second held-out gate:
    target/adam_matrix_l4_b8_900_20260621T005752Z.log
    val_loss=9.394782, train_elapsed_s=900.032, completed_steps=7063.
measured_effect:
  Adam-matrix mode ran more steps than the accepted Aurora baseline but failed
  the held-out objective badly. The accepted Aurora baseline at
  target/linear_bwd_aligned_l4_b8_900_20260620T184019Z.log had
  val_loss=4.031730 and completed_steps=4587. The Adam-matrix run learned
  early but drifted upward later, ending with train loss around 9.27 and
  validation loss 9.394782.
decision:
  Reject after the 900-second gate. Code was reverted; keep Aurora for linear
  matrix weights.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Direct one-thread-per-pack CTA staging for aligned linear backward
  projection tiles.
status: accepted
change:
  The aligned CTA projection staging path now relies on the current shape
  contract where A packs, B packs, and CTA threads are all 256. Each thread
  stages exactly one A pack and one B pack instead of entering stride loops
  whose second iteration is unreachable for the accepted aligned shape.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/cta_stage_direct_l4_b8_20_20260621T012017Z.run.log
    val_loss=8.505538, train_elapsed_s=3.721, completed_steps=20.
  100-step SYNTH screen:
    target/cta_stage_direct_l4_b8_100_20260621T012038Z.log
    val_loss=6.550644, train_elapsed_s=18.996, completed_steps=100.
  900-second held-out gate:
    target/cta_stage_direct_l4_b8_900_20260621T012116Z.log
    val_loss=4.047531, train_elapsed_s=900.052, completed_steps=4614.
measured_effect:
  The 20-step nsys screen showed
  linear_backward_projection_pair_cta_device_scale_kernel dropping from
  711.226580ms to 680.398853ms versus the aligned baseline profile, and
  profiled train time moved from 3.766s to 3.721s. The 900-second gate
  completed 27 more steps than the previous promoted baseline while validation
  loss moved from 4.031730 to 4.047531, a +0.39% change.
decision:
  Promote under the current acceptance rule: validation loss stayed within the
  +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Remove redundant one-warp MS-EDEN pack barrier.
status: accepted
change:
  Removed the block-wide sync after MS-EDEN scale-bit stores in the one-warp
  fp32/NVFP4 transpose pack path. The following FP4 payload store reads only
  lane-local values and warp shuffles, not the stored scales, so the barrier
  was unnecessary for the current 32-thread launch contract.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_no_pack_barrier_l4_b8_20_20260621T045551Z.run.log
    val_loss=8.505538, train_elapsed_s=3.588, completed_steps=20.
  100-step SYNTH screen:
    target/ms_eden_no_pack_barrier_l4_b8_100_20260621T045631Z.log
    val_loss=6.545762, train_elapsed_s=18.300, completed_steps=100.
  900-second held-out gate:
    target/ms_eden_no_pack_barrier_l4_b8_900_20260621T045707Z.log
    val_loss=4.013671, train_elapsed_s=900.101, completed_steps=4788.
measured_effect:
  Against the previous promoted profile
  target/nsys/f16_qk_aligned_stage_l4_b8_20_20260621T043243Z.run.log,
  fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel moved from
  162.245027ms to 161.850259ms over 20 profiled steps, and
  fp32_to_nvfp4_ms_eden_device_scale_kernel moved from 160.688510ms to
  160.560723ms. Profiled train time moved from 3.591s to 3.588s.
  The 900-second gate completed 11 more steps than the previous promoted
  baseline while validation loss moved from 4.012894 to 4.013671, a +0.019%
  change.
decision:
  Promote under the current acceptance rule: validation loss stayed well within
  the +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Reuse row global scales in aligned NVFP4 CTA projection stores.
status: rejected_screen
change:
  Changed the aligned no-bias CTA projection store to load the row global
  scale twice for the two output rows and store adjacent column pairs directly,
  instead of routing all four accumulator values through the generic aligned
  store helper.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass after rerun with completed PTX build.
  20-step nsys screen:
    target/nsys/proj_store_scale_reuse_l4_b8_20_20260621T053351Z.run.log
    val_loss=8.505538, train_elapsed_s=3.588, completed_steps=20.
  100-step SYNTH screen:
    target/proj_store_scale_reuse_l4_b8_100_20260621T053413Z.log
    val_loss=6.546121, train_elapsed_s=18.312, completed_steps=100.
measured_effect:
  The intended kernel moved only from 621.499531ms to 620.544267ms over
  20 profiled steps versus the promoted MS-EDEN barrier baseline, while total
  profiled train time stayed at 3.588s. The 100-step screen was slightly worse
  than the promoted baseline, moving from val_loss=6.545762 and
  train_elapsed_s=18.300 to val_loss=6.546121 and train_elapsed_s=18.312.
decision:
  Reject before the 900-second gate. The tiny projection-kernel delta did not
  improve the short wall-clock or validation screen. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Unroll MS-EDEN global-scale chunk reduction by four.
status: accepted
change:
  In quartet_backward_ms_eden_global_scale_from_chunks_kernel, each thread now
  checks four stride-separated chunk amax values per loop iteration and reduces
  them with max4_f32 before the existing warp/block reduction.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass after rerun with completed PTX build.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_chunk_scale_unroll4_l4_b8_20_20260621T062103Z.run.log
    val_loss=8.505538, train_elapsed_s=3.584, completed_steps=20.
  900-second held-out gate:
    target/ms_eden_chunk_scale_unroll4_l4_b8_900_20260621T062140Z.log
    val_loss=4.002436, train_elapsed_s=900.150, completed_steps=4786.
measured_effect:
  Against the promoted MS-EDEN barrier baseline
  target/nsys/ms_eden_no_pack_barrier_l4_b8_20_20260621T045551Z.run.log,
  quartet_backward_ms_eden_global_scale_from_chunks_kernel moved from
  7.784846ms to 3.215699ms over 20 profiled steps. Profiled train time moved
  from 3.588s to 3.584s. The adjacent device-scale pack kernels were
  effectively unchanged:
    fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel 161.850259ms -> 161.960962ms
    fp32_to_nvfp4_ms_eden_device_scale_kernel 160.560723ms -> 160.685299ms
    rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel 71.623581ms -> 71.743201ms
  The 900-second gate lowered validation loss from 4.013671 to 4.002436 while
  completed steps moved from 4788 to 4786.
decision:
  Promote under the current acceptance rule because held-out validation loss
  improved, even though completed step count decreased by two.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Unroll Aurora Polar Frobenius norm source scan by four.
status: rejected_screen
change:
  In the fused Aurora Polar normalization path, each thread accumulated four
  stride-separated source values per loop before the existing block reduction.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aurora_norm_sum_unroll4_l4_b8_20_20260621T064443Z.run.log
    val_loss=8.505538, train_elapsed_s=3.587, completed_steps=20.
measured_effect:
  Against the promoted MS-EDEN chunk-scale baseline
  target/nsys/ms_eden_chunk_scale_unroll4_l4_b8_20_20260621T062103Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1361.184230ms to
  1360.497890ms over 20 profiled steps, but profiled train time moved from
  3.584s to 3.587s. Neighboring kernels also drifted slightly worse:
    linear_backward_projection_pair_cta_device_scale_kernel 621.664121ms -> 623.041158ms
    fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel 161.960962ms -> 162.229717ms
    fp32_to_nvfp4_ms_eden_device_scale_kernel 160.685299ms -> 160.852854ms
decision:
  Reject before the 900-second gate. The target-kernel delta was too small and
  the short wall-clock screen regressed. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Unroll fixed 32-lane Hadamard transform in MS-EDEN packing.
status: rejected_screen
change:
  Replaced the five-iteration Hadamard loop used by MS-EDEN packing with a
  fixed 1/2/4/8/16 butterfly utility.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_hadamard_unroll_l4_b8_20_20260621T064734Z.run.log
    val_loss=8.505538, train_elapsed_s=3.587, completed_steps=20.
measured_effect:
  Against the promoted MS-EDEN chunk-scale baseline
  target/nsys/ms_eden_chunk_scale_unroll4_l4_b8_20_20260621T062103Z.run.log,
  profiled train time moved from 3.584s to 3.587s. The main pack kernels did
  not improve:
    fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel 161.960962ms -> 162.152785ms
    fp32_to_nvfp4_ms_eden_device_scale_kernel 160.685299ms -> 160.712347ms
    rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel 71.743201ms -> 71.816529ms
decision:
  Reject before the 900-second gate. The compiler/runtime already handles the
  small fixed loop well enough, and the short wall-clock screen regressed.
  Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Skip final CTA projection shared-memory sync after last K tile.
status: rejected_screen
change:
  Changed projection_accumulator and projection_accumulator_aligned to run the
  post-MMA block sync only when another K tile follows.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/proj_cta_skip_final_sync_l4_b8_20_20260621T064931Z.run.log
    val_loss=8.505538, train_elapsed_s=3.586, completed_steps=20.
measured_effect:
  Against the promoted MS-EDEN chunk-scale baseline
  target/nsys/ms_eden_chunk_scale_unroll4_l4_b8_20_20260621T062103Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  621.664121ms to 621.771841ms over 20 profiled steps, and profiled train time
  moved from 3.584s to 3.586s.
decision:
  Reject before the 900-second gate. Removing the final barrier did not improve
  the target kernel or short wall-clock. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Cache flattened attention softmax/probability indices.
status: rejected_screen
change:
  Cached the score index in attention_softmax_forward_kernel's probability
  store loop and cached the log-sum-exp row index in attention_prob_ds_kernel.
  Masked probability and gradient stores still wrote zeros as before.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_log_sum_exp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/attention_index_cache_l4_b8_20_20260621T065123Z.run.log
    val_loss=8.505538, train_elapsed_s=3.585, completed_steps=20.
measured_effect:
  Against the promoted MS-EDEN chunk-scale baseline
  target/nsys/ms_eden_chunk_scale_unroll4_l4_b8_20_20260621T062103Z.run.log,
  attention_prob_ds_kernel moved from 90.620948ms to 90.690734ms over 20
  profiled steps. attention_softmax_forward_kernel moved from 45.188987ms to
  45.180041ms, but profiled train time still moved from 3.584s to 3.585s.
decision:
  Reject before the 900-second gate. The forward softmax delta was tiny, the
  backward probability kernel regressed, and the short wall-clock regressed.
  Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Use aligned CTA projection path for aligned LM-head shapes.
status: accepted
change:
  lm_head_kernel now uses the aligned no-bias NVFP4 CTA projection body when
  token_count, vocab_size, and input_dim are divisible by the CTA M/N/K tile
  sizes. The generic path remains for non-aligned tests and shapes. The CUDA
  module was split into gpt/lm_head/kernels.rs to keep file ownership small.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/lm_head_aligned_path_l4_b8_20_20260621T065355Z.run.log
    val_loss=8.505538, train_elapsed_s=3.584, completed_steps=20.
  900-second held-out gate:
    target/lm_head_aligned_path_l4_b8_900_20260621T065424Z.log
    val_loss=4.016790, train_elapsed_s=900.004, completed_steps=4798.
measured_effect:
  Against the promoted MS-EDEN chunk-scale baseline
  target/nsys/ms_eden_chunk_scale_unroll4_l4_b8_20_20260621T062103Z.run.log,
  lm_head_kernel moved from 113.621970ms to 112.524121ms over 20 profiled
  steps. The 20-step train time stayed at 3.584s. The 900-second gate completed
  12 more steps than the previous promoted baseline while validation loss moved
  from 4.002436 to 4.016790, a +0.359% change.
decision:
  Promote under the current acceptance rule: validation loss stayed within the
  +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Skip final CTA shared-memory barrier in f16 tensor-core matmul bodies.
status: accepted
change:
  Added a shared helper that keeps the CTA tile-reuse barrier between K chunks
  but skips it after the final K chunk before global stores. Applied the helper
  to the six f16 CTA matmul bodies plus the base f16 CTA path.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul_tiled -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log
    val_loss=8.505538, train_elapsed_s=3.582, completed_steps=20.
  900-second held-out gate:
    target/f16_skip_final_cta_sync_l4_b8_900_20260621T074127Z.log
    val_loss=3.993229, train_elapsed_s=900.120, completed_steps=4790.
measured_effect:
  Against the promoted LM-head aligned baseline
  target/nsys/lm_head_aligned_path_l4_b8_20_20260621T065355Z.run.log,
  f16_cta_tc_matmul_f32_kernel moved from 218.561695ms to 216.870461ms over
  20 profiled steps. f16_cta_tc_matmul_f32_rhs_kernel moved from 121.649975ms
  to 121.494588ms, while f16_cta_tc_matmul_f32_a_transposed_rhs_kernel moved
  from 121.392258ms to 121.406391ms. Profiled train time moved from 3.584s to
  3.582s.
  The 900-second gate validation loss improved from 4.016790 to 3.993229
  while completed steps moved from 4798 to 4790.
decision:
  Promote under the current acceptance rule: validation loss improved on the
  fixed 900-second held-out gate.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before profile
experiment: Reuse Aurora NVFP4 encode lane values with half-warp shuffles.
status: rejected_correctness_gate
change:
  Tried replacing the second pair of global reads in Aurora's 16-value NVFP4
  encode group with half-warp shuffles from the values already loaded for
  amax/error calculation.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture:
    aborted after more than 10 minutes. The test process was still running and
    GPU0 was at 100% utilization, so the candidate did not pass the optimizer
    correctness gate.
measured_effect:
  No nsys screen or 900-second gate was run. The candidate failed before
  profiling.
decision:
  Reject and revert. Do not retry this shuffle substitution without first
  isolating why the Aurora optimizer test runtime explodes.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Pair aligned nobias projection stores by shared row scale.
status: rejected_screen
change:
  Changed store_accumulator_aligned to store acc[0]/acc[1] and acc[2]/acc[3]
  as row pairs so each row's input_global_scale was loaded and multiplied once
  instead of once per scalar store.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/proj_store_pair_aligned_l4_b8_20_20260621T081302Z.run.log
    val_loss=8.505538, train_elapsed_s=3.606, completed_steps=20.
measured_effect:
  Against the promoted f16 CTA sync baseline
  target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  621.385509ms to 628.303774ms over 20 profiled steps. Profiled train time
  moved from 3.582s to 3.606s.
decision:
  Reject before the 900-second gate. The target kernel and short wall-clock
  both regressed. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Remove second barrier from attention backward softmax_d reduction.
status: rejected_screen
change:
  Changed softmax_d_kernel's two-warp head reduction so only dim 0 read the
  final shared sum, removing the second block-wide sync from the reduction.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/softmax_d_single_sync_l4_b8_20_20260621T081514Z.run.log
    val_loss=8.505538, train_elapsed_s=3.589, completed_steps=20.
measured_effect:
  Against the promoted f16 CTA sync baseline
  target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log,
  softmax_d_kernel moved from 4.716164ms to 4.709215ms over 20 profiled steps,
  but profiled train time moved from 3.582s to 3.589s.
decision:
  Reject before the 900-second gate. The target-kernel gain was too small and
  the short wall-clock regressed. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Move LM-head aligned/generic selection from device branch to host launcher.
status: rejected_screen
change:
  Added a separate lm_head_aligned_kernel and made the host launcher call it
  when token_count, vocab_size, and input_dim were CTA-aligned. The generic
  kernel remained for non-aligned tests and shapes.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/lm_head_host_aligned_split_l4_b8_20_20260621T081804Z.run.log
    val_loss=8.506057, train_elapsed_s=3.585, completed_steps=20.
measured_effect:
  Against the promoted f16 CTA sync baseline
  target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log,
  lm_head moved from 112.481106ms to 112.720425ms over 20 profiled steps.
  Profiled train time moved from 3.582s to 3.585s.
decision:
  Reject before the 900-second gate. The target kernel and short wall-clock
  both regressed. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Split paired linear backward CTA projection launch into two homogeneous launches.
status: rejected_screen
change:
  Replaced linear_backward_projection_pair_cta_device_scale_kernel with two
  calls to the existing linear_backward_projection_cta_device_scale_kernel,
  one for dinput and one for dweight.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/linear_bwd_split_cta_launch_l4_b8_20_20260621T082033Z.run.log
    val_loss=8.505538, train_elapsed_s=3.594, completed_steps=20.
measured_effect:
  Against the promoted f16 CTA sync baseline
  target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log,
  projection time moved from
  linear_backward_projection_pair_cta_device_scale_kernel=621.385509ms to
  linear_backward_projection_cta_device_scale_kernel=632.232545ms over 20
  profiled steps. Profiled train time moved from 3.582s to 3.594s.
decision:
  Reject before the 900-second gate. The extra launch count lost to the paired
  kernel despite removing the pair branch. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Skip terminal shared-memory barrier in Aurora Polar CTA tile.
status: rejected_screen
change:
  Added a per-tile flag so Aurora Polar CTA matmul skipped the final
  thread::sync_threads() only when the same block had no later tile to stage.
  The conservative form kept the barrier for any block that would continue the
  tile loop.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aurora_polar_skip_terminal_tile_sync_l4_b8_20_20260621T082808Z.run.log
    val_loss=8.505538, train_elapsed_s=3.591, completed_steps=20.
measured_effect:
  Against the promoted f16 CTA sync baseline
  target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1362.072601ms to
  1373.652972ms over 20 profiled steps. Profiled train time moved from
  3.582s to 3.591s.
decision:
  Reject before the 900-second gate. The Aurora kernel and short wall-clock
  both regressed. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Skip terminal shared-memory barrier in shared NVFP4 projection CTA accumulator.
status: rejected_screen
change:
  Changed the generic and aligned NVFP4 projection CTA accumulators to keep
  the post-MMA shared-memory barrier only when another K slice would be staged.
  Each CTA projection block computes one output tile, so the terminal K-slice
  barrier is not needed for correctness.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/projection_cta_skip_final_sync_l4_b8_20_20260621T083131Z.run.log
    val_loss=8.505538, train_elapsed_s=3.582, completed_steps=20.
measured_effect:
  Against the promoted f16 CTA sync baseline
  target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  621.385509ms to 621.284999ms over 20 profiled steps, but lm_head_kernel
  moved from 112.481106ms to 112.769367ms. Profiled train time stayed at
  3.582s.
decision:
  Reject before the 900-second gate. The shared projection change did not
  produce a meaningful short-run wall-clock gain and regressed another
  projection user. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Unroll 32-point MS-EDEN Hadamard transform stages.
status: rejected_screen
change:
  Replaced the fixed five-stage Hadamard while loop in the MS-EDEN pack body
  with explicit shuffle stages for strides 1, 2, 4, 8, and 16.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test nvfp4_quant -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_unroll_hadamard_l4_b8_20_20260621T083411Z.run.log
    val_loss=8.505538, train_elapsed_s=3.584, completed_steps=20.
measured_effect:
  Against the promoted f16 CTA sync baseline
  target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log,
  fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel moved from
  161.869875ms to 162.155047ms, and
  fp32_to_nvfp4_ms_eden_device_scale_kernel moved from 160.623396ms to
  160.712656ms over 20 profiled steps. Profiled train time moved from
  3.582s to 3.584s.
decision:
  Reject before the 900-second gate. The explicit unroll made the target
  kernels and short wall-clock slower. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Restore multi-warp MS-EDEN pack kernels on the current L4/B8 baseline.
status: accepted
change:
  Changed MS-EDEN pack launch shape from one 32-thread CTA per Hadamard chunk
  to one 256-thread CTA packing eight independent 32-value chunks. Each warp
  owns one chunk, and the kernel receives chunk_count so inactive tail warps
  return before writing output.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test nvfp4_quant -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_pack_warp8_current_l4_b8_20_20260621T083902Z.run.log
    val_loss=8.505538, train_elapsed_s=3.435, completed_steps=20.
  100-step SYNTH screen:
    target/ms_eden_pack_warp8_current_l4_b8_100_20260621T083924Z.log
    val_loss=6.548421, train_elapsed_s=17.535, completed_steps=100.
  900-second held-out gate:
    target/ms_eden_pack_warp8_current_l4_b8_900_20260621T083951Z.log
    val_loss=4.010499, train_elapsed_s=900.019, completed_steps=4989.
measured_effect:
  Against the promoted f16 CTA sync baseline
  target/nsys/f16_skip_final_cta_sync_l4_b8_20_20260621T073906Z.run.log,
  fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel moved from
  161.869875ms to 109.203520ms, and
  fp32_to_nvfp4_ms_eden_device_scale_kernel moved from 160.623396ms to
  71.279059ms over 20 profiled steps. The short profile train time moved from
  3.582s to 3.435s.
  The 900-second gate completed 4989 steps versus the previous promoted
  baseline's 4790 steps. Held-out validation moved from 3.993229 to 4.010499,
  a +0.43% change.
decision:
  Promote under the current acceptance rule: held-out validation stayed within
  the +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Split FP32 MS-EDEN source gather paths.
status: accepted_900s
change:
  Split the shared FP32 MS-EDEN Hadamard input helper into separate row-major
  and transposed source helpers. The non-transposed and transposed kernels were
  already separate launch symbols; this removes the per-lane transpose_source
  branch from the hot gather path without changing seeds, scaling, RHT, or pack
  math.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_split_fp32_source_l4_b8_20_20260621T112900Z.run.log
    val_loss=8.506022, train_elapsed_s=3.397, completed_steps=20.
  100-step SYNTH screen:
    target/ms_eden_split_fp32_source_l4_b8_100_20260621T112925Z.log
    val_loss=6.544603, train_elapsed_s=17.426, completed_steps=100.
  900-second held-out gate:
    target/ms_eden_split_fp32_source_l4_b8_900_20260621T112956Z.log
    val_loss=3.947354, train_elapsed_s=900.104, completed_steps=5026.
measured_effect:
  Against the accepted coalesced linear-bias profile
  target/nsys/linear_bias_coalesced_l4_b8_20_20260621T105744Z.run.log,
  fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel moved from
  108.275280ms to 108.253016ms over 20 profiled steps,
  fp32_to_nvfp4_ms_eden_device_scale_kernel moved from 65.408980ms to
  65.337191ms, and rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel
  moved from 63.461071ms to 63.420185ms. The 900-second gate improved held-out
  validation loss from 3.976525 to 3.947354 and completed steps from 5024 to
  5026.
decision:
  Promote. This passes the fixed-wall objective directly: lower held-out
  validation loss and higher completed step count under the same 900-second
  budget.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Use scalar MS-EDEN device global scale in linear-backward projection
  stores.
status: rejected_reverted
change:
  Added a temporary MS-EDEN-only variant of the paired CTA linear-backward
  projection kernel that read scalar device global scales instead of row-global
  scale arrays at the final accumulator store.
verification:
  cargo fmt --all: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass after fixing the transposed error scalar to reuse e_h.global_scale.
  20-step nsys:
    target/nsys/linear_bwd_scalar_input_scale_b16_l4d1024_20_20260622T135504Z.run.log
    val_loss=9.063751, train_elapsed_s=6.072, completed_steps=20.
  100-step SYNTH screen:
    target/linear_bwd_scalar_input_scale_b16_l4d1024_100_20260622T135546Z.log
    val_loss=6.300044, train_elapsed_s=30.387, completed_steps=100.
  900-second held-out gate:
    target/linear_bwd_scalar_input_scale_b16_l4d1024_900_20260622T135632Z.log
    val_loss=3.609159, train_elapsed_s=900.175, completed_steps=2892.
measured_effect:
  Against the accepted baseline
  target/nsys/ms_eden_no_chunk_amax_b16_l4d1024_20_20260622T131123Z.sqlite,
  the projection kernel moved from 1272.332052ms to 1268.428120ms over
  400 calls. The 100-step screen was slightly better than baseline
  val_loss=6.305133 / 30.416s, but the 900-second gate regressed from
  val_loss=3.603050 / 2894 completed steps to val_loss=3.609159 / 2892
  completed steps.
decision:
  Reject and revert. The final gate did not meet the active promotion rule:
  validation was within the 1% noise band but completed step count decreased.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Write residual-chain gradients directly to the next backward
  destination.
status: accepted_900s
change:
  Removed the backward-only device-to-device copies that moved final layer-norm
  d_residual into the last block's d_residual_out and each block's
  d_residual_in into the previous block's d_residual_out or embedding residual.
  The residual add/layer-norm kernels now write those values directly into the
  destination consumed by the next backward step. The old backward device-copy
  helper became unused and was removed.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test residual_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/direct_residual_grad_l4_b8_20_20260621T131424Z.run.log
    val_loss=8.506022, train_elapsed_s=3.470, completed_steps=20.
  100-step SYNTH screen:
    target/direct_residual_grad_l4_b8_100_20260621T131445Z.log
    val_loss=6.543701, train_elapsed_s=17.386, completed_steps=100.
  900-second held-out gate:
    target/direct_residual_grad_l4_b8_900_20260621T131534Z.log
    val_loss=3.954098, train_elapsed_s=900.126, completed_steps=5028.
measured_effect:
  Against the accepted MS-EDEN source-split profile
  target/nsys/ms_eden_split_fp32_source_l4_b8_20_20260621T112900Z.nsys-rep,
  D2D memcpy count moved from 2671 to 2571 over 20 profiled steps and D2D
  memcpy time moved from 63.187351ms to 61.899231ms. The 20-step profiled
  train_elapsed_s was noisier and moved from 3.397 to 3.470, so this should
  not be read as a large profiler win.
  The 100-step screen improved train_elapsed_s from 17.426 to 17.386 and
  validation loss from 6.544603 to 6.543701. The 900-second gate completed 2
  more steps than the accepted baseline while validation loss moved from
  3.947354 to 3.954098, a +0.171% change.
decision:
  Promote under the current acceptance rule: held-out validation stayed within
  the +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Decouple Aurora square Polar tile from rectangular host f16 attention tile.
status: rejected_screen
change:
  Added local square Aurora Polar tile/fragments and changed the shared host
  f16 CTA tile to M128/N32, so attention-side f16 matmuls could test a
  rectangular tile while Aurora kept its square Polar tile contract.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul_tiled -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  20-step plain screen:
    target/f16_attention_rect_aurora_square_l4_b8_20_20260621T115508Z.log
    val_loss=8.506022, train_elapsed_s=3.418, completed_steps=20.
  20-step nsys screen:
    target/nsys/f16_attention_rect_aurora_square_l4_b8_20_20260621T115523Z.run.log
    train_elapsed_s=3.498.
measured_effect:
  Against the accepted MS-EDEN source split profile
  target/nsys/ms_eden_split_fp32_source_l4_b8_20_20260621T112900Z.run.log,
  f16_cta_tc_matmul_f32_kernel moved from 225.104836ms to 233.306115ms,
  f16_cta_tc_matmul_f32_a_transposed_rhs_kernel moved from 126.800414ms to
  134.775889ms, and f16_cta_tc_matmul_f32_rhs_kernel moved from 124.911671ms
  to 127.010860ms over 20 profiled steps. Aurora and linear projection stayed
  roughly flat: aurora_mega_update_cooperative_kernel moved from 1.382451463s
  to 1.381607862s, and linear_backward_projection_pair_cta_device_scale_kernel
  moved from 651.431430ms to 650.909127ms.
decision:
  Reject before the 100-step and 900-second gates and revert the code. The
  attention f16 kernels got slower, and the small Aurora/projection movement
  did not compensate.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Split Aurora Polar f16 RHS staging by row-major/transposed source.
status: rejected_900s
change:
  Split the Aurora Polar f16 shared-memory B staging path into row-major and
  transposed helper bodies, moving the rhs_transposed branch outside the
  per-staged-element index calculation.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys screen:
    target/nsys/aurora_stage_split_rhs_l4_b8_20_20260621T120206Z.run.log
    val_loss=8.506022, train_elapsed_s=3.473, completed_steps=20.
  100-step SYNTH screen:
    target/aurora_stage_split_rhs_l4_b8_100_20260621T120254Z.log
    val_loss=6.547944, train_elapsed_s=17.387, completed_steps=100.
  900-second held-out gate:
    target/aurora_stage_split_rhs_l4_b8_900_20260621T120326Z.log
    val_loss=4.034446, train_elapsed_s=900.001, completed_steps=5026.
measured_effect:
  Against the accepted MS-EDEN source split profile
  target/nsys/ms_eden_split_fp32_source_l4_b8_20_20260621T112900Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1.382451463s to
  1.379752055s, linear_backward_projection_pair_cta_device_scale_kernel moved
  from 651.431430ms to 649.217101ms, and f16_cta_tc_matmul_f32_kernel moved
  from 225.104836ms to 224.361996ms over 20 profiled steps. Profiled train
  time still moved from 3.397s to 3.473s, so the profiler screen was noisy.
  The full 900-second gate matched the accepted completed step count, 5026, but
  worsened held-out validation loss from 3.947354 to 4.034446.
decision:
  Reject and revert the code. The fixed-wall validation result failed the
  promotion rule even though a few kernel totals moved slightly in the 20-step
  profile.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Hoist reciprocal denominator in cross-entropy dlogits loop.
status: rejected_pre_gate
change:
  Replaced per-logit division by the row softmax denominator with one reciprocal
  and a multiply in the dlogits write loop.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/cross_entropy_inv_denom_l4_b8_20_20260621T122202Z.run.log
    val_loss=8.505426, train_elapsed_s=3.479, completed_steps=20.
measured_effect:
  Against the accepted MS-EDEN source split profile
  target/nsys/ms_eden_split_fp32_source_l4_b8_20_20260621T112900Z.run.log,
  cross_entropy_kernel moved from 55.353896ms to 55.338910ms over 20 profiled
  steps. Profiled train time moved from 3.397s to 3.479s.
decision:
  Reject before the 100-step and 900-second gates and revert the code. The
  kernel movement was far below profiler noise and did not justify a longer
  fixed-wall gate.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Write QKV and attention log-sum-exp directly into block tape.
status: rejected_900s
change:
  Routed training forward QKV and attention log-sum-exp outputs directly into
  the per-block tape buffers instead of writing to shared scratch and copying
  those two buffers into tape after attention. Inference/no-tape mode kept the
  original scratch path.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/direct_qkv_lse_tape_l4_b8_20_20260621T122611Z.run.log
    val_loss=8.503760, train_elapsed_s=3.460, completed_steps=20.
  100-step SYNTH screen:
    target/direct_qkv_lse_tape_l4_b8_100_20260621T122649Z.log
    val_loss=6.544517, train_elapsed_s=17.344, completed_steps=100.
  900-second held-out gate:
    target/direct_qkv_lse_tape_l4_b8_900_20260621T122715Z.log
    val_loss=4.002939, train_elapsed_s=900.064, completed_steps=5045.
measured_effect:
  Against the accepted MS-EDEN source split profile
  target/nsys/ms_eden_split_fp32_source_l4_b8_20_20260621T112900Z.run.log,
  device-to-device copy volume dropped from 62296.613 MB to 53796.856 MB over
  20 profiled steps, and device-to-device copy time dropped from 63.244391ms to
  48.260440ms. The 100-step screen was slightly faster, 17.344s versus
  17.426s. The full fixed-wall gate completed 5045 steps versus the accepted
  5026, but held-out validation loss worsened from 3.947354 to 4.002939.
decision:
  Reject and revert the code. The memory-copy reduction was real, but the
  fixed-wall held-out loss worsened by more than the +/-1% acceptance band.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Write MLP up and relu2 activations directly into block tape.
status: rejected_900s
change:
  Routed training forward MLP pre-activation and relu2 activation outputs
  directly into the per-block tape buffers instead of writing to shared scratch
  and copying those two large activation buffers into tape after the MLP.
  Inference/no-tape mode kept the original scratch path.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/direct_mlp_tape_l4_b8_20_20260621T124522Z.run.log
    val_loss=8.506022, train_elapsed_s=3.454, completed_steps=20.
  100-step SYNTH screen:
    target/direct_mlp_tape_l4_b8_100_20260621T124550Z.log
    val_loss=6.543642, train_elapsed_s=17.313, completed_steps=100.
  900-second held-out gate:
    target/direct_mlp_tape_l4_b8_900_20260621T124615Z.log
    val_loss=4.030444, train_elapsed_s=900.161, completed_steps=5071.
measured_effect:
  Against the accepted MS-EDEN source split profile
  target/nsys/ms_eden_split_fp32_source_l4_b8_20_20260621T112900Z.run.log,
  device-to-device copy volume dropped from 62296.613 MB to 39748.035 MB over
  20 profiled steps, and device-to-device copy time dropped from 63.244391ms to
  30.603698ms. The 100-step screen was faster, 17.313s versus 17.426s, with
  similar held-out loss. The full fixed-wall gate completed 5071 steps versus
  the accepted 5026, but held-out validation loss worsened from 3.947354 to
  4.030444.
decision:
  Reject and revert the code. The memory-copy reduction and step-count increase
  were real, but fixed-wall held-out loss worsened by more than the +/-1%
  acceptance band.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: F16 CTA matmul M128/N32 tile shape.
status: rejected_correctness
change:
  Tested changing the global f16 tensor-core CTA tile from M64/N64/K16 to
  M128/N32/K16 with the same 256 threads. The warp map changed from four row
  groups by two column groups to eight row groups by one column group.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul_tiled -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/f16_tile_m128_n32_l4_b8_20_20260621T115024Z.run.log
    step 0 finite, step 1 loss=NaN, heldout_eval val_loss=NaN.
measured_effect:
  The profile showed speedups in several kernels, but the full training path
  became non-finite immediately. Against the accepted MS-EDEN source-split
  profile, aurora_mega_update_cooperative_kernel moved from 1.382451463s to
  1.279894511s and f16_cta_tc_matmul_f32_kernel moved from 225.104836ms to
  215.321531ms over 20 profiled steps, but these timings are not usable
  because training loss became NaN at step 1.
likely_cause:
  Aurora's symmetric Polar Gram path currently assumes the shared f16 CTA tile
  is square: run_symmetric_tiles derives one tile_dim from CTA_M and mirrors
  off-diagonal stores with the same tile shape. A global rectangular f16 tile
  can pass standalone attention/f16 tests while breaking the optimizer path.
decision:
  Reject and revert. Do not change the global f16 CTA tile to a rectangular
  shape while Aurora reuses it. The next valid version would need a separate
  attention-only rectangular f16 kernel or a rectangular-safe Aurora Polar path.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Add aligned fast paths to f16 RHS-transposed CTA matmul variants.
status: rejected_screen
change:
  Tried adding aligned staging and aligned stores to
  f16_cta_tc_matmul_f32_rhs_kernel and
  f16_cta_tc_matmul_f32_a_transposed_rhs_kernel, mirroring the existing aligned
  path used by the plain f32-input CTA matmul body.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul_tiled -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/f16_rhs_aligned_l4_b8_20_20260621T090042Z.run.log
    val_loss=8.505538, train_elapsed_s=3.448, completed_steps=20.
measured_effect:
  Against the accepted multi-warp MS-EDEN baseline
  target/nsys/ms_eden_pack_warp8_current_l4_b8_20_20260621T083902Z.run.log,
  f16_cta_tc_matmul_f32_rhs_kernel moved from 121.833720ms to
  129.082962ms, and f16_cta_tc_matmul_f32_a_transposed_rhs_kernel moved from
  121.801569ms to 126.011788ms over 20 profiled steps. Profiled train time
  moved from 3.435s to 3.448s.
decision:
  Reject before the 900-second gate. The target kernels and short wall-clock
  got slower. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Cache cross-entropy exponentials in dlogits.
status: rejected_screen
change:
  Tried writing exp(logit - row_max) into dlogits during the denominator pass,
  then reading it back during the final dlogits pass. This removed one exp
  calculation per vocab element but added an extra global write/read of the
  full logits-shaped buffer.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/cross_entropy_cache_exp_l4_b8_20_20260621T090411Z.run.log
    val_loss=8.505538, train_elapsed_s=3.449, completed_steps=20.
measured_effect:
  Against the accepted multi-warp MS-EDEN baseline
  target/nsys/ms_eden_pack_warp8_current_l4_b8_20_20260621T083902Z.run.log,
  cross_entropy_kernel moved from 55.344469ms to 70.514102ms over 20 profiled
  steps. Profiled train time moved from 3.435s to 3.449s.
decision:
  Reject before the 900-second gate. Extra global traffic outweighed removing
  the second exp. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Increase MS-EDEN pack CTA from 8 warps to 16 warps.
status: rejected_screen
change:
  Split pack launch constants from the general quant constants and tested
  packing sixteen independent 32-value MS-EDEN chunks per CTA with 512 threads,
  leaving tensor/row amax kernels at the existing 256-thread shape.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test nvfp4_quant -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_pack_warp16_l4_b8_20_20260621T090700Z.run.log
    val_loss=8.505538, train_elapsed_s=3.441, completed_steps=20.
measured_effect:
  Against the accepted 8-warp MS-EDEN pack baseline
  target/nsys/ms_eden_pack_warp8_current_l4_b8_20_20260621T083902Z.run.log,
  fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel moved from
  109.203520ms to 111.253414ms, and
  fp32_to_nvfp4_ms_eden_device_scale_kernel moved from 71.279059ms to
  73.603381ms over 20 profiled steps. Profiled train time moved from
  3.435s to 3.441s.
decision:
  Reject before the 900-second gate. The larger pack CTA reduced block count
  but slowed the target kernels. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Decode paired linear-backward CTA tile coordinates with shifts.
status: accepted
change:
  The paired linear-backward CTA projection kernel no longer computes tile row
  and column with runtime integer division/modulo on every thread. The host
  wrapper now asserts the paired projection grid column count is a power of two
  and passes mask/shift parameters, so the device kernel decodes tile_col with
  bit-and and tile_row with shift.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/linear_bwd_tile_shift_l4_b8_20_20260621T090938Z.run.log
    val_loss=8.505538, train_elapsed_s=3.434, completed_steps=20.
  100-step SYNTH screen:
    target/linear_bwd_tile_shift_l4_b8_100_20260621T090957Z.log
    val_loss=6.550270, train_elapsed_s=17.536, completed_steps=100.
  900-second held-out gate:
    target/linear_bwd_tile_shift_l4_b8_900_20260621T091022Z.log
    val_loss=3.976722, train_elapsed_s=900.158, completed_steps=4993.
measured_effect:
  Against the accepted multi-warp MS-EDEN baseline
  target/nsys/ms_eden_pack_warp8_current_l4_b8_20_20260621T083902Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  628.820011ms to 626.852315ms over 20 profiled steps. Profiled train time
  moved from 3.435s to 3.434s.
  The 900-second gate improved held-out validation loss from 4.010499 to
  3.976722 and completed steps from 4989 to 4993.
decision:
  Promote. This passes the fixed-wall objective directly: lower held-out
  validation loss and higher completed step count under the same 900-second
  budget.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Skip final CTA projection post-MMA barrier.
status: rejected_pre_gate
change:
  In the shared CTA NVFP4 projection accumulator, removed the post-MMA
  thread-block barrier on the final K tile while keeping barriers before
  subsequent shared-memory staging.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/projection_final_sync_cut_l4_b8_20_20260621T133840Z.run.log
    val_loss=8.506022, train_elapsed_s=3.476, completed_steps=20.
measured_effect:
  Against the accepted direct-residual-gradient profile
  target/nsys/direct_residual_grad_l4_b8_20_20260621T131424Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  648.530862ms to 650.302720ms over 20 profiled steps. lm_head_kernel moved
  from 116.768158ms to 117.191220ms. Profiled train time moved from 3.470s to
  3.476s.
decision:
  Reject before the 900-second gate and revert the code. The candidate slowed
  the projection path it was intended to improve.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: M64/N32 projection CTA shape for NVFP4 projection matmuls.
status: rejected_screen
change:
  Tested a 64-row by 32-column projection CTA tile while keeping K=64 and
  512 threads. The warp map changed from 2x8 row/column warp groups to 4x4,
  and the aligned B-pack staging path was guarded for the smaller B tile.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/projection_cta_m64_n32_l4_b8_20_20260621T112337Z.run.log
    val_loss=8.506022, train_elapsed_s=3.398, completed_steps=20.
measured_effect:
  Against the accepted linear-bias coalescing profile
  target/nsys/linear_bias_coalesced_l4_b8_20_20260621T105744Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel regressed from
  652.037706ms to 654.433265ms over 20 profiled steps. lm_head_kernel improved
  from 117.359029ms to 113.551149ms, but mlp_projection_kernel moved from
  69.572840ms to 70.141766ms, mlp_projection_relu2_kernel moved from
  66.691061ms to 66.791260ms, and attention_projection_kernel moved from
  64.989959ms to 65.382372ms. The plain 20-step training log moved from
  3.401s to 3.398s, which is too small to outweigh the mixed kernel result.
decision:
  Reject before the 100-step and 900-second gates. The main linear-backward
  projection kernel regressed and the small wall-clock change is not a reliable
  fixed-wall objective signal. Code was reverted.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Adam no-decay for layer-norm tensors and linear biases.
status: rejected_900s
change:
  Tested nanoGPT-style AdamW parameter grouping for the Adam-managed tensors:
  token embeddings kept weight decay, while layer-norm weights, layer-norm
  biases, and linear biases used zero weight decay. Aurora-managed matrix
  weights were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  100-step SYNTH screen:
    target/adam_no_decay_bias_ln_l4_b8_100_20260621T103611Z.log
    val_loss=6.551615, train_elapsed_s=17.445, completed_steps=100.
  900-second held-out gate:
    target/adam_no_decay_bias_ln_l4_b8_900_20260621T103646Z.log
    val_loss=4.012671, train_elapsed_s=900.142, completed_steps=5004.
measured_effect:
  Against the accepted baseline
  target/ms_eden_inv_scale_l4_b8_900_20260621T093858Z.log, held-out
  validation loss moved from 3.943465 to 4.012671, and completed steps moved
  from 5011 to 5004.
decision:
  Reject and revert the code. This changed optimizer semantics but worsened
  held-out validation by about 1.75% and reduced completed step count under the
  same fixed-wall budget.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Coalesced 32-column linear bias-gradient reduction.
status: accepted_900s
change:
  Replaced the one-block-per-output-column linear_bias_grad_kernel with a
  32-column tile. Each warp lane owns one adjacent output column and walks rows,
  so the warp reads contiguous e[row, col..col+31] values instead of
  row-strided addresses separated by output_dim. Moved the bias-gradient body
  into gpt/linear_backward/bias.rs to keep ownership out of the large
  linear_backward.rs kernel shell.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/linear_bias_coalesced_l4_b8_20_20260621T105744Z.run.log
    val_loss=8.506022, train_elapsed_s=3.401, completed_steps=20.
  100-step SYNTH screen:
    target/linear_bias_coalesced_l4_b8_100_20260621T105810Z.log
    val_loss=6.543840, train_elapsed_s=17.444, completed_steps=100.
  900-second held-out gate:
    target/linear_bias_coalesced_l4_b8_900_20260621T105837Z.log
    val_loss=3.976525, train_elapsed_s=900.005, completed_steps=5024.
measured_effect:
  Against the accepted MS-EDEN reciprocal profile
  target/nsys/ms_eden_inv_scale_l4_b8_20_20260621T093759Z.run.log,
  linear_bias_grad_kernel moved from 41.079015ms to 27.611254ms over 20
  profiled steps. Profiled train time moved from 3.424s to 3.401s.
  The 900-second gate completed 13 more steps than the previous promoted
  baseline. Held-out validation moved from 3.943465 to 3.976525, a +0.84%
  change.
decision:
  Promote under the current acceptance rule: held-out validation stayed within
  the +/-1% no-meaningful-change band and completed step count increased.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before gate
experiment: Coalesced 32-column layer-norm backward parameter reduction.
status: rejected_pre_gate
change:
  Tested the same 32-column coalescing pattern used for linear bias gradients
  on layer_norm_backward_params_kernel. Each warp lane owned one adjacent
  embedding column and walked rows, then warp-0 summed the eight per-column
  partials for d_weight and d_bias.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test layer_norm_backward_params -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test layer_norm_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/layer_norm_param_coalesced_l4_b8_20_20260621T111850Z.run.log
    val_loss=8.505636, train_elapsed_s=3.399, completed_steps=20.
measured_effect:
  Against the accepted coalesced linear-bias profile
  target/nsys/linear_bias_coalesced_l4_b8_20_20260621T105744Z.run.log,
  layer_norm_backward_params_kernel moved from 22.020317ms to 24.767617ms
  over 20 profiled steps.
decision:
  Reject before the 900-second gate and revert the code. The target kernel
  regressed despite the short wall-clock being within profile noise.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Unroll aligned FP16 CTA f32 staging loops.
status: rejected_pre_gate
change:
  In f16_cta_tc_matmul_f32_kernel's aligned f32 staging path, replaced the
  four-iteration per-thread staging loops for A and transposed B tiles with
  explicit fixed-offset unrolled stores.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test f16_tc_matmul_tiled -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/f16_stage_aligned_unroll_l4_b8_20_20260621T095707Z.run.log
    val_loss=8.505588, train_elapsed_s=3.433, completed_steps=20.
measured_effect:
  Against the current accepted MS-EDEN reciprocal profile
  target/nsys/ms_eden_inv_scale_l4_b8_20_20260621T093759Z.run.log,
  f16_cta_tc_matmul_f32_kernel moved from 217.291413ms to 220.397478ms
  over 20 profiled steps. Profiled train time moved from 3.424s to 3.433s.
decision:
  Reject before the 900-second gate and revert the code. The unrolled staging
  shape made the target f16 CTA matmul slower in the current profile.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Hoist row scale loads in aligned NVFP4 CTA projection store.
status: rejected_pre_gate
change:
  Specialized store_accumulator_aligned to compute the two output rows, two
  adjacent output columns, and two row global scales once, then store all four
  accumulator values directly instead of calling store_one_aligned four times.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/projection_store_pair_l4_b8_20_20260621T100124Z.run.log
    val_loss=8.505588, train_elapsed_s=3.424, completed_steps=20.
measured_effect:
  Against the current accepted MS-EDEN reciprocal profile
  target/nsys/ms_eden_inv_scale_l4_b8_20_20260621T093759Z.run.log,
  lm_head_kernel moved from 112.801360ms to 112.570704ms over 20 profiled
  steps, but the larger linear_backward_projection_pair_cta_device_scale_kernel
  moved from 625.867638ms to 626.393080ms. Total profiled train time stayed at
  3.424s.
decision:
  Reject before the 900-second gate and revert the code. The small LM-head win
  did not offset the larger linear-backward projection regression.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Use warp shuffles for Aurora four/six encode pair values.
status: rejected_correctness
change:
  In Aurora's fused four/six encode writeback, replaced the pair-lane global
  reloads used for FP4 payload packing with half-warp shuffles from the value
  already loaded for group amax/error estimation.
verification:
  cargo fmt --all --check: initially failed only on line wrapping.
  cargo fmt --all: pass.
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: timeout.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: timeout in aurora::aurora_mega_update_matches_first_iteration_recurrence.
measured_effect:
  No profiling run. The candidate failed the Aurora optimizer verification gate
  before profiling.
decision:
  Reject and revert. Avoid using this shuffle rewrite unless the Aurora encode
  correctness issue is isolated first.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Skip final post-MMA sync in NVFP4 CTA projection accumulator.
status: rejected_pre_gate
change:
  In projection_accumulator and projection_accumulator_aligned, kept the
  post-MMA block sync only when another k tile remained, on the assumption that
  the sync only protected shared tiles from the next staging step.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/projection_skip_final_sync_l4_b8_20_20260621T101539Z.run.log
    val_loss=8.505588, train_elapsed_s=3.452, completed_steps=20.
measured_effect:
  Against the current accepted MS-EDEN reciprocal profile
  target/nsys/ms_eden_inv_scale_l4_b8_20_20260621T093759Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  625.867638ms to 635.276893ms, lm_head_kernel moved from 112.801360ms to
  114.414818ms, and profiled train time moved from 3.424s to 3.452s.
decision:
  Reject before the 900-second gate and revert the code. The conditional sync
  shape made the CTA projection path slower.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Decode Aurora Polar CTA tile coordinates with power-of-two shifts.
status: rejected_pre_gate
change:
  Replaced the tile_row/tile_col division in run_plain_tiles, run_next_tiles,
  and run_symmetric_tiles with a match-based power-of-two shift/mask helper and
  kept the division path as a fallback for unsupported tile counts.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aurora_tile_decode_shift_l4_b8_20_20260621T092943Z.run.log
    val_loss=8.505538, train_elapsed_s=3.445, completed_steps=20.
measured_effect:
  Against the accepted linear-backward tile-shift profile
  target/nsys/linear_bwd_tile_shift_l4_b8_20_20260621T090938Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1.362562731s to
  1.371600995s over 20 profiled steps. Profiled train time moved from 3.434s
  to 3.445s.
decision:
  Reject before the 900-second gate and revert the code. The helper adds a
  runtime match and did not reduce the Aurora cooperative kernel time.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Delay Aurora transpose index division until transposed slots.
status: rejected_pre_gate
change:
  Moved row/col division in momentum_orient and update_one behind the
  transposed branch so non-transposed matrix slots could use index directly.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aurora_transpose_index_branch_l4_b8_20_20260621T093226Z.run.log
    val_loss=8.505538, train_elapsed_s=3.439, completed_steps=20.
measured_effect:
  Against the accepted linear-backward tile-shift profile
  target/nsys/linear_bwd_tile_shift_l4_b8_20_20260621T090938Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1.362562731s to
  1.364992942s over 20 profiled steps. Profiled train time moved from 3.434s
  to 3.439s.
decision:
  Reject before the 900-second gate and revert the code. The branch shape did
  not improve the current Aurora cooperative kernel profile.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Early return inactive causal attention probability/dS entries.
status: rejected_pre_gate
change:
  In attention_prob_ds_kernel, wrote zero p/ds values and returned immediately
  for masked causal positions, and computed the log-sum-exp index once for
  active positions.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/attention_prob_ds_active_return_l4_b8_20_20260621T093510Z.run.log
    val_loss=8.505538, train_elapsed_s=3.436, completed_steps=20.
measured_effect:
  Against the accepted linear-backward tile-shift profile
  target/nsys/linear_bwd_tile_shift_l4_b8_20_20260621T090938Z.run.log,
  attention_prob_ds_kernel moved from 90.615654ms to 90.675556ms over 20
  profiled steps. Profiled train time moved from 3.434s to 3.436s.
decision:
  Reject before the 900-second gate and revert the code. The branch/return
  shape did not improve the current attention backward profile.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Reuse reciprocal scale inside MS-EDEN pack.
status: accepted_900s
change:
  In the MS-EDEN pack body, compute inv_scale = 1 / (scale * global_scale) once
  per lane and reuse multiplication for x_scaled, hi, and lo instead of
  repeating division by the same scale product.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/ms_eden_inv_scale_l4_b8_20_20260621T093759Z.run.log
    val_loss=8.505094, train_elapsed_s=3.424, completed_steps=20.
  100-step SYNTH screen:
    target/ms_eden_inv_scale_l4_b8_100_20260621T093815Z.log
    val_loss=6.535932, train_elapsed_s=17.490, completed_steps=100.
  900-second held-out gate:
    target/ms_eden_inv_scale_l4_b8_900_20260621T093858Z.log
    val_loss=3.943465, train_elapsed_s=900.102, completed_steps=5011.
measured_effect:
  Against the accepted linear-backward tile-shift profile
  target/nsys/linear_bwd_tile_shift_l4_b8_20_20260621T090938Z.run.log,
  profiled train time moved from 3.434s to 3.424s over 20 steps.
  fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel moved from 109.248375ms
  to 107.043232ms, fp32_to_nvfp4_ms_eden_device_scale_kernel moved from
  71.326169ms to 66.239611ms, and
  rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel moved from
  62.171738ms to 61.669028ms.
  The 900-second gate improved held-out validation loss from 3.976722 to
  3.943465 and completed steps from 4993 to 5011.
decision:
  Promote. This passes the fixed-wall objective directly: lower held-out
  validation loss and higher completed step count under the same 900-second
  budget.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Skip final Aurora Polar FP16 CTA post-MMA barrier.
status: rejected_900s
change:
  Reused the FP16 CTA sync-before-next-K pattern inside the fused Aurora Polar
  Express compute loop so the final K tile skipped the post-MMA thread-block
  barrier. The Aurora iteration count and optimizer math were unchanged.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys screen:
    target/nsys/aurora_polar_sync_before_next_l4_b8_20_20260621T134356Z.run.log
    val_loss=8.506022, train_elapsed_s=3.470, completed_steps=20.
  100-step SYNTH screen:
    target/aurora_polar_sync_before_next_l4_b8_100_20260621T134424Z.log
    val_loss=6.544118, train_elapsed_s=17.384, completed_steps=100.
  900-second held-out gate:
    target/aurora_polar_sync_before_next_l4_b8_900_20260621T134454Z.log
    val_loss=4.009439, train_elapsed_s=900.150, completed_steps=5033.
measured_effect:
  Against the accepted direct-residual-gradient baseline
  target/direct_residual_grad_l4_b8_900_20260621T131534Z.log,
  completed steps increased from 5028 to 5033, but held-out validation loss
  worsened from 3.954098 to 4.009439, about +1.4%.
decision:
  Reject and revert the code. The extra completed steps do not compensate for
  validation loss moving outside the +/-1% no-meaningful-change band.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Use continuous scalar ranges in sweep candidate space.
status: accepted_tooling
change:
  Centralized sweep candidate-space ownership. Kept only build/kernel shape
  choices discrete: batch size, layer count, embedding/head shape, Aurora
  cooperative blocks, and Aurora phases. Changed scalar training knobs to
  range-sampled values instead of small buckets: LR scale, Adam LR scale,
  warmup steps, LR start ratio, AMUSE beta1, and AMUSE rho.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_continuous_candidate_space_20260621T174129Z
measured_effect:
  candidate_0000_ranked.tsv now contains non-anchor scalar values such as
  lr1.7457, alr0.5475, w21, s0.12, b0.46, r0.78 and guided candidates with
  values such as lr1.6826, w86, s0.17, b0.28, r0.59.
decision:
  Accept as sweep infrastructure. This fixes the bucketed scalar search space
  and preserves discrete choices only where compilation or cooperative launch
  constraints make them discrete.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add factorial probe candidates to sweep acquisition pool.
status: accepted_tooling
change:
  Added a factorial proposal source centered on the current best baseline
  candidate. The source ranks factor beliefs by high variance and low
  confidence, then emits low/high probes for the top uncertain factors. The
  acquisition pool now balances guided, factorial, variance, and random
  candidates before the model scores the ranked list.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_factorial_probe_pool_20260621T174549Z
measured_effect:
  candidate_0000_ranked.tsv contains balanced source coverage:
    factorial 8
    guided 8
    random 8
    variance 8
  The selected candidate source was factorial, proving the new source is
  persisted and participates in acquisition rather than only being generated.
decision:
  Accept as sweep infrastructure. This moves the sweep closer to the requested
  factorial-experiment behavior by using uncertainty in the fitted factors to
  place explicit low/high probes around the current best point.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Scale sweep prediction uncertainty by residual model error.
status: accepted_tooling
change:
  Prediction uncertainty now uses residual_std * sqrt(1 + leverage), where
  leverage is x^T covariance x. The previous acquisition uncertainty used only
  leverage, so it ignored the fitted model's residual error.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_residual_scaled_uncertainty_20260621T174749Z
measured_effect:
  candidate_0000_score.txt still selected a model-ranked factorial candidate.
  The selected candidate now reports uncertainty=1.134338, with per-response
  quality_uncertainty=0.631499, speed_uncertainty=1.134338, and
  stability_uncertainty=1.082089.
decision:
  Accept as sweep infrastructure. The acquisition uncertainty now reflects both
  model residual noise and candidate leverage, matching the intended confidence
  and variance-driven sweep behavior more closely.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Make sweep acquisition target fixed-wall validation loss with stability as a survival prior.
status: accepted_tooling
change:
  Candidate acquisition no longer adds speed and stability as separate
  objectives. The score is expected validation-loss quality plus exploration:
  predicted quality is multiplied by a survival prior derived from the
  stability model, and likely failed candidates receive the failure side of the
  expectation. Speed remains reported in candidate score/ranked artifacts but
  does not improve acquisition score by default. Promotion remains full-run
  held-out validation loss only.
  Removed the arbitrary top-4 factorial cutoff. The factorial source now emits
  fractional-factorial rows over every factor currently supported by the fitted
  belief table.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 15 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_val_loss_survival_prior_20260621T175837Z
measured_effect:
  candidate_0000_score.txt includes:
    expected_quality=1.253229
    survival_prior=1.000000
  candidate_0000_ranked.tsv includes expected_quality and survival_prior
  columns while keeping speed columns diagnostic. The ranked pool still has
  balanced source coverage:
    factorial 8
    guided 8
    random 8
    variance 8
decision:
  Accept as sweep infrastructure. This restores the single target:
  lowest held-out validation loss over the fixed wall-clock budget, with
  stability used as a Bayesian-style survival prior rather than a separate
  promotion objective.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Make the screen gate use the validation-loss survival prior.
status: accepted_tooling
change:
  Replaced the raw screen-loss-only gate with a screen decision module. NaN,
  incomplete, and missing-validation screens still reject immediately. A
  candidate now passes the screen when it improves screen validation loss, has
  no screen baseline, or the existing acquisition model says the candidate has
  positive expected validation quality with survival_prior >= 0.5. The gate
  writes screen_decision.env into the trial directory. Full-run baseline
  promotion is unchanged and still requires successful held-out validation loss.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 17 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_screen_gate_target_prior_20260621T180327Z
measured_effect:
  Added screen_gate unit coverage for:
    worse screen loss can pass when expected_quality is positive and
    survival_prior is high.
    worse screen loss still rejects when survival_prior is low.
  Dry-run proposal artifacts still include expected_quality and survival_prior,
  proving the acquisition values used by the screen gate are generated.
decision:
  Accept as sweep infrastructure. Screening now aims at the same target as the
  sweep: spending full 900-second runs on candidates likely to improve held-out
  validation loss, while using stability only as a survival prior.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Keep screen baseline synchronized after promotion.
status: accepted_tooling
change:
  The sweep loop now updates its in-memory screen baseline after a trial
  promotes the full-run baseline. It reads SCREEN_LOSS from the promoted trial's
  screen_decision.env and uses that value for later screen decisions. If the
  screen decision artifact is unavailable, it falls back to rerunning the
  current baseline screen path. This prevents later trials from being gated
  against a stale pre-promotion screen loss.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 18 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_promoted_screen_baseline_20260621T180611Z
measured_effect:
  Added direct runner unit coverage for reading SCREEN_LOSS from a promoted
  trial's screen_decision.env. The dry-run verifies the surrounding sweep
  proposal and artifact path still executes; dry-run intentionally does not
  execute screen/full training stages.
decision:
  Accept as sweep infrastructure. This removes a manual/stale-state failure
  mode from chained sweeps after an automatic baseline promotion.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Count failed real trials toward sweep random-phase progression.
status: accepted_tooling
change:
  failed_build and failed_run outcomes now count as observed sweep samples for
  deciding when to leave the initial random/fuzzing phase. They receive a large
  penalty only for phase accounting; promotion remains successful held-out
  validation loss only, and stability analysis still records the failure signal.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 19 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_failed_outcome_progress_20260621T180930Z
measured_effect:
  Added optimizer unit coverage proving failed_build and failed_run trials
  advance the sweep from random proposal mode into model proposal mode once
  random_trials is satisfied.
decision:
  Accept as sweep infrastructure. Failed builds/runs are real observations for
  the sweep's stability prior and should not trap an automatic sweep in
  endless random sampling.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Align sweep beliefs with validation-loss survival-prior target.
status: accepted_tooling
change:
  Factor beliefs now separate objective direction from survival-prior
  uncertainty. Validation-loss quality models are the only response models that
  can push guided proposal direction. Stability models can still contribute
  uncertainty and variance for survival-prior exploration, but they do not
  create a target direction by themselves. Speed models remain diagnostic and
  are not used by factor beliefs for this target.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 20 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_target_belief_alignment_20260621T181346Z
measured_effect:
  Added unit coverage proving stability-only observations produce no guided
  target direction while still contributing nonzero variance. The dry-run
  proposal artifacts still include expected_quality and survival_prior, so the
  full acquisition path keeps using stability as a survival prior.
decision:
  Accept as sweep infrastructure. This keeps the sweep target fixed on
  held-out validation loss while using stability as a prior for whether a
  candidate is worth spending a full validation run on.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add a base stability prior for sweep candidate scoring.
status: accepted_tooling
change:
  Sweep analysis now records a beta-smoothed binary stability prior from all
  non-dry-run trial outcomes. Candidate scoring uses that prior even when the
  stability regression cannot be fit because all observed outcomes are constant
  failures or constant successes. When a stability model is available, the
  predicted survival value is shrunk toward the base prior according to sample
  count and prediction uncertainty. analysis_summary.md now prints the
  stability prior sample count, positive count, and posterior mean.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 21 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_stability_base_prior_20260621T181741Z
measured_effect:
  Added unit coverage for the edge case where every observed candidate failed
  and no stability regression exists. In that case candidate scoring now gets a
  survival_prior below 0.5 instead of silently falling back to 1.0.
  The dry-run analysis summary reported:
    stability_prior_n=40 stability_prior_positive=36.000
    stability_prior_posterior_mean=0.880952.
decision:
  Accept as sweep infrastructure. This makes stability a real survival prior
  rather than only a fitted response model that disappears in constant-outcome
  histories.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Use balanced Hadamard levels for factorial sweep probes.
status: accepted_tooling
change:
  Replaced the ad hoc factorial high/low bit mixer with a Walsh-Hadamard
  two-level schedule. Factorial candidates now use balanced low/high settings
  across each factor and balanced pairwise cells over a Hadamard block, which
  better matches the factorial experiment structure used to probe main effects
  and interactions.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 23 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_hadamard_factorial_20260621T181911Z
measured_effect:
  Added unit coverage proving the first 16 factorial rows are balanced per
  factor and cover all two-level pairwise cells evenly for the tested factors.
  The dry-run selected a factorial candidate through the normal proposal path
  and emitted the standard analysis/proposal artifacts.
decision:
  Accept as sweep infrastructure. This removes arbitrary factorial scheduling
  and makes the variance-probing proposal source match the intended design.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Persist elapsed time for sweep speed analysis.
status: accepted_tooling
change:
  trials.tsv now appends an elapsed_s column while preserving backward
  compatibility with old rows. Runtime trials store the final elapsed time from
  the parsed training output, rejected screen trials store screen elapsed time,
  and promoted baselines write/read TRAIN_ELAPSED_S. Speed analysis still reads
  logs first, but it now falls back to persisted completed_steps and elapsed_s
  when logs are unavailable.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 27 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_persist_elapsed_speed_20260621T182405Z
measured_effect:
  Added unit coverage for heldout_eval train_elapsed_s parsing, new-row
  elapsed_s roundtrip, old-row parsing without elapsed_s, and full-speed rows
  reconstructed from persisted trial timing when no log can be read. The
  dry-run trials.tsv emitted the new elapsed_s header column.
decision:
  Accept as sweep infrastructure. The sweep can now keep learning speed
  correlations from shared history even when old run logs are not available.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Persist screen validation loss for sweep analysis.
status: accepted_tooling
change:
  trials.tsv now appends a screen_val_loss column while preserving compatibility
  with older rows. Rejected-screen trials store the screen-stage validation
  loss without overloading full-run val_loss, and successful full-run trials
  also retain the screen-stage loss that allowed them to continue. Screen
  quality analysis still reads screen.log first, but falls back to persisted
  screen_val_loss when logs are unavailable.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 28 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_persist_screen_loss_20260621T182712Z
measured_effect:
  Added unit coverage proving screen_quality_rows can reconstruct a screen
  response from persisted screen_val_loss with no screen.log. The dry-run
  trials.tsv emitted the new screen_val_loss header column.
decision:
  Accept as sweep infrastructure. Screen rejections now remain useful
  statistical samples for screen-quality modeling after logs are moved,
  cleaned, or unavailable.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Cache promoted baseline screen loss.
status: accepted_tooling
change:
  Promoted baseline env files now write/read SCREEN_LOSS from the trial's
  screen_val_loss. Baseline::measured_trial keeps that screen value, and the
  sweep runner uses the cached baseline screen loss before launching a baseline
  screen run. Full-run VAL_LOSS remains the only baseline promotion target.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 28 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_baseline_screen_cache_20260621T182952Z
measured_effect:
  Extended baseline promotion coverage to verify SCREEN_LOSS is written and
  reloads through measured_trial(). Dry-run sweep artifacts still emit through
  the normal proposal and analysis path.
decision:
  Accept as sweep infrastructure. Chained sweeps can now reuse known baseline
  screen loss instead of rerunning that baseline screen just to initialize the
  screen gate.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add design-coverage candidates to the sweep proposal pool.
status: accepted_tooling
change:
  Added a coverage proposal source that chooses candidates far from already
  observed trials in normalized factor space. The proposal pool now reserves
  slots for guided, factorial, variance, and coverage proposals, then fills the
  rest with random candidates. This gives the sweep an explicit mechanism to
  probe under-covered regions and reduce model-design uncertainty instead of
  relying only on random sampling.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 29 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_design_coverage_20260621T183410Z
measured_effect:
  Added unit coverage proving the coverage score prefers an uncovered region
  over a near-observed region. Existing proposal-pool coverage now asserts that
  guided, factorial, variance, coverage, and random sources are all present.
  The dry-run candidate_0000_ranked.tsv contained coverage rows in the normal
  proposal artifact.
decision:
  Accept as sweep infrastructure. The sweep now has a dedicated variance-
  reducing design-space exploration source in addition to model-guided and
  random proposals.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Persist screen timing for sweep prior analysis.
status: accepted_tooling
change:
  trials.tsv now appends screen_completed_steps and screen_elapsed_s columns
  while preserving older row formats. Rejected-screen and successful full-run
  trials both persist the screen-stage step count, elapsed time, and validation
  loss separately from the full 900-second run fields. Promoted baselines also
  write/read SCREEN_COMPLETED_STEPS and SCREEN_ELAPSED_S so baseline injection
  does not erase screen-stage evidence.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 30 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_persist_screen_timing_20260621T184027Z
measured_effect:
  Added unit coverage proving screen_speed_rows uses persisted screen-stage
  timing when screen.log is unavailable. The dry-run trials.tsv emitted
  screen_completed_steps and screen_elapsed_s header columns.
decision:
  Accept as sweep infrastructure. The 500-step screen gate can now inform
  survival and speed priors without confusing screen-stage timing with full
  900-second validation timing.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add expected-improvement acquisition to sweep scoring.
status: accepted_tooling
change:
  Regression models now retain the best observed response value and standardized
  best score. Candidate scoring computes probability_improvement and
  expected_improvement from the predicted mean, predictive uncertainty, and
  observed best response. The score still defaults to validation-quality
  acquisition; speed affects ranking only when sweep_speed_weight is explicitly
  nonzero. Proposal artifacts now print expected_speed,
  probability_improvement, and expected_improvement for auditability.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 32 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_expected_improvement_20260621T184532Z
measured_effect:
  Added unit coverage proving the quality scorer reports improvement
  acquisition against the current best observed quality, and that
  sweep_speed_weight changes ranking only when configured. The dry-run
  candidate_0000_ranked.tsv emitted probability_improvement and
  expected_improvement columns.
decision:
  Accept as sweep infrastructure. Candidate selection now combines the fitted
  response mean and predictive variance through an expected-improvement term,
  which is closer to the requested automatic Bayesian-style sweep loop than
  raw z-score ranking alone.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Wire speed objective into guided sweep beliefs.
status: accepted_tooling
change:
  factor_beliefs now includes tokens_per_s response models when
  sweep_speed_weight is nonzero. This keeps the default validation-loss
  objective unchanged, but makes speed-weighted sweeps move guided proposals
  along speed-correlated factors instead of only using speed during final
  ranking.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 33 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_speed_weighted_beliefs_20260621T184735Z
measured_effect:
  Added unit coverage proving a speed-weighted analysis gives batch_size a
  positive direction when larger batches produce more tokens/s. The dry-run
  analysis_beliefs.tsv showed speed-weighted direction terms, and the selected
  candidate score reported expected_speed.
decision:
  Accept as sweep infrastructure. Explicit speed objectives now affect both
  proposal direction and candidate scoring, while the default objective remains
  held-out validation-loss acquisition.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Use structured variance proposals over the sweep domain.
status: accepted_tooling
change:
  Added candidate_space::from_unit so proposal sources can map normalized
  factor coordinates into valid sweep candidates. The variance proposal source
  now evaluates a shifted Halton-style low-discrepancy design over the full
  factor domain, ranks those candidates by predictive uncertainty, and only
  falls back to random generation when needed. This replaces the old
  best-of-small-random-pool variance source.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 36 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_structured_variance_20260621T185113Z
measured_effect:
  Added unit coverage proving normalized coordinates map into valid candidate
  bounds, Halton units cover each factor range, and variance proposals return
  unique structured points. The dry-run candidate_0000_ranked.tsv included
  variance-source candidates selected from the structured scan.
decision:
  Accept as sweep infrastructure. Variance minimization now probes the actual
  multivariable design space deliberately instead of depending on a small
  random candidate pool.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Add adaptive proposal source budgeting.
status: accepted_tooling
change:
  The proposal pool no longer reserves a fixed one-fifth split for guided,
  factorial, variance, coverage, and random candidates. It now computes source
  weights from fitted-model availability, trial maturity, factor confidence,
  and factor variance, then normalizes those weights into candidate counts.
  Immature or uncertain analysis spends more candidates on design probes;
  mature/confident analysis spends more on guided acquisition. Random
  exploration remains an explicit budget share.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 38 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_adaptive_source_budget_20260621T185517Z
measured_effect:
  Added unit coverage proving no response model gets no guided budget and that
  a mature fitted model increases guided allocation. The dry-run ranked
  proposal artifact showed an adaptive source mix for the current history:
  guided=13, variance=6, coverage=5, factorial=5, random=3.
decision:
  Accept as sweep infrastructure. Candidate generation now reacts to confidence
  and variance instead of using a fixed source split.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Emit proposal source summary artifacts.
status: accepted_tooling
change:
  Each proposal now writes candidate_NNNN_sources.tsv next to the existing
  score and ranked candidate artifacts. The source summary records per-source
  candidate counts, whether that source produced the selected candidate, the
  best rank for the source, and the best source-level score/acquisition fields.
  This makes the adaptive source budget auditable without manually counting
  rows in candidate_NNNN_ranked.tsv.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 39 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_source_summary_20260621T185743Z
measured_effect:
  Added unit coverage proving source summaries count ranked candidate sources.
  The dry-run emitted candidate_0000_sources.tsv with the adaptive source mix
  and the best candidate per source.
decision:
  Accept as sweep infrastructure. Proposal allocation and acquisition choices
  are now directly recorded as sweep artifacts.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Preserve screen rejection reasons for stability modeling.
status: accepted_tooling
change:
  trials.tsv now appends screen_reason while preserving older row formats.
  The runner stores the screen gate reason on both rejected-screen and
  full-run trials, and promoted baseline env files write/read SCREEN_REASON.
  Stability rows now treat screen reasons nan, incomplete, and missing_val_loss
  as stability failures while keeping screen_loss_worse as a survived
  quality-screen rejection.
verification:
  cargo fmt --all --check: pass.
  cargo test --bin sweep: pass, 41 tests.
  cargo check --all-targets: pass.
  dry-run sweep:
    target/sweeps/dryrun_screen_reason_20260621T190245Z
measured_effect:
  Added unit coverage proving incomplete screen rejection maps to stability=0
  and screen_loss_worse maps to stability=1. The dry-run trials.tsv emitted the
  new screen_reason header column.
decision:
  Accept as sweep infrastructure. The stability/survival target now preserves
  useful screen-gate failure information instead of flattening all screen
  rejections into one status.
```

```text
date: 2026-06-21
commit: uncommitted
experiment: Stage two K64 NVFP4 projection CTA atoms per shared-memory load.
status: accepted
change:
  Increased the shared-memory projection CTA K stage from 64 to 128 and issued
  two m16n8k64 NVFP4 MMA atoms per stage. This reduces loop/sync overhead for
  aligned GPT projection shapes while keeping the existing one-warp projection
  path as the fallback for K64 or otherwise unaligned shapes.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  100-step SYNTH sanity:
    target/k128_projection_cta_sanity_100_20260621T210354Z.log
    val_loss=6.406075, completed_steps=100, train_elapsed_s=28.584.
  EOS-only 900-second comparison baseline:
    target/eos_baseline_900_20260621T204225Z.log
    val_loss=3.672471, completed_steps=3025, train_elapsed_s=900.296.
  K128 projection CTA 900-second gate:
    target/k128_projection_cta_900_20260621T210437Z.log
    val_loss=3.663287, completed_steps=3071, train_elapsed_s=900.222.
measured_effect:
  Compared to the EOS-only baseline, the K128 projection CTA patch completed
  46 more steps in the fixed 900-second budget and improved held-out validation
  loss by 0.009184. Logged backward enqueue wall timing moved from roughly
  70.4ms at steady state to roughly 69.5ms. Timing fields are host wall timing,
  not isolated GPU kernel time.
decision:
  Promote. The change improves held-out validation loss and completed steps in
  the required 900-second gate while preserving correctness tests and the K64
  fallback path.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before 100-step screen
experiment: Column-major projection CTA tile scheduling for weight-tile reuse.
status: rejected_screen
source:
  Colfax's GEMM scheduling guidance describes threadblock rasterization and
  tile scheduling as a way to improve cache behavior. This candidate tested
  the analogous projection-CTA work order by walking row tiles first for each
  output-column tile, aiming to reuse the larger NVFP4 weight tile across
  neighboring row tiles.
change:
  Temporarily changed generic projection CTA launches to use a transposed
  grid mapping and changed the paired linear-backward projection kernel's 1D
  tile stream to column-major tile order. The per-CTA MMA, staging, and store
  math were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys baseline:
    target/nsys/current_k128_b16_20_20260621T212738Z.run.log
    val_loss=9.213204, train_elapsed_s=5.728, completed_steps=20.
  20-step nsys candidate:
    target/nsys/projection_col_major_b16_20_20260621T213114Z.run.log
    val_loss=9.214709, train_elapsed_s=5.769, completed_steps=20.
measured_effect:
  The largest target regressed. Over 20 profiled steps,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  1217.244ms to 1257.752ms. lm_head_kernel improved from 222.933ms to
  222.015ms, but this was too small to offset the linear-backward regression.
  Overall profiled train time moved from 5.728s to 5.769s.
decision:
  Reject and revert before the 100-step and 900-second gates. The cache-order
  hypothesis helped LM head slightly but made the dominant projection kernel
  slower.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before 100-step screen
experiment: Power-of-two index decode fast path in attention_prob_ds_kernel.
status: rejected_screen
source:
  Colfax FlashAttention material emphasizes reducing attention overhead outside
  the MMA work, including scheduling and softmax-related scalar work. This
  candidate targeted a local scalar/indexing cost in the backward probability
  and dS kernel without changing attention math.
change:
  Temporarily added a seq_len=1024 fast path that decoded score indices with
  shifts and masks instead of runtime division/modulo, and cached the
  log-sum-exp index once per element.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys baseline:
    target/nsys/current_k128_b16_20_20260621T212738Z.run.log
    val_loss=9.213204, train_elapsed_s=5.728, completed_steps=20.
  20-step nsys candidate:
    target/nsys/attention_prob_decode_fast_b16_20_20260621T213559Z.run.log
    val_loss=9.213204, train_elapsed_s=5.723, completed_steps=20.
measured_effect:
  The target kernel did not improve. attention_prob_ds_kernel moved from
  183.381ms to 183.389ms over 20 profiled steps. The small total wall-clock
  movement was noise from unrelated kernels.
decision:
  Reject and revert before the 100-step and 900-second gates. The runtime cost
  is dominated by probability/gradient math and memory traffic, not the score
  index decode.
```

```text
date: 2026-06-21
commit: uncommitted candidate, reverted before 100-step screen
experiment: N=128 projection CTA tile after K=128 staging.
status: rejected_screen
source:
  Colfax's SM12x NVFP4 tutorial uses a 128x128x128 CTA tile for blockscaled
  GEMM. The older local N=128 rejection predated the accepted K=128 staging
  change, so this candidate retested the wider N tile under the current
  two-K-atom shared-memory stage.
change:
  Temporarily changed the generic NVFP4 projection CTA from M32/N64/K128 with
  512 threads to M32/N128/K128 with 1024 threads. The MMA atom, K staging, and
  math were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  ptxas verbose for affected kernels: 0 spills.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys baseline:
    target/nsys/current_goal_b16_20_20260621T214016Z.run.log
    val_loss=9.213204, train_elapsed_s=5.720, completed_steps=20.
  20-step nsys candidate:
    target/nsys/projection_n128_k128_b16_20_20260621T214521Z.run.log
    val_loss=9.213204, train_elapsed_s=5.898, completed_steps=20.
measured_effect:
  The wider tile regressed the dominant projection path. Over 20 profiled
  steps, linear_backward_projection_pair_cta_device_scale_kernel moved from
  1214.207ms to 1351.379ms. lm_head_kernel moved from 222.632ms to 233.517ms,
  mlp_projection_kernel from 121.813ms to 133.809ms,
  mlp_projection_relu2_kernel from 122.478ms to 132.851ms, and
  attention_projection_kernel from 115.695ms to 129.237ms. Aurora was flat.
decision:
  Reject and revert before the 100-step and 900-second gates. In this kernel,
  the extra warps/thread count and wider N tile reduce effective throughput
  despite matching the Colfax tutorial CTA extent in N.
```

```text
date: 2026-06-21
commit: uncommitted candidate
experiment: Skip invariant causal-mask zero stores in attention probability kernels.
status: accepted
source:
  Colfax FlashAttention-4 notes that Blackwell attention can be limited by
  softmax/scalar work and memory movement outside the GEMMs, and highlights
  scheduling and causal-mask overhead as optimization targets.
change:
  In the forward causal softmax, store probabilities only for key <= query
  instead of rewriting the upper-triangle zeros. In backward probability/dS
  generation, return early for masked entries and cache the row log-sum-exp
  index once. The scratch probability and dS buffers are zero-initialized and
  the masked upper triangle is invariant, so the math consumed by later dense
  matmuls is unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test causal_attention_backward_tc -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys baseline:
    target/nsys/current_goal_b16_20_20260621T214016Z.run.log
    val_loss=9.213204, train_elapsed_s=5.720, completed_steps=20.
  20-step nsys candidate:
    target/nsys/causal_mask_skip_stores_b16_20_20260621T215208Z.run.log
    val_loss=9.213204, train_elapsed_s=5.649, completed_steps=20.
  100-step SYNTH screen:
    target/causal_mask_skip_stores_b16_100_20260621T215234Z.log
    val_loss=6.407254, train_elapsed_s=28.197, completed_steps=100.
  900-second held-out gate:
    target/causal_mask_skip_stores_b16_900_20260621T215315Z.log
    val_loss=3.644228, train_elapsed_s=900.002, completed_steps=3116.
measured_effect:
  Over 20 profiled steps, attention_prob_ds_kernel moved from 183.346ms to
  125.637ms, and attention_softmax_forward_kernel moved from 91.454ms to
  64.040ms. The dominant linear_backward_projection_pair_cta_device_scale_kernel
  was effectively flat at 1214.207ms to 1219.402ms, and Aurora was flat at
  1593.264ms to 1594.792ms. The 900-second gate improved validation loss from
  3.663287 to 3.644228 while completed steps increased from 3071 to 3116.
decision:
  Promote. The change improved held-out validation loss and completed more
  steps under the fixed 900-second budget.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Increase cross-entropy row worker count on the current NextLat
  SYNTH path.
status: accepted_current_nextlat
change:
  Increased CROSS_ENTROPY_THREADS_PER_BLOCK from 256 to 1024. This keeps the
  same dense softmax/loss/dlogits math and only changes the per-row parallel
  scan width for the 32k-vocab loss kernel.
rejected_side_checks:
  Zero-sized Aurora padding slots were correct but negligible:
    target/nsys/nextlat_synth_zero_padding_20_20260622T000633Z.run.log
    aurora_mega_update_cooperative_kernel moved from 102.041ms/step to
    101.970ms/step.
  AURORA_MATRIX_PHASES=7 removed the all-dummy phase but did not improve:
    target/nsys/nextlat_synth_phase7_20_20260622T000803Z.run.log
    aurora_mega_update_cooperative_kernel stayed at 102.012ms/step.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  20-step nsys baseline:
    target/nsys/nextlat_synth_current_20_20260622T000101Z.run.log
    val_loss=9.154721, train_elapsed_s=6.449, completed_steps=20.
  20-step nsys CE-512:
    target/nsys/nextlat_synth_ce512_20_20260622T001257Z.run.log
    cross_entropy_kernel moved from 5.458ms/step to 4.343ms/step.
  20-step nsys CE-1024:
    target/nsys/nextlat_synth_ce1024_20_20260622T001333Z.run.log
    val_loss=9.155092, train_elapsed_s=6.412, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_ce1024_synth_100_20260622T001356Z.log
    val_loss=6.531735, train_elapsed_s=32.061, completed_steps=100.
  900-second held-out gate:
    target/nextlat_ce1024_synth_900_20260622T001447Z.log
    val_loss=3.811654, train_elapsed_s=900.324, completed_steps=2745.
measured_effect:
  Cross entropy moved from 109.152ms over 20 profiled steps to 58.188ms,
  or 5.458ms/step to 2.771ms/step. Against the current NextLat SYNTH 900s
  baseline target/nextlat_synth_default_900s_20260621T234043Z.log, held-out
  validation loss moved from 3.816722 to 3.811654 and completed steps moved
  from 2726 to 2745.
decision:
  Promote for the current NextLat branch. Held-out validation loss improved and
  completed step count increased under the same fixed 900-second SYNTH budget.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Remove unused saved residual_out tape copy.
status: measured_not_promoted
change:
  Removed the forward-tape residual_out field and the D2D copy at the end of
  each transformer block. The saved forward tensor was not read by backward;
  the similarly named d_residual_out gradient buffer remains in use.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_synth_no_residual_out_tape_20_20260622T003939Z.run.log
    val_loss=9.155092, train_elapsed_s=6.404, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_no_residual_out_tape_synth_100_20260622T004018Z.log
    val_loss=6.529115, train_elapsed_s=32.047, completed_steps=100.
  900-second held-out gate:
    target/nextlat_no_residual_out_tape_synth_900_20260622T004101Z.log
    val_loss=3.815764, train_elapsed_s=900.281, completed_steps=2745.
measured_effect:
  D2D copies in the 20-step profile moved from 2579 copies / 147.761ms to
  2495 copies / 140.330ms. 64MiB copies moved from 735 to 651, matching one
  removed hidden-state tape copy per block. The 900-second gate stayed inside
  the +/-1% noise band versus the current CE1024 baseline, but did not gain
  completed steps.
decision:
  Do not promote on its own. The cleanup is correct and reduces measured D2D
  traffic, but it did not improve the fixed-wall validation target or completed
  step count at the 900-second endpoint.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Minimize unused forward-tape FP32 saves on the current NextLat
  SYNTH path.
status: accepted_current_nextlat
change:
  Removed unused saved forward tensors from the training tape:
    - block residual_out from the prior cleanup,
    - block residual_in,
    - embedding_residual,
    - saved layer-norm normalized outputs.
  The remaining layer-norm tape keeps residual, mean, and inv_std, which are
  the values consumed by layer-norm backward. Gradient buffers such as
  d_residual_out and d_normalized are unchanged.
verification:
  cargo fmt --all: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_synth_tape_min_20_20260622T010327Z.run.log
    val_loss=9.155092, train_elapsed_s=6.380, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_tape_min_synth_100_20260622T010353Z.log
    val_loss=6.528718, train_elapsed_s=31.946, completed_steps=100.
  900-second held-out gate:
    target/nextlat_tape_min_synth_900_20260622T010439Z.log
    val_loss=3.810689, train_elapsed_s=900.294, completed_steps=2757.
measured_effect:
  Against the CE1024 current NextLat baseline
  target/nextlat_ce1024_synth_900_20260622T001447Z.log, held-out validation
  loss moved from 3.811654 to 3.810689 and completed steps moved from 2745 to
  2757. In the 20-step nsys profile, D2D copies moved from 2579 copies /
  147.761ms to 2201 copies / 114.530ms.
decision:
  Promote for the current NextLat branch. Held-out validation loss improved and
  completed step count increased under the same fixed 900-second SYNTH budget.
  This is an accepted active-NextLat baseline candidate; pre-NextLat validation
  results are not protected baselines for this branch.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: NextLat LR sweep with widened batch candidates through B32.
status: measured_no_promotion
change:
  Added batch candidates 12, 20, 24, 28, and 32 to the sweep search space while
  keeping the current NextLat SYNTH objective and 500-step/180-second screen.
result:
  Current promoted baseline remains trial_0024:
    val_loss=3.656983, completed_steps=2785, screen_loss=5.057242,
    B16/L4/d1024/h16,
    log=target/sweeps/nextlat_lr_20260622T035322Z/trial_0024/train.log.
  Newly measured widened-batch screens:
    trial_0029 B12/L4/d1024: screen_loss=5.285259, completed_steps=500.
    trial_0031 B20/L4/d1024: screen_loss=5.639343, completed_steps=469.
    trial_0033 B32/L4/d1024: screen_loss=5.258968, completed_steps=331.
    trial_0030 B32/L8/d2048: rejected before steps with
      DriverError(2, "out of memory").
decision:
  Do not promote any widened-batch candidate from these measurements. The B32
  domain is enabled and can remain in the sweep, but the current evidence still
  favors the B16/L4/d1024 baseline for fixed-wall validation. The next useful
  work is profiling/optimizing the promoted B16 shape, not continuing blind
  high-batch screens.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: NCU SOL/register profile for promoted NextLat B16/L4/d1024 shape.
status: measured
profile:
  target/ncu/promoted_b16_top_kernels_20260622T060914Z.txt
result:
  linear_backward_projection_pair_cta_device_scale_kernel:
    no local/shared memory spilling reported.
    registers/thread=40, theoretical occupancy=100%.
    The largest profiled instance was grid=(24192,1,1)x(512,1,1),
    duration=22.33ms, memory throughput=88.04%, compute throughput=53.74%,
    achieved occupancy=99.51%, L2 hit rate=99.39%.
    This points at memory/L2 traffic and layout/reuse, not register spilling.
  aurora_mega_update_cooperative_kernel:
    no local/shared memory spilling reported.
    launch=(180,3,1)x(256,1,1), duration=107.51ms.
    registers/thread=80, theoretical occupancy=50%, achieved occupancy=47.87%,
    memory throughput=63.29%, compute throughput=27.79%, L2 hit rate=99.10%.
    NCU reports occupancy limited by registers.
decision:
  Use this as the next optimization guide. Do not chase register spills; none
  were measured. Linear backward needs memory/layout/reuse work. Aurora needs
  either lower register pressure or a differently staged update path.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted after gate
experiment: Replace projection CTA staging loops with explicit per-thread slots.
status: rejected_900s
change:
  Replaced the constant-bounded shared-memory staging loops in
  projection_cta/stage.rs with explicit per-thread A-pack, B-pack, A-scale, and
  B-scale stores. The tile shape, shared-memory layout, native NVFP4 MMA path,
  and math were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/projection_stage_explicit_b16_l4d1024_20_20260622T062219Z.run.log
    val_loss=9.069621, train_elapsed_s=6.341, completed_steps=20.
  100-step SYNTH screen:
    target/projection_stage_explicit_b16_l4d1024_100_20260622T062250Z.log
    val_loss=6.348930, train_elapsed_s=31.754, completed_steps=100.
  900-second held-out gate:
    target/projection_stage_explicit_b16_l4d1024_900_20260622T062341Z.log
    val_loss=3.658568, train_elapsed_s=900.036, completed_steps=2773.
measured_effect:
  Short-profile target kernel improved:
    linear_backward_projection_pair_cta_device_scale_kernel moved from
    1.407983884s to 1.384849441s over 20 profiled steps.
  The fixed-wall objective did not improve:
    promoted baseline trial_0024 has val_loss=3.656983 and completed_steps=2785.
    this candidate had val_loss=3.658568 and completed_steps=2773.
decision:
  Reject and revert. The short nsys improvement did not translate into the
  900-second held-out objective, and completed step count dropped.
```

```text
date: 2026-06-22
commit: uncommitted candidate
experiment: Reuse B operand staging across adjacent dW projection CTA row tiles.
status: accepted_900s
change:
  The paired linear-backward CTA kernel now maps the dW half to row-pair CTAs.
  For each K chunk it stages the B-side NVFP4 operand/scales once, stages and
  multiplies the first A row tile, then stages and multiplies the adjacent A row
  tile before advancing K. The dX half and training math are unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/projection_dweight_rowpair_b16_l4d1024_20_20260622T064813Z.run.log
    val_loss=9.069621, train_elapsed_s=6.325, completed_steps=20.
  100-step SYNTH screen:
    target/projection_dweight_rowpair_b16_l4d1024_100_20260622T064845Z.log
    val_loss=6.341741, train_elapsed_s=31.657, completed_steps=100.
  900-second held-out gate:
    target/projection_dweight_rowpair_b16_l4d1024_900_20260622T064932Z.log
    val_loss=3.642029, train_elapsed_s=900.306, completed_steps=2779.
measured_effect:
  Against promoted baseline
  target/nsys/nextlat_promoted_b16_l4d1024_20_20260622T060358Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  1.407983884s to 1.360383029s over 20 profiled steps. The full 20-step
  training profile moved from 6.368s to 6.325s.
  Against baseline trial_0024, held-out validation loss improved from
  3.656983 to 3.642029 under the fixed 900-second SYNTH budget.
decision:
  Promote. This directly improves the fixed-wall held-out objective; lower
  validation loss wins even though completed steps moved from 2785 to 2779.
```

```text
date: 2026-06-22
commit: uncommitted candidate
experiment: Reuse B operand staging across adjacent dX and dW projection CTA row tiles.
status: accepted_900s
change:
  Extended the row-pair CTA mapping from only the dW half of the paired
  linear-backward projection kernel to both halves. The dX half now also stages
  the B-side NVFP4 operand/scales once per K chunk and computes two adjacent row
  tiles before advancing K. The GEMM math, Quartet/MS-EDEN operands, and output
  layout are unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/projection_both_rowpair_b16_l4d1024_20_20260622T071103Z.run.log
    val_loss=9.069621, train_elapsed_s=6.277, completed_steps=20.
  100-step SYNTH screen:
    target/projection_both_rowpair_b16_l4d1024_100_20260622T071131Z.log
    val_loss=6.344444, train_elapsed_s=31.415, completed_steps=100.
  900-second held-out gate:
    target/projection_both_rowpair_b16_l4d1024_900_20260622T071216Z.log
    val_loss=3.664893, train_elapsed_s=900.011, completed_steps=2803.
measured_effect:
  Against the accepted dW-only row-pair profile
  target/nsys/projection_dweight_rowpair_b16_l4d1024_20_20260622T064813Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  1.360383029s to 1.311057025s over 20 profiled steps. The full 20-step
  training profile moved from 6.325s to 6.277s.
  Against the accepted dW-only 900-second baseline, held-out validation moved
  from 3.642029 to 3.664893, a +0.628% change, while completed steps increased
  from 2779 to 2803.
decision:
  Promote under the active noise-band rule. Validation loss stayed within the
  +/-1% band and completed step count increased.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before screen
experiment: Four-row CTA reuse for paired linear-backward projection.
status: rejected_pre_gate
change:
  Tested grouping four adjacent row tiles per CTA for both dX and dW halves of
  linear_backward_projection_pair_cta_device_scale_kernel. The candidate staged
  the B-side NVFP4 operand/scales once and then computed four A row tiles before
  advancing K. Math and output layout were unchanged.
verification:
  cargo fmt --all --check: pass before profiling.
  cargo check --all-targets: pass before profiling.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/projection_rowquad_b16_l4d1024_20_20260622T073307Z.run.log
    val_loss=9.069621, train_elapsed_s=6.667, completed_steps=20.
measured_effect:
  Against the accepted both-row-pair profile
  target/nsys/projection_both_rowpair_b16_l4d1024_20_20260622T071103Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel regressed from
  1.311057025s to 1.717597668s over 20 profiled steps. Full 20-step training
  time regressed from 6.277s to 6.667s.
decision:
  Reject before 100-step and 900-second gates. Code was reverted to the accepted
  both-row-pair projection path.
```

```text
date: 2026-06-22
commit: uncommitted candidate
experiment: Reuse B operand staging across adjacent LM-head row tiles.
status: accepted_900s
change:
  The aligned LM-head CTA path now launches with row-pair grid scheduling and
  calls the existing aligned no-bias row-pair projection body. Each CTA stages
  the vocab-side B operand/scales once per K chunk, computes one token-row tile,
  then computes the adjacent token-row tile before advancing K. The generic
  non-aligned LM-head path is unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/lm_head_rowpair_b16_l4d1024_20_20260622T074337Z.run.log
    val_loss=9.069621, train_elapsed_s=6.142, completed_steps=20.
  100-step SYNTH screen:
    target/lm_head_rowpair_b16_l4d1024_100_20260622T074404Z.log
    val_loss=6.342079, train_elapsed_s=31.334, completed_steps=100.
  900-second held-out gate:
    target/lm_head_rowpair_b16_l4d1024_900_20260622T074454Z.log
    val_loss=3.626539, train_elapsed_s=900.213, completed_steps=2808.
measured_effect:
  Against the accepted both-row-pair projection profile
  target/nsys/projection_both_rowpair_b16_l4d1024_20_20260622T071103Z.run.log,
  lm_head_kernel moved from 231.831103ms to 213.578780ms over 21 profiled
  calls. The 20-step profiled training time moved from 6.277s to 6.142s.
  Against the previous 900-second baseline, held-out validation improved from
  3.664893 to 3.626539 and completed steps increased from 2803 to 2808.
decision:
  Promote. The change directly improves the fixed-wall held-out objective with
  lower validation loss and higher completed step count.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before screen
experiment: Reuse B operand staging across adjacent forward projection row tiles.
status: rejected_pre_gate
change:
  Temporarily added aligned row-pair projection bodies for affine and relu2 CTA
  projections, then routed attention QKV/c_proj, MLP up/down, and NextLat
  projection launches through row-pair grid scheduling when shapes were CTA
  aligned. Math and output layouts were unchanged.
verification:
  cargo fmt --all --check: pass after revert.
  cargo check --all-targets: pass after revert.
  cargo oxide build --arch sm_120a: pass before profiling.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test next_latent_concat_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/forward_projection_rowpair_b16_l4d1024_20_20260622T080918Z.run.log
    val_loss=9.069621, train_elapsed_s=6.170, completed_steps=20.
measured_effect:
  Against the accepted LM-head row-pair profile
  target/nsys/lm_head_rowpair_b16_l4d1024_20_20260622T074337Z.run.log,
  the full 20-step profiled training time regressed from 6.142s to 6.170s.
  attention_projection_kernel moved from 114.244753ms to 139.069133ms,
  mlp_projection_kernel moved from 120.210061ms to 129.817442ms, and
  mlp_projection_relu2_kernel moved from 120.764964ms to 123.191866ms.
  nextlat_projection_kernel improved from 76.575046ms to 69.161765ms, but this
  was too small to offset the broader forward projection regressions.
decision:
  Reject before 100-step and 900-second gates. Code was reverted to the accepted
  LM-head row-pair baseline.
```

```text
date: 2026-06-22
commit: uncommitted candidate
experiment: Reuse B operand staging across adjacent NextLat projection row tiles.
status: accepted_900s
change:
  The aligned NextLat projection CTA path now launches with row-pair grid
  scheduling and uses an affine row-pair projection body. The path stages the B
  operand/scales once per K chunk, computes one token-row tile, then computes
  the adjacent token-row tile before advancing K. Attention and MLP projection
  paths remain on the prior accepted scheduling after the broader row-pair
  experiment regressed.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test next_latent_concat_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/nextlat_projection_rowpair_b16_l4d1024_20_20260622T081805Z.run.log
    val_loss=9.069621, train_elapsed_s=6.135, completed_steps=20.
  100-step SYNTH screen:
    target/nextlat_projection_rowpair_b16_l4d1024_100_20260622T081823Z.log
    val_loss=6.346216, train_elapsed_s=31.298, completed_steps=100.
  900-second held-out gate:
    target/nextlat_projection_rowpair_b16_l4d1024_900_20260622T081906Z.log
    val_loss=3.628989, train_elapsed_s=900.120, completed_steps=2811.
measured_effect:
  Against the accepted LM-head row-pair profile
  target/nsys/lm_head_rowpair_b16_l4d1024_20_20260622T074337Z.run.log,
  nextlat_projection_kernel moved from 76.575046ms to 69.238443ms over 20
  profiled steps. The full 20-step profiled training time moved from 6.142s to
  6.135s.
  Against the previous 900-second baseline, held-out validation moved from
  3.626539 to 3.628989, a +0.0675% change inside the active +/-1% noise band,
  while completed steps increased from 2808 to 2811.
decision:
  Promote under the runtime-change acceptance rule. Validation loss stayed well
  inside the active noise band and completed step count increased.
```

```text
date: 2026-06-22
commit: temporary build-only candidate, reverted before screen
experiment: Increase Aurora cooperative block count from 180 to 188.
status: rejected_pre_gate
change:
  Built the current B16/L4/d1024 baseline with AURORA_COOPERATIVE_BLOCKS=188
  instead of the promoted 180. The hypothesis was that 188 blocks across the X
  dimension and matrix_count=3 would exactly fill the observed register-limited
  cooperative-launch ceiling of roughly 3 blocks per 188 SMs. Optimizer math and
  training settings were unchanged.
verification:
  AURORA_COOPERATIVE_BLOCKS=188 cargo oxide build --arch sm_120a: pass.
  20-step nsys:
    target/nsys/aurora_blocks188_b16_l4d1024_20_20260622T084021Z.run.log
    val_loss=9.067668, train_elapsed_s=6.182, completed_steps=20.
measured_effect:
  Against the accepted NextLat row-pair profile
  target/nsys/nextlat_projection_rowpair_b16_l4d1024_20_20260622T081805Z.run.log,
  the full 20-step profiled training time regressed from 6.135s to 6.182s.
  aurora_mega_update_cooperative_kernel moved from 2.022918141s to
  2.078683438s over 20 profiled calls.
decision:
  Reject before 100-step and 900-second gates. Filling the apparent cooperative
  block ceiling did not improve the runtime path, so keep AURORA_COOPERATIVE_BLOCKS=180.
```

```text
date: 2026-06-22
commit: uncommitted candidate
experiment: Fuse post-RoPE QKV f16 tape save into RoPE kernel.
status: accepted_900s
change:
  Added an apply_rope_save_f16_kernel training path that rotates Q/K in place
  and writes the post-RoPE Q/K/V f16 tape values while the QKV values are live.
  Removed the standalone block-level save_qkv_f16 conversion call. The no-tape
  inference path still uses the original apply_rope_kernel.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/rope_qkv_f16_fused_b16_l4d1024_20_20260622T084442Z.run.log
    val_loss=9.069621, train_elapsed_s=6.122, completed_steps=20.
  100-step SYNTH screen:
    target/rope_qkv_f16_fused_b16_l4d1024_100_20260622T084500Z.log
    val_loss=6.345064, train_elapsed_s=31.239, completed_steps=100.
  900-second held-out gate:
    target/rope_qkv_f16_fused_b16_l4d1024_900_20260622T084543Z.log
    val_loss=3.642559, train_elapsed_s=900.292, completed_steps=2819.
measured_effect:
  Against the accepted NextLat row-pair profile
  target/nsys/nextlat_projection_rowpair_b16_l4d1024_20_20260622T081805Z.run.log,
  fp32_to_f16_kernel moved from 51.755272ms over 441 calls to 36.069541ms over
  357 calls. The original apply_rope_kernel entry was replaced by
  apply_rope_save_f16_kernel, which took 22.568906ms over 84 calls. Full
  20-step profiled training time moved from 6.135s to 6.122s.
  Against the previous 900-second baseline, held-out validation moved from
  3.628989 to 3.642559, a +0.3740% change inside the active +/-1% noise band,
  while completed steps increased from 2811 to 2819.
decision:
  Promote under the runtime-change acceptance rule. Validation loss stayed
  inside the active noise band and completed step count increased.
```

```text
date: 2026-06-22
commit: rejected uncommitted candidate, code reverted
experiment: Fuse MLP up pre-activation f16 tape save into the MLP up projection store.
status: rejected_900s
change:
  Tested a training-only mlp_projection_relu2_save_f16_kernel that wrote the
  f16 MLP pre-activation tape value while the FP32 pre-activation was live in
  the MLP up projection store. This removed the separate save_mlp_up_f16
  fp32_to_f16 launch for that tape buffer.
verification:
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test l3_mlp -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/mlp_up_f16_fused_b16_l4d1024_20_20260622T090923Z.run.log
    val_loss=9.069621, train_elapsed_s=6.121, completed_steps=20.
  100-step SYNTH screen:
    target/mlp_up_f16_fused_b16_l4d1024_100_20260622T090957Z.log
    val_loss=6.340821, train_elapsed_s=31.200, completed_steps=100.
  900-second held-out gate:
    target/mlp_up_f16_fused_b16_l4d1024_900_20260622T091045Z.log
    val_loss=3.652935, train_elapsed_s=900.243, completed_steps=2819.
measured_effect:
  Against the accepted RoPE/QKV f16 fusion profile
  target/nsys/rope_qkv_f16_fused_b16_l4d1024_20_20260622T084442Z.run.log,
  fp32_to_f16_kernel moved from 36.069541ms over 357 calls to 13.408325ms over
  273 calls, but the fused MLP up projection took 134.617849ms versus the prior
  mlp_projection_relu2_kernel at 120.749054ms. The full 20-step profiled train
  time was effectively flat, 6.122s to 6.121s.
decision:
  Reject and revert code. The 900-second gate stayed inside the active +/-1%
  noise band but did not increase completed steps, so it does not satisfy the
  runtime-change acceptance rule.
```

```text
date: 2026-06-22
commit: rejected uncommitted candidate, code reverted
experiment: Stage both A row tiles before MMA in paired linear-backward CTA projection.
status: rejected_900s
change:
  Tested a dual-A shared-memory path for the existing two-row paired
  linear_backward_projection_pair_cta_device_scale_kernel. The candidate kept
  B operand reuse and the same math, but staged A0 and A1 into separate shared
  buffers before the MMA phase to remove the middle A-restage barrier.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture: pass.
  20-step nsys:
    target/nsys/linear_dual_a_rowpair_b16_l4d1024_20_20260622T093239Z.run.log
    val_loss=9.069621, train_elapsed_s=6.121, completed_steps=20.
  100-step SYNTH screen:
    target/linear_dual_a_rowpair_b16_l4d1024_100_20260622T093303Z.log
    val_loss=6.342345, train_elapsed_s=31.236, completed_steps=100.
  900-second held-out gate:
    target/linear_dual_a_rowpair_b16_l4d1024_900_20260622T093344Z.log
    val_loss=3.649023, train_elapsed_s=900.032, completed_steps=2819.
measured_effect:
  Against the accepted RoPE/QKV f16 fusion profile
  target/nsys/rope_qkv_f16_fused_b16_l4d1024_20_20260622T084442Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  1.265443216s to 1.255295497s over 20 profiled steps. The full 20-step
  profiled train time stayed effectively flat at 6.121s versus 6.122s.
decision:
  Reject and revert code. The 900-second gate stayed inside the active +/-1%
  noise band but did not increase completed steps, so it does not satisfy the
  runtime-change acceptance rule.
```
```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Widen projection CTA tile from N64 to N128.
status: rejected_screen
change:
  Temporarily changed the shared NVFP4 projection CTA geometry from
  M=32/N=64/K=128 with 512 threads to M=32/N=128/K=128 with 1024 threads.
  The intent was to reuse each staged A tile across twice as many output
  columns and reduce the L2/cache pressure reported by NCU for
  linear_backward_projection_pair_cta_device_scale_kernel.
verification:
  cargo fmt --all: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/projection_n128_b16_l4d1024_20_20260622T095621Z.run.log
    val_loss=9.069621, train_elapsed_s=6.222, completed_steps=20.
measured_effect:
  Against the current accepted profile
  target/nsys/rope_qkv_f16_fused_b16_l4d1024_20_20260622T084442Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel regressed from
  1.265443216s to 1.338326443s over 20 profiled steps. Full profiled train
  time regressed from 6.122s to 6.222s.
decision:
  Reject before the 900-second gate. Wider N reduced CTA count but made the
  target kernel and short wall-clock slower. Code was reverted to the accepted
  N64/512-thread projection CTA geometry.
```

```text
date: 2026-06-22
commit: uncommitted candidate
experiment: Sort Aurora mega optimizer slots by estimated Polar Express cost.
status: accepted_900s
change:
  Before padding the Aurora pointer table, sort real optimizer slots by a
  simple Polar Express work proxy: min(rows, cols)^2 * max(rows, cols). The
  mega kernel runs three slots per phase and grid-syncs after each phase, so
  packing high-cost matrices into the same phase lanes reduces phase idle time.
  The per-weight optimizer math, learning-rate multipliers, and buffers are
  unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/aurora_slot_sort_b16_l4d1024_20_20260622T100114Z.run.log
    val_loss=9.069621, train_elapsed_s=6.035, completed_steps=20.
  100-step SYNTH screen:
    target/aurora_slot_sort_b16_l4d1024_100_20260622T100135Z.log
    val_loss=6.347749, train_elapsed_s=30.798, completed_steps=100.
  900-second held-out gate:
    target/aurora_slot_sort_b16_l4d1024_900_20260622T100219Z.log
    val_loss=3.632965, train_elapsed_s=900.232, completed_steps=2858.
measured_effect:
  Against the current accepted profile
  target/nsys/rope_qkv_f16_fused_b16_l4d1024_20_20260622T084442Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 2.022238524s to
  1.914045610s over 20 profiled steps. Full profiled train time moved from
  6.122s to 6.035s.
  Against notes/sweep_baseline.env, held-out validation improved from
  3.642559 to 3.632965 and completed steps increased from 2819 to 2858 under
  the fixed 900-second SYNTH budget.
decision:
  Promote. This improves the primary fixed-wall held-out objective and also
  completes more steps.
```

```text
date: 2026-06-22
commit: uncommitted build-only candidate, reverted by rebuild
experiment: Reduce Aurora matrix phases from 8 to 7 after slot sorting.
status: rejected_screen
change:
  Rebuilt the accepted slot-sorted Aurora path with AURORA_MATRIX_PHASES=7.
  Source code, model shape, dataset, optimizer math, and training
  hyperparameters were unchanged. The intent was to remove the padded dummy
  phase now that optimizer slots are sorted by estimated Polar Express cost.
verification:
  AURORA_MATRIX_PHASES=7 cargo oxide build --arch sm_120a: pass.
  20-step nsys:
    target/nsys/aurora_slot_sort_phase7_b16_l4d1024_20_20260622T102046Z.run.log
    val_loss=9.069621, train_elapsed_s=6.046, completed_steps=20.
measured_effect:
  Against the accepted slot-sort profile
  target/nsys/aurora_slot_sort_b16_l4d1024_20_20260622T100114Z.run.log,
  aurora_mega_update_cooperative_kernel regressed from 1.914045610s to
  1.915806901s over 20 profiled steps. Full profiled train time regressed from
  6.035s to 6.046s.
decision:
  Reject before the 900-second gate. The phase-count change made the target
  kernel and short wall-clock slightly slower, so it is not a useful runtime
  candidate.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Sync-only path for padded Aurora mega slots.
status: rejected_screen
change:
  Added a zero-length slot branch in the Aurora mega body that executed only
  the cooperative grid barriers needed to match real slots, skipping dummy
  momentum, Polar, update, and requant work. The real optimizer slots and math
  were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/aurora_dummy_sync_b16_l4d1024_20_20260622T103013Z.run.log
    val_loss=9.069621, train_elapsed_s=6.143, completed_steps=20.
measured_effect:
  Against the accepted slot-sort profile
  target/nsys/aurora_slot_sort_b16_l4d1024_20_20260622T100114Z.run.log,
  aurora_mega_update_cooperative_kernel regressed from 1.914045610s to
  1.940239652s over 20 profiled steps. Full profiled train time regressed from
  6.035s to 6.143s.
decision:
  Reject before the 100-step and 900-second gates. The branch reduced dummy
  slot arithmetic but worsened the real cooperative kernel path, likely from
  added control flow/register pressure. Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Power-of-two row decode hint for MS-EDEN packing.
status: rejected_screen
change:
  Passed a host-computed destination-row shift hint into MS-EDEN pack kernels
  and used shift/mask row decoding when the destination row length was a power
  of two. The intent was to reduce per-lane division in the hot FP32/NVFP4
  transpose pack paths without changing RHT seeds, scaling, correction, or
  layout.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/ms_eden_pow2_row_decode_b16_l4d1024_20_20260622T104512Z.run.log
    val_loss=9.069621, train_elapsed_s=6.143, completed_steps=20.
measured_effect:
  Against the accepted slot-sort profile
  target/nsys/aurora_slot_sort_b16_l4d1024_20_20260622T100114Z.run.log,
  the largest fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel call moved
  from 124.613ms to 124.672ms over 20 profiled steps. Full profiled train time
  regressed from 6.035s to 6.143s.
decision:
  Reject before the 100-step and 900-second gates. The target hot pack kernel
  did not improve, and short wall-clock regressed. Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Alias paired projection second-row B loads to tile0.
status: rejected_screen
change:
  In projection_accumulator_aligned_row_pair, changed the second-row MMA loop
  to load B fragments/scales through tile0 instead of tile1. The two tiles have
  the same column base and warp/thread column mapping, so this was intended as
  a hot-loop address simplification with unchanged math.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/projection_b_alias_b16_l4d1024_20_20260622T105607Z.run.log
    val_loss=9.069621, train_elapsed_s=6.146, completed_steps=20.
measured_effect:
  Against the accepted slot-sort profile
  target/nsys/aurora_slot_sort_b16_l4d1024_20_20260622T100114Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel regressed from
  1.268096s to 1.305495s over 20 profiled steps. Full profiled train time
  regressed from 6.035s to 6.146s.
decision:
  Reject before the 100-step and 900-second gates. The compiler/current
  instruction schedule favored the original tile1 expression. Code was
  reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Register-reuse B fragments across paired projection row tiles.
status: rejected_screen
change:
  In projection_accumulator_aligned_row_pair, explicitly loaded the two B
  fragments and B scale packs for K=0/1 into registers before computing the
  first row tile, then reused those register values for the second row tile
  after staging A1. The intent was to reduce shared-memory/L1 traffic in the
  row-pair MMA loop without changing math or tile geometry.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/projection_b_register_reuse_b16_l4d1024_20_20260622T105820Z.run.log
    val_loss=9.069621, train_elapsed_s=6.120, completed_steps=20.
measured_effect:
  Against the accepted slot-sort profile
  target/nsys/aurora_slot_sort_b16_l4d1024_20_20260622T100114Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel regressed from
  1.268096s to 1.280178s over 20 profiled steps. Full profiled train time
  regressed from 6.035s to 6.120s.
decision:
  Reject before the 100-step and 900-second gates. The saved shared-memory B
  reloads did not offset the added register pressure/schedule change. Code was
  reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate
experiment: Fuse layer-norm residual f16 tape save into layer-norm forward.
status: accepted_900s
change:
  Added a gpt_layer_norm_save_residual_f16_kernel variant that writes the
  layer-norm residual tape as f16 while the forward layer-norm kernel already
  has the residual values loaded. Block ln_1, block ln_2, and final ln_f now
  use that fused path when forward tape is present, then copy only mean and
  inv_std through the tape stats path. The no-tape forward path still uses the
  original gpt_layer_norm_kernel.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test layer_norm_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test layer_norm_backward_params -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/layer_norm_residual_f16_fused_b16_l4d1024_20_20260622T110521Z.run.log
    val_loss=9.063751, train_elapsed_s=6.130, completed_steps=20.
  100-step SYNTH screen:
    target/layer_norm_residual_f16_fused_b16_l4d1024_100_20260622T110558Z.log
    val_loss=6.305291, train_elapsed_s=30.696, completed_steps=100.
  900-second held-out gate:
    target/layer_norm_residual_f16_fused_b16_l4d1024_900_20260622T110658Z.log
    val_loss=3.611782, train_elapsed_s=900.244, completed_steps=2869.
measured_effect:
  Against the accepted slot-sort profile
  target/nsys/aurora_slot_sort_b16_l4d1024_20_20260622T100114Z.run.log,
  fp32_to_f16_kernel dropped from 357 calls / 36.026ms to 168 calls /
  25.204ms over 20 profiled steps. The fused layer-norm save kernel added
  189 calls / 13.033ms, so the local conversion-plus-layer-norm region moved
  from 46.541ms to 39.235ms. The short full-profile wall time was noisy and
  moved from 6.035s to 6.130s, so the candidate was sent through the fixed-wall
  gate rather than promoted from profiler data.
  Against the previous promoted baseline
  target/aurora_slot_sort_b16_l4d1024_900_20260622T100219Z.log, held-out
  validation improved from 3.632965 to 3.611782 and completed steps increased
  from 2858 to 2869.
decision:
  Promote. The candidate satisfies the fixed 900-second objective directly:
  lower held-out validation loss and higher completed step count.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Derive cross-entropy dlogits row amax from target probability.
status: rejected_screen
change:
  Replaced the post-dlogits row amax reduction in cross_entropy_kernel with the
  exact identity max(abs(dlogits[row])) = (1 - p_target) / token_count, using
  the target probability already available from the row max and softmax
  denominator. Loss and dlogits math were otherwise unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test loss -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/cross_entropy_exact_amax_b16_l4d1024_20_20260622T112908Z.run.log
    val_loss=9.063751, train_elapsed_s=6.136, completed_steps=20.
measured_effect:
  Against the accepted layer-norm residual f16 profile
  target/nsys/layer_norm_residual_f16_fused_b16_l4d1024_20_20260622T110521Z.run.log,
  cross_entropy_kernel was unchanged in practice: 58.182917ms baseline versus
  58.188720ms candidate over 21 calls. The main hot kernels were also
  unchanged or slightly slower within noise.
decision:
  Reject before the 100-step and 900-second gates. The exact simplification was
  correct but did not produce a measurable runtime win in the profiled kernel.
  Code was reverted to the measured baseline path.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Double-stage A operands for aligned row-pair NVFP4 projection CTAs.
status: accepted_900s
change:
  Added separate shared storage for the second row tile's A packs/scales in the
  aligned row-pair projection helper. The row-pair path now stages B, A0, and
  A1 before the compute section, then computes both row tiles before the final
  synchronization. This keeps the same tile geometry and Quartet/NVFP4 MMA
  math while reducing the middle restage/sync sequence in row-pair projection
  kernels.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test next_latent -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/projection_pair_double_a_stage_b16_l4d1024_20_20260622T113442Z.run.log
    val_loss=9.063751, train_elapsed_s=6.117, completed_steps=20.
  100-step SYNTH screen:
    target/projection_pair_double_a_stage_b16_l4d1024_100_20260622T113523Z.log
    val_loss=6.304021, train_elapsed_s=30.612, completed_steps=100.
  900-second held-out gate:
    target/projection_pair_double_a_stage_b16_l4d1024_900_20260622T113607Z.log
    val_loss=3.614733, train_elapsed_s=900.254, completed_steps=2872.
measured_effect:
  Against the accepted layer-norm residual f16 profile
  target/nsys/layer_norm_residual_f16_fused_b16_l4d1024_20_20260622T110521Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  1303.588048ms to 1289.235967ms over 400 calls. lm_head_kernel moved from
  220.756825ms to 218.412010ms over 21 calls, and nextlat_projection_kernel
  moved from 71.176803ms to 70.927641ms over 63 calls. The 100-step screen
  improved from val_loss=6.305291 / 30.696s to val_loss=6.304021 / 30.612s.
  The 900-second held-out gate completed 2872 steps versus the previous 2869,
  while validation moved from 3.611782 to 3.614733, a +0.082% change inside
  the active 1% noise band.
decision:
  Promote under the active fixed-wall rule: validation stayed within the 1%
  noise band and completed step count increased.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Row-wise lower-triangle scheduling for attention probability/dS.
status: rejected_screen
change:
  Replaced the flat full-square attention_prob_ds launch with one block per
  causal row. Threads walked only key <= query while preserving the same dense
  scratch layout for p and ds.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/attention_prob_ds_rowwise_b16_l4d1024_20_20260622T124713Z.run.log
    val_loss=9.063751, train_elapsed_s=6.099, completed_steps=20.
measured_effect:
  Against the accepted B-reuse projection profile
  target/nsys/projection_pair_b_reuse_b16_l4d1024_20_20260622T120319Z.run.log,
  attention_prob_ds_kernel regressed from 125.634467ms to 127.512996ms over
  80 calls. Other top kernels moved slightly faster in the same run, consistent
  with profiler noise, but the targeted kernel was worse.
decision:
  Reject before the 100-step and 900-second gates. Reducing masked work did not
  pay for the row-wise loop and launch geometry change. Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Reuse one staged projection B tile across four row tiles.
status: rejected_screen
change:
  Added a row-quad linear-backward projection CTA path for fully aligned row
  groups. The candidate staged B once, computed two row tiles, restaged only A,
  then computed two more row tiles. The intent was to reduce repeated B-tile
  L2/shared staging in the memory/L2-limited projection kernel.
verification:
  cargo fmt --all: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/projection_row_quad_b16_l4d1024_20_20260622T125436Z.run.log
    val_loss=9.063751, train_elapsed_s=6.173, completed_steps=20.
measured_effect:
  Against the accepted B-reuse projection profile
  target/nsys/projection_pair_b_reuse_b16_l4d1024_20_20260622T120319Z.run.log,
  the projection kernel regressed from
  linear_backward_projection_pair_cta_device_scale_kernel 1274.205747ms to
  linear_backward_projection_quad_cta_device_scale_kernel 1350.659734ms over
  400 calls. Full profiled train time moved from 6.103s to 6.173s.
decision:
  Reject before the 100-step and 900-second gates. B-tile reuse across four row
  tiles did not offset the extra accumulator pressure and additional A restage
  synchronization. Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before profile
experiment: Force Aurora mega slot dispatcher out of line.
status: rejected_codegen
change:
  Added #[inline(never)] to aurora/fused/mega/slot.rs launch_slot to test
  whether keeping slot-table pointer decoding outside the cooperative entry
  would reduce register pressure in aurora_mega_update_cooperative_kernel.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  PTX inspection after build:
    rust_kernels_cuda.ptx still had aurora_mega_update_cooperative_kernel
    calling aurora_matrix_update_body directly, not launch_slot.
measured_effect:
  No runtime profile was taken because the intended call-boundary change did
  not materialize in generated PTX.
decision:
  Reject as a codegen no-op for the intended optimization. Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Increase NVFP4 projection CTA K tile from 128 to 256.
status: rejected_screen
change:
  Doubled NVFP4_PROJECTION_CTA_K so each projection CTA would stage a deeper K
  tile and perform more MMA work per staging/sync iteration. Projection math,
  M/N tile shape, row-pair scheduling, and Quartet/NVFP4 instructions were
  unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test next_latent -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/projection_cta_k256_b16_l4d1024_20_20260622T122956Z.run.log
    val_loss=9.063751, train_elapsed_s=6.230, completed_steps=20.
measured_effect:
  Against the accepted B-reuse projection profile
  target/nsys/projection_pair_b_reuse_b16_l4d1024_20_20260622T120319Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel regressed from
  1274.205747ms to 1354.755238ms over 400 calls. Projection users also
  regressed: lm_head_kernel moved from 217.835379ms to 221.227757ms,
  nextlat_projection_kernel moved from 70.815732ms to 74.149479ms,
  mlp_projection_kernel moved from 126.827987ms to 141.803886ms, and
  attention_projection_kernel moved from 120.025257ms to 136.344658ms.
decision:
  Reject before the 100-step and 900-second gates. The deeper K tile increased
  shared-memory/register/scheduling pressure enough to dominate the reduced
  stage/sync count. Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Add aligned fast paths for FP16 RHS TC matmul variants.
status: rejected_screen
change:
  Mirrored the existing aligned f16_cta_tc_matmul_f32_kernel path in the
  row-major RHS and A-transposed/RHS variants, using unguarded staging and
  aligned stores when m, n, and k are CTA-aligned. Matmul math and launch
  geometry were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/f16_rhs_aligned_b16_l4d1024_20_20260622T123355Z.run.log
    val_loss=9.063751, train_elapsed_s=6.114, completed_steps=20.
measured_effect:
  Against the accepted B-reuse projection profile
  target/nsys/projection_pair_b_reuse_b16_l4d1024_20_20260622T120319Z.run.log,
  f16_cta_tc_matmul_f32_a_transposed_rhs_kernel regressed from 254.775616ms
  to 263.069520ms over 160 calls, and f16_cta_tc_matmul_f32_rhs_kernel
  regressed from 247.851231ms to 251.097628ms over 164 calls.
decision:
  Reject before the 100-step and 900-second gates. The extra aligned branch
  and duplicated staging code increased code/register/schedule pressure more
  than the removed per-element guards helped. Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Reuse RoPE sin/cos inside attention scatter_dqkv_kernel.
status: rejected_screen
change:
  Computed the RoPE sine/cosine pair once per scatter thread and reused it for
  both d_q and d_k raw-gradient rotation instead of recomputing the same angle
  in each helper call. Scatter math and output layout were unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/scatter_rope_sincos_reuse_b16_l4d1024_20_20260622T123854Z.run.log
    val_loss=9.063751, train_elapsed_s=6.101, completed_steps=20.
  100-step SYNTH screen:
    target/scatter_rope_sincos_reuse_b16_l4d1024_100_20260622T123939Z.log
    val_loss=6.298502, train_elapsed_s=30.553, completed_steps=100.
measured_effect:
  Against the accepted B-reuse projection profile
  target/nsys/projection_pair_b_reuse_b16_l4d1024_20_20260622T120319Z.run.log,
  scatter_dqkv_kernel moved from 18.399166ms to 18.348118ms over 80 calls.
  The 100-step screen had effectively unchanged validation loss but slightly
  slower elapsed time, moving from 6.298800 / 30.533s to 6.298502 / 30.553s.
decision:
  Reject before the 900-second gate. The local kernel win was too small to
  improve the fixed-step screen and did not justify a full fixed-wall gate.
  Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Skip all-padding Aurora mega phases with a separate active slot count.
status: rejected_screen
change:
  Passed active Aurora matrix slot count separately from padded slot count, then
  skipped only phases where every matrix lane mapped to padding. Partial padding
  phases still executed the zero-sized body to preserve the cooperative
  grid-sync sequence.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/aurora_active_slot_skip_b16_l4d1024_20_20260622T122519Z.run.log
    val_loss=9.063751, train_elapsed_s=6.109, completed_steps=20.
measured_effect:
  Against the accepted B-reuse projection profile
  target/nsys/projection_pair_b_reuse_b16_l4d1024_20_20260622T120319Z.run.log,
  aurora_mega_update_cooperative_kernel regressed from 1942.311211ms to
  1943.581891ms over 20 calls. The projection and FP16 TC kernels also moved
  slightly slower within the same screen.
decision:
  Reject before the 100-step and 900-second gates. Skipping the final
  all-padding phase did not reduce runtime; any saved empty-body sync work was
  outweighed by added control/signature pressure or profiler noise. Code was
  reverted.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Hoist Aurora transpose branches out of momentum/update element loops.
status: rejected_screen
change:
  Split momentum orientation and master update element paths into
  direct/transposed variants so transposed indexing would be selected once per
  matrix instead of inside each element update.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/aurora_transpose_branch_hoist_b16_l4d1024_20_20260622T115640Z.run.log
    val_loss=9.063751, train_elapsed_s=6.120, completed_steps=20.
measured_effect:
  Against the accepted double-A projection profile
  target/nsys/projection_pair_double_a_stage_b16_l4d1024_20_20260622T113442Z.run.log,
  aurora_mega_update_cooperative_kernel regressed from 1941.656078ms to
  1942.165400ms over 20 calls, and
  linear_backward_projection_pair_cta_device_scale_kernel also moved from
  1289.235967ms to 1290.500232ms over 400 calls.
decision:
  Reject before the 100-step and 900-second gates. The branch hoist increased
  code/register/schedule pressure enough to lose the small per-element branch
  removal. Code was reverted.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Reuse staged B fragments across both row tiles in row-pair projection.
status: accepted_900s
change:
  Combined the two row-pair accumulator k-atom loops so the staged B fragment
  and B scale are loaded once from shared memory and consumed by both row
  accumulators. The tile shape, row-pair scheduling, Quartet/NVFP4 MMA
  instruction, and output math are unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test lm_head -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test next_latent -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/projection_pair_b_reuse_b16_l4d1024_20_20260622T120319Z.run.log
    val_loss=9.063751, train_elapsed_s=6.103, completed_steps=20.
  100-step SYNTH screen:
    target/projection_pair_b_reuse_b16_l4d1024_100_20260622T120426Z.log
    val_loss=6.298800, train_elapsed_s=30.533, completed_steps=100.
  900-second held-out gate:
    target/projection_pair_b_reuse_b16_l4d1024_900_20260622T120514Z.log
    val_loss=3.621134, train_elapsed_s=900.116, completed_steps=2880.
measured_effect:
  Against the accepted double-A projection profile
  target/nsys/projection_pair_double_a_stage_b16_l4d1024_20_20260622T113442Z.run.log,
  linear_backward_projection_pair_cta_device_scale_kernel moved from
  1289.235967ms to 1274.205747ms over 400 calls. lm_head_kernel moved from
  218.412010ms to 217.835379ms over 21 calls, and nextlat_projection_kernel
  moved from 70.927641ms to 70.815732ms over 63 calls. The 100-step screen
  improved from val_loss=6.304021 / 30.612s to val_loss=6.298800 / 30.533s.
  The 900-second held-out gate completed 2880 steps versus the previous 2872,
  while validation moved from 3.614733 to 3.621134, a +0.177% change inside
  the active 1% noise band.
decision:
  Promote under the active fixed-wall rule: validation stayed within the 1%
  noise band and completed step count increased.
```

```text
date: 2026-06-22
commit: uncommitted
experiment: Skip transformed chunk-amax writes in internal MS-EDEN device-scale
  linear-backward pack paths.
status: accepted_900s
change:
  Added internal no-chunk-amax MS-EDEN device-scale pack kernels and routed
  linear backward operand quantization through them. The existing public
  device-scale APIs keep their chunk-amax side effect for comparison tests.
  The fast path is used only after the amax buffer has already been consumed
  to derive the device global scale, so packed FP4 bytes, FP8 scales, row
  global scales, seeds, RHT, and MMA consumers are unchanged.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test ms_eden_transpose -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/ms_eden_no_chunk_amax_b16_l4d1024_20_20260622T131123Z.run.log
    val_loss=9.063751, train_elapsed_s=6.076, completed_steps=20.
  100-step SYNTH screen:
    target/ms_eden_no_chunk_amax_b16_l4d1024_100_20260622T131212Z.log
    val_loss=6.305133, train_elapsed_s=30.416, completed_steps=100.
  900-second held-out gate:
    target/ms_eden_no_chunk_amax_b16_l4d1024_900_20260622T131306Z.log
    val_loss=3.603050, train_elapsed_s=900.088, completed_steps=2894.
measured_effect:
  Against the accepted row-pair B-reuse profile
  target/nsys/projection_pair_b_reuse_b16_l4d1024_20_20260622T120319Z.run.log,
  fp32_transpose_to_nvfp4_ms_eden_device_scale moved from 250.567290ms to
  242.841270ms over 400 calls, fp32_to_nvfp4_ms_eden_device_scale moved from
  147.024446ms to 135.792200ms, rowwise_nvfp4_transpose_to_nvfp4_ms_eden moved
  from 160.446773ms to 156.856497ms, and nvfp4_transpose_to_nvfp4_ms_eden
  moved from 22.074919ms to 21.435099ms. Profiled 20-step wall moved from
  6.103s to 6.076s with identical 20-step validation loss. The 900-second gate
  improved held-out validation loss from 3.621134 to 3.603050 and completed
  steps from 2880 to 2894.
decision:
  Promote. This passes the fixed-wall objective directly: lower held-out
  validation loss and higher completed step count under the same 900-second
  budget.
```
```text
date: 2026-06-22
commit: uncommitted candidate, reverted before gate
experiment: Split dweight-dominant final-head linear-backward projection launch.
status: rejected_screen
change:
  Added a temporary single-projection row-pair device-scale CTA kernel and
  routed linear backward through two leaner launches only when dweight tile
  count exceeded dinput tile count. In the current B16/L4/d1024 Llama2-vocab
  run this isolated the final LM-head backward projection, whose combined grid
  was 12096 tiles: 4096 dinput tiles plus 8000 dweight tiles.
verification:
  cargo fmt --all: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test qkv_projection_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/final_head_split_projection_b16_l4d1024_20_20260622T142614Z.run.log
    val_loss=9.063751, train_elapsed_s=5.981, completed_steps=20.
measured_effect:
  Against the fresh current profile
  target/nsys/current_b16_l4d1024_20_20260622T142037Z.run.log, the combined
  projection kernel had 400 calls / 1238.082617ms total. The split candidate
  had paired projection 380 calls / 834.898054ms plus single projection
  40 calls / 411.756732ms, for 1246.654786ms total. The final-head work split
  into 4096-tile and 8000-tile single-projection launches around 10.2-10.3ms
  each, which was slower than the original combined 12096-tile call at about
  20.2ms.
decision:
  Reject before the 100-step and 900-second gates. The extra launch and lower
  argument/control pressure did not beat the accepted paired projection path.
  Code was reverted; the note is kept to avoid repeating this final-head split.
```
```text
date: 2026-06-22
commit: uncommitted candidate, accepted after gate
experiment: Compact Aurora symmetric Gram tile scheduling.
status: accepted
change:
  Changed the Aurora Polar Express symmetric Gram stage from iterating the full
  square tile domain and skipping lower-triangle tiles to iterating a compact
  upper-triangle tile domain. The cooperative launch shape is unchanged; only
  the in-kernel tile index mapping changed.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1:
    harness failed before exercising kernels with DriverError(301, "file not found").
    The generated PTX exists at the common test harness path; release runtime
    checks below are the acceptance evidence for this candidate.
  20-step nsys:
    target/nsys/aurora_compact_tri_b16_l4d1024_20_20260622T145452Z.run.log
    val_loss=9.063751, train_elapsed_s=5.923, completed_steps=20.
  100-step SYNTH screen:
    target/aurora_compact_tri_b16_l4d1024_100_20260622T145530Z.log
    val_loss=6.298866, train_elapsed_s=29.640, completed_steps=100.
  900-second held-out gate:
    target/aurora_compact_tri_b16_l4d1024_900_20260622T145614Z.log
    val_loss=3.575694, train_elapsed_s=900.209, completed_steps=2967.
measured_effect:
  Against the fresh current profile
  target/nsys/current_clean_b16_l4d1024_20_20260622T143653Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1914.892ms to 1750.067ms
  over 20 calls, and profiled train time moved from 5.968s to 5.923s.
  Against the promoted 900-second baseline
  target/ms_eden_no_chunk_amax_b16_l4d1024_900_20260622T131306Z.log,
  held-out validation loss improved from 3.603050 to 3.575694 and completed
  steps increased from 2894 to 2967.
decision:
  Promote. This passes the active rule directly: lower held-out validation loss
  and higher completed step count under the same 900-second SYNTH budget.
```
```text
date: 2026-06-22
commit: uncommitted candidate, reverted before screen
experiment: Replace compact Aurora upper-triangle sqrt mapping with integer binary search.
status: rejected_launch
change:
  Replaced the compact upper-triangle tile inverse-sqrt mapping with a small
  integer binary search over row starts. The tile domain and math were intended
  to remain unchanged.
verification:
  cargo fmt --all --check: pass after formatting.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test linear_backward_projection_cta -- --ignored --nocapture --test-threads=1: pass when run serially after PTX build.
  20-step nsys launch:
    target/nsys/aurora_compact_tri_intmap_b16_l4d1024_20_20260622T151702Z.run.log
    failed before step 0 with DriverError(720, "too many blocks in cooperative launch").
measured_effect:
  No kernel timing. The extra integer control flow likely increased register
  pressure or otherwise reduced cooperative launch residency below the required
  grid shape.
decision:
  Reject before 100-step and 900-second gates. Code was reverted to the
  accepted sqrt-based compact mapping.
```
```text
date: 2026-06-22
commit: uncommitted candidate, reverted after 20-step screen
experiment: Add aligned fast path to F16 RHS and A-transposed-RHS TC matmul staging.
status: rejected_screen
change:
  Added aligned staging/store branches for
  f16_cta_tc_matmul_f32_rhs_kernel and
  f16_cta_tc_matmul_f32_a_transposed_rhs_kernel when m, n, and k are already
  multiples of the CTA tile sizes. The intent was to remove bounds checks from
  the aligned attention-backward FP16 TC matmul shapes.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test f16_tc_matmul -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test f16_tc_matmul_tiled -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test block_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test forward -- --ignored --nocapture --test-threads=1: pass.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p gpt2-nvfp4 --test causal_attention_backward -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/f16_rhs_aligned_b16_l4d1024_20_20260622T152402Z.run.log
    val_loss=9.063751, train_elapsed_s=5.928, completed_steps=20.
measured_effect:
  Against the accepted compact-triangle profile
  target/nsys/current_after_compact_tri_b16_l4d1024_20_20260622T151411Z.run.log,
  f16_cta_tc_matmul_f32_a_transposed_rhs_kernel moved from 255.595400ms to
  262.777947ms over 160 calls. f16_cta_tc_matmul_f32_rhs_kernel moved from
  251.316697ms to 253.286365ms over 164 calls. Profiled train time moved from
  5.935s to 5.928s, which is within noise and contradicted by the regressed
  target kernels.
decision:
  Reject before 100-step and 900-second gates. The candidate made both target
  kernels slower in the 20-step nsys screen, so the code was reverted and only
  this note is kept.
```
```text
date: 2026-06-22
commit: uncommitted candidate, accepted after gate
experiment: Pack Aurora mega slot metadata into one descriptor buffer.
status: accepted_900s
change:
  Replaced the Aurora mega launch contract's split slot metadata buffers
  (seven pointer arrays, rows, cols, and learning-rate multipliers) with one
  AuroraSlotDescriptor buffer. Slot ordering, phase geometry, scratch layout,
  optimizer math, model shape, and training hyperparameters were unchanged.
  The intent was to reduce always-live pointer-array bases in the cooperative
  Aurora entry and lower register pressure without adding a CPU round trip.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  Generated PTX: aurora_mega_update_cooperative_kernel entry parameters moved
    from param_0..40 to param_0..22, and PTX b64 registers moved from 69 to 41.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/aurora_slot_descriptor_b16_l4d1024_20_20260622T153325Z.run.log
    val_loss=9.063751, train_elapsed_s=5.915, completed_steps=20.
  100-step SYNTH screen:
    target/aurora_slot_descriptor_b16_l4d1024_100_20260622T153354Z.log
    val_loss=6.300856, train_elapsed_s=29.597, completed_steps=100.
  900-second held-out gate:
    target/aurora_slot_descriptor_b16_l4d1024_900_20260622T153447Z.log
    val_loss=3.599322, train_elapsed_s=900.128, completed_steps=2973.
measured_effect:
  Against the accepted compact-triangle profile
  target/nsys/current_after_compact_tri_b16_l4d1024_20_20260622T151411Z.run.log,
  aurora_mega_update_cooperative_kernel moved from 1753.570775ms to
  1748.781893ms over 20 calls. The full profiled training run moved from
  5.935s to 5.915s.
  Against the prior promoted 900-second baseline
  target/aurora_compact_tri_b16_l4d1024_900_20260622T145614Z.log, held-out
  validation loss moved from 3.575694 to 3.599322 (+0.66%, inside the active
  1% noise band) while completed steps increased from 2967 to 2973.
decision:
  Promote under the active runtime-change rule: validation loss stayed within
  the 1% noise band and completed step count increased under the fixed
  900-second SYNTH budget.
```

```text
date: 2026-06-22
commit: uncommitted candidate, reverted after 20-step screen
experiment: Remove unreachable Aurora mega slot-count guard from device entry.
status: rejected_screen
change:
  Removed the aurora_mega_update_cooperative_kernel slot_count parameter and
  the in-kernel `slot < slot_count` guard. The host launch already asserts
  slot_count is an exact multiple of AURORA_MATRIX_PHASES and launches
  gridDim.y = slot_count / AURORA_MATRIX_PHASES, so the guard is unreachable
  in the accepted launch contract.
verification:
  cargo fmt --all --check: pass.
  cargo check --all-targets: pass.
  cargo oxide build --arch sm_120a: pass.
  Generated PTX: aurora_mega_update_cooperative_kernel entry parameters moved
    from param_0..22 to param_0..21, but PTX b64 registers stayed at 41.
  CUDA_DEVICE_INDEX=0 timeout 300 cargo test -p rust-kernels-cuda --test optimizer -- --ignored --nocapture --test-threads=1: pass.
  20-step nsys:
    target/nsys/aurora_slot_guard_removed_b16_l4d1024_20_20260622T155619Z.run.log
    val_loss=9.063751, train_elapsed_s=5.923, completed_steps=20.
measured_effect:
  Against the accepted descriptor profile
  target/nsys/aurora_slot_descriptor_b16_l4d1024_20_20260622T153325Z.run.log,
  aurora_mega_update_cooperative_kernel regressed from 1748.781893ms to
  1751.096652ms over 20 calls. Profiled train time moved from 5.915s to
  5.923s.
decision:
  Reject before the 100-step and 900-second gates. The candidate trimmed one
  entry parameter but did not reduce b64 register count and made the target
  kernel slower. Code was reverted; keep this note to avoid repeating this
  cleanup.
```
