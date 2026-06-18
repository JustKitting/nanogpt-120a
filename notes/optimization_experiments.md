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

Current first target:

- Reach at least 100k tokens/s sustained.
- This makes a 10B-token run day-scale instead of week-scale.

## Current Baseline

Last measured AMUSE 2k Shakespeare run:

```text
batch_size = 8
seq_len = 1024
tokens_per_step = 8192
late-step time with logged loss sync ~= 0.72 s
late-step time without loss sync ~= 0.54 s
throughput excluding loss sync ~= 15k tokens/s
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

## Current Bottlenecks

From recent logs and profiling:

- Optimizer is too expensive, roughly 290 ms/step in late AMUSE runs.
- Aurora dominates optimizer time, roughly 280 ms/step.
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
- Fusion or removal of Aurora helper kernels.
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
dataset:
command:
tokens_per_step:
steps:
log_interval:
eval_interval:
final_train_loss:
final_val_loss:
tokens_per_second:
top_kernels:
notes:
```
