# G3 — Tuning Guide

Task-oriented walkthrough of every training knob in `configs/pretrain_a100_400m.yaml`: what each one does, how it interacts with the others, and — most importantly — how to *measure* its effect before you trust it.

## 1. 60-second summary

The Mamba-3-Lite training surface is deliberately small: one YAML file, one dataclass, one training loop. There are exactly four families of knobs that matter — **chunk_size** (algorithmic work/memory trade in the SSD), **the learning-rate schedule** (lr, warmup_steps, min_lr_ratio), **batch geometry** (micro_batch_size × gradient_accumulation_steps), and **the three performance switches** (grad_checkpoint, compile/compile_mode, ssd_dispatch). Every knob has an expected effect and a way to observe it: `utils/logging.py:TrainingLogger.log` prints windowed `loss`/`ppl`/`tps` every `log_every` steps, `--dry-run` validates wiring in 2 steps, and small-scope A/B sweeps (Section 10) decide between two settings without burning a full run. **There are no benchmarks in this repo** — every throughput effect quoted below is `[INFERENCE]`, derived from tensor shapes and the algorithm, not measured.

## 2. Measure first: the discipline

You can only tune what you measure, and this repo gives you exactly one instrument: `utils/logging.py:TrainingLogger.log`, called every `log_every` steps (default 50) from `training/pretrain.py:Pretrainer.train`. Each line looks like:

```
step=   3500 | loss=6.8214 | ppl=916.32 | lr=3.00e-04 | tps=142,331
```

The metrics are windowed (averaged over the last `log_every` micro-steps), `ppl = exp(avg_loss)`, and

$$tps = \frac{\text{log\_every} \times \text{seq\_len} \times \text{batch\_size}}{\text{elapsed}}$$

where `seq_len` is `max_seq_len` (2048) and `batch_size` is the **micro** batch — so tps is model-work tokens/sec through forward+backward, unaffected by accumulation (2 micro-steps of 2× tokens take 2× time; the ratio is the same). The same dict goes to WandB when `WANDB_PROJECT` is set. Full contract: [R10 — logging](../reference/10-logging.md).

Three habits before touching any knob:

1. **Prove the wiring first.** `python3 training/pretrain.py --dry-run` forces `max_steps=2` (`training/pretrain.py:main`) and exercises config → model → dataset → loop. On the CPU/Mac dev box, add `--no-compile` (Section 7) or the 2 steps are mostly compile time.
2. **Sweep small.** A/B decisions (Section 10) use a scratch config with `total_steps` ~2–3k and a separate `save_dir`, never the 256k production schedule.
3. **Record, then believe.** Keep a table of (variable, value, loss@N, median tps, notes). With no `.benchmarks/` in the tree, your own log rows are the only throughput evidence.

## 3. chunk_size — the SSD knob

**What it is.** `chunk_size` is a field of `models/transformer.py:ModelConfig` (default 64) consumed by `models/ssd_complex.py:ssd_complex_chunkwise` (via `models/mamba_block.py:Mamba3Block.chunk_size`). The sequence mixer replaces the O(T) sequential scan with two levels: intra-chunk matmuls over windows of C tokens plus an inter-chunk scan over `T/C` chunks. `ssd_complex_chunkwise` pads T to a multiple of C (`pad = (C - (T % C)) % C`), so C need not divide T.

**The tradeoff** (derived in [T8 — scaling/efficiency](../theory/08-scaling-efficiency.md), all throughput columns `[INFERENCE]`):

| C | chunks | L bytes/layer¹ | intra-chunk GEMM | inter-chunk scan | throughput² |
|---|---|---|---|---|---|
| 32 | 64 | 128 MiB | 32×32 | 64 hops | ~baseline; best for short seqs |
| 64 | 32 | 256 MiB | 64×64 | 32 hops | **baseline (default)** |
| 128 | 16 | 512 MiB | 128×128 | 16 hops | +5–10% |
| 256 | 8 | 1024 MiB | 256×256 | 8 hops | +10–15%; OOM risk at seq 8k / batch ≥ 32 |

¹ The materialized causal-decay matrix $L[l,s] = e^{A_{cs}[l]-A_{cs}[s]}\mathbf{1}[l\ge s]$ is $C\times C$ per (batch, chunk, head): $|L| = 8\,B\,H\,T\,C$ bytes at complex64 (derived in [T8 — scaling/efficiency](../theory/08-scaling-efficiency.md)). Larger C = **more L memory but bigger, tensor-core-friendlier GEMMs and fewer sequential chunk hops**. With `grad_checkpoint: true` (default) L is rebuilt in backward instead of stored — see Section 6. ² SKILLS.md Skill 3's numbers, `[INFERENCE]` — no measured sweep exists in this tree.

**Two hard constraints on the Triton path.** `chunk_size` becomes the constexpr `BLOCK_C` consumed by `tl.arange`, so it must be a **power of two** — and it must be **≤ 256**, enforced by `models/ssd_triton.py:_check_block_dims`:

```python
def _check_block_dims(P: int, N: int, chunk_size: int) -> None:
    for name, dim in (("P", P), ("N", N), ("chunk_size", chunk_size)):
        if dim > _MAX_BLOCK:
            raise ValueError(
                f"per_chunk_ssd_triton: {name}={dim} exceeds the {_MAX_BLOCK}-cap. "
                f"Use ssd_dispatch='pytorch' for this config."
            )
```

A C > 256 or non-power-of-two C does **not** crash a triton run: `Mamba3Block._ssd_with_dispatch` catches the exception, warns once per block, and falls back to the PyTorch path — silently slower, still correct. The PyTorch dispatch has no cap (it just runs out of memory as L grows). **How to A/B:** Section 10, with `chunk_size` as the one variable.

## 4. The learning-rate family

All three fields live in `training/pretrain.py:TrainingConfig` and the `training:` YAML section: `lr` (3e-4), `warmup_steps` (2000), `min_lr_ratio` (0.05). The schedule is built in `training/pretrain.py:Pretrainer.__init__`:

```python
warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
cosine = CosineAnnealingLR(self.optimizer, T_max=config.max_steps - config.warmup_steps, eta_min=config.lr * config.min_lr_ratio)
self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps])
```

**Trajectory.** Warmup is *linear from 1% of peak*: effective lr at step $t$ is $\text{lr}\cdot(0.01 + 0.99\,t/\text{warmup\_steps})$ — 3e-6 at step 0, reaching 3e-4 at step 2000 (0.8% of the 256k schedule). Then cosine decay to `eta_min = lr × min_lr_ratio` = 1.5e-5:

$$lr_t = \eta_{\min} + \tfrac12(\text{lr}-\eta_{\min})\bigl(1 + \cos(\pi \tfrac{t-2000}{256000-2000})\bigr)$$

The 0.01× start is the stability margin: the first steps see raw gradient directions, and ramping from a small value keeps the first optimizer updates small.

**Too-high lr looks like early NaN.** The failure signature is a loss that spikes or stalls in the first few hundred steps and then a `[nan-guard]` line: `training/pretrain.py:train_step` checks `torch.isnan(loss)` before backward and returns `None` instead of stepping; `Pretrainer.train` counts consecutive NaN micro-steps, and after `nan_guard_max_consecutive` (5) it restores the latest checkpoint — or aborts with `RuntimeError` if none exists. Full semantics: [T7 — numerical stability](../theory/07-numerical-stability.md).

**Why 3e-4?** `[INFERENCE]` — no sweep exists in the repo. The rationale that survives arithmetic: (a) 3e-4 is the standard peak for the 100M–1B parameter class with Adam-family optimizers; (b) **beta2=0.95** (not 0.999) shortens the second-moment EMA window to $1/(1-\beta_2) = 20$ steps, so $v_t$ tracks recent gradients closely and the adaptive update $\eta/(\sqrt{\hat v_t}+\epsilon)$ is *less damped* — a smaller peak lr is warranted than with β₂=0.999; (c) warmup absorbs the early-stability risk and the NaN guard is the safety net. Treat 3e-4 as the default arm of an A/B, not as law: if loss is flat but stable try `lr: 6e-4`; if it spikes, halve it. Measure via the `lr=` and `loss=` columns.

## 5. Batch geometry

Tokens per optimizer step:

$$\text{tokens/step} = \text{micro\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{max\_seq\_len} = 16 \times 2 \times 2048 = 65{,}536$$

**The micro-batch is the memory lever, accumulation is the stability lever.** Activations scale with `micro_batch_size` (one micro-batch is live at a time; accumulated gradients are just summed — negligible memory). So:

- **Raise `micro_batch_size`** when the card has spare memory: bigger GEMMs, better tensor-core utilization, fewer micro-steps per optimizer step. Cost: activation memory grows linearly.
- **Raise `gradient_accumulation_steps`** for a larger effective batch (more stable gradients) without touching memory. Cost: the optimizer (and clipping) only runs at accumulation boundaries, so more accumulation lengthens wall-clock per optimizer step.
- Keep `micro_batch_size × gradient_accumulation_steps` constant and the *number* of optimizer steps for a fixed token budget is constant — you trade memory against GEMM efficiency, not step count (see [T8](../theory/08-scaling-efficiency.md) §5 for the step-count arithmetic).

Grad clipping: `nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)` runs **once per optimizer step**, on the accumulated gradients (`train_step`, guarded by `is_opt_step`) — the correct place, since clipping per micro-step would distort the accumulation. `max_grad_norm = 1.0` is the production default.

**Measure it:** tps (Section 2) is the honest throughput number; peak VRAM comes from `nvidia-smi` on the A100 box. The A100 runbook ([G2 — training runbook](02-training-runbook.md)) covers the 80 GB budget.

## 6. grad_checkpoint — one global boolean

`grad_checkpoint` is a **single boolean applied uniformly to all 28 blocks** — there is no every-4th-layer cadence in this repo. `training/pretrain.py:Pretrainer.__init__` injects it into the model config via `config.model_config.setdefault("grad_checkpoint", config.grad_checkpoint)`, and `models/mamba_block.py:Mamba3Block.forward` wraps the whole block:

```python
if self.grad_checkpoint and self.training:
    return torch.utils.checkpoint.checkpoint(
        self._forward_impl, x, use_reentrant=False
    )
```

**Effect:** activations for the block's forward are not stored; the backward re-runs `_forward_impl` (SSD included) once per block. Cost ≈ one extra forward pass ≈ **~33% more FLOPs**; saving ≈ **~1.4 GiB per layer** at the production batch (derived in [T8](../theory/08-scaling-efficiency.md) §6.4 — `[INFERENCE]`, no measured profile). The L matrix of Section 3 is the biggest thing not stored.

**When to flip it:** memory-bound runs (batch 32+, seq 8k) → keep `true`; if FLOP-bound with headroom, `false` buys ~25–30% throughput `[INFERENCE]`. The CLI override is `--no-checkpoint`; the YAML key wins either way (`setdefault` only fills a *missing* key — an explicit `grad_checkpoint: false` is respected).

**Compile interplay:** `use_reentrant=False` is torch.compile-compatible — the recompute runs inside the compiled graph, so the two features compose; the compile cost is paid once, the recompute is a graph replay per layer per backward.

## 7. torch.compile modes

`training/pretrain.py:Pretrainer.__init__` compiles once, before training:

```python
if config.compile_model and hasattr(torch, "compile"):
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", config.compile_mode)
    self._log(f"Compiling model with torch.compile (mode={compile_mode})...")
    training_model = torch.compile(training_model, mode=compile_mode, fullgraph=False)
```

- **`compile_mode: "max-autotune"`** (default) — spends *compile time* autotuning every kernel so steady-state steps run fast. Right for production, wrong for experiments: `--dry-run` with compile enabled mostly measures **compilation**, not training.
- **`reduce-overhead`** — cudagraphs-style launch reduction; good when the step is launch-bound (small batches), less kernel tuning.
- **`"default"`** — cheaper compile, middling runtime; use for sweeps.
- **`TORCH_COMPILE_MODE`** env var overrides the YAML at run time: `TORCH_COMPILE_MODE=reduce-overhead python3 training/pretrain.py ...`.
- **`--no-compile`** — skip entirely. On the CPU/Mac dev box, compilation buys nothing and costs minutes: **always pass `--no-compile` for smoke tests and dry-runs**.

Compile only affects the FLOP-utilization ledger, never the FLOP count — its effect shows in tps, not loss.

## 8. Triton dispatch

The one sanctioned Triton kernel fuses the per-chunk SSD work (L construction, the two complex GEMMs, the state update) into one program per (batch, chunk, head) — see [R3 — SSD Triton](../reference/03-ssd-triton.md). Enabling it is a **two-part opt-in**:

1. `ssd_dispatch: "triton"` under `model:` in the YAML (a `ModelConfig` field), **and**
2. `ENABLE_TRITON_KERNELS=1` in the environment — otherwise `training/pretrain.py:_enforce_triton_env_var` force-rewrites the config back to `"pytorch"` with a single `[warn]` line, no error.

```bash
ENABLE_TRITON_KERNELS=1 python3 training/pretrain.py --config configs/pretrain_a100_400m.yaml
```

Outside the harness (notebooks, tests), a triton-less box prints one warning per block on the first forward and runs PyTorch — watch for the warn line if you set `triton` and see no speedup.

**Env knobs:** `TRITON_PER_CHUNK_NUM_STAGES` (default 1) and `TRITON_PER_CHUNK_NUM_WARPS` (default 4), read in `_per_chunk_ssd_triton_forward`; both are per-launch tuning, `[INFERENCE]` without a benchmark.

**When to prefer `pytorch`:** small models (per-(B, c, H) launch overhead dominates), debugging (the backward of `_PerChunkSSDTriton` recomputes `models/ssd_triton.py:per_chunk_ssd_pytorch`, so gradients are easier to follow on the eager path), dims above the 256-cap or non-power-of-two, and any box without Triton. The kernel's backward is exact but reference-speed — measure end-to-end before assuming the step is faster.

## 9. Data side

Three facts shape any tuning run:

1. **No shuffling.** `training/pretrain.py:Pretrainer.train` builds `DataLoader(dataset, batch_size=..., num_workers=0, drop_last=True)` with no `shuffle=True`: every epoch scans windows in the same deterministic order (see [R8 — dataset](../reference/08-dataset.md)). Comparisons are *not* confounded by data order — but curves reflect one fixed order.
2. **The data mix is fixed upstream.** The YAML's `data.data_mix: "mamba2-default"` field is documentation for the pipeline, not a knob `pretrain.py` reads (it reads only `train_data_path`): fineweb-edu 0.50 / fineweb 0.20 / the-stack-python 0.15 / openmath-instruct-2 0.10 / arxiv 0.05 — see [R11 — data pipeline](../reference/11-data-pipeline.md). A/B arms must share the same data path or loss comparisons are meaningless.
3. **EOS-separated shards caveat.** The workspace pipeline's raw on-disk format (EOS-separated `uint32` records, per `LLM/shared_data/`) is **not** directly loadable — it must be converted to packed `torch.long` tensors (`tests/e2e_gpu_smoke.py:_build_synthetic_shard` shows the format). Once packed, EOS id 50,256 is just another token: no document handling, windows overlap by one token, every position of `y` is trained on.

## 10. A/B test protocol

The protocol that decides between two settings:

1. **Fix one variable.** Change exactly one field between arm A and arm B; hold data, seed, schedule, and all other knobs identical.
2. **Scope the run.** Copy the production YAML to a scratch config: `total_steps: 3000`, `save_dir: "checkpoints/ab_chunk64"`, `log_interval: 50`, `save_interval: 100000`. Sweeps run with `--no-compile` so compile time doesn't pollute tps — or with compile on *for both arms*, never one.
3. **Warm up, then measure.** Discard the first ~500 steps' tps (compile, cudnn benchmark, and autotune settle in); record loss@2000 and the median of the last 20 tps rows.
4. **Record.** One row per arm: variable, value, loss@N, median tps, notes.

**Worked example — chunk_size 64 vs 128.** Arm A is the shipped config:

```bash
# wiring check (CPU box): 2 steps, no compile
python3 training/pretrain.py --config configs/pretrain_a100_400m.yaml --dry-run --no-compile
```

Copy the YAML to `configs/ab_chunk128.yaml`, edit `chunk_size: 128` (model section) plus the scratch fields above, then on the A100:

```bash
python3 training/pretrain.py --config configs/pretrain_a100_400m.yaml --no-compile   # arm A (C=64)
python3 training/pretrain.py --config configs/ab_chunk128.yaml --no-compile         # arm B (C=128)
```

Expected `[INFERENCE]`: B shows +5–10% tps (bigger intra-chunk GEMMs, half the chunk hops) at the cost of 2× the L materialization (512 MiB/layer vs 256 MiB — irrelevant with grad checkpointing, decisive without). Loss at step 2000 should match within noise — the chunked algorithm computes the *same function* as the scan (pinned by `tests/test_ssd.py::test_chunkwise_matches_naive_complex`), so chunk_size is pure throughput/memory, never a loss lever. If B's loss diverges, something else changed. The same template applies to `lr` (loss curves), `micro_batch_size` × accumulation (tps and VRAM), and `compile_mode` (tps after warmup).

## 11. Pitfalls

1. **`--dry-run` with compile enabled measures compilation, not training** — it forces `max_steps=2` (`training/pretrain.py:main`) and the first steps include autotune. Use `--dry-run --no-compile` for wiring checks.
2. **`ssd_dispatch='triton'` can silently mean `'pytorch'`.** Missing `ENABLE_TRITON_KERNELS=1` force-backs with one warn line; a triton-less box falls back per block. Check the warn line, not the config.
3. **chunk_size C > 256 or non-power-of-two silently degrades triton runs** to per-block PyTorch fallback (`_check_block_dims` raises, the block catches). Still correct, just not what you configured.
4. **All throughput numbers here are `[INFERENCE]`.** No `.benchmarks/` exists; SKILLS.md's +5–10% / +10–15% figures and this doc's table are estimates. Your own tps rows are the only measurements.
5. **The data-mix YAML field is inert** — changing `data_mix` changes nothing in `pretrain.py`; only `train_data_path` matters there.
6. **grad_checkpoint is all-or-nothing per block**; a "checkpoint every 4th layer" plan is a code change, not a config.

## 12. Tests that pin this doc's claims

- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` — chunked == naive scan, so chunk_size cannot change the loss function.
- `tests/test_ssd_triton.py::TestPerChunkSsdImportSurface::test_check_block_dims_raises_value_error_on_too_large_dim` — the 256-cap.
- `tests/test_ssd_triton.py::TestEnableTritonKernelsForceBack::test_triton_dispatch_forced_back_when_env_var_missing` / `::test_triton_dispatch_passes_through_when_env_var_set` — the two-part opt-in.
- `tests/test_train_step.py::test_train_step_on_tiny_model` — the accumulation/clip/schedule plumbing on CPU.
- `tests/e2e_gpu_smoke.py` (CUDA + triton) — check 7 exercises a full `Pretrainer` dry-run with triton dispatch.

Related: [R1 — ModelConfig](../reference/01-model-config.md), [R7 — pretrain CLI](../reference/07-pretrain-cli.md), [R10 — logging](../reference/10-logging.md), [T8 — scaling/efficiency](../theory/08-scaling-efficiency.md) (the derivations behind every `[INFERENCE]` here), [G2 — training runbook](02-training-runbook.md) (the full A100 launch), [G1 — quickstart](01-quickstart.md).

## 13. Anchors cited

- `models/transformer.py:ModelConfig`
- `models/mamba_block.py:Mamba3Block.forward`
- `models/mamba_block.py:Mamba3Block.chunk_size`
- `models/ssd_complex.py:ssd_complex_chunkwise`
- `models/ssd_triton.py:_check_block_dims`
- `models/ssd_triton.py:per_chunk_ssd_pytorch`
- `training/pretrain.py:TrainingConfig`
- `training/pretrain.py:Pretrainer.__init__`
- `training/pretrain.py:Pretrainer.train`
- `training/pretrain.py:train_step`
- `training/pretrain.py:_enforce_triton_env_var`
- `training/pretrain.py:main`
- `utils/logging.py:TrainingLogger.log`
