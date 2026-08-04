# Mamba-3-Lite — A100 Training Runbook — launch, monitor, recover

Operational guide for launching, monitoring, and recovering a real pre-training run of the ~434M-parameter Mamba-3-Lite model on a single A100 80 GB box.

## 1. 60-second summary

After reading this you can run the full pre-training job end to end: verify the box, decide Triton vs PyTorch SSD dispatch, pass the 8-check GPU smoke test, launch `training/pretrain.py` under `nohup`/`tmux` with WandB tracking, read the `step | loss | ppl | lr | tps` heartbeat, survive NaN excursions via the built-in rollback, resume after a crash, and shut down without corrupting a checkpoint. Two facts bite everyone: the NaN guard rewinds you to the **latest complete checkpoint** (not your explicit `--resume` step), and checkpoint "atomicity" is only crash-*tolerance* — a Ctrl-C mid-save can tear a triple, which is silently skipped on resume.

## 2. Before you start: the A100 pre-flight checklist

All commands below are GPU/A100-specific; they are marked **[GPU-only]**.

### 2.1 CUDA / torch / GPU sanity — [GPU-only]

```bash
nvidia-smi                       # driver, ECC, free VRAM, no zombie processes
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import triton; print(triton.__version__)"   # optional, only if you plan Triton dispatch
```

Expect a CUDA torch build, `True`, and `NVIDIA A100-SXM4-80GB`. `triton` imports only on Linux+CUDA — on this Mac dev box it will not, which is fine ([Mamba-3-Lite — SSD Reference](../references/ssd-reference.md)). Also confirm the token shards exist: the run reads `data.train_data_path` (default `data/pretrain_chinchilla`, a directory of `shard_*.bin` tensors produced by the workspace pipeline through `data/prepare_data.py` against `LLM/shared_data/`). If that path is missing, `training/pretrain.py:PretrainDataset` silently falls back to **dummy random tokens** with a warning — your loss would decay on noise. Verify with `ls data/pretrain_chinchilla`.

### 2.2 The `ENABLE_TRITON_KERNELS` decision

Dispatch is configured by the `ssd_dispatch` key (`"pytorch"` default, `models/transformer.py:ModelConfig.ssd_dispatch`; the shipped yaml does not set it). The Triton path additionally requires the environment gate: `training/pretrain.py:_enforce_triton_env_var` force-backs `ssd_dispatch='triton'` to `"pytorch"` — with a warning — unless `ENABLE_TRITON_KERNELS=1`. So:

- **PyTorch dispatch is the safe default.** No env var, no kernel compile, works everywhere. Correctness is identical (see the parity check in 2.3); you only give up the fused kernel's throughput ([Mamba-3-Lite — SSD Reference](../references/ssd-reference.md), [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md)).
- **Triton dispatch requires both** `ssd_dispatch: triton` in the `model:` section of the yaml you launch *and* `ENABLE_TRITON_KERNELS=1`. Get this backwards and you get a silent force-back, not an error.
- Start a first A100 run with PyTorch dispatch; add Triton after the pipeline is proven (Section 9).

### 2.3 The 8-check e2e GPU smoke — [GPU-only]

```bash
ENABLE_TRITON_KERNELS=1 python3 tests/e2e_gpu_smoke.py
```

This is the whole box, exercised before you burn 12+ hours. Each check proves one layer of the stack:

| # | Check (`tests/e2e_gpu_smoke.py`) | Proves |
|---|---|---|
| 1 | `check_environment` | torch sees CUDA; device name/capability/memory print; triton importable; `ENABLE_TRITON_KERNELS=1` visible |
| 2 | `check_data_pipeline` | `shard_*.bin` → `PretrainDataset` → DataLoader → GPU tensors: window shapes `(32,)` in / `(32,)` target, batch `(4, 32)` on CUDA |
| 3 | `check_model_pytorch` | full model forward on GPU, PyTorch dispatch: output `(2, 16, 128)`, finite logits, param count prints |
| 4 | `check_model_triton` | Triton forward: every block uses the kernel, **no** fallback block (asserted) |
| 5 | `check_triton_vs_pytorch` | dispatch parity: Triton vs PyTorch vs naive O(T) scan all within `1e-3` max-abs |
| 6 | `check_training_step` | 4 real BF16-autocast steps via `training/pretrain.py:train_step`: finite, decreasing losses; prints peak VRAM (`torch.cuda.max_memory_allocated()`) |
| 7 | `check_checkpoint` | `utils/checkpoint.py:CheckpointManager` save/load round-trip on a fresh tied model, `strict=False`, drift `< 1e-4` |
| 8 | `check_pretrainer_dry_run` | the full `Pretrainer` loop (2 micro-batches, Triton) saves a discoverable checkpoint |

Passing output ends with `E2E SMOKE: ALL CHECKS PASSED`. On a CPU box this test cannot run (check 1 asserts CUDA); the CPU suite is `python3 -m pytest tests/ -q` → 37 tests collected (32 passed, 5 GPU-skipped).

## 3. Launching the run

The canonical command ([Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md)):

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
export WANDB_PROJECT=mamba3-lite            # optional: enables WandB (Section 10.3)
export WANDB_RUN_NAME=a100-run-01           # optional: names the WandB run
export ENABLE_TRITON_KERNELS=1              # only if ssd_dispatch: triton is in your yaml
nohup python3 training/pretrain.py --config configs/pretrain_a100_400m.yaml \
      > run.log 2>&1 &
```

`training/pretrain.py:main` reads the yaml, folds `model:`/`training:`/`data:` into a `TrainingConfig`, and builds the 434M-parameter `Mamba3Transformer` (logged: `Parameters: 433,662,400 total`), AdamW with decay/no-decay groups, `SequentialLR` warmup→cosine, and `torch.compile` (mode `max-autotune`). `nohup` + `run.log` (or `tmux`/`screen`) survives SSH drops; `tail -f run.log` is your window into the run. Everything is single-process and synchronous — no daemon, no worker pool — so `nohup` alone suffices.

- **First launch:** run once with `--dry-run` (caps `max_steps=2` — one optimizer step at the default accumulation of 2) as the fastest wiring check short of the GPU smoke.
- **Data note:** shards elsewhere? `--data-path /path/to/shard_dir`; checkpoints go to `training.save_dir` (`checkpoints/pretrain_a100`) or `--checkpoint-dir`.
- **WandB:** `WANDB_PROJECT` triggers `wandb.init` inside `utils/logging.py:TrainingLogger.__init__` at trainer construction — a broken auth aborts at startup, not mid-run (Section 10.3).

## 4. What you watch

### 4.1 The log line

`training/pretrain.py:Pretrainer.train` calls `utils/logging.py:TrainingLogger.log` every `log_every` (50) steps, and the logger prints one line per flush. Sample (illustrative):

```
step=  4000 | loss=6.8341 | ppl=927.43 | lr=2.84e-04 | tps=187,432
```

Each field, from `TrainingLogger.log`:
- **`step`** — the global optimizer step. One optimizer step = `gradient_accumulation_steps` (2) micro-batches of `batch_size` (16) windows of `max_seq_len` (2048) tokens: $16 \times 2 \times 2048 = 65{,}536$ tokens/step. This is also the WandB x-axis, stable across resumes.
- **`loss`** — the window-averaged cross-entropy, `ignore_index=-100`. Subtleties ([Mamba-3-Lite — Training Reference](../references/training-reference.md)): the window holds exactly one loss (caller and logger share the `% log_every` gate), and it is the per-micro-batch CE — `training/pretrain.py:train_step` logs `ce_loss_val` **before** dividing by the accumulation count.
- **`ppl`** — $\exp(\text{avg\_loss})$: exp after averaging (Jensen: ≤ the mean of per-step perplexities; with window size 1 they coincide).
- **`lr`** — `scheduler.get_last_lr()[0]`: linear warmup from $0.01 \times \text{lr}$ over 2,000 steps, then cosine to $\text{lr} \times 0.05 = 1.5\times10^{-5}$ at step 256,000 ([Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md)).
- **`tps`** — tokens/sec: $\text{tps} = \dfrac{\text{log\_every} \times \text{seq\_len} \times \text{batch\_size}}{\text{elapsed}} = \dfrac{50 \times 2048 \times 16}{T_{\text{win}}}$. One window is 1,638,400 tokens; `tps` catches stalls and regressions. The step-0 line's `elapsed` includes startup — discard it.

### 4.2 Expected loss trajectory — [INFERENCE]

A uniform-initialized model predicts $\ln(\text{vocab}) = \ln(50{,}257) \approx 10.82$ — expect the first logged loss near that. From there: a fast drop through warmup (steps 0–2,000) as the LR climbs to $3\times10^{-4}$, then a long, progressively flattening cosine decay. There are no benchmarks in this tree, so any *specific* end-of-run number is an estimate, not a measurement: at 65,536 tokens/step the run consumes $256{,}000 \times 65{,}536 \approx 16.8\text{B}$ tokens — twice the 8.0B "Chinchilla-optimal" figure in the yaml header comment (which matches `micro_batch_size=8`; the comment is stale, the code arithmetic is authoritative, [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md)). A plausible end-state is loss well below 5 / ppl in the low tens — `[INFERENCE]`. Watch the *shape* instead: monotone-ish decay, no high-loss plateau, no NaN excursions. If loss barely moves after 10k steps, something structural is wrong (dummy data — see 2.1 — or a broken schedule, e.g. `warmup_steps=0`).

### 4.3 tps sanity

Anchor number: **65,536 tokens per optimizer step**. A 50-step window is 1,638,400 tokens, so $\text{tps} = 1{,}638{,}400 / T_{\text{win}}$. The yaml's throughput hint (12–15 h for 256k steps) implies roughly $16.8\text{B} / (43{,}200\text{–}54{,}000\,\text{s}) \approx 310\text{–}390\text{k}$ tps steady-state — `[INFERENCE]`, no benchmark backs it, assuming Triton dispatch + `torch.compile` + gradient checkpointing. Use your smoke-run `tps` as the baseline; a stable figure within ~20% of it is healthy. A 10× collapse means CPU fallback (see 2.2), a hung compile, or data-loading stalls.

## 5. NaN recovery

Two layers, both in `training/pretrain.py`:

**Per-step detection** — `train_step` computes the loss inside autocast, then:

```python
if config.nan_guard and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
    log(f"[nan-guard] NaN/Inf at micro_step={micro_step}, opt_steps={opt_steps}. Skipping backward.")
    optimizer.zero_grad(set_to_none=True)
    return None, opt_steps
```

A NaN/Inf loss skips backward, zeroes grads, and returns `None` — the step does not advance `global_step` and nothing is logged.

**Streak rollback** — `Pretrainer.train` counts consecutive `None`s in `nan_guard_streak`. At `nan_guard_max_consecutive = 5`:

```python
latest = self._find_latest_checkpoint()
if latest is not None:
    self._log(f"[nan-guard] {nan_guard_streak} consecutive NaN/Inf — restoring checkpoint step {latest}.")
    global_step = self.load_checkpoint(latest)
else:
    self._log("[nan-guard] No checkpoint to restore from. Aborting.")
    raise RuntimeError("NaN/Inf with no checkpoint to restore from")
nan_guard_streak = 0
```

**What 5 consecutive NaNs mean:** one-off spikes are absorbed silently (skip backward, keep going); five in a row is a *systemic* excursion — a divergent complex recurrence, an unstable LR, a corrupted shard, or a dtype blowup. The guard rewinds: `training/pretrain.py:Pretrainer.load_checkpoint` restores weights, optimizer moments, scheduler state, and `_opt_steps` from the **latest complete checkpoint**, and the loop continues from that step — at most ~4,000 steps lost (one `save_every` interval). The LR resumes at the checkpointed position; no warmup replay. If no checkpoint exists yet, the guard raises `RuntimeError` rather than training on garbage ([Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) for the rationale). Repeated NaN-rollback loops mean stop and investigate — the guard heals the symptom, not the disease.

## 6. Resuming

The CLI flag is `--resume <step>` (not `--resume-from`):

```bash
python3 training/pretrain.py --config configs/pretrain_a100_400m.yaml --resume 120000
```

`training/pretrain.py:main` calls `Pretrainer.load_checkpoint(120000)` — weights, AdamW moments, scheduler, and the optimizer-step counter all come back from the triple at step 120,000.

**The pitfall (restated from [Mamba-3-Lite — Training Reference](../references/training-reference.md)):** `train()` then *unconditionally* auto-discovers the latest complete checkpoint and loads it:

```python
latest = self._find_latest_checkpoint()
if latest is not None:
    try:
        global_step = self.load_checkpoint(latest)
    except Exception as exc:
        self._log(f"[warn] Could not load checkpoint: {exc}")
```

So if the directory's newest complete triple is step 180,000, an explicit `--resume 120000` is **overridden** — you get 180,000. Auto-resume is a feature (a plain relaunch continues the run, no flags) and a trap for selective resumption. To resume an older step, point `--checkpoint-dir` at a directory whose latest complete checkpoint *is* that step (e.g. an archive copy). Also note `--no-checkpoint` disables **gradient** checkpointing, not disk checkpoints.

## 7. Monitoring VRAM

The A100 80 GB has enormous headroom for 434M parameters. Derived budget (no benchmarks): float32 weights $433.7\text{M} \times 4\text{B} \approx 1.7\text{GB}$ (autocast does not change parameter storage), AdamW moments ≈3.5 GB, gradients ≈1.7 GB, plus activations — the term gradient checkpointing shrinks dramatically (recompute for memory; the config ships `grad_checkpoint: true`, [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md)). Realistic peak is well under 15 GB `[DERIVED]`. Watch `nvidia-smi` and, in code, `torch.cuda.max_memory_allocated()` — the e2e smoke's check 6 prints exactly this (`peak VRAM: … MB`).

**OOM handling, in order of preference:**

1. **Reduce `micro_batch_size`** (16 → 8) and, to keep the same global batch, double `gradient_accumulation_steps` (2 → 4). Tokens/step stays 65,536; only the micro-batch footprint shrinks.
2. Reduce `max_seq_len` only if you accept a different run configuration.
3. **Never "fix" OOM by disabling gradient checkpointing** — that *increases* activation memory. Keep `grad_checkpoint: true`; it is your memory safety net. Consider `--no-checkpoint` only after measuring headroom and wanting the speed.

## 8. Shutting down cleanly

The loop saves synchronously every `save_every = 4000` steps (plus a `tag="final"` checkpoint at the end, via `training/pretrain.py:Pretrainer.save_checkpoint`). To stop: wait for a `Checkpoint saved at step N` line, then Ctrl-C (or `kill` the pid). The loop is single-threaded; nothing else is mid-flight.

**The atomicity caveat:** `utils/checkpoint.py:CheckpointManager.save` writes the three files **directly** to their final paths — `model_step_N.safetensors`, `optim_step_N.pt`, then `meta_step_N.json` — there is no `tmp + os.rename` sequence (the docstrings say "atomic", the code is not, [Mamba-3-Lite — Training Reference](../references/training-reference.md)). A Ctrl-C between the writes leaves a partial triple. Recovery is by design: `utils/checkpoint.py:CheckpointManager.latest_step` only returns steps where `CheckpointManager._checkpoint_complete` sees all three files, so a torn step is invisible and the next launch resumes the previous complete one.

**Verify a triple** before relying on it:

```bash
ls checkpoints/pretrain_a100/ | tail -n 12     # model_step_N.safetensors, optim_step_N.pt, meta_step_N.json for the same N
```

All three must exist for step N. If only two do, that step is dead weight — delete it and confirm the previous triple is intact.

## 9. Triton dispatch on the A100

To switch the real run to the fused kernel: add `ssd_dispatch: triton` to the `model:` section of the yaml you launch, and launch with `ENABLE_TRITON_KERNELS=1` (see 2.2 — without it, `_enforce_triton_env_var` force-backs to pytorch with a warning; with it, nothing changes unless the yaml also asks for triton). The smoke test's check 4 already asserted every block uses the kernel.

Facts that matter ([Mamba-3-Lite — SSD Reference](../references/ssd-reference.md)):

- **The 256-cap:** the kernel's constexpr block sizes `BLOCK_C/P/N` are capped at 256 (`models/ssd_triton.py:_MAX_BLOCK`). The shipped config (chunk 64, D 64, N 64) is far under it; a config exceeding it raises a `ValueError` telling you to use `ssd_dispatch='pytorch'`.
- **Fallback warning:** if the kernel path raises for a block, `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` prints a **one-shot** per-block warning — `[Mamba3Block {i}] ssd_dispatch='triton' unavailable (…); falling back to 'pytorch' for this block.` — and continues. Training still works, but that block is on the slow path.
- **When the pytorch path is fine:** always, correctness-wise — check 5 keeps the two dispatches within `1e-3`, and pytorch is the shipped default. Use pytorch for debugging, CPU work, config experiments, and small runs; reserve Triton for throughput-hunting on the A100/H100. Env knobs if you tune the kernel: `TRITON_PER_CHUNK_NUM_STAGES` (default 1) and `TRITON_PER_CHUNK_NUM_WARPS` (default 4; the smoke test pins 2).

## 10. Troubleshooting

1. **Compile hangs / very slow first step.** `torch.compile(mode="max-autotune")` plus the first Triton kernel compilation happens on the first forward — minutes of apparent silence is normal once. Beyond ~30 minutes: `TORCH_COMPILE_MODE=reduce-overhead` (or `default`) overrides the yaml's `compile_mode`, or relaunch with `--no-compile`. A stale/corrupt triton cache (`~/.triton/cache`) can also wedge compilation — clear it and retry.
2. **TF32 nondeterminism.** On CUDA, `training/pretrain.py:Pretrainer.__init__` enables matmul TF32, cuDNN TF32, and `torch.set_float32_matmul_precision("high")`. TF32 drops matmul mantissa bits: two runs differ slightly, and results differ from a strict-FP32 run. Expected and acceptable for pre-training (and why the parity check's tolerance is `1e-3`, not `0`). For bit-exact debugging only: set both TF32 flags to `False` and precision to `"highest"` — this changes the shipped config, so revert after debugging.
3. **WandB auth failures.** `WANDB_PROJECT` set → `wandb.init` runs at logger construction, and only `ImportError` is caught — a bad API key or no network **raises and aborts at startup**. Fixes, in order: `wandb login` on the box first; or launch without `WANDB_PROJECT` (console logging still works); or `WANDB_MODE=offline WANDB_DISABLED=true` (what the smoke test does) to defer syncing.
4. **Loss stuck near $\ln(\text{vocab})$.** Check for the dummy-data fallback (2.1) and `warmup_steps ≥ 1` (`warmup_steps=0` silently trains at ~1–5% of the intended LR, [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md)).
5. **Repeated NaN rollbacks** (see 5): suspect a corrupt shard (re-run `data/prepare_data.py`), a too-high LR after a config edit, or an unstable `chunk_size`/`state_dim` change; the guard keeps healing, but the run stalls.

## What the tests verify

- `tests/e2e_gpu_smoke.py` — the 8 checks of Section 2.3, the only place the whole A100 stack is exercised; GPU-only (5 skipped on CPU).
- `tests/test_train_step.py::test_train_step_on_tiny_model` — `training/pretrain.py:train_step` direct: finite loss, parameters changed, NaN-guard `None` path.
- `tests/test_ssd_triton.py::TestEnableTritonKernelsForceBack` — the `_enforce_triton_env_var` contract: force-back when `ENABLE_TRITON_KERNELS` is unset, pass-through when `"1"`.
- `tests/test_doc_refs.py` — the machine checker validating every `file.py:Symbol` anchor in this and every other doc, including the ones above.

Related reading: config mechanics in [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md); checkpoint format and the resume quirk in [Mamba-3-Lite — Training Reference](../references/training-reference.md); the logger's exact math in [Mamba-3-Lite — Training Reference](../references/training-reference.md); NaN-guard and dtype rationale in [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md); throughput economics in [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md). First-timer end-to-end walkthrough: [Mamba-3-Lite — Quickstart](quickstart.md); knob-by-knob effects: [Mamba-3-Lite — Tuning Guide](tuning.md).

## References

- [Mamba-3-Lite — Pretrain CLI](pretrain-cli.md) — config mechanics behind every command in this runbook.
- [Mamba-3-Lite — Training Reference](../references/training-reference.md) — checkpoint format, the resume quirk, and the logger's exact math.
- [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) — NaN-guard and dtype rationale.
- [Mamba-3-Lite — SSD Reference](../references/ssd-reference.md) — the Triton kernel contract and the 256-cap.
- [Mamba-3-Lite — Quickstart](quickstart.md) — first-timer end-to-end walkthrough.
- [Mamba-3-Lite — Tuning Guide](tuning.md) — knob-by-knob effects.
