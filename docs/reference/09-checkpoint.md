# Checkpointing — `CheckpointManager` and the 3-File Format

Reference doc for `utils/checkpoint.py:CheckpointManager`: how Mamba-3-Lite persists model weights, optimizer state, and training metadata, and how the training loop discovers and restores them.

## 60-second summary

Every saved step writes **three files** into the checkpoint directory: `model_step_N.safetensors` (weights, via `safetensors.torch.save_file`), `optim_step_N.pt` (optimizer state, via `torch.save`), and `meta_step_N.json` (step, LR-scheduler state, optimizer-step count, tag, and the full `TrainingConfig`). `utils/checkpoint.py:CheckpointManager.save` first walks the model state dict and materializes **aliased tensors into independent copies** by tracking `data_ptr()` — this is mandatory, not an optimization: `safetensors.save_file` refuses dicts whose entries share storage, and Mamba-3-Lite's weight tying makes `lm_head.weight` the *same tensor* as `embed.weight`. `utils/checkpoint.py:CheckpointManager.load` restores weights with `load_state_dict(strict=False)` (reporting missing/unexpected keys), loads optimizer state only if the file exists, and returns the meta dict. `utils/checkpoint.py:CheckpointManager.latest_step` returns the highest step for which **all three** files exist, which is also the mechanism `training/pretrain.py:Pretrainer.train` uses to auto-resume. **Honesty note:** the module docstring says "Atomic safetensors checkpoint manager", but the code writes each file *directly* — there is no `tmp → os.rename` anywhere. Crash safety comes from the write order (weights → optimizer → meta) plus the all-three-files completeness check, not from atomic rename.

## Why it exists

Pre-training runs for 256,000 steps and is interrupted routinely (preemption, OOM, NaN excursions). The checkpoint layer has three jobs:

1. **Resume**: restart from the last good step without losing optimizer momentum or LR-schedule position — momentum alone is often worth more than the weights themselves for the first few hundred steps after a restart.
2. **Rollback**: the NaN guard in `training/pretrain.py:train_step` skips backward on NaN/Inf losses; after `nan_guard_max_consecutive = 5` consecutive bad micro-steps, `training/pretrain.py:Pretrainer.train` restores the latest checkpoint and continues from that step instead of aborting.
3. **Weight-tying correctness**: because `models/transformer.py:Mamba3Transformer` sets `self.lm_head.weight = self.embed.weight` (`weight_tying=True`), a naive `save_file` of `model.state_dict()` would feed safetensors two dict entries sharing one storage — which `save_file` rejects with `RuntimeError: Some tensors share memory`. The data_ptr dedup pass exists to make saving *possible*.

## Intuition

Think of a checkpoint as a **transaction with three ledger entries** that must all be present to count. The weights are the product, the optimizer `.pt` is the momentum ledger, and the JSON is the receipt (step number, schedule position, config snapshot). Because a crash can interrupt the transaction at any point, the manager never trusts a single file: a step "exists" only when all three files are on disk, and discovery always picks the *largest complete* step.

The dedup pass is easiest to misread. "Dedup" here does **not** mean "store the tied weight once and save space" — the saved file contains **two independent copies** of the 51.46M-parameter embedding/head tensor (the second alias is cloned). What the pass guarantees is that no two entries in the dict handed to `save_file` share storage, which is a hard requirement of the safetensors format. Measured on the dev box (2026-08-04): passing an aliased dict raises; passing the deduped dict saves fine, with both copies present in the file.

## The 3-file format

| File | Writer | Contents | Size (full 434M model, float32) |
|---|---|---|---|
| `model_step_N.safetensors` | `safetensors.torch.save_file` | every `state_dict()` entry, deduped to independent storage | ~1.94 GB [DERIVED: 433,662,400 + 51,463,168 (tied pair counted twice) entries × 4 B] |
| `optim_step_N.pt` | `torch.save` | `optimizer.state_dict()` — per-parameter `exp_avg` / `exp_avg_sq` plus the two param-group configs | depends on optimizer moments |
| `meta_step_N.json` | `json.dump(..., default=str)` | `{"step": N, "scheduler": {...}, "opt_steps": ..., "tag": ..., "config": {...}}` | a few KB |

Note the **asymmetry of formats**: weights are safetensors (self-describing, memory-mappable, no pickle), optimizer state is `torch.save` (pickle-based but loaded with `weights_only=True`), and metadata is plain JSON. Only the model file participates in step discovery (`_list_steps` globs `model_step_*`); the other two are companions that gate completeness.

## Code walkthrough

### `utils/checkpoint.py:CheckpointManager.save` — write the triple

```python
def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
         extra_meta: Optional[dict] = None, state_dict: Optional[dict] = None) -> None:
    state = state_dict if state_dict is not None else model.state_dict()
    seen_ptrs: set = set()
    deduped: dict = {}
    for k, v in state.items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            deduped[k] = v.contiguous().clone()
        else:
            seen_ptrs.add(ptr)
            deduped[k] = v.contiguous()
    save_file(deduped, self.save_dir / f"model_step_{step}.safetensors")
    torch.save(optimizer.state_dict(), self.save_dir / f"optim_step_{step}.pt")
    meta: dict = {"step": step}
    if extra_meta:
        meta.update({k: v for k, v in extra_meta.items() if k != "step"})
    self._write_json(self.save_dir / f"meta_step_{step}.json", meta)
```

Semantics, in order:

1. **Dedup pass** — iterate `state_dict()` in key order; the first tensor at a given `data_ptr()` is kept as `v.contiguous()` (a no-op view for an already-contiguous tensor, so it *retains the original storage*), every later key sharing that pointer becomes `v.contiguous().clone()`. For the tied pair, `embed.weight` (registered first) keeps the live storage and `lm_head.weight` is cloned, so the two file entries are byte-identical but storage-independent. The pass also quietly fixes non-contiguous views by materializing them.
2. **Weights** — `save_file(deduped, ...)` writes directly to the final path (see Honesty check).
3. **Optimizer** — `optimizer.state_dict()` is pickled with `torch.save`. For the tied pair this contains **one** moment pair, because `training/pretrain.py:Pretrainer.__init__` builds its optimizer from a deduplicated parameter list (`id(p)` set), so `lm_head.weight` never appears twice.
4. **Meta** — the caller's `step` key is filtered out (`k != "step"`); the manager's `step` argument always wins. `_write_json` serializes with `indent=2` and `default=str` (a stringification fallback for non-JSON values — see Pitfalls).

`training/pretrain.py:Pretrainer.save_checkpoint` is the production caller:

```python
def save_checkpoint(self, step: int, tag: str = "") -> None:
    state = self.raw_model.state_dict()
    extra_meta = {"scheduler": self.scheduler.state_dict(), "opt_steps": self._opt_steps,
                  "tag": tag or f"step_{step}", "config": asdict(self.config)}
    self.ckpt_manager.save(self.raw_model, self.optimizer, step, extra_meta=extra_meta, state_dict=state)
```

It passes `self.raw_model` (the uncompiled module) rather than `self.model`, so `torch.compile` wrapper parameters never leak into the file, and it records the `SequentialLR` scheduler state, the optimizer-step counter, a human tag, and the full `TrainingConfig` snapshot (which is what makes the JSON a self-contained run record).

### `utils/checkpoint.py:CheckpointManager.load` — restore the triple

```python
def load(self, model: torch.nn.Module, step: int, device: str = "cuda",
         optimizer: Optional[torch.optim.Optimizer] = None, strict: bool = True) -> dict:
    weight_path = self.save_dir / f"model_step_{step}.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}\nAvailable steps: {self._list_steps()}")
    weights = load_file(str(weight_path), device=device)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        msg = f"[checkpoint] {len(missing)} missing key(s): {missing[:5]}{'…' if len(missing) > 5 else ''}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    ...
    if optimizer is not None:
        optim_path = self.save_dir / f"optim_step_{step}.pt"
        if optim_path.exists():
            optimizer.load_state_dict(torch.load(optim_path, map_location=device, weights_only=True))
        else:
            logger.warning("[checkpoint] no optimiser state at %s — optimizer will start from scratch", optim_path)
    meta_path = self.save_dir / f"meta_step_{step}.json"
    meta: dict = json.load(open(meta_path)) if meta_path.exists() else {"step": step}
    return meta
```

Key behaviors:

- **Weights load with `strict=False` always**; the `strict` parameter only controls whether a mismatch *raises* or *warns*. With `strict=True` (the manager default) any missing/unexpected key is a `RuntimeError`; `training/pretrain.py:Pretrainer.load_checkpoint` passes `strict=False`, so a config change that alters the module list degrades to warnings instead of a crash.
- **Weight tying survives the round trip.** `load_state_dict` is called with the default `assign=False`, which copies values *in place* into the existing parameters via `param.copy_(...)`. Because the live model still has `lm_head.weight is embed.weight` (one object, one storage), the two file copies (equal values) are copied into the same storage twice — the model stays tied and consistent. Verified empirically: a freshly constructed tied model with corrupted weights loaded from a deduped file ends with `lm_head.weight is embed.weight` and equal values. If the code ever switched to `assign=True`, the second key would replace the shared parameter with a fresh unaliased tensor and silently break tying — it does not.
- **Optimizer state is optional**: if `optimizer is None` or the `.pt` file is missing, loading proceeds and logs a warning ("optimizer will start from scratch"). `weights_only=True` blocks arbitrary pickle gadgets.
- **The return value is the meta dict** — `{"step": N, "scheduler": ..., "opt_steps": ..., "tag": ..., "config": ...}` — with `{"step": step}` as the fallback when the JSON is absent.

`training/pretrain.py:Pretrainer.load_checkpoint` turns the meta dict back into training state:

```python
def load_checkpoint(self, step: int) -> int:
    meta = self.ckpt_manager.load(self.raw_model, step, device=str(self.device), optimizer=self.optimizer, strict=False)
    if "scheduler" in meta:
        self.scheduler.load_state_dict(meta["scheduler"])
    if "opt_steps" in meta:
        self._opt_steps = meta["opt_steps"]
    resumed_step = meta.get("step", step)
    return resumed_step
```

The LR schedule resumes exactly where it left off (`SequentialLR` state carries the epoch counters and base LRs), the optimizer-step counter is restored, and the returned step becomes the loop's `global_step`.

### `utils/checkpoint.py:CheckpointManager.latest_step` — completeness-gated discovery

```python
def latest_step(self) -> Optional[int]:
    steps = self._list_steps()
    return next((s for s in sorted(steps, reverse=True) if self._checkpoint_complete(s)), None)

def _list_steps(self) -> list:
    return [int(p.stem.removeprefix("model_step_"))
            for p in self.save_dir.glob("model_step_[0-9]*.safetensors")]

def _checkpoint_complete(self, step: int) -> bool:
    return all((self.save_dir / n).exists() for n in [
        f"model_step_{step}.safetensors", f"optim_step_{step}.pt", f"meta_step_{step}.json"])
```

Steps are parsed from the weight-file names only (`model_step_[0-9]*.safetensors` — non-numeric suffixes never match), sorted descending, and the first step whose three companion files all exist wins. A step whose write was interrupted is simply invisible to discovery.

## Honesty check: "Atomic" is not what the code does

The module docstring reads:

```python
"""Atomic safetensors checkpoint manager with shared-tensor dedup and step discovery."""
```

and the class docstring says "Save/load model checkpoints. Files: model_step_N.safetensors, optim_step_N.pt, meta_step_N.json." The plan for this doc (docs/docs_expansion_plan.md, R9) likewise lists "atomicity (tmp→rename)" as a topic. **The code does not do that.** Verbatim from `utils/checkpoint.py:CheckpointManager.save`:

- `save_file(deduped, self.save_dir / f"model_step_{step}.safetensors")` — direct write to the final path;
- `torch.save(optimizer.state_dict(), self.save_dir / f"optim_step_{step}.pt")` — direct write;
- `self._write_json(self.save_dir / f"meta_step_{step}.json", meta)` — and `_write_json` opens the target path directly (`with open(tmp, "w")`; the parameter is *named* `tmp` but no temporary file is ever created — the `tempfile` import at the top of the module is dead code).

There is no `tmp → os.rename` sequence anywhere in the file. The resilience properties the code *actually* has are:

1. **Write order**: weights → optimizer → meta, with the JSON written *last*. A crash mid-save leaves the meta file missing, so the step fails `_checkpoint_complete` and `latest_step` skips it.
2. **Completeness-gated discovery**: a step counts only when all three files exist, so partially-written triples are never resumed.

That is a crash-*tolerant* design (torn saves are skipped) but not an *atomic* one: each individual file can be observed in a half-written state by a concurrent reader, and the window between files is unprotected. If the docstring's claim is ever needed for real (e.g., multi-process readers), the fix is the rename pattern the docstring implies. [VERIFIED against `utils/checkpoint.py`; the discrepancy is deliberate — this reference documents the code as written.]

## Resume flow in the training loop

`training/pretrain.py:Pretrainer.train` auto-discovers and restores at startup:

```python
global_step = 0
latest = self._find_latest_checkpoint()
if latest is not None:
    try:
        global_step = self.load_checkpoint(latest)
    except Exception as exc:
        self._log(f"[warn] Could not load checkpoint: {exc}")
```

(`_find_latest_checkpoint` is a one-line delegate to `utils/checkpoint.py:CheckpointManager.latest_step`.) A failed load — e.g. a torn-but-complete triple — degrades to a warning and training restarts from scratch rather than aborting. The CLI path in `training/pretrain.py:main` additionally supports `--resume <step>` (an explicit `trainer.load_checkpoint(int(args.resume))` before `train()`); note that `train()` then re-runs auto-discovery, so an explicit `--resume` to an *older* step is overridden by the latest complete step (see Pitfalls). The same restore path is used by the NaN guard: after 5 consecutive NaN/Inf micro-steps, `Pretrainer.train` calls `load_checkpoint(latest)` and resets `global_step` to the resumed step, discarding the poisoned run's optimizer momentum.

## Pitfalls

1. **Torn-but-complete triples are undetectable.** The completeness check is existence-only. If a crash truncates `model_step_N.safetensors` *after* the meta file was written, all three files exist, `latest_step` returns `N`, and `load` fails at deserialization (safetensors raises on truncated data). `Pretrainer.train` catches this and warns, but the checkpoint is unrecoverable — the only defense is disk-level atomicity, which the code does not implement.
2. **The dedup clone does not save space.** The tied pair is written twice (~411.8 MB of the ~1.94 GB model file). Calling this "dedup" is accurate only in the sense of *storage independence*: `safetensors.save_file` hard-rejects shared-storage dicts, so the clone is what makes saving a tied model possible at all. Do not "optimize" the pass into actually skipping the second key — `load_state_dict(strict=False)` would then report `lm_head.weight` as missing on every load.
3. **In-memory tying is preserved only because `assign=False`.** `load_state_dict` copies into the existing aliased parameter, so tying survives. Switching to `assign=True` (or calling `load_state_dict` on a model built with `weight_tying=False`) would silently un-tie or misassign. If you load into a non-tied model, the two copies are still written to the two distinct parameters — correct, just redundant.
4. **Optimizer state is `torch.save`, not safetensors.** It is loaded with `weights_only=True`, so version-skew or a missing file means "optimizer starts from scratch" (warned, not fatal). Also, the optimizer's moment tensors are keyed by *parameter index* over the deduplicated parameter list — a run whose model config changes the parameter order will restore moments onto the wrong parameters without any error.
5. **The meta JSON round-trips through `default=str`.** The scheduler state that `Pretrainer.save_checkpoint` stores contains only JSON-native numbers and lists today, so `json.load` → `scheduler.load_state_dict` round-trips. If a tensor ever lands in `extra_meta`, it will be stringified and the load will fail (or silently corrupt). And a *torn* meta JSON raises `json.JSONDecodeError` inside `load` — there is no fallback for a corrupt-but-present file.
6. **`--resume` an older step? The loop will re-discover.** `main()`'s explicit `--resume N` is followed by `train()`'s unconditional auto-discovery of `latest_step()`, so resuming to a step older than the latest complete checkpoint does not stick. The practical resume is: point `--checkpoint-dir` at the directory and let auto-discovery pick the newest triple.
7. **Save is synchronous and single-threaded.** The three writes happen inline in `save()`; there is no background flush, so checkpoint I/O stalls the training loop, and two concurrent `save` calls to the same directory would interleave files.

## Tests

- `tests/e2e_gpu_smoke.py:check_checkpoint` (check 7 of 8, CUDA + triton-gated): saves step 1 with `CheckpointManager`, asserts `model_step_1.safetensors` exists, builds a *fresh* tied `Mamba3Transformer`, loads with `strict=False`, and asserts round-trip max-abs drift `< 1e-4`. This exercises the dedup path end to end (fresh model with `weight_tying=True` + aliased dict → dedup → save → load → in-place copy).
- `tests/e2e_gpu_smoke.py:check_pretrainer_dry_run` (check 8 of 8): runs a 2-step `Pretrainer` dry run with a synthetic shard and asserts `_find_latest_checkpoint()` returns a real step after training — the auto-save side of the loop.
- There is **no CPU unit test** for `CheckpointManager` (the CPU suite's 28 tests cover model/SSD behavior); on a CPU-only box both e2e checks are skipped, which is why the dedup/tying behavior verified here was confirmed with an ad-hoc round-trip script rather than a committed test.

## Cross-links

- Weight tying and the model's parameter layout: [reference/04-transformer.md](04-transformer.md) and [theory/07-numerical-stability.md](../theory/07-numerical-stability.md) (weight-tying param math, NaN-guard rationale).
- The full training loop, LR schedule, and NaN guard: [reference/07-pretrain-cli.md](07-pretrain-cli.md) (concurrent), [reference/12-config-reference.md](12-config-reference.md) (concurrent, every `TrainingConfig` field), [reference/08-dataset.md](08-dataset.md) (concurrent, data loader).
- Operational practice — when checkpoints are written, how to resume, NaN recovery: [guides/02-training-runbook.md](../guides/02-training-runbook.md) and [guides/01-quickstart.md](../guides/01-quickstart.md) (to be written).
