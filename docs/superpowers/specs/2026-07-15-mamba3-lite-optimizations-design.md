# Mamba-3-Lite — Correctness, Test, and Polish

**Status:** Design (awaiting approval)
**Date:** 2026-07-15
**Project:** `LLM/Mamba-3-Lite/`
**Scope:** Three tiers, eleven items, four new files, two files deleted.
**Out of scope:** The complex chunkwise algorithm, MIMO logic, training loop body, data pipeline, README content beyond project structure, LICENSE, requirements.

---

## Background

The Mamba-3-Lite codebase was audited for correctness, test coverage, and code hygiene. The audit found:

1. `inference/generate.py` and `inference/speculative.py` are broken imports carried over from DeepSeek-v3-Lite. They reference `models.transformer.Transformer` and `models.mtp.MTPModule`, neither of which exist in this repo. The Mamba-3 model is named `Mamba3Transformer` and there is no MTP module — `AGENTS.md §2` explicitly forbids MTP. Running either file fails with `ModuleNotFoundError` on the first line.
2. `TrainingConfig.grad_checkpoint` is plumbed from YAML to `pretrain.py`, and the memory estimator models a `factor=2` saving from grad checkpointing, but no `torch.utils.checkpoint.checkpoint` call exists anywhere in the codebase. The flag is silently inactive.
3. `utils/memory.py` contains carry-over code from the DeepSeek-v3-Lite MLA architecture: `_kv_cache_bytes` looks for `kv_lora_rank` and `qk_rope_head_dim` (MLA-specific attributes that Mamba-3 does not have), and `_infer_dim_n_layers` has a dead `getattr(model, "dim", 0)` branch.
4. The 13.7 GB constant in `_detect_overhead_gb` is a magic number with no name or comment.
5. `assert_fits_in_available_gpu` (defined in `utils/memory.py`) is imported in `pretrain.py` but never called.
6. `tests/` is empty despite `SKILLS.md` mandating `python3 -m pytest tests/ -v` as a regression gate and `AGENTS.md §4` reserving the directory for "the future test suite".
7. `models/ssd.py` defines `segsum` but nothing calls it. It is dead code from the Mamba-2 reference, superseded by the inlined cumsum in `ssd_complex_chunkwise`.
8. `Mamba3Block.A` is initialised with `torch.ones` then immediately overwritten with `constant_(-1.0)`. The first write is wasted.
9. `Mamba3Block._init_weights` runs once during `__init__`, but `Mamba3Transformer.apply(self._init_weights)` then re-runs a *different* `_init_weights` on every submodule afterward. The interaction is correct (only Linear/Embedding are touched at the transformer level, so `A=-1` survives) but the order is fragile and undocumented.

This design addresses items 1–9. Items that were considered and rejected on closer inspection:

- **Double `torch.cumsum` in `ssd_complex_chunkwise`** (initially flagged, lines 34 and 38 of `ssd_complex.py`): the two cumsums are over different axes of a permuted view. They are not redundant.
- **`Xc = ...to(torch.complex64)` cast** (initially flagged, line 31): `x` enters as `bfloat16` from the slice; the cast to `complex64` is necessary for the einsum with `Bc` (already `complex64`). Cannot be one-lined further.
- **MIMO `nn.init.eye_` cost**: O(d²) memory where d=1024, runs once at init. Not a hotspot.
- **`_load_shard` LRU cache eviction cost**: list `pop(0)` is O(n) for n=2. Not a hotspot.
- **`_amp_context` CPU branch**: returns a no-op context manager. Leaving as-is — explicit `autocast("cpu", enabled=False)` documents intent.

---

## Tier 1 — Correctness fixes

### 1.1 Delete broken `inference/`

**Files removed:**
- `inference/generate.py` (108 lines)
- `inference/speculative.py` (57 lines)
- `inference/__pycache__/` (build artifact)

**Why safe:** `inference/` has no callers. `training/pretrain.py`, `models/*`, `utils/*`, and `tests/*` do not import from it. Grep confirms: `grep -rn "from inference" --include="*.py" .` returns zero hits.

**README change:** the "Project structure" tree block in `README.md` includes a `inference/` subtree entry. Remove the four lines covering `inference/generate.py`, `inference/speculative.py`, and the surrounding tree. No other README changes.

### 1.2 Wire `torch.utils.checkpoint` in `Mamba3Block.forward`

**File:** `models/mamba_block.py`

**Change:** split the existing `forward` into a public `forward` (which conditionally checkpoints) and a private `_forward_impl` (which holds the current body). Add `self.grad_checkpoint = cfg.get("grad_checkpoint", False)` in `__init__`. Three lines of net diff.

```python
def __init__(self, cfg: dict, layer_idx: int = 0):
    # ... existing body
    self.grad_checkpoint = cfg.get("grad_checkpoint", False)
    # ... rest

def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.grad_checkpoint and self.training:
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, use_reentrant=False
        )
    return self._forward_impl(x)

def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    # ... existing forward body, unchanged
```

`use_reentrant=False` is the PyTorch 2.1+ path and avoids the re-entrance warning. The conditional ensures eval-mode forward (used by `test_train_step` and the gradient-checkpointing test setup) skips the wrapper.

### 1.3 Clean up `utils/memory.py`

**File:** `utils/memory.py`

**Three changes:**

**a) Replace the MLA carry-over in `_kv_cache_bytes` (lines 15–23):**

```python
def _kv_cache_bytes(model: nn.Module, seq_len: int, batch_size: int, dtype_bytes: int = 2) -> int:
    # ponytail: Mamba-3 has no KV cache; included for the estimator signature only.
    return 0
```

**b) Remove the dead `dim` branch in `_infer_dim_n_layers` (lines 31–36):**

```python
def _infer_dim_n_layers(model: nn.Module) -> tuple[int, int]:
    hd = model.embed.embedding_dim if hasattr(model, "embed") else 0
    nl = len(model.layers) if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList) else 0
    return hd, nl
```

**c) Name the 13.7 GB constant:**

```python
# Approx peak overhead from CUDA context + NCCL + caching allocator
# (A100 80GB, PyTorch 2.x). Empirically <= 17% of device total.
STATIC_PYTORCH_OVERHEAD_GB = 13.7
```

Use the named constant in `_detect_overhead_gb`.

### 1.4 Drop the unused `assert_fits_in_available_gpu` import

**File:** `training/pretrain.py`

**Change:** remove `assert_fits_in_available_gpu` from the `utils.memory` import (line 18). Keep `estimate_model_memory_gb` in the import. The function definition in `utils/memory.py` stays — it may be useful to future callers and removing the definition would be a wider change.

**Verify with grep before implementing:** `grep -n "assert_fits_in_available_gpu" training/pretrain.py`. Must show only the import line. If any call site exists, abort and surface the finding.

---

## Tier 2 — Test suite

All four files live under `tests/`. Run with `python3 -m pytest tests/ -v`. Target runtime: <2s on CPU. No fixtures, no parametrize — just direct module-level tests.

### 2.0 Additive helper in `models/ssd.py`

`ssd_naive` is real-valued. The chunkwise path is complex. Add a 5-line mirror:

```python
def ssd_naive_complex(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
) -> torch.Tensor:
    """O(T) sequential complex SSM scan — reference oracle for ssd_complex_chunkwise."""
    B_, T, H, D = x.shape
    A_bar = _discretise(dt, A)
    s = torch.zeros(B_, H, N, D, dtype=torch.complex64, device=x.device)
    ys = []
    for t in range(T):
        s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s \
            + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
        ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
    return torch.stack(ys, dim=1)
```

Note: `N = B_t.shape[-1]` is computed at the top (mirrors `ssd_naive`).

### 2.1 `tests/test_ssd.py`

Three tests, ~30 lines total:

- `test_chunkwise_matches_naive_complex` — the equivalence test from `SKILLS.md` Skill 2, on `complex64` tensors, B=2, T=16, H=2, D=4, N=4, chunk_size=4. `atol=1e-4`.
- `test_chunkwise_handles_uneven_T` — T=20 with chunk_size=4 (last chunk partial), no crash, output shape matches input.
- `test_chunkwise_handles_T_equal_to_chunk` — T=4 with chunk_size=4 (single chunk), no crash.

### 2.2 `tests/test_mimo.py`

Two tests, ~15 lines:

- `test_mimo_identity_init` — after `MIMO(d_model, n_heads, head_dim)`, `mimo(x) == x` to `atol=1e-6` for any input. Catches accidental init drift.
- `test_mimo_shape_and_finite` — random B=2, T=8, output shape unchanged, all values finite.

### 2.3 `tests/test_transformer.py`

Two tests, ~25 lines:

- `test_mamba3_transformer_forward` — the README verification snippet lifted into pytest: tiny config, integer input ids, assert `(B, T, vocab)` output shape and param count in a sane range.
- `test_mamba3_transformer_accepts_dict_config` — `Mamba3Transformer({"vocab_size": ..., ...})` works equivalently to `Mamba3Transformer(ModelConfig(...))`.

### 2.4 `tests/test_train_step.py`

One test, ~30 lines. Uses the `_build_minimal_pretrainer()` factory added to `pretrain.py` (see below). Asserts `out is not None` and `math.isfinite(out["loss"])`. Imports needed: `import math`, `import torch`, `from training.pretrain import _build_minimal_pretrainer`.

**Factory in `pretrain.py`:** add `from models.transformer import Mamba3Transformer, ModelConfig` to the existing import on line 15 (modify the import line, do not add a duplicate).

```python
def _build_minimal_pretrainer(model_config: dict) -> "Pretrainer":
    """Build a CPU-only Pretrainer with compile/grad-checkpoint disabled.
    Used by tests/test_train_step.py to avoid the heavy __init__ side effects.
    The data_path arg is intentionally absent — train_step does not touch the dataset.
    """
    p = Pretrainer.__new__(Pretrainer)
    p.config = None  # not used by train_step
    p.device = torch.device("cpu")
    p.amp_dtype = torch.float32
    p._opt_steps = 0
    p._log = lambda msg: None  # swallow log output during tests; called as self._log(msg) so a 1-arg lambda is correct
    p._amp_context = lambda: torch.amp.autocast("cpu", enabled=False)
    p.model = Mamba3Transformer(ModelConfig(**model_config))
    p.raw_model = p.model
    p.optimizer = AdamW(p.model.parameters(), lr=1e-3, fused=False)
    p.scheduler = LambdaLR(p.optimizer, lr_lambda=lambda s: 1.0)
    p.nan_guard = True
    p.gradient_accumulation_steps = 1
    p.max_grad_norm = 1.0
    return p
```

Add at module level in `pretrain.py`, just above the `Pretrainer` class. The factory is additive; it does not change `Pretrainer.__init__`.

---

## Tier 3 — Numerics + init polish

### 3.1 Delete dead `segsum` from `models/ssd.py`

**File:** `models/ssd.py` lines 8–14.

**Verify with grep first:** `grep -rn "segsum" --include="*.py" .` must return only the definition. If any caller exists, abort and surface the finding. Otherwise delete the 7 lines.

### 3.2 Document init order in `Mamba3Block._init_weights`

**File:** `models/mamba_block.py`

Add a one-line comment at the top of `_init_weights`:

```python
def _init_weights(self):
    # ponytail: sets A=-1 here; transformer.apply() re-inits only Linear/Embedding,
    # so A=-1 survives the second pass.
    nn.init.constant_(self.A, -1.0)
    nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
    nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
```

### 3.3 `torch.ones` → `torch.empty` for the `A` parameter

**File:** `models/mamba_block.py:44`

```python
self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
```

The next line (`nn.init.constant_(self.A, -1.0)`) overwrites all values. Behaviour-preserving.

---

## Verification

After implementation, the following must all succeed:

```bash
# 1. Tests pass
python3 -m pytest tests/ -v

# 2. README smoke check (post-T1.2)
python3 -c "
import torch
from models.transformer import Mamba3Transformer, ModelConfig
cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                  head_dim=16, state_dim=8, chunk_size=4, ffn_dim=128,
                  max_seq_len=32, dtype='fp32', weight_tying=True)
m = Mamba3Transformer(cfg)
x = torch.randint(0, 100, (2, 16))
y = m(x)
assert y.shape == (2, 16, 100)
print('forward ok')
"

# 3. Imports clean (post-T1.4 and T1.1)
grep -n "assert_fits_in_available_gpu" training/pretrain.py  # should match nothing
grep -rn "from inference" --include="*.py" .  # should match nothing
grep -rn "from models.mtp" --include="*.py" .  # should match nothing
```

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `Mamba3Block.forward` ends up duplicated and the wrapper drifts from the body | `_forward_impl` holds the **only** body. The public `forward` is the four-line wrapper. CI test `test_mamba3_transformer_forward` exercises the wrapper in eval mode (skips checkpointing); `test_train_step` exercises training mode and triggers the checkpoint path. |
| `Pretrainer` internals change, breaking the factory | The factory `_build_minimal_pretrainer` is the single point of contact. If `train_step` signature changes, the test breaks at the call site, not in the factory. |
| Removing the inference files breaks a forgotten caller | `grep -rn "from inference" --include="*.py" .` is the verification step. If non-empty, abort. |
| The 13.7 GB constant is wrong for a different GPU | The constant is a `static method`-level default that gets multiplied by `0.17` of `total_memory`. For A100 80GB: `0.17 * 80 = 13.6`. The name `STATIC_PYTORCH_OVERHEAD_GB` reflects that it is a *static* floor, not a per-GPU value. Acceptable for an estimator. |
| `segsum` is referenced somewhere I missed | The grep is part of the spec; if it returns hits, the work item is re-scoped. |

---

## Out of scope

- The complex chunkwise algorithm (`ssd_complex_chunkwise` body).
- The MIMO mixer logic and shape.
- The training loop body, optimizer config, scheduler.
- The data pipeline shim (`data/prepare_data.py`).
- README content beyond the project structure tree.
- `LICENSE`, `requirements.txt`, `pytest.ini`.
- Speculative decoding (per `AGENTS.md §2`, MTP is excluded).
- `inference/generate.py` and `inference/speculative.py` revival (per design call, deleted).

---

## File-level diff summary

| File | Change type | Net LoC |
|------|-------------|---------|
| `inference/generate.py` | deleted | -108 |
| `inference/speculative.py` | deleted | -57 |
| `models/ssd.py` | edited (delete segsum, add ssd_naive_complex) | +5 |
| `models/ssd_complex.py` | unchanged | 0 |
| `models/mamba_block.py` | edited (grad-ckpt wrapper, init comment, torch.empty) | +9 |
| `models/mimo.py` | unchanged | 0 |
| `models/transformer.py` | unchanged | 0 |
| `training/pretrain.py` | edited (drop import, add factory) | +18 |
| `utils/checkpoint.py` | unchanged | 0 |
| `utils/logging.py` | unchanged | 0 |
| `utils/memory.py` | edited (drop MLA, drop dead dim, name constant) | -3 |
| `tests/__init__.py` | new (empty, for pytest discovery) | 0 |
| `tests/test_ssd.py` | new | +30 |
| `tests/test_mimo.py` | new | +15 |
| `tests/test_transformer.py` | new | +25 |
| `tests/test_train_step.py` | new | +30 |
| `README.md` | edited (drop inference/ tree) | -4 |

**Net:** -46 LoC removed, +170 LoC added (mostly new tests), +1 deleted dir. The codebase shrinks in production code (-50 LoC) and grows in test coverage (+125 LoC of regression net).

---

## Commit plan

Single PR with the following commit sequence, each independently green:

1. `chore: delete dead inference/ files carried over from DeepSeek-v3-Lite` (T1.1 + README edit)
2. `feat(models): wire gradient checkpointing in Mamba3Block` (T1.2)
3. `chore(utils): clean up memory.py for Mamba-3, name 13.7 GB constant` (T1.3)
4. `chore(pretrain): drop unused assert_fits import` (T1.4)
5. `test(ssd): add chunkwise-vs-naive complex equivalence + edge cases` (T2.0, T2.1)
6. `test: add MIMO, transformer, and train-step regression tests` (T2.2, T2.3, T2.4)
7. `chore: drop dead segsum, document init order, skip wasted torch.ones` (T3.1, T3.2, T3.3)

Tests added in commit 5 must pass *before* commit 5 lands (i.e., the existing ssd_complex_chunkwise must be correct; the new tests verify the existing behaviour and protect against future regressions).

---

## What this spec does NOT do

- Does not change model architecture.
- Does not change training hyperparameters.
- Does not change the data pipeline.
- Does not add MoE, MTP, attention, or any non-SSM components (would violate `AGENTS.md §2`).
- Does not add `mamba-ssm` or any other external dependency.
- Does not run the full 8B-token training run (no GPU in dev environment; verification is via tests + smoke check).
