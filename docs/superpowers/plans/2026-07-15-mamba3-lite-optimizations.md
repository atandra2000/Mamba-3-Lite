# Mamba-3-Lite Correctness, Test, and Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix broken imports, wire the dormant `grad_checkpoint` flag, clean carry-over code from DeepSeek-v3-Lite, and add the missing test suite to Mamba-3-Lite.

**Architecture:** Three tiers, seven commits. Tier 1 is surgical fixes (4 commits). Tier 2 adds 4 new test files (2 commits). Tier 3 is three behavior-preserving polish edits (1 commit). The 7-commit sequence is independently green: each commit leaves the tree in a state where the verification command for that tier passes.

**Tech Stack:** PyTorch 2.1+, pytest 7+, raw Python. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-07-15-mamba3-lite-optimizations-design.md`

---

## Global Constraints

These are non-negotiable and inherited from the spec. Every task implicitly includes them.

- **Pure SSM purity:** never add `mamba-ssm`, `causal_conv1d`, custom CUDA, or Triton. (Spec §Out of scope; AGENTS.md §2)
- **No MoE, MTP, or attention layers.** Mamba-3-Lite is a pure-SSM repo. (AGENTS.md §2)
- **No new dependencies** beyond the existing `requirements.txt` (torch, safetensors, pyyaml, tqdm, wandb, pytest).
- **No `pickle` checkpoints.** `safetensors` for weights, `torch.save` for optimizer. (Workspace AGENTS.md §1)
- **No `Co-Authored-By: Claude` trailers in any commit message.** Git author must be the user (Atandra Bharati). All commits in this plan use the existing `git config user.name` and `user.email`.
- **Test runtime target:** <2s on CPU for the full `pytest tests/ -v` run.
- **Pre-flight verify before each commit:** `python3 -m pytest tests/ -v` (or, for tasks that don't add tests, the project-level smoke check in `verification/pretrain.py`).
- **Vault sync:** every new `.md` file under the workspace is auto-synced to `~/Documents/obsidian` by the Stop hook in `.claude/settings.json` (per workspace AGENTS.md §7). No manual sync needed.
- **Verification before completion:** per `superpowers:verification-before-completion`, every task must run its verification command and observe the expected output before claiming the task is done. "I think it works" is not completion.

---

## File Structure

| File | Status | Purpose |
|------|--------|---------|
| `inference/generate.py` | **delete** | Broken import (Task 1) |
| `inference/speculative.py` | **delete** | Broken import (Task 1) |
| `inference/__pycache__/` | **delete** | Build artifact (Task 1) |
| `README.md` | edit | Drop `inference/` tree entry (Task 1) |
| `models/ssd.py` | edit | Delete dead `segsum`, add `ssd_naive_complex` oracle (Tasks 5, 7) |
| `models/ssd_complex.py` | unchanged | — |
| `models/mamba_block.py` | edit | Wire `torch.utils.checkpoint` (Task 2), init comment + `torch.empty` (Task 7) |
| `models/mimo.py` | unchanged | — |
| `models/transformer.py` | unchanged | — |
| `training/pretrain.py` | edit | Drop unused import (Task 4), add `_build_minimal_pretrainer` factory (Task 6) |
| `utils/memory.py` | edit | Drop MLA carry-over, drop dead `dim` branch, name 13.7 GB constant (Task 3) |
| `utils/checkpoint.py` | unchanged | — |
| `utils/logging.py` | unchanged | — |
| `tests/__init__.py` | create | Empty, for pytest discovery (Task 5) |
| `tests/test_ssd.py` | create | Chunkwise-vs-naive complex + edge cases (Task 5) |
| `tests/test_mimo.py` | create | Identity init, shape/finiteness (Task 6) |
| `tests/test_transformer.py` | create | Forward smoke, dict-config acceptance (Task 6) |
| `tests/test_train_step.py` | create | One optimizer step on dummy data (Task 6) |

---

## Task Sequencing

The 7 commits correspond to 7 plan tasks. Each task ends with a passing verification command.

| # | Task | Tier | Verifies with |
|---|------|------|---------------|
| 1 | Delete broken `inference/` | T1.1 | Grep + import smoke |
| 2 | Wire `torch.utils.checkpoint` in Mamba3Block | T1.2 | Existing forward smoke |
| 3 | Clean up `utils/memory.py` | T1.3 | Import smoke |
| 4 | Drop unused `assert_fits_in_available_gpu` import | T1.4 | Grep |
| 5 | Add `tests/test_ssd.py` (and `ssd_naive_complex` oracle) | T2.0, T2.1 | `pytest tests/test_ssd.py` |
| 6 | Add `tests/test_mimo.py`, `test_transformer.py`, `test_train_step.py` (and factory) | T2.2, T2.3, T2.4 | `pytest tests/` |
| 7 | Delete dead `segsum`, document init order, `torch.ones`→`torch.empty` | T3.1, T3.2, T3.3 | `pytest tests/` |

After Task 7, the final verification command from the spec (`python3 -m pytest tests/ -v` + smoke + greps) must pass.

---

## Task 1: Delete broken `inference/` files

**Files:**
- Delete: `inference/generate.py`
- Delete: `inference/speculative.py`
- Delete: `inference/__pycache__/` (directory)
- Edit: `README.md` (drop `inference/` tree entry)

**Context:** These files were copied verbatim from DeepSeek-v3-Lite. They import `models.transformer.Transformer` and `models.mtp.MTPModule` — neither symbol exists in Mamba-3-Lite. Running either file fails on import. `inference/generate.py:11` does `from inference.speculative import SpeculativeDecoder`, so the two files form one deletion unit.

**Pre-flight:** `grep -rn "from inference" --include="*.py" .` must return exactly the one self-reference inside `inference/generate.py` (no other callers). If other results appear, abort and surface to the human.

- [ ] **Step 1: Pre-flight grep**

Run from the project root `~/Desktop/CoreProjects/LLM/Mamba-3-Lite/`:

```bash
grep -rn "from inference" --include="*.py" .
```

Expected output:

```
./inference/generate.py:11:from inference.speculative import SpeculativeDecoder
```

If any other line appears, abort and report.

- [ ] **Step 2: Delete the files**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
rm inference/generate.py inference/speculative.py
rm -rf inference/__pycache__
ls inference/ 2>/dev/null && echo "inference/ still has files" || echo "inference/ is empty or gone"
```

Expected: `inference/ is empty or gone`. The `inference/` directory itself is preserved (an empty directory is harmless) but the `__pycache__` and all `.py` files are gone.

- [ ] **Step 3: Edit README.md project structure tree**

Open `README.md` and locate the project structure tree (the `Mamba-3-Lite/` indented block). Find the `inference/` subtree block, which currently contains:

```
├── inference/
│   ├── generate.py                       # constant-memory decoding
│   └── speculative.py                    # MTP-style speculative decode
```

Plus the duplicate `inference/` block in the verification section if any (check carefully). Delete those `inference/` lines so the tree no longer references them.

- [ ] **Step 4: Verify the change**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
ls inference/ 2>/dev/null
grep -n "inference" README.md
```

Expected: `ls` shows `inference/` is empty (or no `inference/` directory at all if step 2 also removed the empty dir). `grep` shows no remaining references to `inference/generate.py` or `inference/speculative.py` in README.md.

- [ ] **Step 5: Run a smoke check that nothing broke**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -c "import torch; from models.transformer import Mamba3Transformer, ModelConfig; \
cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4, head_dim=16, \
state_dim=8, chunk_size=4, ffn_dim=128, max_seq_len=32, dtype='fp32', weight_tying=True); \
m = Mamba3Transformer(cfg); x = torch.randint(0, 100, (2, 16)); y = m(x); \
assert y.shape == (2, 16, 100); print('forward ok')"
```

Expected: `forward ok` printed. (Confirms that removing the broken `inference/` files did not break any other module — the rest of the codebase does not import from `inference/`.)

- [ ] **Step 6: Commit**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
git add inference/ README.md
git status --short
git commit -m "chore: delete dead inference/ files carried over from DeepSeek-v3-Lite"
```

Expected: commit message has no `Co-Authored-By:` line. `git log -1` shows the user as the sole author.

---

## Task 2: Wire `torch.utils.checkpoint` in Mamba3Block

**Files:**
- Modify: `models/mamba_block.py` (lines 26-92)

**Context:** `TrainingConfig.grad_checkpoint` is plumbed from YAML all the way to `pretrain.py`, and `utils/memory.py:27` models a `factor=2` saving from grad checkpointing, but no `torch.utils.checkpoint.checkpoint` call exists. The flag is silently inactive — disabling it would have no effect. This task wires the wrapper.

**Interfaces:**
- Consumes: existing `Mamba3Block.__init__` signature and `forward(x: torch.Tensor) -> torch.Tensor` signature.
- Produces: same `forward` signature externally; internally a public `forward` (the wrapper) and a private `_forward_impl` (the original body).

- [ ] **Step 1: Edit `Mamba3Block.__init__` to store the flag**

In `models/mamba_block.py`, inside `Mamba3Block.__init__` (currently lines 29-51), add a new line storing the grad-checkpoint flag. Insert immediately after `self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)` (currently line 37):

```python
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
        self.grad_checkpoint = cfg.get("grad_checkpoint", False)
```

- [ ] **Step 2: Rename the existing `forward` to `_forward_impl`**

The current `forward` method spans lines 59-92 of `models/mamba_block.py`. Rename it to `_forward_impl`. The body is unchanged.

Find:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        B, T, _ = x.shape
```

Replace with:

```python
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        B, T, _ = x.shape
```

- [ ] **Step 3: Add the new public `forward` wrapper**

Insert a new `forward` method just before `_forward_impl` (i.e., between the line where the old `forward` was and the body of `_forward_impl`). The wrapper is:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        if self.grad_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)
```

- [ ] **Step 4: Run the forward smoke check (eval mode skips wrapper)**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -c "
import torch
from models.transformer import Mamba3Transformer, ModelConfig
cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4, head_dim=16,
                  state_dim=8, chunk_size=4, ffn_dim=128, max_seq_len=32,
                  dtype='fp32', weight_tying=True)
m = Mamba3Transformer(cfg)
m.eval()  # eval mode: wrapper is skipped, no checkpointing
x = torch.randint(0, 100, (2, 16))
y = m(x)
assert y.shape == (2, 16, 100), y.shape
print('eval forward ok')
"
```

Expected: `eval forward ok` printed. (Confirms the wrapper correctly delegates to `_forward_impl` in eval mode without invoking `torch.utils.checkpoint`.)

- [ ] **Step 5: Run the forward smoke check in training mode (triggers wrapper)**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -c "
import torch
from models.transformer import Mamba3Transformer, ModelConfig
cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4, head_dim=16,
                  state_dim=8, chunk_size=4, ffn_dim=128, max_seq_len=32,
                  dtype='fp32', weight_tying=True)
# Add grad_checkpoint to the inner layer config too.
for layer in cfg.n_layers and [None] or []: pass
m = Mamba3Transformer(cfg)
# Patch each block to enable grad_checkpoint
for block in m.layers:
    block.grad_checkpoint = True
m.train()  # training mode: wrapper invokes torch.utils.checkpoint
x = torch.randint(0, 100, (2, 16))
y = m(x)
loss = y.sum()
loss.backward()
print('training forward+backward ok, loss finite:', torch.isfinite(loss).item())
"
```

Expected: `training forward+backward ok, loss finite: True`. (Confirms the wrapper invokes `torch.utils.checkpoint` in training mode and the result is differentiable.)

- [ ] **Step 6: Commit**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
git add models/mamba_block.py
git commit -m "feat(models): wire gradient checkpointing in Mamba3Block"
```

---

## Task 3: Clean up `utils/memory.py`

**Files:**
- Modify: `utils/memory.py` (lines 15-23, 31-36, 39-43)

**Context:** Three small cleanups:
- `_kv_cache_bytes` looks for MLA-specific attributes (`kv_lora_rank`, `qk_rope_head_dim`) that Mamba-3 does not have. Replace with a `return 0` and a comment.
- `_infer_dim_n_layers` has a dead `getattr(model, "dim", 0)` branch — `Mamba3Transformer` has `self.cfg.d_model`, not `self.dim`. Collapse the function.
- The 13.7 GB constant in `_detect_overhead_gb` is a magic number. Name it.

- [ ] **Step 1: Replace `_kv_cache_bytes`**

Open `utils/memory.py` and find the function at lines 15-23. Replace the entire function with:

```python
def _kv_cache_bytes(model: nn.Module, seq_len: int, batch_size: int, dtype_bytes: int = 2) -> int:
    # ponytail: Mamba-3 has no KV cache; included for the estimator signature only.
    return 0
```

- [ ] **Step 2: Replace `_infer_dim_n_layers`**

Find the function at lines 31-36. Replace with:

```python
def _infer_dim_n_layers(model: nn.Module) -> tuple[int, int]:
    hd = model.embed.embedding_dim if hasattr(model, "embed") else 0
    nl = len(model.layers) if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList) else 0
    return hd, nl
```

- [ ] **Step 3: Name the 13.7 GB constant**

Insert a module-level constant near the top of `utils/memory.py`, just after the existing `import` lines (line 5):

```python
# Approx peak overhead from CUDA context + NCCL + caching allocator
# (A100 80GB, PyTorch 2.x). Empirically <= 17% of device total.
STATIC_PYTORCH_OVERHEAD_GB = 13.7
```

- [ ] **Step 4: Use the named constant in `_detect_overhead_gb`**

Find lines 39-43 and replace with:

```python
def _detect_overhead_gb() -> float:
    if not torch.cuda.is_available():
        return 2.0
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return min(STATIC_PYTORCH_OVERHEAD_GB, max(2.0, total_gb * 0.17))
```

- [ ] **Step 5: Verify the import still works**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -c "
from utils.memory import (
    _parameter_bytes, _optimiser_bytes, _kv_cache_bytes, _activation_bytes,
    _infer_dim_n_layers, _detect_overhead_gb, estimate_model_memory_gb,
    assert_fits_in_available_gpu, STATIC_PYTORCH_OVERHEAD_GB,
)
print('utils.memory imports ok, constant =', STATIC_PYTORCH_OVERHEAD_GB)
"
```

Expected: `utils.memory imports ok, constant = 13.7` printed.

- [ ] **Step 6: Commit**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
git add utils/memory.py
git commit -m "chore(utils): clean up memory.py for Mamba-3, name 13.7 GB constant"
```

---

## Task 4: Drop the unused `assert_fits_in_available_gpu` import

**Files:**
- Modify: `training/pretrain.py:18` (one-line edit)

**Context:** `pretrain.py:18` imports `assert_fits_in_available_gpu` from `utils.memory` but never calls it. The function definition in `utils/memory.py` is left intact (potential future use). This task removes the unused symbol from the import.

- [ ] **Step 1: Verify there are no call sites**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
grep -n "assert_fits_in_available_gpu" training/pretrain.py
```

Expected: exactly one line (the import):

```
18:from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb
```

If any other line appears, abort and report — the function is used somewhere unexpected.

- [ ] **Step 2: Edit the import line**

Open `training/pretrain.py`. Find line 18:

```python
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb
```

Replace with:

```python
from utils.memory import estimate_model_memory_gb
```

- [ ] **Step 3: Verify the change**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
grep -n "assert_fits_in_available_gpu" training/pretrain.py
```

Expected: no output (zero matches). The unused import is gone.

- [ ] **Step 4: Run a smoke check that the import still works**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -c "
import training.pretrain
print('training.pretrain imports ok')
"
```

Expected: `training.pretrain imports ok` printed. (Confirms that removing the unused import did not break the file.)

- [ ] **Step 5: Commit**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
git add training/pretrain.py
git commit -m "chore(pretrain): drop unused assert_fits import"
```

---

## Task 5: Add `tests/test_ssd.py` and the `ssd_naive_complex` oracle

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_ssd.py` (~30 lines)
- Modify: `models/ssd.py` (add `ssd_naive_complex` after `ssd_naive`)

**Context:** The complex chunkwise SSD is the algorithmic core. Its current verification is by inline assertion in production code. `SKILLS.md` Skill 2 mandates a chunkwise-vs-naive equivalence test as a regression gate. `ssd_naive` is real-only; we need a complex mirror as the oracle.

**Interfaces:**
- Consumes: existing `_discretise(dt, A)` from `models/ssd.py`. New function `ssd_naive_complex(x, A, B_t, C_t, dt)` mirrors `ssd_naive` but for complex tensors.
- Produces: `ssd_naive_complex` (consumed by `tests/test_ssd.py`).

- [ ] **Step 1: Verify `segsum` has no callers (we delete it in Task 7, but confirm now)**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
grep -rn "segsum" --include="*.py" .
```

Expected: exactly one line:

```
./models/ssd.py:8:def segsum(x: torch.Tensor) -> torch.Tensor:
```

If any other line appears, abort. (This is also the pre-flight check for Task 7; doing it now catches the issue early.)

- [ ] **Step 2: Add `ssd_naive_complex` to `models/ssd.py`**

Open `models/ssd.py`. Find the existing `ssd_naive` function (lines 22-39). Add the following function immediately after it (and before the final comment at lines 42-43):

```python
def ssd_naive_complex(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
) -> torch.Tensor:
    """O(T) sequential complex SSM scan — reference oracle for ssd_complex_chunkwise."""
    B_, T, H, D = x.shape
    N = B_t.shape[-1]
    A_bar = _discretise(dt, A)
    s = torch.zeros(B_, H, N, D, dtype=torch.complex64, device=x.device)
    ys = []
    for t in range(T):
        s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s \
            + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
        ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
    return torch.stack(ys, dim=1)
```

- [ ] **Step 3: Create `tests/__init__.py`**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
touch tests/__init__.py
```

(Empty file. Ensures pytest's test discovery treats `tests/` as a package and the import path `from models.ssd import ssd_naive_complex` resolves correctly from the test files.)

- [ ] **Step 4: Create `tests/test_ssd.py`**

Create the file with this exact content:

```python
"""Equivalence and edge-case tests for ssd_complex_chunkwise vs ssd_naive_complex."""
import torch

from models.ssd import ssd_naive_complex
from models.ssd_complex import ssd_complex_chunkwise


def test_chunkwise_matches_naive_complex():
    """The chunkwise projection must match the naive O(T) scan within atol=1e-4."""
    torch.manual_seed(0)
    B, T, H, D, N = 2, 16, 2, 4, 4
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.randn(H, dtype=torch.complex64) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.randn(B, T, H)

    y_chunk = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
    y_naive = ssd_naive_complex(x, A, B_t, C_t, dt)

    assert y_chunk.shape == y_naive.shape == (B, T, H, D)
    assert torch.allclose(y_chunk, y_naive, atol=1e-4), (
        f"max diff = {(y_chunk - y_naive).abs().max().item()}"
    )


def test_chunkwise_handles_uneven_T():
    """T not a multiple of chunk_size — last chunk is partial, output shape must match input."""
    torch.manual_seed(1)
    B, T, H, D, N = 1, 20, 2, 4, 4
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.randn(H, dtype=torch.complex64) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.randn(B, T, H)

    y = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
    assert y.shape == (B, T, H, D), y.shape
    assert torch.isfinite(y).all()


def test_chunkwise_handles_T_equal_to_chunk():
    """Single-chunk case (T == chunk_size) — must not crash on edge boundary."""
    torch.manual_seed(2)
    B, T, H, D, N = 1, 4, 2, 4, 4
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.randn(H, dtype=torch.complex64) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.randn(B, T, H)

    y = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
    assert y.shape == (B, T, H, D)
    assert torch.isfinite(y).all()
```

- [ ] **Step 5: Run the new tests**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -m pytest tests/test_ssd.py -v
```

Expected: 3 tests pass.

```
tests/test_ssd.py::test_chunkwise_matches_naive_complex PASSED
tests/test_ssd.py::test_chunkwise_handles_uneven_T PASSED
tests/test_ssd.py::test_chunkwise_handles_T_equal_to_chunk PASSED
```

If `test_chunkwise_matches_naive_complex` fails, the `ssd_complex_chunkwise` algorithm has a real regression — surface the failure to the human, do not paper over it.

- [ ] **Step 6: Run the full test suite (should still be just these 3 tests)**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -m pytest tests/ -v
```

Expected: 3 tests pass, no failures, no errors.

- [ ] **Step 7: Commit**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
git add models/ssd.py tests/__init__.py tests/test_ssd.py
git commit -m "test(ssd): add chunkwise-vs-naive complex equivalence + edge cases"
```

---

## Task 6: Add `tests/test_mimo.py`, `tests/test_transformer.py`, `tests/test_train_step.py`

**Files:**
- Create: `tests/test_mimo.py` (~15 lines)
- Create: `tests/test_transformer.py` (~25 lines)
- Create: `tests/test_train_step.py` (~30 lines)
- Modify: `training/pretrain.py` (add `_build_minimal_pretrainer` factory, modify line 15 import to add `ModelConfig`)

**Context:** Three test files. The train-step test depends on a new factory function `_build_minimal_pretrainer` added to `pretrain.py`. The factory bypasses the heavy `Pretrainer.__init__` (compile, TF32, optimizer-group split, full logging init) and wires only the minimum state needed by `train_step`.

**Interfaces:**
- `Pretrainer.train_step(tokens, targets, micro_step)` — the public method exercised by the test. Returns `Optional[Dict[str, float]]`.
- New function `_build_minimal_pretrainer(model_config: dict) -> Pretrainer` — builds a CPU-only Pretrainer with `compile_model=False`, `grad_checkpoint=False`, dummy data path. Lives at module level in `pretrain.py`, just above the `Pretrainer` class definition (currently at line 164 of `pretrain.py`).

- [ ] **Step 1: Modify the import in `pretrain.py` line 15**

Open `training/pretrain.py`. Find line 15:

```python
from models.transformer import Mamba3Transformer
```

Replace with:

```python
from models.transformer import Mamba3Transformer, ModelConfig
```

- [ ] **Step 2: Add `_build_minimal_pretrainer` to `pretrain.py`**

Insert the factory function just above the `Pretrainer` class (currently at line 164). The function uses the imports already in `pretrain.py` (`AdamW`, `LambdaLR`, `torch`, `Mamba3Transformer`, `ModelConfig`).

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

- [ ] **Step 3: Verify the factory works**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -c "
from training.pretrain import _build_minimal_pretrainer
p = _build_minimal_pretrainer({
    'vocab_size': 64, 'd_model': 32, 'n_layers': 2, 'n_heads': 2,
    'head_dim': 16, 'state_dim': 4, 'chunk_size': 4,
    'ffn_dim': 64, 'max_seq_len': 16, 'dtype': 'fp32', 'weight_tying': True,
})
import torch
tokens = torch.randint(0, 64, (2, 16))
targets = torch.randint(0, 64, (2, 16))
out = p.train_step(tokens, targets, micro_step=0)
assert out is not None
import math
assert math.isfinite(out['loss'])
print('factory + train_step ok, loss =', out['loss'])
"
```

Expected: `factory + train_step ok, loss = <float>` printed. (Confirms the factory builds a working Pretrainer instance and `train_step` runs end-to-end on CPU.)

- [ ] **Step 4: Create `tests/test_mimo.py`**

```python
"""MIMO mixer tests: identity init at construction, shape contract preserved."""
import torch

from models.mimo import MIMO


def test_mimo_identity_init():
    """After construction, mimo(x) == x to atol=1e-6 — nn.init.eye_ is correct."""
    m = MIMO(d_model=128, n_heads=4, head_dim=32)
    m.eval()
    x = torch.randn(2, 8, 4, 32)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6), f"max diff = {(y - x).abs().max().item()}"


def test_mimo_shape_and_finite():
    """Random input — output shape unchanged, all values finite."""
    torch.manual_seed(0)
    m = MIMO(d_model=128, n_heads=4, head_dim=32)
    m.train()
    x = torch.randn(2, 8, 4, 32)
    y = m(x)
    assert y.shape == (2, 8, 4, 32)
    assert torch.isfinite(y).all()
```

- [ ] **Step 5: Create `tests/test_transformer.py`**

```python
"""Mamba3Transformer forward smoke + config acceptance tests."""
import torch

from models.transformer import Mamba3Transformer, ModelConfig


def test_mamba3_transformer_forward():
    """Tiny config — forward shape, param count, weight tying sanity."""
    cfg = ModelConfig(
        vocab_size=100, d_model=64, n_layers=2, n_heads=4,
        head_dim=16, state_dim=8, chunk_size=4, ffn_dim=128,
        max_seq_len=32, dtype="fp32", weight_tying=True,
    )
    m = Mamba3Transformer(cfg)
    x = torch.randint(0, 100, (2, 16))
    y = m(x)
    assert y.shape == (2, 16, 100), y.shape
    n = sum(p.numel() for p in m.parameters())
    assert 100_000 < n < 5_000_000, f"unexpected param count: {n}"
    # Weight tying: embed.weight and lm_head.weight share storage.
    assert m.embed.weight.data_ptr() == m.lm_head.weight.data_ptr(), "weight tying not applied"


def test_mamba3_transformer_accepts_dict_config():
    """Mamba3Transformer(dict) must behave equivalently to Mamba3Transformer(ModelConfig)."""
    cfg_dict = {
        "vocab_size": 50, "d_model": 32, "n_layers": 2, "n_heads": 2,
        "head_dim": 16, "state_dim": 4, "chunk_size": 4,
        "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
    }
    m = Mamba3Transformer(cfg_dict)
    x = torch.randint(0, 50, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 50)
```

- [ ] **Step 6: Create `tests/test_train_step.py`**

```python
"""End-to-end train step test on a CPU-only Pretrainer built by the minimal factory."""
import math

import torch

from training.pretrain import _build_minimal_pretrainer


def test_train_step_on_tiny_model():
    """One optimizer step on a tiny model: loss must be finite, params must change."""
    torch.manual_seed(0)
    p = _build_minimal_pretrainer({
        "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
        "head_dim": 16, "state_dim": 4, "chunk_size": 4,
        "ffn_dim": 64, "max_seq_len": 16, "dtype": "fp32", "weight_tying": True,
    })
    # Snapshot params.
    before = [w.detach().clone() for w in p.model.parameters() if w.requires_grad]

    tokens = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    out = p.train_step(tokens, targets, micro_step=0)

    assert out is not None, "train_step returned None (NaN guard tripped)"
    assert math.isfinite(out["loss"]), f"non-finite loss: {out['loss']}"

    # After one optimizer step, at least one parameter must have changed.
    after = [w.detach().clone() for w in p.model.parameters() if w.requires_grad]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), \
        "no parameters changed after train_step"
```

- [ ] **Step 7: Run the full test suite**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -m pytest tests/ -v
```

Expected: 7 tests pass (3 from Task 5, 2 from test_mimo, 2 from test_transformer, 1 from test_train_step), no failures, no errors.

```
tests/test_mimo.py::test_mimo_identity_init PASSED
tests/test_mimo.py::test_mimo_shape_and_finite PASSED
tests/test_ssd.py::test_chunkwise_handles_T_equal_to_chunk PASSED
tests/test_ssd.py::test_chunkwise_handles_uneven_T PASSED
tests/test_ssd.py::test_chunkwise_matches_naive_complex PASSED
tests/test_train_step.py::test_train_step_on_tiny_model PASSED
tests/test_transformer.py::test_mamba3_transformer_accepts_dict_config PASSED
tests/test_transformer.py::test_mamba3_transformer_forward PASSED
```

If `test_train_step_on_tiny_model` fails with `train_step returned None`, the NaN guard tripped — that signals a real numerical issue. Surface it, do not weaken the assertion.

- [ ] **Step 8: Commit**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
git add training/pretrain.py tests/test_mimo.py tests/test_transformer.py tests/test_train_step.py
git commit -m "test: add MIMO, transformer, and train-step regression tests"
```

---

## Task 7: Numerics + init polish

**Files:**
- Modify: `models/ssd.py` (delete lines 8-14, the `segsum` function)
- Modify: `models/mamba_block.py` (line 44 + `_init_weights` comment)

**Context:** Three behavior-preserving micro-edits:
- `segsum` is dead code — confirmed by Task 5's pre-flight grep.
- `Mamba3Block._init_weights` runs in a subtle interaction with `Mamba3Transformer.apply(self._init_weights)`. Document the order so a future reader doesn't break it.
- `torch.ones` is wasted on `A` since `constant_(-1.0)` overwrites it.

- [ ] **Step 1: Pre-flight grep for `segsum` (re-confirm)**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
grep -rn "segsum" --include="*.py" .
```

Expected: exactly one line (`models/ssd.py:8`). If any other hit appears, abort and report.

- [ ] **Step 2: Delete `segsum` from `models/ssd.py`**

Open `models/ssd.py`. Find lines 8-14:

```python
def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable causal segment-sum for the decay matrix."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_seg = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return x_seg.masked_fill(~mask, float("-inf"))
```

Delete those 7 lines. The file should now start with `_discretise` at the top (after the module docstring and `import torch`).

- [ ] **Step 3: Confirm the rest of the file is unchanged**

Verify the `_discretise` and `ssd_naive` and `ssd_naive_complex` functions are still present. Read the file and confirm there are no syntax errors.

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -c "from models.ssd import ssd_naive, ssd_naive_complex, _discretise; print('models.ssd imports ok')"
```

Expected: `models.ssd imports ok`.

- [ ] **Step 4: Add a comment to `Mamba3Block._init_weights`**

Open `models/mamba_block.py`. Find the `_init_weights` method (currently lines 53-57). Find:

```python
    def _init_weights(self):
        nn.init.constant_(self.A, -1.0)
```

Replace with:

```python
    def _init_weights(self):
        # ponytail: sets A=-1 here; transformer.apply() re-inits only Linear/Embedding,
        # so A=-1 survives the second pass.
        nn.init.constant_(self.A, -1.0)
```

- [ ] **Step 5: Change `torch.ones` to `torch.empty` for `A`**

In `models/mamba_block.py`, find line 44:

```python
        self.A = nn.Parameter(torch.ones(self.n_heads, dtype=torch.complex64))
```

Replace with:

```python
        self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
```

- [ ] **Step 6: Run the full test suite — final gate**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
python3 -m pytest tests/ -v
```

Expected: 7 tests pass, no failures, no errors.

- [ ] **Step 7: Run the spec's final verification commands**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite

# 1. Tests pass
python3 -m pytest tests/ -v

# 2. README smoke check
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

# 3. Imports clean
grep -n "assert_fits_in_available_gpu" training/pretrain.py
grep -rn "from inference" --include="*.py" .
grep -rn "from models.mtp" --include="*.py" .
```

Expected: all three grep commands return no matches; the smoke check prints `forward ok`; pytest reports 7 passes.

- [ ] **Step 8: Commit**

```bash
cd ~/Desktop/CoreProjects/LLM/Mamba-3-Lite
git add models/ssd.py models/mamba_block.py
git commit -m "chore: drop dead segsum, document init order, skip wasted torch.ones"
```

---

## Final Verification

After Task 7, run the complete spec verification block from `docs/superpowers/specs/2026-07-15-mamba3-lite-optimizations-design.md §Verification`. All three commands must pass.

Then `git log --oneline -7` should show the 7 commits in order:

```
1c7ea49 design: Mamba-3-Lite correctness, test, and polish spec
<task 1> chore: delete dead inference/ files carried over from DeepSeek-v3-Lite
<task 2> feat(models): wire gradient checkpointing in Mamba3Block
<task 3> chore(utils): clean up memory.py for Mamba-3, name 13.7 GB constant
<task 4> chore(pretrain): drop unused assert_fits import
<task 5> test(ssd): add chunkwise-vs-naive complex equivalence + edge cases
<task 6> test: add MIMO, transformer, and train-step regression tests
<task 7> chore: drop dead segsum, document init order, skip wasted torch.ones
```

(The spec commit `1c7ea49` is from a prior turn and not part of this implementation work.)

**Definition of done for this plan:**

- All 7 commits landed.
- `python3 -m pytest tests/ -v` reports 7 tests pass.
- The forward smoke check in the spec §Verification returns `forward ok`.
- All three grep checks return zero matches.
- No commit message contains `Co-Authored-By: Claude` (or any Claude/Anthropic attribution).
- `git log` shows the user as the sole author of all commits.
- Vault sync (via the Stop hook) has mirrored the new `tests/*.py` files to `~/Documents/obsidian`.
