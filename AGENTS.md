# AGENTS.md — Mamba-3-Lite

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific rules only.

> **Project:** `LLM/Mamba-3-Lite/` · **Type:** State-Space Model (Mamba-3)
> **Scale:** ~404M params · 8.0B Chinchilla-optimal tokens · 12–15h on A100 80GB
> **Stack:** PyTorch ≥2.1, no `mamba-ssm`, no custom CUDA, no Triton
> (see rule #1 below — currently a deliberate purity constraint; the
> carve-out is in place for future sanctioned kernels).
> **Hardware:** A100 80GB (no offloading).
> **Architecture detail:** see `README.md` in this folder and **`SSD.md`**
> (authoritative complex-SSD walkthrough). Cross-project helper: root
> `AGENTS.md §2.13` (`mamba2-ssd-engineer`).

## 1. Subagent: `mamba2-ssd-engineer` (also covers Mamba-3)

**Triggers:** "Explain complex-valued SSD", "Why is N halved with complex
states?", "How does MIMO head mixing replace SISO?", "Why no causal conv
in Mamba-3?", "Tune chunk_size for throughput."

**Knows cold:**
- Faithful Mamba-3 (Dao & Gu, 2025) reproduction succeeding Mamba-2 with
  three architectural breakthroughs — all in pure PyTorch today:
  1. **Complex-valued SSD state spaces.** N=64 complex64 — half the state
     dimension of Mamba-2 (N=128) for parity perplexity. Two real
     sub-states packed into one complex state.
  2. **MIMO (Multi-Input Multi-Output) head mixing.** Fully-connected mixer
     across SSM heads replaces the classical SISO constraint.
  3. **Zero causal convolution.** The memory-bound `causal_conv1d` pass is
     eliminated; replaced by a purely chunked linear projection.
- 28 layers · vocab 50,257 (LLaMA tokenizer) · ~404M params · 8.0B-token
  Chinchilla run planned.
- Training: BF16 + `torch.compile` + TF32. FA2 disabled (we use the
  chunkwise recurrence). NaN guard with checkpoint rollback.
- **Pure SSM** — no attention layers, no MoE, no MTP. Deliberate
  separation from HyMo (the portfolio's hybrid attention/SSM project).

**Triton kernel contract:**

- **Sanctioned Triton paths:**
  - `models/ssd_triton.py` → `per_chunk_ssd_triton` (fused per-chunk
    pass for the complex-SSD chunkwise recurrence; fuses `L` materialisation,
    `Y_diag`, and per-chunk `state` into a single Triton kernel; opt-in
    via `ssd_dispatch='triton'` + `ENABLE_TRITON_KERNELS=1`; see
    `documentation/ssd_triton.md` for design, the A100-box verification
    checklist, and the v2 backward plan).
- When a kernel is added: place it in `models/<name>_triton.py`, gate
  on `import triton` with `try/except ImportError` setting
  `HAS_TRITON = False`, wrap in a `torch.autograd.Function`, add
  `tests/test_<name>_triton.py` with a CPU-runnable pure-PyTorch
  reference, and add the new path to the sanctioned list in rule #1.

## 2. Hard rules

1. **Raw PyTorch by default; custom Triton kernels are first-party for
   sanctioned hot paths.** Bulk of the codebase (complex SSD, MIMO
   mixer, chunkwise projection, RMSNorm, embeddings) stays raw
   PyTorch. No HuggingFace Trainer, no Lightning, no high-level
   wrappers. The sanctioned Triton paths are listed in §1 above. No
   new component gets a custom kernel without updating this file and
   adding a `documentation/<name>.md` plan.
   - **Conflict-resolution note:** the previous "no Triton" hard rule
     (rule #5 in earlier versions of this file) is **superseded** by
     this rule. The current project ships one sanctioned Triton path
     (`per_chunk_ssd_triton`); the carve-out is in use.
   - **`ssd_dispatch` opt-in is two-layered.** A `ssd_dispatch='triton'`
     on the model config AND `ENABLE_TRITON_KERNELS=1` in the
     environment are both required. Missing either one forces the
     dispatch back to `'pytorch'` with a one-line warning (per-block
     warn-and-fallback in `Mamba3Block`, and a process-level guard in
     `training/pretrain.py:_enforce_triton_env_var`).
2. **Always** read `SSD.md` before answering complex-SSD algorithm
   questions — it is the authoritative reference (covers complex
   recurrence, MIMO mixer, and chunkwise projection).
3. **Always** verify the regression tests pass after any change to
   `models/ssd.py` — the chunkwise linear projection must match the
   naive O(T) scan oracle exactly.
4. **Never** pack a state size that doesn't decompose cleanly into real
   pairs (N must be even for complex packing to be exact).
5. **Never** suggest adding MoE, MTP, or attention layers. Pure SSM
   repo (successor to Mamba-2).
6. **Never** let a Triton kernel silently fall back to the raw-PyTorch
   path during a default-config training run. The opt-in is explicit
   (per-kernel config key + `ENABLE_TRITON_KERNELS=1` env-var). If
   the kernel fails to compile or throws at runtime, the run must
   surface a clear error, not a silent fallback.
7. **Always** add a unit test in `tests/` for any new Triton kernel
   path. The test must run on CPU (using the pure-PyTorch reference)
   without `triton` installed. GPU-only behaviour is gated behind
   `@pytest.mark.gpu` and is auto-skipped on CPU-only machines.
8. **Concise comments only.** Docstrings and inline comments must
   justify non-obvious code, not restate it. A docstring is at most
   three short lines unless the function is a public API. Inline
   comments appear only when the code itself is opaque. Verifiable
   targets per file:
   - **Public function docstring:** ≤ 3 lines, or one short paragraph.
   - **Module docstring:** ≤ 6 lines.
   - **Inline comment density:** ≤ 1 comment per ~10 lines of code on
     average; comments that say what the next line does
     (`# compute x`, `# loop over rows`) are forbidden.
   - **Section banners** (`# ---- ... ----`) are reserved for the top
     level of a file (≤ 3 per file) and inside kernels to delimit
     named algorithm phases.
   Violations are reviewable on `wc -l <file>` and `grep -c '^[[:space:]]*#' <file>`.

## 3. Numerical-stability rules

- NaN guard with rollback (mirrors DeepSeek-v3-Lite pattern).
- Recurrent state stays in `complex64`; logits in FP32.
- Selective scan's gating sigmoid kept in FP32.
- Complex64 multiplications monitored for NaN; `torch.view_as_complex` on
  raw real pairs can fail silently if the imaginary stride is wrong.

## 4. Files

- `models/ssd.py` — complex SSD block + MIMO mixer + chunkwise projection.
- `training/`, `inference/`, `data/`, `scripts/`, `tests/`, `documentation/`.
- `SSD.md` — authoritative algorithm reference.
- `README.md`, `requirements.txt`, `pytest.ini`, `LICENSE`.

## 5. Known caveats

- Full 8B-token run not yet started.
- FA2 is disabled — the chunkwise linear projection is used instead.
- Complex states mean a downstream user can't trivially swap in a
  real-only recurrence without N→2N resize and a re-init.
