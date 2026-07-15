# AGENTS.md — Mamba-3-Lite

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific rules only.

> **Project:** `LLM/Mamba-3-Lite/` · **Type:** State-Space Model (Mamba-3)
> **Scale:** ~404M params · 8.0B Chinchilla-optimal tokens · 12–15h on A100 80GB
> **Stack:** PyTorch ≥2.1, no `mamba-ssm`, no custom CUDA, no Triton.
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
  three architectural breakthroughs — **all implemented in pure PyTorch**:
  1. **Complex-valued SSD state spaces.** N=64 complex64 — half the state
     dimension of Mamba-2 (N=128) for parity perplexity. Two real
     sub-states packed into one complex state.
  2. **MIMO (Multi-Input Multi-Output) head mixing.** Fully-connected mixer
     across SSM heads replaces the classical SISO (single-input
     single-output) constraint — cross-head communication for free.
  3. **Zero causal convolution.** The memory-bound `causal_conv1d` pass is
     eliminated; replaced by a purely chunked linear projection.
- 28 layers · vocab 50,257 (LLaMA tokenizer) · ~404M params · 8.0B-token
  Chinchilla run planned.
- Training: BF16 + `torch.compile` + TF32. FA2 disabled (we use the
  chunkwise recurrence). NaN guard with checkpoint rollback.
- **Pure SSM** — no attention layers, no MoE, no MTP. Deliberate
  separation from FusionLLM.

## 2. Hard rules

1. **Never** suggest adding MoE, MTP, or attention layers. Pure SSM repo
   (successor to Mamba-2).
2. **Always** read `SSD.md` before answering complex-SSD algorithm
   questions — it is the authoritative reference (covers complex recurrence,
   MIMO mixer, and chunkwise projection).
3. **Always** verify the regression tests pass after any change to
   `models/ssd.py` — the chunkwise linear projection must match the naive
   O(T) scan oracle exactly.
4. **Never** pack a state size that doesn't decompose cleanly into real
   pairs (N must be even for complex packing to be exact).
5. **Never** recover `mamba-ssm`, custom CUDA, or Triton dependencies —
   purity is part of the project headline.

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
- Complex states mean a downstream user can't trivially swap in a real-only
  recurrence without N→2N resize and a re-init.
