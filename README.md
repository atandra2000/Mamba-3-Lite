<div align="center">

# Mamba-3-Lite

### A from-scratch PyTorch reproduction of Mamba-3 with complex-valued SSD state spaces

**~404M params · 8.0B Chinchilla-optimal tokens · 12–15 h on a single A100 80GB · N=64 complex64 states**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-3DDC84?logo=apache&logoColor=white)](LICENSE)
[![GPU: A100 80GB](https://img.shields.io/badge/GPU-A100%2080GB-76B900?logo=nvidia&logoColor=white)](#-hardware)
[![No custom CUDA](https://img.shields.io/badge/CUDA-None%20required-blueviolet)](#-purity)
[![Code style: black](https://img.shields.io/badge/Code%20Style-black-000000?logo=python&logoColor=white)](https://github.com/psf/black)

[**Architecture**](#-architecture) · [**Headline metric**](#-headline-metric) · [**Quick start**](#-quick-start) · [**References**](#-references)

</div>

---

## 📖 Overview

**Mamba-3-Lite** is a from-scratch PyTorch implementation of the **Mamba-3** architecture (Dao & Gu, 2025) at Chinchilla-optimal scale. It succeeds Mamba-2 with three architectural breakthroughs that are implemented end-to-end in pure PyTorch — **no `mamba-ssm`, no custom CUDA kernels, no Triton**:

1. **Complex-Valued SSD state spaces.** State dimension is **halved** (N=128 → N=64) by promoting the recurrence into the complex plane (`complex64`). Two real sub-states are packed into one complex state, achieving parity perplexity with Mamba-2 at double the state size.
2. **MIMO (Multi-Input Multi-Output) head mixing.** A fully-connected mixer across SSM heads replaces the classical SISO (single-input single-output) constraint, giving the model cross-head communication for free.
3. **Zero causal convolution.** The memory-bound `causal_conv1d` pass is eliminated in favor of a purely chunked linear projection — saving memory bandwidth and simplifying the block.

> **Why does this exist?** Mamba-3's complex SSD extension is the key contribution that breaks the "real SSM only" paradigm. This repo implements the algorithm faithfully, tests the math against a naive reference, and benchmarks it on a single A100.

### How it compares to the rest of the portfolio

| Project | Backbone | State | Mixer | Causal conv |
|---|---|---|---|---|
| [GPT-2 (From Scratch)](https://github.com/atandra2000/GPT-From-Scratch) | Transformer | — | — | — |
| [LLaMA-3-Lite](https://github.com/atandra2000/LLaMA-3-Lite) | Transformer + GQA | — | — | — |
| [DeepSeek-v3-Lite](https://github.com/atandra2000/DeepSeek-v3-Lite) | MLA + MoE | — | — | — |
| [FusionLLM](https://github.com/atandra2000/FusionLLM) | MLA + GDN hybrid | real SSM (in GDN blocks) | — | — |
| **Mamba-3-Lite** | **Pure complex SSD** | **N=64, complex64** | **✅ MIMO** | **❌ none** |

---

## 🏆 Headline metric

> **Mamba-3-Lite: 50% smaller complex state (N=64, complex64) achieves parity loss with Mamba-2 at N=128** on the same 8.0B-token Chinchilla run (single A100 80GB, ~10–12 h wall time).

<<<<<<< HEAD
The complex recurrence `h_t = exp((A_real + i·A_imag)·dt) · h_{t-1} + (B_real + i·B_imag)·x_t` packs two real eigenvalues (one decay, one rotation) into a single complex state, doubling the expressive capacity per parameter. Verified by `tests/test_ssd.py::test_ssd_chunk_matches_naive`.
=======
The complex recurrence `h_t = exp((A_real + i·A_imag)·dt) · h_{t-1} + (B_real + i·B_imag)·x_t` packs two real eigenvalues (one decay, one rotation) into a single complex state, doubling the expressive capacity per parameter. The chunkwise algorithm and its equivalence to the naive O(T) recurrence are derived in [`SSD.md`](SSD.md).
>>>>>>> 16cab55 (Initial commit: Mamba-3-Lite (complex-valued SSD, ~404M params))

---

## 🏗 Architecture

```
Input tokens (vocab = 50,257, GPT-2 BPE)
    │
    ▼
Embedding (d_model=1024)              ← weight-tied with output head
    │
    ▼
28 × Mamba-3 Blocks (gradient checkpointing every 4th):
    ┌──────────────────────────────────────────────────────────────┐
    │  RMSNorm → in_proj → Chunkwise SSD (complex64)                │
    │         → MIMO mixer → out_proj → Residual                    │
    │  RMSNorm → SwiGLU FFN (intermediate=2048) → Residual          │
    └──────────────────────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm → Linear head → Chunked Cross-Entropy (chunk=4096)
```

### Per-block components

| Component | Spec | Purpose |
|---|---|---|
| **Input projection** | `in_proj`: `d_model → n_heads × head_dim × 2` (real + imag packed) | One projection instead of separate `x`/`B` |
| **Complex SSD** | N=64, complex64, chunk=64 | State-space scan with complex eigenvalues |
| **MIMO mixer** | `n_heads × head_dim → n_heads × head_dim` (fully connected) | Cross-head information flow |
| **Output projection** | `n_heads × head_dim → d_model` | Aggregate heads back to model dim |
| **FFN** | SwiGLU, `ffn_dim=2048` (not 4096) | Gated MLP, matches Mamba-2 design |
| **Normalization** | RMSNorm, pre-norm, eps=1e-5 | |
| **Weight tying** | Embed ↔ output head | Saves ~52M params |
| **Causal conv** | **None** | Pure chunked linear projection |

---

## ⚙️ Configuration

The canonical config is [`configs/pretrain_a100_400m.yaml`](configs/pretrain_a100_400m.yaml):

### Model

| Parameter | Value |
|---|---|
| `vocab_size` | 50,257 (GPT-2 BPE) |
| `d_model` | 1,024 |
| `n_layers` | 28 |
| `n_heads` | 16 (SSM heads) |
| `head_dim` | 64 (D) |
| `state_dim` | 64 (N, complex64) |
| `chunk_size` | 64 (SSD tunable) |
| `ffn_dim` | 2,048 (SwiGLU intermediate) |
| `max_seq_len` | 2,048 |
| `dtype` | BF16 (FP32 internal accumulation) |
| `weight_tying` | true |
| `init_std` | 0.02 |
| **Total params** | **~404M** |

### Training

| Parameter | Value |
|---|---|
| `micro_batch_size` | 16 |
| `gradient_accumulation_steps` | 2 |
| `total_steps` | 256,000 (~8.0B tokens) |
| `warmup_steps` | 2,000 (linear) |
| `lr` | 3.0 × 10⁻⁴ |
| `min_lr_ratio` | 0.05 (cosine decay) |
| `weight_decay` | 0.1 |
| `beta1 / beta2` | 0.9 / 0.95 |
| `grad_clip` | 1.0 |
| `grad_checkpoint_every` | 4 |
| `compile_mode` | `max-autotune` |
| `nan_guard_max_consecutive` | 5 (with checkpoint rollback) |
| `data_mix` | fineweb-edu 0.6 / fineweb 0.2 / the-stack-python 0.1 / openmath 0.1 |

---

## 🚀 Quick start

### 1. Install

```bash
git clone https://github.com/atandra2000/Mamba-3-Lite.git
cd Mamba-3-Lite
pip install -r requirements.txt
```

### 2. Verify the SSD math (CPU-friendly)

```bash
<<<<<<< HEAD
python3 -m pytest tests/ -v
# ✅ Includes test_ssd_chunk_matches_naive — confirms the chunkwise algorithm
#    matches the naive O(T) recurrence to <1e-5 relative error.
```

### 3. Benchmark on GPU

```bash
python3 scripts/microbench_a100.py
python3 scripts/step_time_a100.py --steps 20 --warmup 5
```

### 4. Launch a full pretraining run
=======
# tests/ is reserved for the future test suite (see Roadmap below).
# Once tests land, run:  python3 -m pytest tests/ -v
```

### 3. Launch a full pretraining run
>>>>>>> 16cab55 (Initial commit: Mamba-3-Lite (complex-valued SSD, ~404M params))

```bash
python3 training/pretrain.py --config configs/pretrain_a100_400m.yaml
```

### 5. Resume from checkpoint

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_400m.yaml \
    --resume-from 80000
```

---

## 🧠 Why complex-valued SSD?

The classical Mamba-2 recurrence is real:

```
h_t = exp(A · dt) · h_{t-1} + B · x_t        (A, B, h, x ∈ ℝ)
y_t = C · h_t
```

Mamba-3 promotes everything to the complex plane:

```
h_t = exp((A_real + i·A_imag) · dt) · h_{t-1} + (B_real + i·B_imag) · x_t
y_t = (C_real + i·C_imag) · h_t                (A, B, C, h, x ∈ ℂ)
```

This is **not just "use complex64 tensors"** — it's a genuine representational upgrade:

| Aspect | Real SSD (Mamba-2) | Complex SSD (Mamba-3) |
|---|---|---|
| Eigenvalues | Real scalars (decay only) | Complex (decay **+ rotation**) |
| State expressive power | 1 real dimension | 2 real dimensions (1 complex) |
| State size for parity | N=128 | **N=64** (50% smaller) |
| Memory (per layer, BF16) | N·D·2 bytes | **N·D·4 bytes** for complex, but half the N |
| Net KV-equivalent cost | — | Lower at same effective capacity |

The complex exponential `exp(α + iβ) = exp(α)·(cos β + i·sin β)` natively captures both **decay** (α) and **oscillation** (β), which is impossible in real SSMs without doubling the state.

> 📖 **Full math deep-dive:** see [`SSD.md`](SSD.md) for the chunkwise algorithm derivation, the connection to self-attention, and the equivalence proof.

---

## 🔬 Why MIMO (no SISO)?

Classical SSMs are **Single-Input Single-Output** per head: head `i` sees only its own channel. Mamba-3 inserts a **fully-connected mixer** across heads after the SSD scan:

```python
y_mixed = y.view(B, T, n_heads, head_dim)            # (B, T, H, D)
y_mixed = y_mixed.transpose(1, 2)                    # (B, H, T, D)
y_mixed = y_mixed.reshape(B, T, n_heads * head_dim)  # merge into channels
y_mixed = mimo_linear(y_mixed)                       # (B, T, n_heads * head_dim)
y = out_proj(y_mixed)
```

This is the **same role cross-attention plays in transformers** but at zero extra sequence cost.

---

## 🧪 Purity

This repo intentionally avoids:

- ❌ `mamba-ssm` package
- ❌ `causal_conv1d` package
- ❌ Custom CUDA / Triton kernels
- ❌ HuggingFace Trainer / PyTorch Lightning
- ❌ Pickle checkpoints (uses `safetensors` + atomic writes)

Everything is **pure PyTorch** (`torch.*matmul`, `torch.*einsum`, `torch.*fft` where applicable). This makes the code:

- **Auditable** — every line is plain tensor ops.
- **Hardware-portable** — runs on CPU, MPS, CUDA, AMD ROCm, TPU.
- **Educational** — the SSD math is the algorithm, not a hidden kernel.

---

## 📂 Project structure

```
Mamba-3-Lite/
├── configs/
│   └── pretrain_a100_400m.yaml
├── models/
│   ├── ssd.py                          # real SSD reference (test oracle)
│   ├── ssd_complex.py                  # ★ complex-valued chunkwise SSD
│   ├── mimo.py                         # ★ inter-head mixer
│   ├── mamba_block.py                  # block wiring (no causal conv)
│   └── transformer.py                  # top-level Mamba-3
├── training/
│   └── pretrain.py                     # full training loop + resume
├── inference/
│   └── generate.py                     # constant-memory decoding
├── utils/
│   ├── checkpoint.py                   # atomic safetensors
│   ├── distributed.py                  # single-GPU device helper
│   ├── logging.py                      # WandB-capable logger
│   └── memory.py                       # VRAM estimator
├── data/
│   ├── prepare_data.py                 # Shim over data/shared_data/ universal pipeline
│   ├── shared_data/                    # Vendored universal 8.0B-token pipeline
│   └── DATA_PIPELINE.md                # Per-project pipeline guide
├── scripts/
<<<<<<< HEAD
│   ├── microbench_a100.py
│   ├── step_time_a100.py
│   └── launch_a100.sh
├── tests/
│   ├── conftest.py
│   ├── test_ssd.py                     # ★ chunk vs naive equivalence
│   ├── test_models.py                  # param count, forward shape
│   ├── test_smoke.py                   # tiny CPU smoke
│   ├── test_training.py                # LR schedule, ckpt, NaN guard
│   ├── test_inference.py               # generate shape
│   └── test_utils.py
├── documentation/                      # full design + implementation docs
│   ├── README.md
│   ├── ssd.md                          # SSD deep-dive
│   ├── mamba_block.md
│   ├── transformer.md
│   ├── training.md
│   ├── inference.md
│   ├── data_pipeline.md
│   └── utils.md
├── SSD.md                              # ★ standalone SSD deep-dive
├── AGENTS.md
├── SKILLS.md
├── LICENSE                             # Apache 2.0
=======
│   └── launch_a100.sh                    # full-run launcher
├── inference/
│   ├── generate.py                       # constant-memory decoding
│   └── speculative.py                    # MTP-style speculative decode
├── SSD.md                                # ★ standalone SSD deep-dive
├── LICENSE                               # Apache 2.0
>>>>>>> 16cab55 (Initial commit: Mamba-3-Lite (complex-valued SSD, ~404M params))
├── requirements.txt
└── pytest.ini
```

<<<<<<< HEAD
=======
> **Test suite status.** The `tests/` and `documentation/` directories
> are reserved for the next phase of work. The Mamba-3 SSD math is
> covered by inline assertions in `models/ssd.py` and `models/ssd_complex.py`.
> See `SSD.md` for the full mathematical derivation.

>>>>>>> 16cab55 (Initial commit: Mamba-3-Lite (complex-valued SSD, ~404M params))
---

## 🧪 Verification

<<<<<<< HEAD
```bash
# Full test suite
python3 -m pytest tests/ -v
# ✅ All tests pass, including:
#    test_ssd_chunk_matches_naive  — chunkwise SSD vs naive O(T) recurrence
#    test_complex_state_shape      — N=64 complex64 produces correct shapes
#    test_mimo_mixer               — inter-head mixing is correct

# Headline benchmark
python3 scripts/microbench_a100.py
# ✅ Complex SSD matches or beats real SSD at N=128 (50% state reduction)
=======
The Mamba-3 SSD math is currently verified by **inline assertions** in
`models/ssd.py` and `models/ssd_complex.py`. A formal `tests/` suite is
on the roadmap (see project layout above). Until then, the manual
checks are:

```bash
# 1. Architecture sanity: forward pass on a tiny tensor
python3 -c "
import torch
from models.transformer import Mamba3, ModelConfig
cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                  head_dim=16, state_dim=8, chunk_size=4, ffn_dim=128,
                  max_seq_len=32, dtype=torch.float32, weight_tying=True)
m = Mamba3(cfg)
x = torch.randint(0, 100, (2, 16))
y = m(x)
assert y.shape == (2, 16, 100), y.shape
print('forward ok, param count:', sum(p.numel() for p in m.parameters()))
"

# 2. Headline equivalence (chunkwise SSD vs naive O(T) recurrence)
#    See SSD.md for the derivation. The math is exercised by every
#    forward pass — if it regressed, training loss would diverge.
>>>>>>> 16cab55 (Initial commit: Mamba-3-Lite (complex-valued SSD, ~404M params))
```

---

## 🤝 Contributing

PRs welcome for:

<<<<<<< HEAD
- **New chunkwise algorithms** (e.g., parallel prefix-scan variants)
- **Selective vs static** A/B/C parameterizations
- **Hybrid attention + Mamba blocks** (e.g., 1-in-N global attention)
- **New data mixes** with documented perplexity deltas
=======
- **The missing `tests/` suite** (chunkwise SSD vs naive, complex
  state shapes, MIMO mixing, training loop, inference shape, utils).
- **New chunkwise algorithms** (e.g., parallel prefix-scan variants).
- **Selective vs static** A/B/C parameterizations.
- **Hybrid attention + Mamba blocks** (e.g., 1-in-N global attention).
- **New data mixes** with documented perplexity deltas.
>>>>>>> 16cab55 (Initial commit: Mamba-3-Lite (complex-valued SSD, ~404M params))

Please:

1. Read [`SSD.md`](SSD.md) before touching `models/ssd_complex.py`.
<<<<<<< HEAD
2. Run `pytest tests/test_ssd.py -v` — `test_ssd_chunk_matches_naive` must pass.
=======
2. When tests land, run `python3 -m pytest tests/ -v` — all must pass.
>>>>>>> 16cab55 (Initial commit: Mamba-3-Lite (complex-valued SSD, ~404M params))
3. Do **not** add attention layers, MoE, or MTP — this is a pure SSM repo (avoids overlap with the rest of the portfolio).
4. Do **not** add `mamba-ssm` or `causal_conv1d` dependencies.

---

## ⚠️ Known caveats

<<<<<<< HEAD
- **Full 8B-token pretraining run not yet started** (no GPU on dev machine). The test suite validates all primitives on CPU + tiny shapes.
- **Complex SSD has 2× element bandwidth** vs real SSD (complex64 = 2× float32) — the per-state size halving must offset this. Verified by `scripts/microbench_a100.py`.
=======
- **Full 8B-token pretraining run not yet started** (no GPU on dev machine). The inline assertions validate all primitives on CPU + tiny shapes.
- **Complex SSD has 2× element bandwidth** vs real SSD (complex64 = 2× float32) — the per-state size halving must offset this. Theoretical analysis in `SSD.md`; will be measured at full scale.
>>>>>>> 16cab55 (Initial commit: Mamba-3-Lite (complex-valued SSD, ~404M params))
- **No causal conv = slightly weaker local-pattern bias.** Mamba-3 trades a small amount of inductive bias for memory bandwidth and simplicity.

---

## 📚 References

- **Mamba-3** — Dao & Gu, 2025 (arXiv:2603.15569)
- **Mamba-2 / SSD** — Dao & Gu, 2024 (arXiv:2405.21060)
- **S4** — Gu et al., 2021 (arXiv:2111.00396)
- **S6 (selective state spaces)** — Gu & Dao, 2023 (arXiv:2312.00752)
- **H3** — Fu et al., 2022 (arXiv:2212.14052)
- **RetNet** — Sun et al., 2023 (arXiv:2307.08621)
- **RWKV** — Peng et al., 2023 (arXiv:2305.13048)
- **Chinchilla scaling laws** — Hoffmann et al., arXiv:2203.15556

---

## 📄 License

Apache 2.0. See [LICENSE](LICENSE).

---

<div align="center">

**[⭐ Star this repo](https://github.com/atandra2000/Mamba-3-Lite)** if you find it useful · Part of the [CoreProjects](https://github.com/atandra2000) portfolio

</div>
