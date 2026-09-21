# Mamba-3-Lite — Glossary

Two tables: the **notation** used across the docs and code (all values from
`configs/pretrain_a100_400m.yaml` and the derived accounting in
[`../concepts/block-and-stability.md`](../concepts/block-and-stability.md)), then
terms grouped by subsystem. Symbols are cited as `file.py:Symbol` anchors; the
[doc gate](../README.md) resolves every one.

## Notation

| Symbol | Meaning | Value in this repo |
|---|---|---|
| $L$ | Transformer layers (`n_layers`) | 28 |
| $d$ | Model width (`d_model`) | 1024 |
| $H$ | SSM heads (`n_heads`) | 16 |
| $D$ | Per-head channel dim (`head_dim`) | 64 |
| $N$ | SSM state dim (`state_dim`), **complex** | 64 (→ 128 real sub-states) |
| $C$ | SSD chunk size (`chunk_size`) | 64 (the one tunable knob) |
| $f$ | SwiGLU intermediate (`ffn_dim`) | 2048 = 2·d (Mamba-family 2× convention, not 4×) |
| $V$ | Vocab size | 50,257 (GPT-2 BPE; EOS/PAD = 50,256) |
| $T$ | Sequence length (`max_seq_len`) | 2048 |
| $\bar A$ | Per-head discretized transition $\exp(\text{softplus}(\Delta)\,A)$ | `models/ssd_complex.py:_discretise` |
| $L$ matrix | Intra-chunk causal segment matrix $L[l,s]=\exp(A_{cs}[l]-A_{cs}[s])\cdot\mathbf 1[l\ge s]$ | `models/ssd_complex.py:ssd_complex_chunkwise` |

## Canonical numbers

| Quantity | Value | Source |
|---|---|---|
| Total parameters | 433,662,400 exact (~434M) | config comment + parameter-count window test (`tests/test_transformer.py`) |
| Per-layer parameters | 13,649,936 = in_proj 5,259,264 + MIMO 1,048,576 + out_proj 1,048,576 + FFN 6,291,456 + A 16 + 2 norm gains 2,048 | [`../concepts/block-and-stability.md`](../concepts/block-and-stability.md) §5.3 |
| Tied embedding/head | 51,463,168 = 50,257 × 1024 (one shared `Parameter`) | `models/transformer.py:Mamba3Transformer` |
| `in_proj` width | 5,136 = $H(D+4N+1)$ = 16 × 321 | `models/mamba_block.py:Mamba3Block` |
| Tokens per optimizer step | 65,536 = 16 micro × 2 accum × 2048 seq | `configs/pretrain_a100_400m.yaml` |
| Steps / token exposures | 256,000 steps → 16.78B token exposures ≈ 2 epochs over the 8.0B-token corpus | `configs/pretrain_a100_400m.yaml` |
| Optimizer | lr 3e-4, cosine decay to 5% (`min_lr_ratio 0.05`), warmup 2,000 steps, betas (0.9, 0.95), wd 0.1 (≥2-dim params only — `A` and norm gains are no-decay), grad clip 1.0 | `training/pretrain.py:Pretrainer` |
| Checkpoint cadence | every 4,000 micro-steps; three files per step (`model_step_N.safetensors`, `optim_step_N.pt`, `meta_step_N.json`) | `training/pretrain.py:TrainingConfig`, `utils/checkpoint.py:CheckpointManager` |
| NaN guard | skip step before backward; rollback after 5 consecutive non-finite losses (`nan_guard_max_consecutive`) | `training/pretrain.py:train_step` |
| Shards | 50M tokens each, raw uint32 LE, ≈190 MiB; 160 shards cover the 8.0B corpus | `data/data_config.yaml`, [`../training.md`](../training.md) |
| Hardware target | 1× A100 80GB SXM, ~12–15 h at 35–40% MFU `[INFERENCE]` — design targets, no run yet | `configs/pretrain_a100_400m.yaml` header |

## State spaces & SSD

- **State space model (SSM)** — a linear recurrence $h_t = \bar A_t h_{t-1} + \bar B_t x_t$, $y_t = C_t h_t$; the arc from RNN through S4/S6 to Mamba is traced in [`../concepts/state-space-foundations.md`](../concepts/state-space-foundations.md).
- **Selective scan (S6)** — making $B$, $C$, $\Delta$ functions of the token so the model chooses what to remember; Mamba-2's contribution.
- **SSD theorem / structured state-space duality** — the recurrence's output equals an attention-like form with an exponential kernel; chunking turns the O(T) scan into matmuls (§4 of [`../concepts/ssd-theory.md`](../concepts/ssd-theory.md)).
- **Chunkwise evaluation** — split $T$ into chunks of $C$; per chunk compute intra-chunk outputs ($Y^{\mathrm{diag}}$ via the causal segment matrix $L$), per-chunk end states, then a cheap chunk-level scan carrying states forward (`decay_chunk`). Implemented in `models/ssd_complex.py:ssd_complex_chunkwise`; proven equal to the naive scan by `tests/test_ssd.py::test_chunkwise_matches_naive_complex`.
- **O(T) oracle** — `models/ssd_complex.py:ssd_naive_complex`, the unrolled sequential recurrence that the chunkwise path must match exactly (AGENTS.md rule #3).
- **`per_chunk_ssd_pytorch`** — the eager PyTorch entry point of the chunkwise pass; the default dispatch target.

## Complex states (the Mamba-3 move)

- **Complex64 state** — the recurrence runs in `torch.complex64`; there is no BF16 complex dtype, which is why the block casts slices to FP32 before packing (`torch.complex(B_real, B_imag)`).
- **N-halving packing** — one complex state component = two real sub-states (a decay + a rotation), so N=64 complex carries 128 real sub-states — the parity argument vs Mamba-2's N=128. A *design claim from the Mamba-3 paper*; this repo proves correctness against the oracle, not perplexity parity (no pretrain run yet — see [`../AUDIT.md`](../AUDIT.md)).
- **Even-N constraint** — complex packing is exact only if N decomposes into real pairs; AGENTS.md hard rule #4 forbids odd N.
- **Per-head complex $A$** — a single complex64 scalar per head (init $-1.0$, i.e. real −1, imag 0) producing both damping $e^{\alpha\Delta}$ and rotation $\beta\Delta$: a damped oscillator. Mamba-2's $A$ is a real $(H, N)$ tensor — decay only, no rotation. See `models/mamba_block.py:Mamba3Block`.
- **Wirtinger gradients** — autograd over complex tensors; `A` is the only complex parameter, its gradient is complex64, and being 1-D it lands in the optimizer's no-decay group.
- **Contractive-at-init** — with $\Re A = -1$ and softplus > 0, every per-step decay factor has magnitude ≤ 1; the forward pass is a composition of contractions until weight drift pushes $\Re A > 0$ (the overflow mechanism the NaN guard exists for).

## Block anatomy

- **`in_proj` packing** — one bias-free `Linear(1024 → 5136)` emitting, head-major: content `x_ssm` (HD), `B_real`/`B_imag`, `C_real`/`C_imag` (4·HN), and `dt` (H). Slice table in [`../concepts/block-and-stability.md`](../concepts/block-and-stability.md) §4.2.
- **MIMO head mixing** — `models/mimo.py:MIMO`: flatten heads, one bias-free `Linear(H·D → H·D)` initialized to the identity (`nn.init.eye_`, flagged `_identity_init` so the blanket re-init skips it); the one place heads communicate. Replaces the SISO constraint.
- **No causal convolution** — Mamba-1/2's depthwise conv (kernel 4) is dropped entirely; locality now reaches the gates only through the recurrent state. An inductive-bias tradeoff whose quality impact is `[INFERENCE]`.
- **RMSNorm, pre-norm** — scale-invariant, mean-free; both lanes start with a norm, residuals stay exact identities.
- **SwiGLU FFN (2× expansion)** — `ffn_gate_up` split into gate/up, `SiLU(gate) · up`, `ffn_down`; 2,048-wide per the Mamba-family 2× convention.
- **Gradient checkpointing** — one checkpoint per block forward, `use_reentrant=False`, gated on `self.training`; no per-layer cadence exists (a single global boolean).

## Precision & stability

- **BF16 autocast (CUDA only)** — projection GEMMs and logits run BF16; the CPU path (what tests exercise) runs FP32. The residual stream stays an FP32 accumulator via per-add promotion.
- **FP32 gating region** — the `.float()` casts after slicing are load-bearing: `torch.complex` needs FP32 inputs and the complex scan must never see BF16-rounded arguments.
- **complex64 state** — `models/ssd_complex.py:ssd_complex_chunkwise` operands; autocast never casts complex, so the scan keeps FP32-pair precision.
- **TF32** — enabled for real FP32 GEMMs on Ampere; complex cuBLAS GEMMs are unaffected. The Triton kernel's `tl.dot` precision is an open GPU-box verification item.
- **No GradScaler** — BF16's FP32-matched range makes loss scaling unnecessary; FP16 was rejected on range, not precision.
- **NaN guard + rollback** — finiteness checked on the FP32 loss *before* backward; a poisoned micro-step zeroes the whole accumulation window; 5 consecutive failures restore the last complete checkpoint (`CheckpointManager.latest_step()` refuses torn saves).

## Data & tokenizer

- **GPT-2 BPE** — pretrained tokenizer, vocab 50,257, EOS/PAD 50,256, never retrained (`--skip-train-tokenizer` is forced on).
- **Shard format** — flat raw uint32 little-endian, 50M tokens, ≈190 MiB, atomic `.tmp` + rename, SHA-256 verified into the manifest.
- **Corpus cap** — 8.0B tokens (Chinchilla-optimal for ~434M params); 160 shards.
- **Dedup** — SHA-256 exact + Bloom filters, 256 hash buckets, 200k capacity per bucket, 1e-3 error rate.
- **Data mix** — the 5-source "mamba2-default" table in the yaml is *documented intent*; the live recipe is the workspace 7-source `shared_data/config/mixture.yaml` — reconcile before a real run (honesty note in [`../training.md`](../training.md)).

## Triton & dispatch

- **Sanctioned kernel** — `models/ssd_triton.py:per_chunk_ssd_triton`, the only custom kernel: fuses `L` materialisation, `Y_diag`, and per-chunk states into one kernel; the complex multiply is expanded into four real `tl.dot` GEMMs.
- **Two-layer opt-in** — `ssd_dispatch='triton'` on the model config **and** `ENABLE_TRITON_KERNELS=1` in the environment; missing either silently stays on the PyTorch path with a warning, and a kernel runtime error is surfaced, never swallowed during a run.
- **`HAS_TRITON` gate** — kernel code sits inside `if HAS_TRITON:`; docs must cite the module-level wrapper, not JIT-internal symbols (the doc gate checks this).

## Acronyms

| Acronym | Expansion |
|---|---|
| SSD | Structured State-Space Dual(ity) — the chunkable linear-recurrence family |
| SSM | State space model |
| SISO / MIMO | Single-/multi-input multi-output — here, the per-head independence constraint vs cross-head mixing |
| ZOH | Zero-order hold — continuous → discrete step |
| MFU | Model FLOPs utilization |
| BF16 / FP16 / FP32 / TF32 / complex64 | Floating formats; complex64 = FP32 real + FP32 imag |
| CE | Cross-entropy (the training loss, computed in FP32) |
| E2E | End-to-end (GPU smoke path, marker-gated) |
