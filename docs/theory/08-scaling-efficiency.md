# Scaling, Memory, and Throughput

This doc derives the 433,662,400-parameter count from the module code (not from memory), justifies the 8.0B-token Chinchilla-optimal training budget, and analyzes where the model's memory and FLOPs actually go — every throughput and MFU number is marked `[INFERENCE]`, because no `.benchmarks/` directory exists in this tree.

## 1. 60-second summary

After reading this doc you will understand: exactly where each of the 433,662,400 parameters lives (per-layer table plus the tied embedding and the final norm, summed to the last unit); why 8.0B tokens is the right *data* budget under the Chinchilla ≈20-tokens-per-parameter rule; why chunking caps the materialized causal-decay matrix at $C \times C$ per chunk instead of $T \times T$ (256 MiB vs 8 GiB per layer at the production config); why `complex64`'s 2× element bandwidth is exactly cancelled by halving $N$ from Mamba-2's 128 to 64; how `torch.compile` modes, micro-batch × accumulation, and grad checkpointing are wired in `training/pretrain.py:Pretrainer.__init__` and `models/mamba_block.py:Mamba3Block.forward`; what `models/ssd_triton.py:per_chunk_ssd_triton` fuses and why its backward is exact but slow; and the honest arithmetic behind the "12–15 h at 35–40% MFU" goal — which, at 6N FLOPs on a single A100, does not fit an 8.0B-token budget even at 100% MFU.

## 2. Why it exists

Every headline number in the repo's config header — "~434M params", "8.0B Chinchilla-optimal tokens", "~12-15 h at 35-40% MFU", "256000 steps ≈ 8.0B tokens" — is asserted, and at least two do not survive arithmetic. The parameter count hides a subtlety (the final RMSNorm's 1,024 parameters are invisible to the "28 × 13.65M + 51.5M" shorthand); the step count hides another (256,000 steps × 65,536 tokens/step exposes 16.78B tokens, ~2.1 epochs); and the wall-clock target hides a third (at 6N FLOPs/token, 8.0B tokens cannot run on one A100 in 12–15 h even at 100% MFU). This doc makes every one of those numbers *derivable* — from module shapes, the config, and first-principles FLOP accounting — so a discrepancy with a real run is diagnosable, not mysterious.

## 3. Intuition

Think of training cost as three ledgers with different units. The **parameter ledger** counts weights; it is fixed by architecture and is what the optimizer stores. The **memory ledger** counts live bytes during a training step; it is dominated by activations, which is why chunking and grad checkpointing matter. The **FLOP ledger** counts arithmetic; it determines wall-clock time through the machine's peak rate and the achieved fraction of it (MFU). The ledgers are related but not interchangeable: halving the state dimension cuts memory but barely changes the FLOP ledger (the projections dominate); chunking cuts the memory ledger quadratically in the chunk's local window but adds a small FLOP constant; `torch.compile` improves only the third ledger's utilization, never the first two.

## 4. The parameter ledger, derived

All numbers below are computed from the modules, not quoted. The model is `models/transformer.py:Mamba3Transformer`; each of its 28 blocks is a `models/mamba_block.py:Mamba3Block`.

**Per-layer components** (d_model 1024, H=16 heads, D=head_dim 64, N=state_dim 64, ffn_dim 2048):

| Component | Shape | Params | Derivation |
|---|---|---|---|
| `in_proj` | (5136, 1024) | 5,259,264 | width $H(D+4N+1) = 16(64+256+1) = 5136$; $5136 \times 1024$ |
| `out_proj` | (1024, 1024) | 1,048,576 | $H \cdot D = 1024$ both sides |
| `mimo.mix` | (1024, 1024) | 1,048,576 | MIMO is a full $H\cdot D \to H\cdot D$ linear (`models/mimo.py:MIMO`) |
| `ffn_gate_up` | (4096, 1024) | 4,194,304 | SwiGLU: $2 \times$ ffn_dim 2048 |
| `ffn_down` | (1024, 2048) | 2,097,152 | ffn_dim → d_model |
| `A` | (16,) complex64 | 16 | one complex scalar per head; `numel()` counts one element |
| `norm1` + `norm2` | (1024,) × 2 | 2,048 | RMSNorm gains |
| **Per layer** | | **13,649,936** | sum of the seven rows |

Sum: 5,259,264 + 1,048,576 + 1,048,576 + 4,194,304 + 2,097,152 + 16 + 2,048 = 13,649,936. Verified by instantiating the model and summing `p.numel()` per block — the table matches the module to the unit.

**Whole model:**

$$28 \times 13{,}649{,}936 = 382{,}198{,}208 \quad (\text{blocks})$$
$$\underbrace{50{,}257 \times 1024}_{= 51{,}463{,}168} \quad (\text{tied embed/head})$$
$$1024 \quad (\text{final } \texttt{norm\_f})$$
$$\boxed{\;382{,}198{,}208 + 51{,}463{,}168 + 1{,}024 = 433{,}662{,}400\;}$$

Three facts make this exact. (1) **The head is not an extra 51.5M**: `Mamba3Transformer` sets `self.lm_head.weight = self.embed.weight`, and `training/pretrain.py:count_parameters` sums over `model.parameters()`, which yields a shared `Parameter` object once — the tied weight is counted exactly once. (2) **The final `norm_f` contributes 1,024**; the config's rounded shorthand "28 × 13.65M + 51.5M" omits it — 13.65M × 28 + 51.5M = 433.66M rounds to the same headline, but the exact sum needs the norm. (3) The per-layer 13,649,936 reproduces the config's "13.65M/layer" comment to four significant figures.

## 5. The data ledger: Chinchilla tokens-per-parameter

Chinchilla (Hoffmann et al., 2022) fitted scaling laws $L(N,D) = a N^{-\alpha} + b D^{-\beta} + c$ over model sizes and token budgets and found that, for fixed compute, the loss-minimizing split keeps the ratio $D/N$ roughly constant at $\approx 20$ tokens per parameter. Derivation: under the compute constraint $C \approx 6ND$ (Section 7), minimizing $L$ over the split gives $N \propto C^{1/(\alpha+\beta)}$, $D \propto C^{1/(\alpha+\beta)}$ with fitted exponents $\alpha \approx 0.34$, $\beta \approx 0.28$ — hence a constant optimal ratio $D^*/N^* \approx 20$. Applying it here:

$$20 \times 433{,}662{,}400 = 8{,}673{,}248{,}000 \approx 8.7\ \text{B tokens.}$$

The config targets $D = 8.0 \times 10^9$ tokens — a round-numbered corpus, slightly *under* the heuristic (ratio $8.0\times10^9 / 4.336624\times10^8 \approx 18.4$ tokens/param). Whether 18.4 vs 20 matters for final loss is `[INFERENCE]`: the scaling laws are smooth, the exponent on $D$ is small, and no measured loss curve exists in this tree. The design rationale to record: 8.0B is "Chinchilla-optimal" in the sense of being within ~10% of the 20:1 heuristic at a round corpus size — the *quality* claim that this split is actually optimal for this architecture is `[INFERENCE]` and untested here.

**Step-count arithmetic worth doing twice.** Tokens per optimizer step: `micro_batch_size 16 × gradient_accumulation_steps 2 × max_seq_len 2048 = 65,536`. The yaml's `total_steps: 256000` therefore exposes

$$256{,}000 \times 65{,}536 = 16{,}777{,}216{,}000 \approx 16.8\ \text{B token-exposures} \approx 2.10 \times \text{the 8.0B corpus.}$$

If the intent is exactly 8.0B exposures, the step count should be $\lceil 8.0\times10^9 / 65{,}536 \rceil = 122{,}071$; the 256k schedule instead cycles the corpus ~2.1 times (`training/pretrain.py:PretrainDataset` has no epoch cap; `Pretrainer.train` loops until `max_steps`). Whether a second epoch helps or hurts is `[INFERENCE]` — flag it, don't hide it.

## 6. The memory ledger: chunkwise vs sequential, complex64, checkpointing

### 6.1 Sequential scan: bandwidth-bound, not capacity-bound

The oracle `models/ssd_complex.py:ssd_naive_complex` keeps a live state of shape $(B, H, N, D)$ in `complex64`. At the production config that state is

$$B \cdot H \cdot N \cdot D \cdot 8\ \text{B} = 16 \cdot 16 \cdot 64 \cdot 64 \cdot 8 = 8\ \text{MiB},$$

and every one of the $T = 2048$ steps reads it and writes it back: $2 \times 8$ MiB per step, so $32$ GiB of HBM traffic per layer per batch. The scan never *materializes* a big tensor — the problem is moving the same 8 MiB 2048 times, an arithmetic intensity near 1 FLOP per byte. Chunking replaces "one 8 MiB round trip per token" with "a few big GEMMs per chunk".

### 6.2 Chunkwise: the $C \times C$ cap on the causal mask

The chunkwise path (`models/ssd_complex.py:ssd_complex_chunkwise`) builds, for every chunk, the causal decay matrix $L[l,s] = e^{A_{cs}[l] - A_{cs}[s]}\mathbf{1}[l \ge s]$. Materialized across all chunks (as the PyTorch dispatch does), that is one $(B, n_c, H, C, C)$ complex64 tensor:

$$|L|_{\text{bytes}} = 8\, B\, n_c\, H\, C^2 = 8\, B\, H\, T\, C \qquad (n_c = T/C).$$

At $B{=}16, H{=}16, T{=}2048$: $C{=}64 \Rightarrow 256$ MiB per layer. The reason to chunk at all is what this formula does *not* contain: a full attention-style causal mask would be $T \times T$ per head,

$$8\, B\, H\, T^2 = 8 \cdot 16 \cdot 16 \cdot 2048^2 \approx 8\ \text{GiB per layer,}$$

so chunking cuts the mask ledger by exactly $T/C = 32$ — the mask size is $O(T \cdot C)$, not $O(T^2)$ (the compute-side half of this bargain, and the full derivation of $L$, is in [04-chunkwise-algorithm.md](04-chunkwise-algorithm.md)).

The other complex64 intermediates: the token content `Xc`, `Bc`, and `Cc` are promoted to complex64 inside the chunkwise function, so each carries 2× the bytes of its float32 source (`x_ssm` is 134 MiB as float32, 268 MiB as complex64 at the production batch). The full per-layer SSD intermediate set — `L` (256 MiB) + `Xc`/`Bc`/`Cc`/`Y_diag`/`states` (268 MiB each) — is roughly 1.6 GiB per layer, ~45 GiB across 28 layers, which is why Section 6.4 exists.

### 6.3 complex64 bandwidth, and why halving N pays for it

A `complex64` element is 8 bytes — two float32s. For a *fixed shape*, complex64 therefore moves 2× the bytes of float32. The offsetting move is the N-halving: Mamba-2's state is real with $N{=}128$; Mamba-3's is complex with $N{=}64$ (the capacity argument — two real coordinates per complex element — is derived in [03-complex-ssd.md](03-complex-ssd.md)). The state bytes are

$$\text{Mamba-3: } 8\,B\,H\,D\,N = 8 \cdot 16 \cdot 16 \cdot 64 \cdot 64 = 8\ \text{MiB},$$
$$\text{Mamba-2: } 4\,B\,H\,D\,(2N) = 4 \cdot 16 \cdot 16 \cdot 64 \cdot 128 = 8\ \text{MiB}.$$

Identical: the 2× per-element bandwidth of complex64 is cancelled exactly by halving the element count — $8 \times 64 = 4 \times 128$ bytes per (b, h, p) row — and the same cancellation holds for every state-shaped tensor. (Watch the units: complex64 is 8 B, not 16 B — that mistake doubles every estimate below.)

### 6.4 Grad checkpointing: recompute vs store

`models/mamba_block.py:Mamba3Block.forward` wraps the whole block in `torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)` whenever `grad_checkpoint and self.training`, and the yaml sets `grad_checkpoint: true` — so **every** layer is checkpointed (the "every Nth layer" cadence some configs use is not what this repo does; the knob is all-or-nothing per block). The tradeoff, quantified:

- **Store (no checkpointing):** keep every intermediate the backward needs — about 1.6 GiB/layer from Section 6.2, ≈ 45 GiB for 28 layers. With AdamW state (~3.4 GiB) + weights + grads + CUDA context, this fits an 80 GB card but is the dominant consumer.
- **Recompute (checkpointing):** store only the block input and output — $2 \times B \cdot T \cdot d\_model \times 4$ B = 256 MiB per layer, ≈ 7 GiB total. In the backward pass, `_forward_impl` is re-run once per layer, so the FLOP cost is one extra forward pass: total ≈ forward + backward + re-forward, i.e. roughly 1.5× the no-checkpoint FLOPs (the standard ~33% overhead). The SSD's L matrix is the biggest thing *not* stored — it is rebuilt during the recompute.

The right framing: grad checkpointing trades the memory ledger against the FLOP ledger at a rate of ~1.4 GiB saved per layer for ~1/3 more arithmetic. On an A100 80 GB it is the difference between a comfortable run and an OOM risk. `[INFERENCE]` no memory profile exists in this tree; the 1.6 GiB/layer figure is derived from tensor shapes, not measured.

## 7. The FLOP ledger: what the throughput levers actually do

### 7.1 The 6N rule and the MFU budget

A dense transformer does ≈ 2 FLOPs of forward per parameter per token (one MAC per weight = 2 FLOPs) and ≈ 4 FLOPs of backward, so training FLOPs ≈ $6ND$. For this model the projection stack dominates (in_proj 1024→5136, out_proj, MIMO, SwiGLU: ≈ 29 MFLOP/token/layer forward, consistent with $6 \times 13.65$M), while the SSD einsums add only ~1–2 MFLOP/token/layer (their cost is $O(T \cdot C)$, subquadratic). Hence

$$\text{FLOPs} \approx 6 \times 4.336624\times10^8 \times 8.0\times10^9 = 2.08\times10^{19}.$$

MFU is delivered FLOPs over peak FLOPs; the A100 80GB SXM BF16 dense peak is 312 TFLOPS (TF32 tensor cores: 156; the SSD's complex64 math runs at FP32/TF32 precision, so the *effective* peak for the mix is between 156 and 312). The resulting wall-clock table, all `[INFERENCE]`:

| MFU | Time for 8.0B tokens | Time for 16.8B exposures |
|---|---|---|
| 100% | 18.5 h | 38.9 h |
| 40% | 46.3 h | 97.2 h |
| 35% | 53.0 h | 111.2 h |

The config header's "~12–15 h at 35–40% MFU" does **not** appear in this table — 12–15 h would require 123–154% MFU of the BF16 peak. Under the 6N estimate, an 8.0B-token run on one A100 cannot finish in 15 h at any achievable utilization; the goal must have assumed a FLOP/token constant below 6N, or predated the current batch/step configuration. State it plainly: **12–15 h is a goal, not a prediction; if a real run hits it, the first thing to re-derive is the FLOP/token constant.** This is exactly the "MFU without measured FLOPs is a guess" pitfall from Section 12.

### 7.2 torch.compile: modes, wiring, and the cost of compiling

`training/pretrain.py:Pretrainer.__init__` compiles the model once, before training, and only on the CUDA path:

```python
training_model = raw_model
if config.compile_model and hasattr(torch, "compile"):
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", config.compile_mode)
    self._log(f"Compiling model with torch.compile (mode={compile_mode})...")
    training_model = torch.compile(training_model, mode=compile_mode, fullgraph=False)
```

The mode is read from the yaml (`training.compile_mode: "max-autotune"` in `configs/pretrain_a100_400m.yaml`), so it is genuinely wired from config — `main()` maps it into `TrainingConfig.compile_mode` — and `TORCH_COMPILE_MODE` overrides it at run time. `fullgraph=False` allows graph breaks (the Triton dispatch's fallback path is a graph-break candidate); `mode="max-autotune"` spends *compile* time autotuning kernels so *training* steps run fast. The lever's effect is on MFU only — fewer launches, fused ops — never on the 2.08e19 FLOP count.

The cost: `max-autotune` compile time is not training time. Compilation happens inside `Pretrainer.__init__`, and the first forward is a Triton compilation; a `--dry-run` (2 steps) with compile enabled therefore measures mostly compilation. Benchmark with `--no-compile` or a lighter `compile_mode`.

### 7.3 Batch arithmetic and the optimizer split

Tokens per step: 16 × 2 × 2048 = 65,536 (Section 5). The same `Pretrainer.__init__` builds the optimizer with a decay/no-decay split and a dedup that matters for the tied weight:

```python
seen = set()
all_params = []
for p in self.model.parameters():
    pid = id(p)
    if pid not in seen:
        seen.add(pid)
        all_params.append(p)
decay_params = [p for p in all_params if p.dim() >= 2]
no_decay_params = [p for p in all_params if p.dim() < 2]
self.optimizer = AdamW([
    {"params": decay_params, "weight_decay": config.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=config.lr, betas=(config.beta1, config.beta2), fused=False)
```

Weight-decay is applied to 2-D weights (every `Linear`, the embed/head) and withheld from 1-D parameters — the RMSNorm gains and the complex `A` vector — so the tied embed/head is decayed exactly once (the `id()` dedup guarantees the shared `Parameter` appears in one group). AdamW then stores two float32 moments per parameter: ~3.4 GiB for 433,662,400 parameters, the fixed part of the memory ledger that neither chunking nor checkpointing can shrink.

## 8. Code walkthrough: where the scaling knobs live

The three levers of Sections 6–7 are each one line in the config or one branch in the code.

**The block's shape knobs** (`models/mamba_block.py:Mamba3Block.__init__`): the projection width is computed, not hardcoded:

```python
in_dim = self.n_heads * (self.head_dim + 4 * self.state_dim + 1)
self.in_proj = nn.Linear(self.d_model, in_dim, bias=False)
```

$16 \times (64 + 256 + 1) = 5136$ — the row of the parameter table. `chunk_size` defaults to 64 (`cfg.get("chunk_size", 64)`), and `grad_checkpoint` defaults to False at the block level, flipped on by the training config (Section 6.4).

**The compile + optimizer wiring** (`training/pretrain.py:Pretrainer.__init__`) is Sections 7.2–7.3 verbatim, and `training/pretrain.py:count_parameters` — `sum(p.numel() for p in model.parameters())` — is the exact counter Section 4 used. `Pretrainer.__init__` logs its result: `Parameters: 433,662,400 total / 433,662,400 trainable` at the production config — the runtime echo of the table in Section 4.

**The Triton dispatch guard** (`training/pretrain.py:_enforce_triton_env_var`) force-backs `ssd_dispatch='triton'` to `'pytorch'` unless `ENABLE_TRITON_KERNELS=1` is set, and `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` additionally falls back per block (one warning) if the kernel path raises. The Triton path is therefore an *opt-in performance layer*; correctness never depends on it.

## 9. The Triton path economics

`models/ssd_triton.py:per_chunk_ssd_triton` is the public entry point; it calls the autograd `Function` `models/ssd_triton.py:_PerChunkSSDTriton`, whose forward launches one Triton program per `(B, chunk, H)` — 8192 programs at the production config. What the kernel fuses is exactly the per-chunk work the PyTorch dispatch does as three separate tensor passes plus a materialized `L`:

1. **`L` construction** — `tl.cumsum` over the chunk axis of the split real/imag `A_log`, then `exp` of the pairwise difference with a causal mask, all in registers/SRAM;
2. **`Cb = Cc @ Bcᵀ` and `Y_diag = (L ⊙ Cb) @ Xc`** — the complex GEMMs decomposed into real `tl.dot` pairs (4 real GEMMs per complex GEMM);
3. **`state = Xcᵀ @ (decay_states ⊙ Bc)`** — the third `tl.dot`.

The HBM win: the PyTorch path writes `L` (256 MiB/layer, Section 6.2) to HBM and reads it back for the einsum; the kernel builds `L` on-chip and consumes it immediately, so the only HBM traffic per program is loading that chunk's `Bc`/`Cc`/`Xc`/`A_log`/`decay_states` and writing `Y_diag` + `state`. One unavoidable host cost: `models/ssd_triton.py:_view_real_imag` splits each complex64 tensor into two contiguous float32 buffers (Triton has no complex pointer type), a one-time copy per forward.

**The backward is exact but slow.** `_PerChunkSSDTriton.backward` does not run a kernel. It re-executes the pure-PyTorch reference `models/ssd_triton.py:per_chunk_ssd_pytorch` on detached inputs and seeds `torch.autograd.grad` with the *true* downstream gradients:

```python
g_y = grad_y_diag if grad_y_diag is not None else torch.zeros_like(Y_diag_ref)
g_s = grad_state if grad_state is not None else torch.zeros_like(state_ref)
grads = torch.autograd.grad(
    (Y_diag_ref, state_ref), (b, c, x, a, d),
    grad_outputs=(g_y, g_s), allow_unused=True,
)
```

This is exact — the gradients match the PyTorch dispatch grad-for-grad (machine-checked, Section 13) — because it recomputes the same einsum math rather than approximating it. The economics: the forward is fast (fused kernel), the backward is a full reference forward *plus* autograd through it, i.e. several times the fused forward's cost and none of its memory savings. The v2 plan is a fused backward kernel with the same recompute structure moved into Triton; the design and status live in [../reference/03-ssd-triton.md](../reference/03-ssd-triton.md) (the migrated successor of the retired kernel design doc).

## 10. chunk_size as the tunable knob

`chunk_size` is the one hyperparameter that trades the memory ledger against the FLOP ledger directly, and the only one with a hard bound on the Triton path. Derived numbers at $B{=}16, H{=}16, T{=}2048$ (complex64, 8 B):

| C | n_chunks | L bytes/layer | intra-chunk GEMM size | inter-chunk scan length |
|---|---|---|---|---|
| 32 | 64 | 128 MiB | 32×32 | 64 chunks |
| 64 | 32 | 256 MiB | 64×64 | 32 chunks |
| 128 | 16 | 512 MiB | 128×128 | 16 chunks |
| 256 | 8 | 1,024 MiB | 256×256 | 8 chunks |

Reading the table: L grows linearly in C ($O(T \cdot C)$), the per-chunk GEMMs get bigger and more tensor-core-friendly, and the scan over chunks shrinks (fewer, longer hops). Smaller C minimizes memory but starves the GEMMs (C=32 blocks are marginal for `tl.dot`); larger C saturates the tensor cores but grows the L ledger and the $O(C^2)$ intra-chunk build; C=256 is the Triton cap via `_check_block_dims`, and the PyTorch dispatch has no cap (it just runs out of memory). The repo's C=64 default sits near the knee of the memory curve while keeping 64×64 GEMMs above tensor-core minimum sizes. The *perf* columns (what MFU each C delivers) are `[INFERENCE]`; there is no measured sweep in the tree, and [../guides/03-tuning.md](../guides/03-tuning.md) describes how to run one.

## 11. The data mix at scale

The corpus is 8.0B tokens (Section 5), assembled by `data/DATA_PIPELINE.md` with the universal workspace pipeline (`data/prepare_data.py` shim → vendored `shared_data`):

| Source | Weight | Tokens |
|---|---:|---:|
| FineWeb-Edu | 0.50 | 4.00 B |
| FineWeb | 0.20 | 1.60 B |
| the-stack-python | 0.15 | 1.20 B |
| OpenMathInstruct-2 | 0.10 | 0.80 B |
| arxiv | 0.05 | 0.40 B |
| **Total** | **1.00** | **8.00 B** |

The design intent (all `[INFERENCE]` for quality): a small model is data-hungry but capacity-poor, so the mix leans heavily on filtered web text (FineWeb-Edu at half the corpus) to maximize signal per token; code and math (0.25 combined) supply structured, self-checkable patterns; arxiv supplies long-range formal text that exercises the SSM's state. The tokenizer is GPT-2 BPE (vocab 50,257, EOS/PAD id 50,256), the smallest of the portfolio's tokenizers — the physical corpus is therefore larger per token than a LLaMA-3 tokenizer's, a storage consideration documented in `data/DATA_PIPELINE.md`. And `data.max_tokens: 8000000000` caps the pipeline; the 256k-step schedule still exposes the corpus ~2.1 times (Section 5).

## 12. Pitfalls

1. **The param count includes the tied embed exactly once.** `lm_head.weight` *is* `embed.weight` (same `Parameter` object, `data_ptr` identity), so `count_parameters` sees it once — and the exact total is 433,662,400, *including the final `norm_f`'s 1,024*: the rounded shorthand "28 × 13.65M + 51.5M" leaves you 1,024 short. Also: the config *filename* `pretrain_a100_400m.yaml` and the test name in Section 13 still say "400m" — historical, do not rename.
2. **MFU without measured FLOPs is a guess.** There is no `.benchmarks/` in this tree. Every MFU number here is derived from the 6N rule and the 312 TFLOPS A100 peak, and the derived wall-clock (46–53 h at 35–40% for 8.0B tokens) contradicts the 12–15 h goal. Treat the goal as a target and the table as the arithmetic; a real run is the only arbiter.
3. **torch.compile `max-autotune` compile time is not training time.** Compilation happens in `Pretrainer.__init__` and the first step includes Triton autotuning; `--dry-run` (2 steps) with compile enabled mostly measures compilation. Benchmark with `--no-compile` or a lighter `compile_mode`.
4. **complex64 is 8 bytes, not 16.** Every memory number in Sections 6–10 uses 8 B (two float32s); using 16 B doubles the ledger and breaks the "N=64 complex = N=128 real bytes" equality.
5. **256,000 steps ≠ 8.0B tokens.** At 65,536 tokens/step the schedule exposes 16.78B (2.1 epochs); exactly 8.0B needs ~122,071 steps.
6. **Grad checkpointing is all-or-nothing per block in this code** — every layer recomputes when `grad_checkpoint: true` (yaml default), at ~+33% FLOPs. A per-Nth-layer cadence is a code change, not a config field.
7. **The Triton path has a 256-cap** on `P`/`N`/`chunk_size` (`_check_block_dims`) and requires `ENABLE_TRITON_KERNELS=1` or `_enforce_triton_env_var` silently force-backs to PyTorch. And the fused *forward* is fast while the backward is the reference implementation — measure end-to-end before assuming the whole step is faster.
8. **MFU baseline ambiguity.** 312 TFLOPS is the BF16 dense peak; the SSD's complex64 math runs at FP32/TF32 precision (156 TFLOPS TF32 peak), so the *achievable* ceiling for this model is below 312 TF. State which peak an MFU number is against.

## 13. Tests

The claims of this doc are pinned by three kinds of tests:

- **The block-dimension cap and the production shape** — `tests/test_ssd_triton.py::TestPerChunkSsdImportSurface::test_check_block_dims_accepts_production_404m_shape` calls `_check_block_dims(P=64, N=64, chunk_size=64)` and must pass; the *name* says "404m" (historical), the dims are the production 64/64/64 — do not rename. Its sibling `tests/test_ssd_triton.py::TestPerChunkSsdImportSurface::test_check_block_dims_raises_value_error_on_too_large_dim` pins the 256-cap (`P=512` raises `ValueError`).
- **The Triton backward's exactness** — `tests/test_ssd_triton.py::TestPerChunkSsdAutogradPlumbing::test_backward_gradcheck_cpu` gradchecks the autograd `Function` on CPU with the kernel forward substituted by the reference, and `test_backward_propagates_to_content_path_cpu` asserts the token-content path gets a nonzero gradient (the `None`-grad-for-`Xc` regression); on GPU, `tests/test_ssd_triton.py::TestPerChunkSsdKernelGPU::test_backward_matches_pytorch_dispatch` compares Triton gradients grad-for-grad against PyTorch. The env-var guard is pinned by `tests/test_ssd_triton.py::TestEnableTritonKernelsForceBack::test_triton_dispatch_forced_back_when_env_var_missing`.
- **The chunkwise-vs-oracle equivalence** (the memory analysis of Section 6 is only sound because the chunkwise path computes the same function): `tests/test_ssd.py::test_chunkwise_matches_naive_complex` and `tests/test_ssd.py::test_chunkwise_matches_naive_time_varying_dt` — the latter is the regression for the inter-chunk propagation form (see [04-chunkwise-algorithm.md](04-chunkwise-algorithm.md)).

The parameter table in Section 4 is verified by construction — it is computed from the modules, and `Pretrainer.__init__` logs the same 433,662,400 via `count_parameters`; there is no dedicated unit test pinning the count, so the table is the authoritative derivation.

## Anchors cited

- `models/mamba_block.py:Mamba3Block`
- `models/mamba_block.py:Mamba3Block.__init__`
- `models/mamba_block.py:Mamba3Block.forward`
- `models/mamba_block.py:Mamba3Block._forward_impl`
- `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`
- `models/mimo.py:MIMO`
- `models/ssd_complex.py:ssd_complex_chunkwise`
- `models/ssd_complex.py:ssd_naive_complex`
- `models/ssd_triton.py:_check_block_dims`
- `models/ssd_triton.py:_PerChunkSSDTriton`
- `models/ssd_triton.py:_PerChunkSSDTriton.backward`
- `models/ssd_triton.py:_view_real_imag`
- `models/ssd_triton.py:per_chunk_ssd_pytorch`
- `models/ssd_triton.py:per_chunk_ssd_triton`
- `models/transformer.py:Mamba3Transformer`
- `training/pretrain.py:PretrainDataset`
- `training/pretrain.py:Pretrainer`
- `training/pretrain.py:Pretrainer.__init__`
- `training/pretrain.py:Pretrainer.train`
- `training/pretrain.py:TrainingConfig`
- `training/pretrain.py:_enforce_triton_env_var`
- `training/pretrain.py:count_parameters`
