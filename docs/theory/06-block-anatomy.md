# The Mamba-3 Residual Block: Anatomy of One Layer

This doc dissects the composable unit of Mamba-3-Lite — `models/mamba_block.py:Mamba3Block` — deriving the input-projection layout, the exact slice boundaries, the `A` parameterization, and the full tensor-shape data flow through one of the 28 blocks that make up `models/transformer.py:Mamba3Transformer`.

## 1. 60-second summary

After reading this doc you will understand: why one block is `RMSNorm → in_proj → complex-SSD → MIMO → out_proj → +residual → RMSNorm → SwiGLU → +residual`; how the single input projection packs token content, complex input/output weights, and step sizes into one tensor of width $H(D + 4N + 1)$ and exactly where each slice lives; why `x_ssm` stays float32 while `B_t`/`C_t` become complex64; why `A` is one complex scalar per head (init −1.0) instead of Mamba-2's per-state real `A`; what dropping the causal convolution costs and buys; how gradient checkpointing is wired (and why it only fires in training mode); and the shape of every tensor at every stage of the block.

## 2. Why it exists

A 434M-parameter model is not a single mathematical object — it is a stack. `models/transformer.py:Mamba3Transformer` builds `n_layers = 28` identical blocks and passes the residual stream through them in a loop. The block is therefore the *unit of composition*: almost everything you will ever debug — a shape error, a dtype mismatch, a gradient that fails to flow, a checkpoint that won't load — lives inside one block, not in the transformer shell. The shell contributes three things (embedding, final norm, tied head); the block contributes everything else.

This doc is the map for that unit. It answers the questions that come up the moment you modify anything: *"I changed the state_dim — which slice moves?"*, *"Why does this tensor have 5136 features?"*, *"Is this complex64 tensor really complex?"*, *"Why is my FFN only 2048 wide?"*, *"Did my new Linear just get re-initialized?"*. The math section derives the projection width from first principles so the slice boundaries are consequences, not memorized constants; the walkthrough then traces one token-batch through every line of `models/mamba_block.py:Mamba3Block._forward_impl`.

## 3. Intuition

Think of a block as a **two-lane highway with on-ramps and off-ramps**. The main road is the *residual stream*: a $(B, T, 1024)$ tensor that flows through all 28 blocks unchanged in shape, accumulating refinements. At each block, two service lanes branch off, do work, and merge back with an add:

- **Lane 1 (sequence mixing)**: `RMSNorm → in_proj → complex-SSD → MIMO → out_proj`. This lane is *linear* (every op is a matrix multiply or a linear recurrence) and it is the only lane that mixes *across time* — token $t$'s output depends on every token $s \le t$ through the recurrent state. This is where memory lives.
- **Lane 2 (channel mixing)**: `RMSNorm → SwiGLU →` back. This lane is *position-wise*: it applies the same nonlinear function to every token independently, with no cross-time information at all. It is the transformer-FFN half of the architecture.

Both lanes start with a norm (pre-norm ordering) and end by adding back onto the residual stream. Because the residual add is the identity map, gradients flow backward through the stream unimpeded — each lane's contribution to the loss is computed independently and the norm in front of each lane prevents the lane's internal activations from drifting in magnitude.

The mental model for the SSD lane: the projection takes one token vector and *packs* it into the arguments of a linear recurrence — the content to remember ($x$), the complex write weights ($B$), the complex read weights ($C$), and the step size ($\Delta t$). The recurrence itself is a per-head complex-valued state machine described in [docs/theory/01-ssm-foundations.md](01-ssm-foundations.md) and [docs/theory/03-complex-ssd.md](03-complex-ssd.md); the block does not care *how* the recurrence is evaluated (sequential scan, chunkwise einsums, or a Triton kernel) — it only cares about the interface: in $(B,T,H,D)$-shaped tensors, out a $(B,T,H,D)$ tensor of real outputs.

## 4. Math / layout: the input projection

### 4.1 Deriving the width $H(D + 4N + 1)$

The SSM branch needs, per head, a complete set of recurrence arguments. Let $H$ = number of heads, $D$ = head dimension (per-head channel width of the token content), $N$ = state dimension (complex states per head). Per head:

- **token content**: $D$ real values — the slice of the token that this head will write into and read from its state;
- **input weights $B$**: $N$ complex values — but complex64 stores real and imaginary parts separately, so the projection must emit $2N$ real values;
- **output weights $C$**: $N$ complex values, again $2N$ real values;
- **step size $\Delta t$**: $1$ real value.

Per head: $D + 2N + 2N + 1 = D + 4N + 1$ real features. Across $H$ heads:

$$\boxed{\;W_{\text{in\_proj}} = H\,(D + 4N + 1).\;}$$

With the model's verified dimensions ($H = 16$, $D = 64$, $N = 64$): $W = 16 \times (64 + 4\cdot64 + 1) = 16 \times 321 = 5136$. The projection is `nn.Linear(1024, 5136, bias=False)` — no bias, because a bias on the packed tensor would add a constant to $x$ but an *offset* to `dt`, which is wrong: `dt` is a learned step size whose semantics come from `softplus`, and a bias there is an unprincipled prior. The parameter count of the projection alone is

$$d_{\text{model}} \times W = 1024 \times 5136 = 5{,}259{,}264,$$

which is the largest single matrix in the layer (see Section 5.3 for how the pieces sum to the verified 13,649,936 params/layer).

### 4.2 The exact slice boundaries

`models/mamba_block.py:Mamba3Block._forward_impl` slices the packed tensor in head-major order. With $P = H\cdot D$, $Q = H\cdot N$, the offsets are:

| Slice | Python expression in `_forward_impl` | Closed form | Numeric (H=16, D=N=64) |
|---|---|---|---|
| `x_ssm` | `proj[..., :H * D]` | $[0,\; HD)$ | $[0,\; 1024)$ |
| `B_real` | `proj[..., H*D : H*D + H*N]` | $[HD,\; HD + HN)$ | $[1024,\; 2048)$ |
| `B_imag` | `proj[..., H*D + H*N : H*D + 2*H*N]` | $[HD + HN,\; HD + 2HN)$ | $[2048,\; 3072)$ |
| `C_real` | `proj[..., H*D + 2*H*N : H*D + 3*H*N]` | $[HD + 2HN,\; HD + 3HN)$ | $[3072,\; 4096)$ |
| `C_imag` | `proj[..., H*D + 3*H*N : H*D + 4*H*N]` | $[HD + 3HN,\; HD + 4HN)$ | $[4096,\; 5120)$ |
| `dt` | `proj[..., -H:]` | $[HD + 4HN,\; HD + 4HN + H)$ | $[5120,\; 5136)$ |

Two consistency checks make the table self-evidently right. First, the slices are contiguous and exhaustive: $HD + 4HN + H = H(D + 4N + 1) = W$. Second, `dt`'s `-H:` form is exactly the closed form: $HD + 4HN = 1024 + 4096 = 5120$ and $5136 - 16 = 5120$. The layout inside each $HN$-wide slice is head-major — head 0's $N$ values, then head 1's — which is why the downstream `.reshape(B, T, H, N)` (Section 6) is a no-copy view, not a permutation.

### 4.3 dtypes: float32 content, complex64 weights

The projection output `proj` is (nominally) float32, but the slices are *not* used uniformly:

- `x_ssm = proj[..., :H*D].reshape(B, T, H, D).float()` — token content, real, float32;
- `B_t = torch.complex(B_real, B_imag).reshape(B, T, H, N)` and likewise `C_t` — complex64;
- `dt = proj[..., -H:].float()` — real, float32.

The explicit `.float()` calls are load-bearing. Under the training loop's BF16 autocast (see [docs/theory/07-numerical-stability.md](07-numerical-stability.md)), the `in_proj` matmul executes in BF16, so `proj` arrives as bfloat16; the casts restore float32 before the complex path. The SSD core (`models/ssd_complex.py:ssd_complex_chunkwise`) maintains a complex64 state — there is no bfloat16 complex dtype in PyTorch — so this is exactly where BF16 ends and the float32 "gating region" begins. `torch.complex(real, imag)` builds a *new* complex64 tensor from two real tensors, so it imposes no stride constraint (contrast `view_as_complex`, Section 7).

### 4.4 The `A` parameter: per-head complex scalar vs Mamba-2's per-state

`models/mamba_block.py:Mamba3Block` declares

```python
self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
nn.init.constant_(self.A, -1.0)
```

one complex scalar per head, constant-initialized to $-1.0$. The recurrence uses it through the discretization `models/ssd_complex.py:_discretise`:

$$\bar A_t = e^{\operatorname{softplus}(\Delta_t)\, A},$$

where $A = \alpha + i\beta$ is a complex number. The per-step decay/rotation of the state is

$$|e^{(\alpha + i\beta)\Delta}| = e^{\alpha\Delta}, \qquad \arg\big(e^{(\alpha + i\beta)\Delta}\big) = \beta\Delta \pmod{2\pi},$$

so a single complex parameter produces *both* damping ($e^{\alpha\Delta}$, with $\alpha = -1$ at init) and oscillation ($\beta\Delta$ radians per token) — a damped oscillator (derived in [docs/theory/03-complex-ssd.md](03-complex-ssd.md)). Mamba-2's `A` is a real tensor of shape $(H, N)$ — a different decay rate per state component per head — which can only decay, never rotate. The tradeoff:

| | Mamba-2 | Mamba-3 (this repo) |
|---|---|---|
| Shape | $(H, N)$ real | $(H,)$ complex64 |
| Parameters (numel) | $H \cdot N = 1024$ | $H = 16$ — a $64\times$ reduction |
| Float storage | 1024 | 32 (16 complex × 2) — a $32\times$ reduction |
| Capability | per-state decay rates | per-head decay **and** rotation |

The design bet: per-head diversity in *frequency and damping* matters less than the per-state selectivity that `B`/`C` already provide (each of the $N$ state components is written and read with its own learned complex weight), so collapsing $A$ to one oscillator per head buys a large parameter cut (part of why the layer is only 13.65M params, [docs/theory/08-scaling-efficiency.md](08-scaling-efficiency.md)) and introduces rotation — a capability real-valued per-state $A$ simply does not have. A is also 1-dimensional, so it lands in the no-decay group of the optimizer (`training/pretrain.py:Pretrainer.__init__` splits on `p.dim() >= 2`), alongside the norm gains.

## 5. The components

### 5.1 Zero causal convolution

Mamba-1 and Mamba-2 apply a depthwise **causal convolution** (kernel size 4) to the projected input *before* the scan. Its job was local feature mixing: each channel's value at position $t$ becomes a learned combination of positions $t{-}3 \ldots t$, and — critically in Mamba-1 — `B`, `C`, and `dt` were computed from the *convolved* activations, giving every gate a 4-token local context window. It was also the mechanism that let the model use position information locally without the full state.

Mamba-3-Lite's block contains **no convolution at all** — `models/mamba_block.py:Mamba3Block` has no conv module, and the README's component table states `Causal conv | None`. The slot it occupied is filled by the per-token linear projection + chunkwise SSD: the recurrence's own mixing now carries local context (a token at position $s$ influences $y_t$ through the state for every $t \ge s$), and the chunked evaluation reorganizes that mixing into intra-chunk matmuls ([docs/theory/04-chunkwise-algorithm.md](04-chunkwise-algorithm.md)) rather than a sliding window. The *honest* framing: this is a deliberate inductive-bias tradeoff, and its quality impact is `[INFERENCE]` — this repo has no `.benchmarks/` tree, so nothing here measures the difference.

- **Given up**: the explicit 4-gram bias. In Mamba-3 the gates are strictly per-token at the projection level; local context reaches them only *through* the recurrent state, which is a softer, learned notion of locality than a hard conv window.
- **Gained**: (1) one fewer hyperparameter (kernel size); (2) memory bandwidth — the conv forces two extra full-sequence activation passes through HBM per layer, and in a chunked-SSD world that cost is pure overhead rather than expressivity; (3) a cleaner autograd surface (no conv means the only sequence op is the linear recurrence, which is exactly the op with a provably correct chunkwise equivalent).

### 5.2 RMSNorm, pre-norm

Both lanes start with `nn.RMSNorm(self.d_model, eps=self.rms_norm_eps)` ($\varepsilon = 10^{-5}$ by default, `models/transformer.py:ModelConfig.rms_norm_eps`). RMSNorm normalizes by the root-mean-square of the features, *without mean subtraction*:

$$\operatorname{RMSNorm}(x)_j = \frac{x_j}{\sqrt{\frac{1}{d}\sum_{k=1}^{d} x_k^2 \;+\; \varepsilon}}\; \gamma_j,$$

where $\gamma \in \mathbb{R}^d$ is a learned gain (no bias). Two properties justify it over LayerNorm here. First, **scale invariance**: $\operatorname{RMSNorm}(\lambda x) = \operatorname{RMSNorm}(x)$ for $\lambda > 0$, so the pre-norm block is immune to the residual stream's magnitude drift — the stream itself is never normalized, only the lane inputs are. Second, **mean subtraction is uninformative for token representations**: removing the per-token channel mean discards a real signal (the overall "activation level" of a token) and costs an extra reduction and a bias parameter. RMSNorm needs only $\sum x_k^2$ — one reduction — and the gain $\gamma$; LayerNorm needs the mean *and* variance reductions plus gain *and* bias. Placement is **pre-norm**: `norm1` before the SSD lane, `norm2` before the FFN lane, residual added after each, and a final `norm_f` before the head in `models/transformer.py:Mamba3Transformer.forward`. Pre-norm ordering keeps the residual path an exact identity, which is what makes 28-block gradient flow stable without per-block scaling.

### 5.3 SwiGLU FFN

The FFN lane is `models/mamba_block.py:Mamba3Block._ffn`:

```python
def _ffn(self, x: torch.Tensor) -> torch.Tensor:
    gate, up = self.ffn_gate_up(x).chunk(2, dim=-1)
    return self.ffn_down(F.silu(gate) * up)
```

One projection `ffn_gate_up: Linear(d_model, 2·ffn_dim)` (no bias) is split down the middle into `gate` and `up`, each $(B, T, \texttt{ffn\_dim})$; the nonlinearity is the gated product $\operatorname{SiLU}(\text{gate}) \odot \text{up}$; `ffn_down: Linear(ffn_dim, d_model)` closes the lane.

**Why `ffn_dim = 2048`, not 4096.** The README's component table states "SwiGLU, `ffn_dim=2048` (not 4096) — matches Mamba-2 design". Reading that claim against the code: `models/transformer.py:ModelConfig` sets `ffn_dim = 2048` and the config comment says "SwiGLU intermediate (NOT 4096)". 2048 = $2 \times d_{\text{model}}$; the Transformer-FFN convention is $4\times$ (which would be 4096 here). Mamba-2 has no separate FFN — its block is SSM-only with an inner dimension of $2 \times d_{\text{model}}$ (expansion factor 2). So the correct statement is: **the FFN uses the Mamba family's expansion-factor-2 budget** ($\texttt{ffn\_dim} = 2 \cdot d_{\text{model}}$), consistent with the README's "matches Mamba-2 design" — it is the 2× convention, not the 4× convention. The cost difference at this scale:

$$\underbrace{1024 \times 4096}_{g/u} + \underbrace{2048 \times 1024}_{down} = 6{,}291{,}456 \text{ params (2×)} \quad\text{vs}\quad 12{,}582{,}912 \text{ (4×)}.$$

The 2× FFN (6.29M params) closes the per-layer accounting exactly: in_proj 5,259,264 + MIMO 1,048,576 + out_proj 1,048,576 + FFN 6,291,456 + $A$ 16 + two RMSNorm gains 2,048 = **13,649,936**, the verified per-layer figure ($28 \times 13{,}649{,}936 + 51{,}463{,}168$ tied embed = 433,662,400 total, [docs/theory/08-scaling-efficiency.md](08-scaling-efficiency.md)). The FFN is the largest single cost center at ~46% of the layer — which is exactly why it is the natural target for the layer's only nonlinear position-wise computation: the SSD lane is linear and mixes time; the FFN is nonlinear and does not.

### 5.4 Gradient checkpointing

`models/mamba_block.py:Mamba3Block.forward` is a thin dispatcher:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """(B, T, d_model) -> (B, T, d_model)."""
    if self.grad_checkpoint and self.training:
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, use_reentrant=False
        )
    return self._forward_impl(x)
```

Checkpointing wraps the *entire block forward* as one recompute segment: in the backward pass, the saved activations are discarded and `_forward_impl` is re-run to regenerate them, trading one extra forward pass per block per step for a large activation-memory cut. `use_reentrant=False` selects the non-reentrant implementation (no `torch.no_grad` + in-place state dance, and it forbids in-place modification of the inputs inside the segment). The gate `and self.training` means evaluation never recomputes — inference runs the plain forward.

What is freed, per block at full config ($B = 16$, $T = 2048$): `proj` $(B,T,5136)$ ≈ 673 MB, `x_ssm` ≈ 134 MB, `B_t`/`C_t` as complex64 ≈ 537 MB each, the SSD output ≈ 134 MB, the gate/up projection ≈ 537 MB, the gated product ≈ 268 MB — on the order of **~2.8 GB per block**, so the 28-block activation stack drops from ~28× to roughly one block's worth (each checkpointed segment must keep only its input, which is the previous block's output). These are *derived from shapes*, not measured — there is no `.benchmarks/` tree.

**One correction to the expansion plan's outline**: it mentions "the every-4th-layer policy in the config". The code implements **no per-layer cadence** — `grad_checkpoint` is a single global boolean. The yaml sets `training.grad_checkpoint: true`; `training/pretrain.py:Pretrainer.__init__` merges it into the model config (`config.model_config.setdefault("grad_checkpoint", config.grad_checkpoint)`), and the CLI flag `--no-checkpoint` can force it off; every block then carries the same flag (asserted by `tests/test_grad_checkpoint.py::test_grad_checkpoint_propagates_to_blocks`). If a selective cadence ever appears, it will be a per-`layer_idx` decision inside `Mamba3Block` — it does not exist in the current tree.

## 6. Code walkthrough: full data flow

Everything above assembles into `models/mamba_block.py:Mamba3Block._forward_impl`. Reading it top to bottom, with shapes at full config ($B, T$ arbitrary; $H{=}16, D{=}64, N{=}64, f{=}2048, d{=}1024$, `chunk_size` $C{=}64$):

```python
def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.shape
    H, D, N = self.n_heads, self.head_dim, self.state_dim

    residual = x
    h = self.norm1(x)

    proj = self.in_proj(h)
    x_ssm = proj[..., :H * D].reshape(B, T, H, D).float()

    B_real = proj[..., H * D:H * D + H * N].float()
    B_imag = proj[..., H * D + H * N:H * D + 2 * H * N].float()
    B_t = torch.complex(B_real, B_imag).reshape(B, T, H, N)

    C_real = proj[..., H * D + 2 * H * N:H * D + 3 * H * N].float()
    C_imag = proj[..., H * D + 3 * H * N:H * D + 4 * H * N].float()
    C_t = torch.complex(C_real, C_imag).reshape(B, T, H, N)

    dt = proj[..., -H:].float()

    y = self._ssd_with_dispatch(x_ssm, B_t, C_t, dt)

    y = self.mimo(y)
    y = y.reshape(B, T, H * D)
    y = self.out_proj(y)
    x = residual + y

    residual = x
    h = self.norm2(x)
    h = self._ffn(h)
    x = residual + h

    return x
```

The shape table:

| # | Stage | Code | Shape | dtype |
|---|---|---|---|---|
| 1 | block input (residual stream) | `x` | $(B, T, 1024)$ | float32 |
| 2 | SSM lane norm | `self.norm1(x)` | $(B, T, 1024)$ | float32 |
| 3 | packed projection | `self.in_proj(h)` | $(B, T, 5136)$ | float32 (BF16 under autocast) |
| 4 | content slice | `proj[..., :H*D]` → `x_ssm` | $(B, T, 16, 64)$ | float32 (`.float()`) |
| 5 | complex input weights | `B_real/B_imag` → `B_t` | $(B, T, 16, 64)$ | complex64 |
| 6 | complex output weights | `C_real/C_imag` → `C_t` | $(B, T, 16, 64)$ | complex64 |
| 7 | step sizes | `proj[..., -H:]` → `dt` | $(B, T, 16)$ | float32 (`.float()`) |
| 8 | complex-SSD scan | `self._ssd_with_dispatch(...)` | $(B, T, 16, 64)$ — `Y.real` | float32 |
| 9 | MIMO head mixing | `self.mimo(y)` | $(B, T, 16, 64)$ | float32 |
| 10 | head flatten + project | `reshape(B,T,1024)` → `out_proj` | $(B, T, 1024)$ | float32 |
| 11 | SSM residual add | `x = residual + y` | $(B, T, 1024)$ | float32 |
| 12 | FFN lane norm | `self.norm2(x)` | $(B, T, 1024)$ | float32 |
| 13 | gate/up projection | `self.ffn_gate_up(h)` → `chunk(2)` | gate, up: $(B, T, 2048)$ | float32 |
| 14 | gated nonlinearity | `F.silu(gate) * up` | $(B, T, 2048)$ | float32 |
| 15 | FFN down-project | `self.ffn_down(...)` | $(B, T, 1024)$ | float32 |
| 16 | FFN residual add | `x = residual + h` | $(B, T, 1024)$ | float32 |

Notes on the walkthrough:

- **Row 8's return contract**: `models/ssd_complex.py:ssd_complex_chunkwise` returns the *real part* of the complex output, sliced back to the unpadded $T$ — the block never sees complex outputs. The complex-to-real handoff is deliberate: everything downstream (`MIMO`, `out_proj`, the residual stream, the head) is real-valued, and `Y.real` is what the recurrence's output equation produces for the real token stream. The internal chunkwise algebra — including the inter-chunk `decay_chunk` propagation with its strict-tril structure — is derived end to end in [docs/theory/04-chunkwise-algorithm.md](04-chunkwise-algorithm.md); the block merely passes the full-sequence arguments through.
- **Row 9**: `models/mimo.py:MIMO` — the head-mixing layer built by `models/mamba_block.py:Mamba3Block.__init__` — flattens $(B, T, H, D) \to (B, T, H{\cdot}D)$ in `models/mimo.py:MIMO.forward`, applies a single bias-free `Linear(H·D, H·D)` initialized to identity (`nn.init.eye_`), and reshapes back. This is the one place heads *talk to each other* — the SSD recurrence keeps heads independent, and MIMO is the linear cross-head mixer (theory in [docs/theory/05-mimo-mixing.md](05-mimo-mixing.md)). Its identity init is preserved against `models/transformer.py:Mamba3Transformer._init_weights`' blanket re-init via the `_identity_init` flag.
- **Rows 1–16 are per-block**. `models/transformer.py:Mamba3Transformer.forward` runs `embed → for layer in layers → norm_f → lm_head`; the SSD state is created fresh inside each chunkwise call (zero initial state), so the 28 blocks are **28 independent recurrences** — only the $(B, T, 1024)$ residual stream passes between them. There is no cross-block state, which is what makes the blocks freely composable and checkpointable.
- **Dispatch**: row 8 routes through `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`, which calls the chunkwise path directly when `ssd_dispatch != "triton"`, and otherwise tries the Triton backend with a one-shot warning + PyTorch fallback on any exception.

## 7. Pitfalls

1. **Slice arithmetic is off-by-one-hostile.** All five `B`/`C` slices have the same width $HN$, so a single misplaced index (say `H*D + H*N + 1` for `B_imag`) produces **no shape error** — it silently shifts every head's real/imag data by one element, corrupting the state machine. The only guard is the closed-form table in Section 4.2; when you change `N` or `D`, re-derive the offsets rather than editing one slice.
2. **`.float()` after slicing is load-bearing.** Under BF16 autocast, `proj` is bfloat16; without the casts, `torch.complex(B_real, B_imag)` operates on BF16 pairs (there is no BF16 complex dtype) and the SSD's complex64 state math silently loses precision or errors. Never "optimize away" these casts.
3. **Reshape order: $(B,T,H,N)$, never $(B,T,N,H)$.** The slice is head-major (head 0's $N$ values first), so `.reshape(B, T, H, N)` is a free view. Reshaping to $(B,T,N,H)$ also succeeds — and transposes state and head axes silently, producing garbage that allclose-style tests will catch only if they run.
4. **`torch.complex` vs `view_as_complex`.** The code deliberately builds complex tensors with `torch.complex(real, imag)` — no stride-2 requirement. If you "simplify" by interleaving the slices and calling `view_as_complex`, you inherit the stride-2 constraint: a non-unit-stride buffer either raises or, worse, misreads pairs.
5. **Gradient checkpointing requires training mode.** The `and self.training` gate means `model.eval()` silently bypasses the checkpoint path. A test that calls `backward()` in eval mode will run the un-checkpointed forward (correct gradients, no recompute) — and a memory regression will not reproduce.
6. **Anything you add to the block gets re-initialized.** `models/transformer.py:Mamba3Transformer._init_weights` recursively re-inits every `nn.Linear`/`nn.Embedding` to $\mathcal{N}(0, 0.02)$; if you add a Linear with a special init (like MIMO's eye), set `module._identity_init = True` or your init is overwritten at construction.
7. **`A` is the only complex parameter.** Its gradient is complex64 (Wirtinger), and being 1-D it sits in the optimizer's no-decay group — if you ever inspect optimizer state for `A`, expect no weight decay and complex-valued moments.

## 8. Tests

- `tests/test_transformer.py::test_mamba3_transformer_forward` — end-to-end block stack: builds a 2-layer `models/transformer.py:Mamba3Transformer`, checks the output shape $(2, 16, 100)$, the parameter-count window, and — the block-relevant invariant — that `embed.weight.data_ptr() == lm_head.weight.data_ptr()` (weight tying).
- `tests/test_transformer.py::test_mamba3_transformer_accepts_dict_config` — exercises the dict-config path that `models/mamba_block.py:Mamba3Block(cfg.__dict__, ...)` relies on.
- `tests/test_grad_checkpoint.py::test_grad_checkpoint_propagates_to_blocks` — asserts *every* block receives `grad_checkpoint=True` (the uniform-flag behavior of Section 5.4, not a per-layer policy).
- `tests/test_grad_checkpoint.py::test_grad_checkpoint_explicit_false_disables` — the flag-off path.
- `tests/test_grad_checkpoint.py::test_grad_checkpoint_actually_triggers_training_mode` — trains through the checkpoint path and asserts at least one parameter receives a finite gradient, proving the recompute backward actually flows.
- `tests/test_mimo.py::test_mimo_identity_init`, `test_mimo_identity_survives_transformer_init` — the identity-init contract of the SSM lane's tail, including its survival of `_init_weights`.
- The SSD core itself — the block's row-8 workhorse — is machine-proven against the O(T) oracle in `tests/test_ssd.py::test_chunkwise_matches_naive_complex` (walked through in [docs/theory/01-ssm-foundations.md](01-ssm-foundations.md) and [docs/theory/04-chunkwise-algorithm.md](04-chunkwise-algorithm.md)).

The block is the whole model: 28 copies, two lanes, one stream. Everything else is plumbing.
