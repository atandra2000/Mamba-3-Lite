# Mamba-3-Lite — Block Anatomy, Numerical Stability, and Scaling

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

The mental model for the SSD lane: the projection takes one token vector and *packs* it into the arguments of a linear recurrence — the content to remember ($x$), the complex write weights ($B$), the complex read weights ($C$), and the step size ($\Delta t$). The recurrence itself is a per-head complex-valued state machine described in [docs/concepts/state-space-foundations.md](../concepts/state-space-foundations.md) and [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md); the block does not care *how* the recurrence is evaluated (sequential scan, chunkwise einsums, or a Triton kernel) — it only cares about the interface: in $(B,T,H,D)$-shaped tensors, out a $(B,T,H,D)$ tensor of real outputs.

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

The explicit `.float()` calls are load-bearing. Under the training loop's BF16 autocast (see [docs/concepts/block-and-stability.md](../concepts/block-and-stability.md)), the `in_proj` matmul executes in BF16, so `proj` arrives as bfloat16; the casts restore float32 before the complex path. The SSD core (`models/ssd_complex.py:ssd_complex_chunkwise`) maintains a complex64 state — there is no bfloat16 complex dtype in PyTorch — so this is exactly where BF16 ends and the float32 "gating region" begins. `torch.complex(real, imag)` builds a *new* complex64 tensor from two real tensors, so it imposes no stride constraint (contrast `view_as_complex`, Section 7).

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

so a single complex parameter produces *both* damping ($e^{\alpha\Delta}$, with $\alpha = -1$ at init) and oscillation ($\beta\Delta$ radians per token) — a damped oscillator (derived in [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md)). Mamba-2's `A` is a real tensor of shape $(H, N)$ — a different decay rate per state component per head — which can only decay, never rotate. The tradeoff:

| | Mamba-2 | Mamba-3 (this repo) |
|---|---|---|
| Shape | $(H, N)$ real | $(H,)$ complex64 |
| Parameters (numel) | $H \cdot N = 1024$ | $H = 16$ — a $64\times$ reduction |
| Float storage | 1024 | 32 (16 complex × 2) — a $32\times$ reduction |
| Capability | per-state decay rates | per-head decay **and** rotation |

The design bet: per-head diversity in *frequency and damping* matters less than the per-state selectivity that `B`/`C` already provide (each of the $N$ state components is written and read with its own learned complex weight), so collapsing $A$ to one oscillator per head buys a large parameter cut (part of why the layer is only 13.65M params, [docs/concepts/block-and-stability.md](../concepts/block-and-stability.md)) and introduces rotation — a capability real-valued per-state $A$ simply does not have. A is also 1-dimensional, so it lands in the no-decay group of the optimizer (`training/pretrain.py:Pretrainer.__init__` splits on `p.dim() >= 2`), alongside the norm gains.

## 5. The components

### 5.1 Zero causal convolution

Mamba-1 and Mamba-2 apply a depthwise **causal convolution** (kernel size 4) to the projected input *before* the scan. Its job was local feature mixing: each channel's value at position $t$ becomes a learned combination of positions $t{-}3 \ldots t$, and — critically in Mamba-1 — `B`, `C`, and `dt` were computed from the *convolved* activations, giving every gate a 4-token local context window. It was also the mechanism that let the model use position information locally without the full state.

Mamba-3-Lite's block contains **no convolution at all** — `models/mamba_block.py:Mamba3Block` has no conv module, and the README's component table states `Causal conv | None`. The slot it occupied is filled by the per-token linear projection + chunkwise SSD: the recurrence's own mixing now carries local context (a token at position $s$ influences $y_t$ through the state for every $t \ge s$), and the chunked evaluation reorganizes that mixing into intra-chunk matmuls ([docs/concepts/ssd-theory.md](../concepts/ssd-theory.md)) rather than a sliding window. The *honest* framing: this is a deliberate inductive-bias tradeoff, and its quality impact is `[INFERENCE]` — this repo has no `.benchmarks/` tree, so nothing here measures the difference.

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

The 2× FFN (6.29M params) closes the per-layer accounting exactly: in_proj 5,259,264 + MIMO 1,048,576 + out_proj 1,048,576 + FFN 6,291,456 + $A$ 16 + two RMSNorm gains 2,048 = **13,649,936**, the verified per-layer figure ($28 \times 13{,}649{,}936 + 51{,}463{,}168$ tied embed = 433,662,400 total, [docs/concepts/block-and-stability.md](../concepts/block-and-stability.md)). The FFN is the largest single cost center at ~46% of the layer — which is exactly why it is the natural target for the layer's only nonlinear position-wise computation: the SSD lane is linear and mixes time; the FFN is nonlinear and does not.

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

- **Row 8's return contract**: `models/ssd_complex.py:ssd_complex_chunkwise` returns the *real part* of the complex output, sliced back to the unpadded $T$ — the block never sees complex outputs. The complex-to-real handoff is deliberate: everything downstream (`MIMO`, `out_proj`, the residual stream, the head) is real-valued, and `Y.real` is what the recurrence's output equation produces for the real token stream. The internal chunkwise algebra — including the inter-chunk `decay_chunk` propagation with its strict-tril structure — is derived end to end in [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md); the block merely passes the full-sequence arguments through.
- **Row 9**: `models/mimo.py:MIMO` — the head-mixing layer built by `models/mamba_block.py:Mamba3Block.__init__` — flattens $(B, T, H, D) \to (B, T, H{\cdot}D)$ in `models/mimo.py:MIMO.forward`, applies a single bias-free `Linear(H·D, H·D)` initialized to identity (`nn.init.eye_`), and reshapes back. This is the one place heads *talk to each other* — the SSD recurrence keeps heads independent, and MIMO is the linear cross-head mixer (theory in [docs/concepts/mimo.md](../concepts/mimo.md)). Its identity init is preserved against `models/transformer.py:Mamba3Transformer._init_weights`' blanket re-init via the `_identity_init` flag.
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
- The SSD core itself — the block's row-8 workhorse — is machine-proven against the O(T) oracle in `tests/test_ssd.py::test_chunkwise_matches_naive_complex` (walked through in [docs/concepts/state-space-foundations.md](../concepts/state-space-foundations.md) and [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md)).

The block is the whole model: 28 copies, two lanes, one stream. Everything else is plumbing.
---

#Numerical Stability in Mamba-3-Lite: Why Every Dtype, Norm, and Precision Choice Exists

This doc explains the defense layers that keep a 28-block complex-valued recurrence trainable for 256,000 steps: where BF16, TF32, FP32, and complex64 each live in the pipeline, the math behind the NaN guard and checkpoint rollback, and why weight tying and the init scheme are stability features rather than conveniences.

## 60-second summary

After reading this doc you will understand: why the model runs *mixed* precision rather than any single dtype — BF16 activations inside the projection GEMMs, FP32 everywhere that feeds the recurrence, complex64 inside the SSD scan, and an FP32 loss; why BF16 beats FP16 here (range beats precision when the critical path is already FP32); why `torch.set_float32_matmul_precision("high")` (TF32) is safe; exactly which `.float()` casts in `models/mamba_block.py:Mamba3Block._forward_impl` enforce FP32 for the recurrence, and where logits actually end up (BF16 under autocast, with the *loss* promoted to FP32); the mechanism of the NaN guard in `training/pretrain.py:train_step` (check before backward, skip the step, wipe the accumulation window) and the five-consecutive rollback in `training/pretrain.py:Pretrainer.train`; why rollback beats gradient clipping for a recurrence whose decay factors are $\exp$ of trainable sums; and how weight tying (`models/transformer.py:Mamba3Transformer.__init__`) plus the `N(0, 0.02)` / eye / constant-`−1` init keep the residual stream and its gradients well conditioned.

## 1. Why it exists

An SSM layer is a recurrence: every output is a function of every earlier position through products of per-step transition factors, which multiplies numerical sensitivity beyond a feed-forward network's. A rounding error in one step's decay factor is compounded by every later step; one `inf` contaminates the state for the rest of the sequence; and the backward pass differentiates through $\exp$ chains whose intermediates can exceed any sane range. A NaN mid-training wastes hours of A100 time — and in a 256,000-step, 8-billion-token run, *when* a failure happens matters as much as *whether* it can.

This repo treats stability as a layered problem:

1. **Precision placement** — run each computation in the cheapest dtype that cannot corrupt it (BF16 GEMMs, FP32 gating, complex64 state, FP32 loss).
2. **Contractive dynamics by construction** — at init, every decay factor has magnitude ≤ 1, so the recurrence provably cannot amplify on its own.
3. **A NaN guard with rewind** — detect non-finite loss *before* backward, skip the poisoned update, and after five consecutive failures restore the last checkpoint.

This doc derives each layer's shape; [04-chunkwise-algorithm](../concepts/ssd-theory.md) derives the chunkwise scan this stability math is about, [03-complex-ssd](../concepts/ssd-theory.md) why the state is complex, and [02-state-space-duality](../concepts/ssd-theory.md) the `L` matrix the decay analysis reuses.

## 2. Intuition: three defense layers

Think of a **precision budget** and a **failure budget**.

Precision: the only ops that tolerate low precision are the big projection GEMMs (`in_proj`, `out_proj`, `ffn_*`, `mimo`, `lm_head`). Their inputs are renormalized to $O(1)$ scale by RMSNorm, and their outputs are *deltas* added to a residual stream — a 0.4% rounding error in a delta is 0.4% of a small correction, not the state. Everything the recurrence depends on — the `dt` gate, the complex projections, the state — runs in FP32/complex64 ($2^{-24}$ rounding, not $2^{-8}$), and the residual stream is itself an FP32 accumulator: BF16 deltas are promoted on addition, so rounding never compounds across layers.

Failure: a divergence usually starts in the *weights* (a drift of $\Re A$ above zero, a `dt` spike), produces exponentially amplified forward activations, and only then surfaces as a non-finite loss. Clipping bounds the gradient norm but cannot repair the weights that caused the amplification; rewinding to the last checkpoint resets weights, optimizer moments, and scheduler to a known-good state. A five-step streak separates one transient NaN (skip it) from a genuine regime change (rewind).

## 3. The precision landscape

### 3.1 Where each dtype lives — verified flow

On CUDA the forward pass runs inside a BF16 autocast region produced by `training/pretrain.py:Pretrainer._amp_context`:

```python
def _amp_context(self):
    if torch.cuda.is_available():
        return autocast("cuda", dtype=self.amp_dtype)
    else:
        return autocast("cpu", enabled=False)
```

with `self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32` set in `Pretrainer.__init__` — BF16 on an A100 (`sm_80`); on CPU the region is disabled and everything runs FP32 (as the CPU test suite exercises).

Autocast casts *per op category*, not per graph. The verified dtype flow through a model structured exactly like `Mamba3Transformer` (probe run on torch 2.12 under `autocast("cpu", dtype=torch.bfloat16)`; CUDA BF16 autocast uses the same op categorization):

| Stage | Op | Result dtype |
|---|---|---|
| `self.embed(x)` | `nn.Embedding` | **float32** (gather; not in cast lists) |
| `self.norm1(x)` | `nn.RMSNorm` | **float32** (autocast promotes norms) |
| `self.in_proj(h)` | `nn.Linear` | **bfloat16** (matmul category) |
| `x_ssm = … .float()` | explicit cast | **float32** |
| `B_t, C_t = torch.complex(B_real, B_imag)` | cast + pack | **complex64** |
| `dt = proj[..., -H:].float()` | explicit cast | **float32** |
| SSD einsums (`ssd_complex_chunkwise`) | complex64 operands | **complex64** (autocast never casts complex) |
| `Y = Y.real` | view | **float32** |
| `self.mimo(y)` / `self.out_proj(y)` | `nn.Linear` | **bfloat16** |
| `x = residual + y` | elementwise add | **float32** (promotion) |
| `self.norm_f(x)` | `nn.RMSNorm` | **float32** |
| `logits = self.lm_head(h)` | `nn.Linear` | **bfloat16** |
| `F.cross_entropy(logits, …)` | `log_softmax` + `nll_loss` | **float32** (log_softmax promoted) |

Two facts are easy to get wrong:

- **The logits are BF16, not FP32.** The final `Linear` head is *not* outside autocast's scope — it is a matmul like any other, so its output is BF16. What is FP32 is the **loss**: autocast promotes `log_softmax` (hence `cross_entropy`), so the scalar the NaN guard inspects is always FP32. Since probabilities are never fed back into the recurrence, the BF16 rounding of logits is harmless.
- **The residual stream is an FP32 accumulator.** Each block's delta is BF16 but added to an FP32 residual with promotion per block — rounding never compounds across the 28 residual adds.

### 3.2 BF16 vs FP16: range vs precision

An IEEE binary format with $e$ exponent bits and $m$ explicit mantissa bits has unit roundoff $u = 2^{-(m+1)}$ and largest finite value $\approx 2^{2^{e-1}+1}$:

| Format | $e$ | $m$ | $u$ | max finite |
|---|---|---|---|---|
| FP16 | 5 | 10 | $2^{-11} \approx 4.9\times10^{-4}$ | $2^{16} \approx 6.6\times10^{4}$ |
| BF16 | 8 | 7 | $2^{-8} \approx 3.9\times10^{-3}$ | $2^{128} \approx 3.4\times10^{38}$ |
| FP32 | 8 | 23 | $2^{-24} \approx 6.0\times10^{-8}$ | $2^{128}$ |

FP16 has ~3 extra mantissa bits (8× finer relative precision) but its 5-bit exponent caps at 65,504. BF16 shares FP32's exponent field, so its range matches FP32's, paid for with 16 fewer mantissa bits.

Which is "safer" depends on what gets rounded. The precision-critical path never touches 16-bit arithmetic: the SSD runs in complex64 (two FP32 components) and `dt` is FP32. The 16-bit format only rounds projection GEMMs whose inputs are RMSNorm-normalized. What kills FP16 here is range, in three places:

1. **Loss scaling.** FP16 needs dynamic loss scaling (`GradScaler`) to keep underflowing gradients above $6\times10^{-8}$ — an extra feedback loop with its own silent failure mode. BF16 needs none, and the repo's discipline is explicitly "no GradScaler".
2. **Logits.** With 50,257 classes, logits reach tens in magnitude; FP16 values near 65,504 have ULP ≈ 32 — coarse quantization exactly where the softmax gradient is most sensitive.
3. **GEMM products.** Tensor cores accumulate in FP32 internally for both formats, but the products come from 16-bit-rounded inputs. BF16's $2^{-8}$ input rounding is acceptable *because the rounded quantity is a delta, not a state*.

Honest summary: BF16 trades 8× precision for ~10³⁰× range — correct here because the sensitive part of the graph never runs in 16 bits.

### 3.3 TF32: 19-bit mantissa truncation

In `Pretrainer.__init__`, on CUDA:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
```

TF32 is the Ampere-and-later tensor-core format: 8-bit exponent (FP32 range) but only 11 significant bits — 10 explicit plus the implicit one — versus FP32's 24. Its unit roundoff is $u_{\text{tf32}} = 2^{-11} \approx 4.9\times10^{-4}$; TF32 keeps the range and discards 13 mantissa bits, at roughly 8× the FP32 GEMM rate (vendor spec ratio; `[INFERENCE]` — no benchmark exists in this tree). `torch.set_float32_matmul_precision("high")` is torch's way of saying "FP32 matmuls may use TF32", equivalent to `allow_tf32` and additionally letting `torch.compile` pick TF32 kernels.

Why safe here? TF32 only truncates **real FP32 GEMMs**. The GEMMs that dominate run in BF16 under autocast (Section 3.1), and the SSD einsums operate on **complex64** operands: torch dispatches complex matmuls through cuBLAS's complex GEMM routines, which compute in FP32 and are not governed by the TF32 math-mode flag — the recurrence keeps full FP32 internal precision regardless of this setting. TF32 applies to the residual FP32 GEMMs (e.g. the CPU-fallback path where autocast is disabled). One caveat, an open verification item: the Triton dispatch `models/ssd_triton.py:per_chunk_ssd_triton` computes the per-chunk matmuls as *real* FP32 `tl.dot` calls (the complex multiply is expanded into four real GEMMs — `tl.dot(cc_re, bc_t_re) - tl.dot(cc_im, bc_t_im)` and friends), and Triton's default `input_precision` for FP32 dots on NVIDIA is TF32 unless overridden; the kernel does not set it, so the Triton path's internal GEMMs may run at TF32 precision. GPU-box verification item `[INFERENCE]` — this clone's CPU tests cannot exercise the kernel.

### 3.4 Where FP32 is enforced: the `.float()` casts

`models/mamba_block.py:Mamba3Block._forward_impl` enforces FP32 for everything that feeds the recurrence with four explicit casts:

```python
proj = self.in_proj(h)
x_ssm = proj[..., :H * D].reshape(B, T, H, D).float()

B_real = proj[..., H * D:H * D + H * N].float()
B_imag = proj[..., H * D + H * N:H * D + 2 * H * N].float()
B_t = torch.complex(B_real, B_imag).reshape(B, T, H, N)

C_real = proj[..., H * D + 2 * H * N:H * D + 3 * H * N].float()
C_imag = proj[..., H * D + 3 * H * N:H * D + 4 * H * N].float()
C_t = torch.complex(C_real, C_imag).reshape(B, T, H, N)

dt = proj[..., -H:].float()
```

(The plan outline's "gating sigmoid in FP32": this repo has no sigmoid — the gate is `softplus` — and these casts *are* the FP32 enforcement.) The casts are load-bearing:

1. **`torch.complex` requires FP32/FP64 inputs.** Under BF16 autocast `proj` is BF16; packing BF16 slices raises a type error.
2. **Autocast would otherwise round the recurrence's inputs.** `einsum` is in autocast's lower-precision set — empirically, FP32 einsum inputs under BF16 autocast produce BF16 output. With BF16 slices, the scan's products and exponentials would start from $2^{-8}$-rounded values, each decay factor carrying 0.4% relative error into a 64-deep cumsum. The casts guarantee FP32-precision inputs; the complex64 operands inside `models/ssd_complex.py:ssd_complex_chunkwise` are then never touched by autocast, so the whole scan runs in complex64 = FP32-pair arithmetic, and the state never leaves complex64 — only `Y.real` (FP32) escapes.

## 4. The decay bound: why the recurrence cannot amplify (at init)

The discretized transition is `models/ssd_complex.py:_discretise`:

```python
def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.softplus(dt) * A)
```

with `A` initialized to a constant real `−1.0` (verified: `constant_` on complex64 sets real = −1, imag = 0):

```python
self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
nn.init.constant_(self.A, -1.0)
```

Softplus is strictly positive — $\mathrm{softplus}(x)=\ln(1+e^x)\in(0,\infty)$ (empirically $4.5\times10^{-5}$ at $x=-10$, $0.693$ at $x=0$, $10.0$ at $x=10$). With $\Re A=-1$,

$$\ln|\bar a_t| = \mathrm{softplus}(dt_t)\cdot\Re A = -\mathrm{softplus}(dt_t) \le 0,$$

so $|\bar a_t| = e^{-\mathrm{softplus}(dt_t)} \in (0,1]$ — the recurrence is **non-expansive**: no step can amplify the state (strictly contractive except as $dt\to-\infty$). Consequences: the intra-chunk decay $L[l,s]=\exp(A_{\text{cs}}[l]-A_{\text{cs}}[s])\cdot\mathbf{1}[l\ge s]$ ([02-state-space-duality](../concepts/ssd-theory.md)) and the inter-chunk propagation ([04-chunkwise-algorithm](../concepts/ssd-theory.md)) — with $CD[k]:=\sum_{u=0}^{k}\Lambda_u$ ($\Lambda_u$ the chunk-$u$ total log-decay) and $CD[-1]:=0$, the fixed implementation carries chunk $c$'s end-of-chunk state into chunk $z$ with

$$\text{decay\_chunk}[z,c] = \exp\big(CD[z-1]-CD[c]\big)\cdot\mathbf{1}[z>c],$$

and the initial state with $\exp(CD[z-1])$ — both have non-positive real parts at init, so every decay factor is a contraction too. The entire forward pass is then a composition of contractions; blow-up must come from outside the decay path — from the $B_t\otimes x_t$ injection or from *weight drift*.

**The blow-up mechanism (why the guard exists).** $\Re A$ is trainable; nothing pins it below zero. If AdamW drives $\Re A>0$, the per-position amplification is $g = e^{\mathrm{softplus}(dt)\cdot\Re A} > 1$, and over $T=2048$ tokens the last position's memory scale is $g^{T}$: $\Re A=0.02$, $\mathrm{softplus}(dt)=1$ gives $g^{2048}=e^{40.96}\approx 6\times10^{17}$; $\Re A=0.05$ gives $e^{102}\approx 10^{44}$ — instant overflow of complex64 (max $\approx 3.4\times10^{38}$) to `inf`, propagating through every einsum to a NaN loss. The decay bound guarantees stability *conditionally* — at init and while weights stay contractive — and the NaN guard enforces the region constraint.

## 5. The NaN guard: detection, skip, rewind

### 5.1 Detection in `train_step` — before backward

`training/pretrain.py:train_step` (module-level, so tests call it directly) computes the loss inside the autocast region, then checks finiteness *before* `loss.backward()`:

```python
with amp_context:
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100,
    )
    ce_loss_val = float(loss.item())
    loss = loss / config.gradient_accumulation_steps

if config.nan_guard and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
    log(f"[nan-guard] NaN/Inf at micro_step={micro_step}, opt_steps={opt_steps}. Skipping backward.")
    optimizer.zero_grad(set_to_none=True)
    return None, opt_steps

loss.backward()
if is_opt_step:
    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    opt_steps += 1
```

Ordering is the design: the check runs **before** `backward()`, so a poisoned forward can never write NaN gradients into parameter buffers. The check reads the FP32 scalar loss (Section 3.1), so it is reliable and nearly free. It is also *stronger* than skipping one step: `zero_grad(set_to_none=True)` wipes the partial gradients accumulated from earlier micro-steps of the current window — a NaN anywhere in a window discards the window's partial credit, never mixing a clean loss's gradients with a poisoned one's. The returned `None` metric signals the skip to the caller.

### 5.2 The streak and rollback in `Pretrainer.train`

`training/pretrain.py:Pretrainer.train` interprets `None` with a consecutive counter:

```python
metrics = self.train_step(tokens, targets, global_step)
if metrics is None:
    nan_guard_streak += 1
    if nan_guard_streak >= self.config.nan_guard_max_consecutive:
        latest = self._find_latest_checkpoint()
        if latest is not None:
            self._log(f"[nan-guard] {nan_guard_streak} consecutive NaN/Inf — restoring checkpoint step {latest}.")
            global_step = self.load_checkpoint(latest)
        else:
            self._log("[nan-guard] No checkpoint to restore from. Aborting.")
            raise RuntimeError("NaN/Inf with no checkpoint to restore from")
        nan_guard_streak = 0
    continue
nan_guard_streak = 0
```

`nan_guard_max_consecutive = 5` (config default, `training/pretrain.py:TrainingConfig`). One non-finite loss just skips; five in a row declares a regime change and `load_checkpoint(latest)` rewinds the **whole training state** — weights, optimizer moments, scheduler position, `opt_steps` — to the most recent complete checkpoint via `utils/checkpoint.py:CheckpointManager`. `CheckpointManager.latest_step()` returns the highest step whose three files (`model_step_N.safetensors`, `optim_step_N.pt`, `meta_step_N.json`) all exist (`_checkpoint_complete`), so the rewind target is never a torn save. After a rewind, `global_step` resets to the checkpoint's step, the streak clears, and training continues; the triggering batch is consumed, not replayed.

Checkpoint cadence is `save_every = 4000` micro-steps (`save_interval` in `configs/pretrain_a100_400m.yaml`), and step 0 is never saved (`global_step % self.config.save_every == 0 and global_step > 0`), so the rollback discards up to 4,000 micro-steps and aborts (RuntimeError) if no checkpoint exists yet — the guard is only as good as its most recent save. The interval is a stability knob, not bookkeeping (see [Mamba-3-Lite — Training Runbook](../guides/training-runbook.md) for operational semantics).

### 5.3 Why rollback beats clipping — honest rationale

Gradient clipping (`nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)`, `max_grad_norm = 1.0`, on optimizer steps) is present and useful — it bounds the *update* norm against large-but-finite gradients. But it cannot rescue a diverged run:

1. **NaN is not clip-able.** Once the forward produces `inf` (an amplified complex state overflowing $3.4\times10^{38}$), gradients are NaN, and $\text{clip}(\text{NaN})=\text{NaN}$. Clipping only exists between finite values.
2. **The damage is in the weights, not the step.** The Section 4 mechanism is a *parameter regime* ($\Re A>0$). The gradient there points along the amplifying direction, and AdamW's per-parameter normalization means even a fully clipped update keeps moving weights deeper into the bad regime. Clipping fixes step size, not trajectory.
3. **Loss can be finite while gradients are garbage.** The backward differentiates products of $\exp$ factors — `decay_chunk`, `L`, `exp(A_cumsum)` — and $\frac{d}{d\theta}\exp(f) = \exp(f)\,f'$ multiplies already-large exponentials by sums of softplus derivatives. In the amplifying regime these intermediates can be astronomical or NaN while the forward loss still rounds finite (one batch before overflow). A clipped step then silently corrupts weights from a "healthy" reading — exactly the silent failure the rollback targets.

Honest trade-off: rollback throws away up to 4,000 micro-steps and is a coarse instrument; the streak threshold keeps it coarse-on-purpose — one transient NaN (corrupted batch, data glitch) should not discard progress; five in a row signals the optimizer has left the contractive region.

## 6. Weight tying: one parameter, two roles

`models/transformer.py:Mamba3Transformer.__init__`:

```python
self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
...
self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

if cfg.weight_tying:
    self.lm_head.weight = self.embed.weight

self.apply(self._init_weights)
```

The assignment makes `lm_head.weight` *the same `Parameter` object* as `embed.weight` — identical `data_ptr`, not a copy. Consequences:

- **Parameter savings.** The matrix is $50257 \times 1024 = 51{,}463{,}168$ parameters (~51.5M). Untied, the model would have $433{,}662{,}400 + 51{,}463{,}168 = 485{,}125{,}568$; tying saves $\approx 10.6\%$ of parameters, plus $51{,}463{,}168 \times 2 \times 4\text{B} \approx 412\text{ MB}$ of AdamW moments (two FP32 per parameter).
- **Gradients accumulate automatically.** Both paths write through the same tensor, so `backward()` accumulates the embedding-path and head-path contributions into one shared `.grad` — no manual summing exists or is needed (verified: the tied parameter's `.grad` receives both). `Pretrainer.__init__` also deduplicates `model.parameters()` by `id()` when building the decay/no-decay groups, so AdamW holds one state entry.
- **A checkpoint nuance.** `CheckpointManager.save` walks the state dict tracking `data_ptr`:

```python
for k, v in state.items():
    ptr = v.data_ptr()
    if ptr in seen_ptrs:
        deduped[k] = v.contiguous().clone()
    else:
        seen_ptrs.add(ptr)
        deduped[k] = v.contiguous()
```

safetensors refuses to serialize shared storage (verified: `RuntimeError: Some tensors share memory…`), so the second occurrence is **cloned** — the file stores both `embed.weight` and `lm_head.weight` as independent full-size tensors, so the checkpoint is *not* smaller; the dedup exists to keep the file valid and self-contained. The tie survives a round-trip because `load_state_dict` copies values *in place* into the existing (still shared) parameter, so the two names alias the same storage again after `load_checkpoint`. See [Mamba-3-Lite — Training Reference](../references/training-reference.md) for the file format.

## 7. Init scheme: small normal, identity mixer, constant decay

`models/transformer.py:Mamba3Transformer._init_weights`, applied via `self.apply(...)` over the whole tree:

```python
def _init_weights(self, module):
    if getattr(module, "_identity_init", False):
        return
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
```

`init_std = 0.02` (`models/transformer.py:ModelConfig`). Three pieces:

**Why $N(0, 0.02)$ for a residual network.** A Linear with fan-in $d$ acting on unit-variance input gives output variance $\sigma^2 d$; with $\sigma=0.02$, $d=1024$:

$$\text{std}(\text{block delta}) = \sigma\sqrt{d} = 0.02 \times 32 = 0.64.$$

The variance-preserving scale is $1/\sqrt{d}\approx 0.031$; 0.02 is about two-thirds of it, so each block's contribution is *smaller* than the identity path it joins: unrolling the residual stream with roughly independent deltas gives $\mathrm{Var}(x_L) \approx 1 + L\cdot 0.64^2$, i.e. std ≈ 3.5 at $L=28$ — bounded. The stronger effect is backward: the residual Jacobian is $I + \sum\prod\partial\Delta/\partial x$, and small deltas keep it near-identity, so gradients neither vanish nor explode across 28 blocks. Pre-norm RMSNorm reinforces this by renormalizing each block input to unit RMS — which is also why BF16 rounding always operates on $O(1)$-scaled activations.

**Why eye for MIMO.** `models/mimo.py:MIMO` initializes the mixer to identity and flags it:

```python
self.mix = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
nn.init.eye_(self.mix.weight)
---

#Mamba3Transformer._init_weights skips Linears flagged this way.
self.mix._identity_init = True
```

`_identity_init` is the escape hatch: the generic Linear branch would otherwise overwrite the eye with $N(0, 0.02)$ (that is the exact dead-code bug this flag prevents — see [05-mimo-mixing](../concepts/mimo.md)). At init the block behaves like the classical SISO SSM (head $h$'s scan output feeds only head $h$'s projection), the mixing Jacobian is the identity — no scaling distortion of gradients — and cross-head communication grows as training moves `mix` off identity.

**Why a constant $−1$ complex $A$.** This anchors Section 4's contractive dynamics: a real −1 (verified imag = 0) puts every decay factor in $(0,1]$ at init, so the recurrence starts provably non-amplifying, while the imaginary part of $A$ — which encodes rotation, see [03-complex-ssd](../concepts/ssd-theory.md) — is free to train away from zero. `A` is a `Parameter` of the block, not a submodule, so `_init_weights` never touches it; only AdamW moves it.

## 8. The loss path: FP32 cross-entropy, scaled accumulation

```python
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100,
)
ce_loss_val = float(loss.item())
loss = loss / config.gradient_accumulation_steps
```

- `logits.reshape(-1, 50257)` flattens `(B, T, V)`; `targets.reshape(-1)` flattens `(B, T)`; cross-entropy takes the mean over non-ignored positions.
- `ignore_index=-100` is the conventional mask sentinel but **inert in this pipeline**: `PretrainDataset` windows never produce a −100 target (pad id is 50,256; targets are always real token ids).
- `ce_loss_val` is recorded *before* the division (the logged loss is the per-micro-batch mean); the *backpropagated* loss is divided by `gradient_accumulation_steps`. With accumulation 2, two micro-batches of $\tfrac{1}{2}L_i$ sum to $\tfrac{1}{2}(L_1+L_2)$ — the mean over the 32 × 2048-token effective batch, whose gradients AdamW consumes after clipping.

## 9. Pitfalls

1. **NaN checks must run before `backward()`.** The guard's position in `train_step` is the point: after `backward()`, NaN gradients are already in the parameter buffers even if you skip `optimizer.step()`. Any edit must keep the check between the loss and the backward call.
2. **A complex NaN can hide in `.real`.** `torch.isnan` on complex64 checks *both* components (verified: `isnan(1 + NaN·i)` is `True`), so a direct complex check is safe. But the guard only sees the real scalar loss: a NaN in an imaginary component decoupled by the readout (e.g. `C_imag = 0` kills the imag contribution to `Y.real`) leaves the loss finite and the guard silent — it surfaces later as corrupt gradients and trips the guard next step. Debugging a state-side NaN: check `torch.isnan(state)`, never `torch.isnan(state.real)`.
3. **`torch.compile` changes numerics.** `Pretrainer.__init__` wraps with `torch.compile(..., mode=compile_mode)` (`max-autotune` default). Fusion and reassociation change rounding — last-bit differences, not semantic ones. The guard still works (it reads the scalar loss, outside the graph), but a *compiled* CUDA crash (illegal memory access) raises rather than producing NaN — it will not be caught. `--no-compile` is the debug escape.
4. **Rollback needs a recent checkpoint.** The rewind target is the latest *complete* checkpoint (`CheckpointManager.latest_step()` requires all three files), saved every 4,000 micro-steps, and step 0 is never saved. A divergence inside the first 4,000 steps aborts: `RuntimeError("NaN/Inf with no checkpoint to restore from")`. For early crash-tolerance, lower `save_interval`.
5. **The TF32 flag has narrow scope.** It affects real FP32 GEMMs; the complex64 einsums are dispatched through cuBLAS complex GEMMs and keep FP32 arithmetic. The one place TF32-style truncation could reach the recurrence is the Triton kernel's internal real `tl.dot`s, whose precision is Triton's default (not torch's flag) — verify `input_precision` on the GPU box `[INFERENCE]`.
6. **The decay bound is conditional.** $|\bar a_t|\le 1$ holds at init and while $\Re A \le 0$; `A` is unconstrained, so drifted $\Re A>0$ turns the recurrence into an amplifier — the failure the guard exists for. Also, $\exp(-\mathrm{softplus}(dt))$ for very negative `dt` rounds to exactly 1.0 in FP32: no amplification, but no decay either — memory then persists the full sequence length.
7. **Logits are BF16 under autocast.** Don't "fix" the head by casting logits to FP32 manually — the loss path is already FP32 and a `.float()` inside the region is a no-op promotion. If you ever call `softmax` on logits *outside* autocast for sampling, cast to FP32 first.

## 10. Tests

- `tests/test_train_step.py::test_train_step_on_tiny_model` — the end-to-end guard contract on a CPU-only 2-block tiny model: runs `training/pretrain.py:train_step` with `nan_guard=True`, asserts it returns a metric (the guard did **not** trip), `out["loss"]` is finite, and a step changed parameters. This would catch a guard placed after `backward()` (parameters still change, but the check is meaningless) or one tripping spuriously on healthy runs.
- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` / `test_chunkwise_matches_naive_time_varying_dt` — pin the FP32/complex64 recurrence math this stability analysis rests on (chunkwise ≡ naive scan to `atol=1e-4`, float32-vs-complex64 dtypes asserted).
- `tests/test_ssd.py::test_chunkwise_handles_uneven_T` / `test_chunkwise_handles_T_equal_to_chunk` — shape and **finiteness** checks for the SSD edge cases: the cheapest regression net for the exp-chain paths.

---

## References

- [Mamba-3-Lite — SSD Foundations](state-space-foundations.md) — the recurrence and discretization the stability analysis rests on.
- [Mamba-3-Lite — SSD Theory](ssd-theory.md) — the chunkwise algorithm and its `exp`-chain intermediates.
- [Mamba-3-Lite — MIMO Head Mixing](mimo.md) — the identity-init escape hatch from `models/transformer.py:Mamba3Transformer._init_weights`.
- [Mamba-3-Lite — Config Reference](../references/config-reference.md) — `init_std`, `rms_norm_eps`, `grad_checkpoint`, and the training fields the NaN guard reads.
- [Mamba-3-Lite — Training Reference](../references/training-reference.md) — `utils/checkpoint.py:CheckpointManager` rollback mechanics, `utils/logging.py:TrainingLogger` tps math.
- [Mamba-3-Lite — Training Runbook](../guides/training-runbook.md) — operational NaN recovery and VRAM monitoring.
- [Mamba-3-Lite — Tuning Guide](../guides/tuning.md) — measuring the throughput levers this doc derives.
