# Complex-Valued State Spaces

Mamba-3's sequence-mixing primitive packs decay *and* rotation into one complex parameter per head; this doc derives why, walks the code that builds and scans the complex state, and explains complex autograd, `torch.complex` mechanics, and the N-halving parity claim.

## 60-second summary

After reading this doc you will understand:

- A real SSM state can only *scale* per step (`e^{a·dt}`); oscillation requires rotation, which needs two real dimensions per mode. A complex state `h = u + iv` carries both in one scalar: `e^{(α+iβ)dt} = e^{αdt}(cos βdt + i sin βdt)`.
- One complex state = **two real degrees of freedom**, so a 64-dimensional complex state has the same coordinate count as a 128-dimensional real state. The "N=64 complex ≈ N=128 real" parity claim is plausible from that counting argument, but it is an empirical claim from the paper — marked `[INFERENCE]` below — not something this repo's tests verify.
- The repo's recurrence is `h_t = e^{softplus(dt)·A} h_{t-1} + B_t x_t` with per-head complex scalar `A` (init `−1.0`, i.e. pure decay), complex `B_t`/`C_t`, and a real readout `y_t = Re(C_t h_t)`. The state never leaves `complex64`; the logits are FP32 because the readout is a real linear map.
- PyTorch's complex autograd uses (conjugate) Wirtinger derivatives, so `.grad` on a `complex64` tensor is a complex tensor whose `.real`/`.imag` are the two coordinate gradients; the imaginary parts of `A`, `B`, `C` are genuinely trainable even though the loss is real.
- `torch.view_as_real` gives an interleaved stride-2 layout; `models/ssd_triton.py:_view_real_imag` materializes two contiguous float32 buffers for the Triton kernel — the `contiguous()` call is load-bearing.

## Why it exists

A diagonal real state space `h_t = e^{A·dt} h_{t-1}` with `A ∈ ℝ^N` models one thing per mode: a rate of decay (or growth). Its impulse response is a sum of decaying exponentials. Many sequence phenomena — alternating structure, periodic content, anything with a "phase" — are better described by *oscillating* modes, which require eigenvalues off the real axis. A real matrix can only have such eigenvalues in conjugate pairs, which forces a `2×2` real block per oscillating mode. In other words, in real arithmetic an oscillating mode costs **two** state dimensions.

A complex scalar `A = α + iβ` is the minimal object that carries both a decay rate `α` and an angular frequency `β`: after one discrete step of size `dt`, the state is multiplied by `e^{(α+iβ)dt} = e^{αdt}·e^{iβdt}`, i.e. scaled by `e^{αdt}` and rotated by `β·dt`. Mamba-2 used `N=128` real state dimensions per head; Mamba-3 replaces them with `N=64` **complex** state dimensions (`state_dim=64`, `complex64`) — the same number of real coordinates, with rotation for free. The architectural bet, stated in `AGENTS.md` and derived below, is that 64 complex states give perplexity parity with 128 real states while halving the state dimension (and the per-step state bandwidth in the scan).

## Intuition first

Think of a phasor on the complex plane. A complex number `a = e^{α+iβ}` has a radius `e^{α}` and an angle `β`. Multiplying a state by it does two independent things at once:

- `e^{α}` shrinks (α < 0) or grows (α > 0) the magnitude — **decay**;
- `e^{iβ} = cos β + i sin β` turns the state by angle `β` — **rotation**.

So *one* complex parameter is a scale *and* a rotation: two real controls (`α`, `β`) in a single multiplicand. A real scalar can only scale; to rotate you need two coupled real numbers (a `2×2` rotation matrix). That is the whole reason complex states exist here: the sequence model wants dynamics that turn, and the complex plane is the smallest place a turn lives.

The state itself is a little complex vector: each of its `N` complex entries `h_n = u_n + iv_n` is two real numbers that get *coupled* — the rotation mixes `u` and `v` on every step (that coupling is exactly what a real diagonal SSM cannot do). Readout is a real projection `y = Re(C h)`: it folds the two sub-states back into one real number per head-channel.

## Math

### The recurrence

Let `A = α + iβ ∈ ℂ` be a per-head scalar, `dt > 0` the per-token step, and `B_t, C_t ∈ ℂ^{H×N}` complex input/output maps. The repo's recurrence (see `models/ssd_complex.py:ssd_naive_complex`, the O(T) oracle) is

$$h_t = e^{(\alpha + i\beta)\,dt_t}\, h_{t-1} + B_t\, x_t, \qquad y_t = \operatorname{Re}\!\big(C_t\, h_t\big),$$

with `h_t ∈ ℂ^{N×D}` (here `N=64` states × `D=64` channels per head) and a real input `x_t` promoted to complex (imaginary part 0). The transition factor factors into the promised two effects:

$$e^{(\alpha + i\beta)dt} \;=\; e^{\alpha dt}\cdot e^{i\beta dt} \;=\; \underbrace{e^{\alpha dt}}_{\text{scale}}\;\big(\underbrace{\cos(\beta dt) + i\sin(\beta dt)}_{\text{rotation}}\big).$$

This is Euler's formula; it is not an approximation — it *is* the definition of the complex exponential.

### Unpacking: one complex state is two real states

Write `h_t = u_t + iv_t`, `B_t = b^R_t + i b^I_t`. Substituting and separating real/imaginary parts gives the equivalent **real** two-dimensional recurrence

$$
\begin{aligned}
u_t &= e^{\alpha dt}\big(\cos(\beta dt)\, u_{t-1} - \sin(\beta dt)\, v_{t-1}\big) + b^R_t\, x_t,\\
v_t &= e^{\alpha dt}\big(\sin(\beta dt)\, u_{t-1} + \cos(\beta dt)\, v_{t-1}\big) + b^I_t\, x_t,
\end{aligned}
$$

i.e. a `2×2` real transition matrix

$$\begin{pmatrix} u_t \\ v_t \end{pmatrix} = e^{\alpha dt}\underbrace{\begin{pmatrix} \cos(\beta dt) & -\sin(\beta dt) \\ \sin(\beta dt) & \cos(\beta dt) \end{pmatrix}}_{R(\beta dt)} \begin{pmatrix} u_{t-1} \\ v_{t-1} \end{pmatrix} + \begin{pmatrix} b^R_t \\ b^I_t \end{pmatrix} x_t .$$

Two observations fall out:

1. **Two real DOF per complex state.** Each complex entry is two real coordinates, and the recurrence is exactly a real linear recurrence over those coordinates. The complex model at `N=64` is therefore a linear SSM over **128 real coordinates** — the same coordinate count as Mamba-2's `N=128`.
2. **The transition is a scaled rotation.** A real *diagonal* SSM at `N=128` has transition `diag(e^{a_1 dt}, …, e^{a_{128} dt})` with `a_i ∈ ℝ`: it can scale each coordinate independently but can never rotate (a real diagonal matrix has real eigenvalues). The complex model's 128-coordinate transition is block-diagonal with `2×2` scaled-rotation blocks — a strictly richer dynamics class at the same coordinate count, because its eigenvalues `e^{α+ iβ}` may be off the real axis.

A real SSM *can* oscillate, but only by using `2×2` blocks (conjugate eigenvalue pairs), i.e. **two** real states per oscillating mode. So "oscillation capacity" measured in modes is `N_complex` complex states = `N_complex` oscillating modes = `2·N_complex` real states. That is the packing argument in one line: *one complex state carries two real sub-states, so N=64 complex achieves what N=128 real does* — same real DOF, plus the rotation the real diagonal model lacks.

### The parity claim — stated honestly

The stronger statement — that the 64-complex model reaches *perplexity parity* with the 128-real model — is an **empirical** claim from the Mamba-3 paper (Dao & Gu, 2025), not a theorem and not a fact this repository has measured: `[INFERENCE]`. What the counting argument above establishes is *plausibility* (equal real dimension, strictly more expressive transitions), and what the repo's tests establish is only *internal consistency* (chunkwise ≡ naive scan on the complex model, see Tests). No test here compares perplexity against a real-state model; treat "parity" as a paper claim the repo inherits, pending the 8B-token run (see `../theory/08-scaling-efficiency.md`).

### Discretization

The repo discretizes the continuous-time rate `A` with a per-token step via `models/ssd_complex.py:_discretise`:

$$A\_bar = \exp\!\big(\operatorname{softplus}(dt)\cdot A\big), \qquad \operatorname{softplus}(dt) = \log(1 + e^{dt}).$$

Two design points, both deliberate:

- **`softplus` keeps the step positive and smooth.** `softplus(dt) > 0` for all real `dt`, is smooth (unlike `ReLU`), and behaves like `dt` for large `dt`. With `A = −1 + 0i` at init this makes every mode a contraction: `|e^{−softplus(dt)}| = e^{−softplus(dt)} ∈ (0,1)` — no blow-up, no sign flips, gradients flow through the smooth map.
- **Order matters.** The nonlinearity is applied to the *real* `dt` first, then multiplied by the *complex* `A` — `exp(softplus(dt)·A)`, never `exp(softplus(dt·A))`. The softplus operates in `ℝ` (complex `softplus` is not defined in PyTorch); the complex exponential then produces the rotation+scale factor. The same `A_log = softplus(dt)·A` quantity is the seed of every complex exponential inside `models/ssd_complex.py:ssd_complex_chunkwise` (the `L` matrix, `decay_states`, `decay_chunk`), which is why the whole chunkwise algorithm is just a batched, associative version of this one-step factor — detailed in `../theory/04-chunkwise-algorithm.md`.

## Gradients through complex tensors

The loss is real: the SSD returns `Y.real` (float32), then MIMO, `out_proj`, and `lm_head` are real linear layers, and cross-entropy is real. Yet the parameters `A` (complex64), `B_t`, `C_t` (complex64) sit *upstream* of the `.real` truncation. How do real gradients reach them?

PyTorch's complex autograd uses **Wirtinger calculus**. For a real-valued loss `L(z)` with `z = u + iv`, the two Wirtinger derivatives are

$$\frac{\partial L}{\partial z} = \tfrac12\Big(\frac{\partial L}{\partial u} - i\frac{\partial L}{\partial v}\Big), \qquad \frac{\partial L}{\partial \bar z} = \tfrac12\Big(\frac{\partial L}{\partial u} + i\frac{\partial L}{\partial v}\Big),$$

and the *steepest-descent* direction in the `(u,v)` plane is the conjugate one: `−∂L/∂z̄`. PyTorch stores, for a complex input, exactly

$$\text{grad}_z = 2\,\frac{\partial L}{\partial \bar z} = \frac{\partial L}{\partial u} + i\,\frac{\partial L}{\partial v},$$

so that the update `z ← z − η·grad_z` is precisely real gradient descent on the two coordinates. Consequences for this repo:

- **`.grad` on a `complex64` tensor is `complex64`.** Its `.real` is `∂L/∂u` and its `.imag` is `∂L/∂v`. The two packed sub-states are trained as two real coordinates, which is exactly the packing argument in optimizer space.
- **The imaginary parts of `A`, `B`, `C` are genuinely trainable.** Although the loss only sees `Re(C h)`, the rotation makes the real output depend on `β` and on `Im B`/`Im C`: `∂Re(e^{iβdt}h)/∂β ≠ 0`. If the model needs oscillation, gradient descent can push `β` off its init value 0.
- **Non-holomorphic ops are handled.** `z.real`, `.abs`, `torch.view_as_real` are not holomorphic; autograd still propagates through them with the same conjugate-Wirtinger convention (e.g. the backward of `z.real` contributes a purely real `grad_z`). The chain through the SSD — holomorphic ops (`exp`, multiply, einsums) ending in `.real` — is therefore well-defined.
- **Optimizers accept complex parameters.** The `A` parameter (`torch.empty(H, dtype=torch.complex64)` in `models/mamba_block.py:Mamba3Block`) is optimized directly by AdamW; PyTorch optimizers treat a complex parameter as two real coordinates, consistent with the convention above.

### Why the state stays `complex64` while logits are FP32

The state is complex because the *dynamics* are complex — the recurrence is one complex multiply per step and the rotation cannot be expressed as a real scalar. Everything *after* the scan is real by design:

- the readout `y_t = Re(C_t h_t)` is a real linear functional of the 2N real coordinates (the `C` einsum in `ssd_complex_chunkwise` sums over `n` and takes `.real` at the end);
- a probability distribution over the 50,257-token vocabulary is real; complex logits have no meaning for cross-entropy;
- keeping `lm_head` real halves its memory (`50257 × 1024` complex64 would double the head's footprint) and keeps the logit path in FP32 as required by `AGENTS.md` ("Recurrent state stays in complex64; logits in FP32").

So `complex64` is confined to the SSD scan: `B_t`/`C_t` are complex only inside `ssd_complex_chunkwise`, the state tensors (`states`, starting from zeros of dtype `complex64`) never leave it, and `Y.real` hands a float32 tensor to the rest of the block. From there every subsequent op — MIMO, `out_proj`, residual, `norm_f`, `lm_head` in `models/transformer.py:Mamba3Transformer.forward` — is a real FP32 linear map, ending in real FP32 logits. The imaginary part of the scan output carries no probability mass — discarding it is the intended projection, not a lossy truncation.

## `torch` complex mechanics

Three operations carry the whole implementation:

- **`torch.complex(real, imag)`** — builds a `complex64` tensor from two equal-shaped `float32` tensors (real and imaginary parts). Used in `models/mamba_block.py:Mamba3Block._forward_impl` to assemble `B_t` and `C_t`, and in `models/ssd_triton.py` to reassemble `Y_diag`/`state` from the kernel's split float32 outputs. It requires `float32`/`float64` inputs — this is why the slices are `.float()`-cast before assembly (under BF16 autocast the `in_proj` output would be BF16).
- **`torch.view_as_real(z)`** — views a `complex64` tensor of shape `(…, N)` as `float32` of shape `(…, N, 2)`. In memory a complex64 tensor interleaves `[re₀, im₀, re₁, im₁, …]`, so the pair dimension has stride 1 while the element dimension has **stride 2** — the "stride-2 layout" named in `models/ssd_triton.py:_view_real_imag`. Any code that assumes a packed float32 buffer over this view reads every *other* float (i.e. real parts as imag and vice versa).
- **`torch.view_as_complex(real_pairs)`** — the inverse; it requires the pair dimension to have stride 1. Applied to a non-contiguous tensor it silently interleaves wrong elements — the "silent stride bug" `AGENTS.md` warns about ("`torch.view_as_complex` on raw real pairs can fail silently if the imaginary stride is wrong").

## Code walkthrough

### The block: `models/mamba_block.py:Mamba3Block._forward_impl`

The per-head complex scalar `A` is created in `__init__` and initialized to pure decay:

```python
self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
nn.init.constant_(self.A, -1.0)
```

`nn.init.constant_` fills with the scalar cast to `complex64`, i.e. `−1 + 0i`: `α = −1`, `β = 0`. At initialization every mode decays and nothing rotates; rotation must be learned. Note `A` is a **per-head scalar** (shape `(H,)` — 16 complex numbers in the whole model), shared across all `N=64` states of a head, unlike Mamba-2's per-state `A`. Selectivity per token/head comes from `dt`, not from `A`.

The forward path projects and slices. `in_proj` has width `H·(D + 4N + 1)` — one `D`-wide block for the token content, four `N`-wide blocks for the real/imaginary parts of `B` and `C`, and one slot per head for `dt`:

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

The slice boundaries, in `proj` columns: `x` occupies `[0, H·D)`, then four contiguous `H·N` blocks (`B_real`, `B_imag`, `C_real`, `C_imag`), then `dt` in the last `H` columns. This exact layout is the block's ABI — see `../theory/06-block-anatomy.md` for the shape-by-shape dataflow. Three dtypes coexist: `x_ssm` stays `float32` (its imaginary part is 0 when promoted), `B_t`/`C_t` are `complex64`, `dt` is `float32`.

### The discretization: `models/ssd_complex.py:_discretise`

```python
def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.softplus(dt) * A)
```

`dt` is `(B, T, H)` real; `A` is `(H,)` complex64; the product broadcasts to `(B, T, H)` complex64 and the complex exponential yields the scale+rotation factor per token. The oracle `models/ssd_complex.py:ssd_naive_complex` applies this factor in a literal O(T) loop with a `complex64` state of shape `(B, H, N, D)`:

```python
A_bar = _discretise(dt, A)
s = torch.zeros(B_, H, N, D, dtype=torch.complex64, device=x.device)
for t in range(T):
    s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s             + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
    ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
```

Each step is one complex multiply of the whole state (`A_bar·s`, the scale+rotation) plus injection `B_t·x_t`; the readout sums `C_t·h_t` over the `N` states. This sequential scan is the ground truth the chunkwise algorithm must reproduce.

### The chunkwise pass: `models/ssd_complex.py:ssd_complex_chunkwise`

The production path replaces the O(T) loop with the chunked algorithm (derived in `../theory/04-chunkwise-algorithm.md`); what matters here is that *every* recurrence factor is a complex exponential of `A_log = softplus(dt)·A`:

- intra-chunk coupling `L[l,s] = exp(A_cumsum[l] − A_cumsum[s])·1[l≥s]` and `decay_states = exp(A_cumsum[−1] − A_cumsum)`;
- inter-chunk `decay_chunk = exp(cd_seg)·tril` from the per-chunk cumsums;
- the output term `Y_off` weights propagated states by `exp(A_cumsum)`, and `Y = Y_diag + Y_off` is truncated with `Y = Y.real` before being sliced back to `T`.

All state tensors (`states`) are `complex64`; only the final `Y.real` (float32) escapes. When `ssd_dispatch='triton'`, the per-chunk `Y_diag` and `state` are produced by `models/ssd_triton.py:per_chunk_ssd_triton` and the inter-chunk propagation stays in PyTorch.

### The kernel-side split: `models/ssd_triton.py:_view_real_imag`

Triton has no complex type, so the host wrapper splits every complex64 input into two packed float32 buffers:

```python
pair = torch.view_as_real(z.contiguous())
return pair[..., 0].contiguous(), pair[..., 1].contiguous()
```

The `z.contiguous()` first guarantees the interleaved layout is materialized (a permuted complex tensor would have non-trivial strides), and the final `.contiguous()` on `pair[..., 0]`/`pair[..., 1]` copies the stride-2 slices into stride-1 buffers the kernel indexes linearly (the kernel's grid is one program per `(B, c, H)`, reading `Bc`/`Cc`/`Xc`/`A_log`/`decay_states` as flat pointers). The same split feeds the dtype guard (`complex64` required, else `TypeError`). The backward of `models/ssd_triton.py:_PerChunkSSDTriton` recomputes the per-chunk math with `models/ssd_triton.py:per_chunk_ssd_pytorch` seeded with the true `grad_outputs` — so the complex gradients described above are reproduced on the PyTorch path even when the forward ran on GPU. Full kernel mechanics: `../reference/03-ssd-triton.md`.

## Pitfalls

1. **`N` must be even (AGENTS.md rule 4).** The parity argument ("`N=64` complex ≡ `N=128` real") is exact only when the state decomposes cleanly into real pairs; an odd `N` leaves the 2N-real-equivalent packing ambiguous. This is a repo rule, not an enforced assertion — `models/ssd_complex.py` has no even-`N` check, so a config with `state_dim=63` would silently run a model whose "packing" story is broken. Keep `state_dim=64`.
2. **Stride-2 view bugs.** `torch.view_as_real` output has stride 2 on the element dim; feeding it to anything assuming packed float32 (a Triton kernel, a reshape, `view_as_complex` on a non-contiguous pair) reads or interleaves the wrong floats *without error*. This is the "silent stride bug" of `AGENTS.md`. The `contiguous()` calls in `_view_real_imag` exist precisely to prevent it.
3. **`complex64` is 2× the element bandwidth of `float32`.** Every complex multiply is 4 real multiplies + 2 adds, and a complex64 tensor moves twice the bytes of a float32 tensor with the same shape. This is exactly why halving `N` matters: `N=64` complex state `(B,H,64,D)` occupies the same bytes as `N=128` real — the halving pays for the wider dtype. See `../theory/08-scaling-efficiency.md` for the memory analysis.
4. **Discretization order.** `softplus` applies to real `dt`, then the product with complex `A` is exponentiated: `exp(softplus(dt)·A)`. `F.softplus` does not accept complex input, and `exp(softplus(dt·A))` would be a different (and wrong) recurrence. Don't "optimize" the order.
5. **`torch.complex` rejects BF16/float16.** Under BF16 autocast the `in_proj` output is BF16; the `.float()` casts in `Mamba3Block._forward_impl` exist so the complex assembly and the entire SSD run in FP32/complex64 — the "gating FP32" rule of `AGENTS.md`. Dropping the casts breaks under autocast.
6. **Only `.real` escapes the scan.** `Y.real` is a view into the complex result; the imaginary part is discarded by design (it carries no probability mass), but the truncation means the model cannot use `Im(C h)` directly — if a future variant wanted complex logits, the readout and the head would both change.

## Tests

`tests/test_ssd.py` machine-checks the claims of this doc:

- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` — the core equivalence: random complex `x` (dtype `complex64`), `A = torch.randn(H, dtype=torch.complex64) - 1.0` (negative real part → decay), random complex `B_t`/`C_t`, `dt = 0`, `chunk_size=4`; asserts `ssd_complex_chunkwise` output is `float32`, `ssd_naive_complex` output is `complex64`, shapes match, and `allclose(y_chunk, y_naive.real, atol=1e-4)`. Note what this proves: the chunkwise algorithm reproduces the O(T) complex scan *exactly on the real readout* — it verifies the math of this doc, **not** the parity claim (no real-state model is involved).
- `tests/test_ssd.py::test_chunkwise_handles_uneven_T` — `T=20`, random nonzero `dt` (so `softplus(dt)` varies and the rotation term is exercised); checks shape and finiteness.
- `tests/test_ssd.py::test_chunkwise_handles_T_equal_to_chunk` — `T = chunk_size = 4`; the single-chunk degenerate case.

One honest caveat: all three tests use `T` divisible by `chunk_size` (16/4, 20/4, 4/4), so the padding branch in `ssd_complex_chunkwise` (`(C − T%C) % C`) is not directly exercised by this file — the uneven-`T` path is covered only if a future test uses a non-multiple. The parity claim, again, is `[INFERENCE]` from the paper; this repo's tests establish internal consistency, not perplexity parity. The GPU path adds an end-to-end check (`tests/e2e_gpu_smoke.py`, CUDA + triton) and the reference docs `../reference/02-ssd-complex.md` and `../reference/03-ssd-triton.md` carry the full API contracts.

For the surrounding machinery: `../theory/01-ssm-foundations.md` builds the real recurrence from first principles, `../theory/02-state-space-duality.md` motivates why a scan can become a chunked matmul, and `../theory/05-mimo-mixing.md` explains what happens to the real readout after it leaves the scan.
