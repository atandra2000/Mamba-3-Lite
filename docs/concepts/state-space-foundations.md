# Mamba-3-Lite — SSD Foundations: From RNNs to State Space Models

This doc builds the state-space-model (SSM) toolkit from scratch — the linear recurrence, its unrolled convolution form, zero-order-hold discretization, why a diagonal `A` makes the scan parallelizable, the S4 → Mamba-3 arc, and a line-by-line walkthrough of `models/ssd_complex.py:ssd_naive_complex`, the O(T) oracle every faster path in this repo is measured against.

## 1. 60-second summary

After reading this doc you will understand: why any sequence model needs a hidden state and what the linear recurrence `h_t = A h_{t-1} + B x_t, y_t = C h_t` computes; how that discrete recurrence falls out of the continuous ODE `ḣ = A h + B x` via zero-order-hold discretization (and exactly where `dt` enters); why a *diagonal* `A` turns a sequential loop into an associative scan of depth O(log T); and why the repo keeps `ssd_naive_complex` even though the model runs the chunkwise path — it is the ground-truth specification the fast path is proven against in `tests/test_ssd.py::test_chunkwise_matches_naive_complex`. The arc S4 → S6 → Mamba-2 → Mamba-3 reads as a sequence of *constraint removals*: each step gave up a mathematical convenience (time-invariance, real states) to buy expressivity, and each required a new algorithm to stay fast.

## 2. Why it exists

Sequence models face a structural tension. An autoregressive model must, at position $t$, summarize everything it has seen — potentially unbounded context — into a fixed-size representation. Attention solves this by *storing* the whole past (KV cache) and paying O(T) per query, O(T²) per layer, and O(T) memory per layer. Recurrent models instead *compress* the past into a hidden state `h_t` of fixed size, so per-step cost is O(1) regardless of how far back the relevant information lies. The price is that the recurrence is sequential in time: step $t$ cannot start until step $t-1$ finishes.

State space models keep the recurrence's O(1)-per-step inference cost while breaking its sequential training bottleneck. The trick is to make the recurrence **linear** — no nonlinearity inside the state update. Linear recurrences are associative, and associativity buys parallelization: a sequential loop over T steps can be reorganized into a tree of compositions running in O(log T) depth (the associative scan), or, when parameters are constant in time, recognized as a convolution and evaluated with an FFT. Nonlinearity is not lost; it is pushed out of the recurrence into the input/output projections that produce `B`, `C`, and `dt` from the tokens.

This repo needs the foundations for a concrete reason. Mamba-3-Lite's sequence mixer is a *complex-valued* chunkwise SSD (`models/ssd_complex.py:ssd_complex_chunkwise`), an algebraically rearranged, parallel version of exactly this linear recurrence. The rearrangement is only trustworthy because a literal, un-optimized transcription of the recurrence — `models/ssd_complex.py:ssd_naive_complex` — is kept in the tree and machine-checked against the fast path. This doc is the theory that makes that naive function obvious and the equivalence test meaningful.

## 3. Intuition

Think of the hidden state as a **leaky integrator with taps**. A tank of water: `x_t` pours in through a pipe with cross-section `B`; the tank drains through a hole whose size is set by `A`; a dipstick `C` measures the level and reports it as `y_t`. `A` controls how fast the past is forgotten (large negative `A` → fast decay → short memory; `A ≈ 0` → the tank holds everything → long memory, but also accumulates noise). `B` controls how strongly the current token enters the tank; `C` controls how much of the state the output "sees". Three knobs, one tank.

The RNN connection: this is exactly an RNN cell `h_t = f(W_h h_{t-1} + W_x x_t)` with the nonlinearity `f` deleted and the matrices structured. Deleting the nonlinearity looks like a loss of power, but it is the entire point: the update `h ↦ A h + B x_t` is an *affine map*, and affine maps compose associatively — a stack of T affine maps is equivalent to one composed affine map, so the sequence can be processed by composing segments in any order (parallel), not just left-to-right (sequential). The nonlinearity the RNN smuggled into the recurrence reappears, better placed, in the projections that compute `A, B, C, dt` from the input — the "selective" mechanism of Mamba.

The diagonal `A` intuition: a diagonal transition matrix makes the tank into N independent, smaller tanks. State component $n$ evolves as $h_{t,n} = a_n h_{t-1,n} + (B x_t)_n$ — N scalar recurrences that do not talk to each other. Each is a one-dimensional leaky integrator; with complex $a_n$, each becomes a damped oscillator (decay plus rotation per step). Independent scalar recurrences are the easiest possible thing to parallelize, because a parallel prefix scan on scalars is a textbook problem.

## 4. Math / proof

### 4.1 The linear recurrence and its unrolled form

Define the discrete-time linear time-invariant (LTI) system

$$h_t = A\,h_{t-1} + B\,x_t, \qquad y_t = C\,h_t, \qquad h_0 = 0,$$

with state $h_t \in \mathbb{C}^{N}$, input $x_t \in \mathbb{C}^{D}$, output $y_t \in \mathbb{C}^{D}$, and $A \in \mathbb{C}^{N\times N}$ (the transition), $B \in \mathbb{C}^{N\times D}$ (the input map), $C \in \mathbb{C}^{D\times N}$ (the readout). Unroll from $h_0 = 0$:

$$
\begin{aligned}
h_1 &= B x_1,\\
h_2 &= A h_1 + B x_2 = A B x_1 + B x_2,\\
h_3 &= A h_2 + B x_3 = A^2 B x_1 + A B x_2 + B x_3,\\
&\vdots\\
h_t &= \sum_{s=1}^{t} A^{\,t-s}\, B\, x_s,
\end{aligned}
$$

where $A^0 = I$. Multiplying through by $C$:

$$\boxed{\;y_t = \sum_{s \le t} C\, A^{\,t-s}\, B\, x_s.\;}$$

Two facts fall out of this closed form. First, **causality**: $y_t$ depends only on $x_{\le t}$, so the model is autoregressive by construction. Second, **time-invariance ⇒ convolution**: the coefficient of $x_s$ depends only on the lag $t-s$, so $y = k \ast x$ with kernel $k_\tau = C A^{\tau} B$, $\tau \ge 0$. In the frequency domain this is the transfer function $Y(z) = C(zI - A)^{-1}B\, X(z)$ of an LTI filter. That means the entire sequence can be filtered in $O(T \log T)$ via FFT — no sequential loop at all — *provided* the parameters do not change over time. This is the S4 training mode, and it dies the moment parameters become input-dependent (Section 4.3).

### 4.2 Continuous → discrete: zero-order hold

The discrete recurrence is usually presented as a *discretization* of a continuous ODE. Start from

$$\dot h(t) = A\,h(t) + B\,x(t),$$

and assume the input is held constant between samples — the **zero-order hold (ZOH)** assumption: $x(\tau) = x_t$ for $\tau \in [t,\, t + \Delta_t]$, where $\Delta_t$ is the sampling interval at step $t$ (the "dt"). The ODE is linear, so its exact solution over one interval is

$$h(t + \Delta_t) = e^{A\Delta_t} h(t) + \int_0^{\Delta_t} e^{A(\Delta_t - \tau)}\, B\, x_t \, d\tau .$$

Substitute $s = \Delta_t - \tau$ to evaluate the integral:

$$\int_0^{\Delta_t} e^{A(\Delta_t - \tau)} d\tau = \int_0^{\Delta_t} e^{A s}\, ds .$$

For a scalar eigenvalue $a$ (and, since everything below is diagonal, elementwise for $A$), $\int_0^{\Delta} e^{a s} ds = (e^{a\Delta} - 1)/a$, with the limit $\Delta$ as $a \to 0$. Hence

$$h_{t+1} = \underbrace{e^{A\Delta_t}}_{=:\ \bar A_t}\, h_t \;+\; \underbrace{A^{-1}\big(e^{A\Delta_t} - I\big) B}_{=:\ \bar B_t}\; x_t .$$

Two things are visible here. **Where dt enters:** (i) the decay $\bar A_t = e^{A \Delta_t}$, and (ii) the input scale $\bar B_t = A^{-1}(\bar A_t - I) B$. Note the small-$\Delta_t$ limit: $\bar A_t \to I$ and $\bar B_t \to \Delta_t\, B$ — the recurrence reduces to a forward-Euler step with step size $\Delta_t$. So $\Delta_t$ is genuinely a *learned step size*: it chooses how far the continuous dynamics advance per token, which the model can modulate per position.

**Discretization Schemes Comparison.** The zero-order hold assumption is compared below against standard numerical ODE integrators:

| Discretization Method | Discrete State Matrix $\bar A$ | Discrete Input Matrix $\bar B$ | Stability Condition ($\mathrm{Re}(A) < 0$) |
|---|---|---|---|
| **Zero-Order Hold (ZOH)** | $e^{A\Delta_t}$ | $A^{-1}(e^{A\Delta_t}-I)B$ | Unconditionally stable ($\forall \Delta_t > 0 \implies \|\bar A\| < 1$) |
| **Forward Euler** | $I + A\Delta_t$ | $\Delta_t B$ | Conditionally stable ($\Delta_t < 2 / \|A\|$; diverges for large $\Delta_t$) |
| **Bilinear / Tustin** | $(I - \frac{\Delta_t}{2}A)^{-1}(I + \frac{\Delta_t}{2}A)$ | $(I - \frac{\Delta_t}{2}A)^{-1} \Delta_t B$ | Unconditionally stable (maps LHP to unit disk) |

ZOH is chosen because token steps are piecewise constant signals in discrete time, and $e^{A\Delta_t}$ provides exact integration of the continuous ODE over interval $[t, t+\Delta_t]$ without truncation error.

```
Continuous State Trajectory h(t) sampled under Zero-Order Hold (ZOH):

  h(t) ^
       |            /--- h(t+Δt) = e^{A Δt} h(t) + \int_0^{Δt} e^{A (Δt - τ)} B x_t dτ
       |           /
       |   h(t)   /
       |    *----/
       |    |   /|
       |    |  / |
       +----+----+---------------------------------> time t
            t   t+Δt
            |______|  Input x(τ) = x_t held constant over [t, t+Δt]
```

**The absorption convention.** The second factor, $A^{-1}(\bar A_t - I)$, is a diagonal operator that depends only on $A$ and $\Delta_t$. Since $B$ is a learned (and, in Mamba, input-dependent) projection, multiplying $B$ by this diagonal factor is equivalent to choosing a different learned $B$: the family of achievable $\bar B_t$ equals the family of achievable $B$. The Mamba-2/SSD line therefore drops the factor and keeps the recurrence

$$h_{t+1} = \bar A_t\, h_t + B_t\, x_t, \qquad \bar A_t = e^{A \Delta_t},$$

which is precisely what this repo implements in `models/ssd_complex.py:_discretise` (it returns only the decay: `torch.exp(F.softplus(dt) * A)`); the discretization's input scaling is reabsorbed into the learned input map.

**Why `dt` must stay positive, and why softplus.** Stability of the recurrence requires $|\bar A_t| \le 1$ in the relevant norm. With the repo's initialization $A = -1.0$ (per-head complex scalar, `models/mamba_block.py:Mamba3Block`), the decay magnitude is $|e^{A\Delta_t}| = e^{\mathrm{Re}(A)\,\Delta_t} = e^{-\Delta_t}$. If $\Delta_t$ were allowed to go negative, the decay would flip into growth and the state would diverge exponentially over a long sequence — the single most catastrophic failure mode of a recurrent model. So the raw, unconstrained projection value $z_t$ is passed through

$$\Delta_t = \operatorname{softplus}(z_t) = \ln\big(1 + e^{z_t}\big) > 0 \quad \forall z_t,$$

which is (i) strictly positive everywhere, (ii) smooth with derivative $\sigma(z_t) = (1 + e^{-z_t})^{-1} \in (0,1)$ — nonzero everywhere, so no dead gradients, unlike ReLU's exactly-zero flat region, and (iii) asymptotically linear: $\operatorname{softplus}(z) \to z$ as $z \to +\infty$, so large step sizes grow linearly rather than exponentially (no overflow from the step size itself). Contrast with the naive alternative $e^{z}$: it overflows for $z \gtrsim 88$ and has vanishing gradient as $z \to -\infty$. One subtlety worth flagging: $\operatorname{softplus}(0) = \ln 2 \approx 0.693$, so there is **no "dt = 0" state** — even a raw projection of exactly zero yields a step size of $\ln 2$, i.e. a per-token decay of $e^{-\ln 2} = 1/2$ at initialization.

### 4.3 Diagonal A: the associative scan and the convolution link

The closed form (Section 4.1) requires $A^{t-s}$, an $N \times N$ power — expensive and opaque. Now suppose $A$ is **diagonal**: $A = \operatorname{diag}(a_1, \dots, a_N)$. Then $A^{t-s} = \operatorname{diag}(a_1^{t-s}, \dots, a_N^{t-s})$ and the recurrence separates into $N$ independent scalar recurrences,

$$h_{t,n} = a_n\, h_{t-1,n} + (B x_t)_n, \qquad n = 1,\dots,N,$$

and $y_t = C h_t = \sum_n C_{:,n} h_{t,n}$ is a sum over $N$ scalar modes. Two consequences.

**Consequence 1: the parallel scan (selective case).** Even when the parameters are *time-varying* (the selective case: $a_t, b_t$ depend on $x_t$), each scalar recurrence $h_t = a_t h_{t-1} + b_t$ is an affine map $f_t: h \mapsto a_t h + b_t$. Affine maps compose associatively:

$$f_{t_2} \circ f_{t_1}(h) = a_{t_2}\big(a_{t_1} h + b_{t_1}\big) + b_{t_2} = (a_{t_2} a_{t_1})\, h + (a_{t_2} b_{t_1} + b_{t_2}),$$

so a segment $[s..t]$ is summarized by the pair $\big(a_t \cdots a_s,\; \text{accumulated input}\big)$, and the pair-composition rule above is itself associative. Function composition is associative *always* — no approximation involved — so the T sequential steps can be reorganized into a balanced binary tree: a **parallel prefix scan** (Blelloch) that computes every prefix $h_1,\dots,h_T$ in $O(T)$ total work and $O(\log T)$ depth. This is the entire content of "the diagonal recurrence is parallelizable": the sequential dependency chain is real (you cannot know $h_T$ without all earlier inputs) but its *depth* collapses from $T$ to $\log T$, and the work is embarrassingly parallel. The segment summary $(\alpha, \beta)$ is exactly the "chunk primitive" the chunkwise algorithm reuses at chunk granularity — see [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md).

**Consequence 2: convolution (time-invariant case).** When $A, B, C$ are constant in time, the kernel from Section 4.1 becomes a sum of $N$ geometric series:

$$k_\tau = C A^{\tau} B = \sum_{n=1}^{N} C_{:,n}\, B_{n,:}\, a_n^{\tau},$$

i.e. the convolution kernel is a linear combination of $N$ exponentials. Three evaluation strategies exist: FFT convolution ($O(T\log T)$ parallel over time), the scan above ($O(T)$ work, $O(\log T)$ depth), and the sequential recurrence ($O(1)$ per step — the only one usable at generation time, one token at a time). S4's core algorithmic contribution was making the convolution-mode kernel computable in closed form for its special (normal-plus-low-rank, NPLR) parameterization of $A$, which is why it could train in convolution mode. The catch: convolution *requires* time-invariance, and time-invariance is exactly what "selective" SSMs sacrifice.

## 5. Historical arc: S4 → S6 → Mamba-2 → Mamba-3

The arc is best read as a sequence of deliberate constraint removals, each paying for expressivity with a harder computation.

**S4 (Gu et al., 2021) — LTI SSMs, convolution-mode training.** The first practical deep SSM: parameters $A, B, C$ (and a scalar $\Delta$) are learned but **input-independent**. With the NPLR parameterization, $A$ is (almost) diagonal, the kernel $C A^\tau B$ is a sum of exponentials computable in closed form, training runs as a parallel convolution, and inference runs as a constant-time-per-step recurrence. The limitation: with fixed parameters the model cannot *choose* what to remember — on tasks requiring selective attention (selective copying, induction heads), LTI SSMs fail where attention succeeds.

**S6 / Mamba (Gu & Dao, 2023) — selective SSMs.** Make $B$, $C$, and the effective transition all depend on the input: $B_t = B(x_t)$, $C_t = C(x_t)$, $\Delta_t = \Delta(x_t)$, $\bar A_t = e^{A \Delta_t}$. The model now decides per token how hard to write it into state and how to read state out. This breaks convolution (the kernel changes at every position), so training must fall back to the scan; the hardware-aware selective-scan kernel fuses the per-step operations to avoid round-tripping the state through HBM. The recurrence is also made explicitly linear (nonlinearity removed from the state update; gating lives in the projections), which is what keeps the scan valid — and Mamba dissolves the "RNN vs attention" dichotomy: one architecture, O(1)-per-step generation, parallelizable training.

**Mamba-2 / SSD (Dao & Gu, 2024) — state-space duality.** Observe that the selective scan can be rewritten as a *matrix multiplication with structured masks*: the scan is a "semi-separable" matrix times the input sequence, and for a structured $A$ this becomes chunkwise matrix multiplication — big intra-chunk matmuls (Tensor-Core friendly) plus a much shorter scan over chunks. The practical payoff: larger state dimensions (N = 64/128 vs 16 in Mamba-1) become affordable, and state size is what buys long-range memory. See [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md).

**Mamba-3 — complex SSD.** Promote the state into the complex plane. Each complex state component packs two real dimensions with a single complex eigenvalue $a = \alpha + i\beta$, whose transition is

$$e^{a\Delta} = e^{\alpha\Delta}\big(\cos(\beta\Delta) + i\sin(\beta\Delta)\big)$$

— one parameter produces *both* decay ($e^{\alpha\Delta}$) and rotation ($\beta\Delta$ radians per step), i.e. a damped oscillator. A real SSM needs two real state dimensions to express oscillation; the complex SSM gets it in one. The claim this repo tests is the halving: N = 64 complex states achieve parity with N = 128 real states (two real sub-states packed per complex state), at half the state memory. The full derivation is in [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md).

**Where this repo sits.** Mamba-3-Lite implements the complex-SSD recurrence faithfully and audibly: the chunkwise algorithm in pure-PyTorch einsums is the runtime path (`models/ssd_complex.py:ssd_complex_chunkwise`, with an opt-in Triton kernel behind `ssd_dispatch='triton'`), and the naive O(T) scan is kept as the reference oracle the fast path is proven against. Beyond complex states, the repo follows Mamba-3 in replacing the SISO constraint with a MIMO head mixer and dropping the causal convolution ([docs/concepts/block-and-stability.md](../concepts/block-and-stability.md)); scaling, param counts (~434M), and the `[INFERENCE]`-marked throughput targets live in [docs/concepts/block-and-stability.md](../concepts/block-and-stability.md).

## 6. Code walkthrough: `ssd_naive_complex` as the O(T) ground truth

The naive scan is the specification. `models/ssd_complex.py:ssd_naive_complex` transcribes the recurrence of Section 4.2 with no algebraic rearrangement whatsoever:

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
        s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s             + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
        ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
    return torch.stack(ys, dim=1)
```

**Shapes and layout.** The function is written in *head-major* form: `x` is `(B, T, H, D)` (batch, time, heads, head-dim), `B_t`/`C_t` are `(B, T, H, N)` (per token, per head, the N-dimensional input/output mixing vectors), `A` is a per-head complex scalar `(H,)`, and `dt` is `(B, T, H)`. The discretization is one line, `models/ssd_complex.py:_discretise`:

```python
def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.softplus(dt) * A)
```

`F.softplus(dt)` (float32, `(B, T, H)`) is multiplied by `A` (complex64, `(H,)`) — the product promotes to complex64 — and exponentiated to give `A_bar` of shape `(B, T, H)`: one complex decay factor per (batch, token, head). This is exactly $\bar A_t = e^{\operatorname{softplus}(\Delta_t) A}$ from Section 4.2, with the B-absorption convention applied (Section 4.2), so the recurrence uses raw `B_t`.

**The state shape `(B, H, N, D)`.** The state is not a vector but a *matrix per (batch, head)*: the outer product of the N-dimensional write vector `B_t[:, t]` with the D-dimensional token content `x[:, t]`,

$$s^{(b,h)} \leftarrow \bar A^{(b,h)}_t \, s^{(b,h)} + B^{(b,h)}_t \otimes x^{(b,h)}_t \;\in\; \mathbb{C}^{N \times D},$$

implemented as `B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)`: `(B, H, N, 1) × (B, H, 1, D) → (B, H, N, D)` via broadcasting. Each of the N rows is an independently decaying copy of the token content, scaled by $B_n$; the decay `A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s` is a per-head scalar multiply applied to all N rows — the Mamba-3 per-head scalar-A parameterization (contrast Mamba-1's per-state real $a_n$). The output read is the contraction over N: `(C_t[:, t].unsqueeze(-1) * s).sum(dim=-2)` computes $\sum_n C_n s_{n,:} = C_t^\top s$, giving `(B, H, D)` per step.

**Why it is the oracle.** Three properties make this function the ground truth rather than just "a slow version":

1. *Directness*: each line is a term of the recurrence — `A_bar` the discretized decay, the `s` update the state equation, the `ys` append the output equation. No reordering, no factorization, no einsum contraction that could conceal a sign or index error; correctness is established by inspection.
2. *Full generality*: no assumption about T, chunking, or time-invariance; any sequence length works, and it is the definition the chunkwise path must reproduce.
3. *Independent arithmetic*: it accumulates sequentially in canonical order while the chunkwise path reorganizes the same sums into einsum contractions with a different summation order — so agreement to `1e-4` (Section 8) is a genuine check of the algebra, not a tautology.

Caveat: `ssd_naive_complex` is *not* on the model's hot path. In `models/mamba_block.py:Mamba3Block._forward_impl`, the block's input projection is sliced into `x_ssm` (float32), `B_real`/`B_imag` → `B_t` (complex64), `C_real`/`C_imag` → `C_t` (complex64), and `dt` (float32), then routed through `Mamba3Block._ssd_with_dispatch`, which always calls the chunkwise `ssd_complex_chunkwise` (with optional Triton dispatch). The naive scan exists to be the specification the tests hold the fast path to — a "reference oracle", as its own docstring says. (Note also the dtype convention the test relies on: the naive function returns `torch.complex64`, while `ssd_complex_chunkwise` returns `Y.real` as float32 — comparisons must be against `y_naive.real`, see Section 8.)

## 7. Pitfalls

**The naive scan is memory-bound, not compute-bound.** Each step of the loop reads and writes the full state `(B, H, N, D)` of complex64 values (16 bytes per element). At the repo's full config (B = 16, H = 16, N = 64, D = 64, T = 2048), that is 1,048,576 elements ≈ 16 MiB per step, × 2048 steps ≈ 32 GiB of HBM traffic *per layer per batch* — for roughly one complex multiply-add per state element, an arithmetic intensity on the order of 1 flop per byte moved, orders of magnitude below what saturates a GPU's tensor cores. The sequential dependency chain additionally caps latency at T × per-step latency, and the per-step Python loop adds interpreter overhead. This is *why* the chunkwise algorithm exists ([docs/concepts/ssd-theory.md](../concepts/ssd-theory.md)): it replaces the per-token state round-trip with intra-chunk matmuls that keep data in registers. All throughput numbers here are `[INFERENCE]` — there is no `.benchmarks/` directory in this tree.

**`dt` and softplus.** Three traps cluster here. (1) *Never feed raw dt into `exp`.* The projection value is unconstrained; a negative value would make $e^{A\,\Delta_t}$ a growth factor at the init $A = -1$, and the state diverges over the sequence. `softplus` is the positivity guarantee — it belongs inside `_discretise`, and any code path that bypasses it is introducing an instability. (2) *There is no zero step.* $\operatorname{softplus}(0) = \ln 2 \approx 0.693$, so a raw dt of exactly 0 still advances the state by a factor $e^{0.693\,\mathrm{Re}(A)}$ per token; if you intended "hold the state", you must push dt far negative, and even then it only asymptotes to 0. (3) *Overflow.* $\exp(\operatorname{softplus}(\Delta_t) A)$ overflows to inf when $\mathrm{Re}(A)\operatorname{softplus}(\Delta_t) \gtrsim 88$. The $-1.0$ init keeps the exponent ≤ 0 at init (decay ≤ 1), but a learned $A$ with positive real part plus a large dt can blow up — the complex cumsum in the chunkwise path is where this surfaces, part of why the training loop carries a NaN guard (see [docs/concepts/block-and-stability.md](../concepts/block-and-stability.md)). Also remember the dtype promotion: `softplus(dt)` is float32, `A` is complex64, their product complex64 — `x_ssm` staying float32 in `Mamba3Block._forward_impl` while B/C go complex is deliberate, and only `Y.real` (float32) is ever returned from the chunkwise path.

**State-init zeros.** The recurrence starts from `s = torch.zeros(B_, H, N, D)`. This is the *correct* causal choice: before the first token nothing is known, and because the recurrence is linear, zero init introduces no bias — $y_1 = C B x_1$ with no ghost contribution. The traps are elsewhere: (1) *reset per sequence* — a state that survives across sequences leaks context; the naive function creates it fresh inside the call, and the chunkwise path has no `initial_states` parameter (every scan starts from zeros), so neither can leak state. (2) *Layout mismatch* — the chunkwise path's internal state is `(B, H, D, N)` complex64 (note the transposed N/D vs the naive scan's `(B, H, N, D)`); it never leaves the function, so the mismatch is an implementation detail, not a caller contract. (3) *No absolute position.* With zero init and no positional embedding, the model has no absolute-position signal — position must be inferred from the decay/rotation dynamics, which encode *relative* structure; a design property (see [docs/concepts/block-and-stability.md](../concepts/block-and-stability.md)), not an omission to "fix" by initializing the state to something else.

## 8. Tests

The machine proof of scan-vs-chunkwise equivalence is `tests/test_ssd.py::test_chunkwise_matches_naive_complex`:

```python
def test_chunkwise_matches_naive_complex():
    torch.manual_seed(0)
    B, T, H, D, N = 2, 16, 2, 4, 4
    x = torch.randn(B, T, H, D, dtype=torch.complex64)
    A = torch.randn(H, dtype=torch.complex64) - 1.0
    B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
    dt = torch.zeros(B, T, H)
    y_chunk = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
    y_naive = ssd_naive_complex(x, A, B_t, C_t, dt)
    assert y_chunk.dtype == torch.float32
    assert y_naive.dtype == torch.complex64
    assert y_chunk.shape == y_naive.shape == (B, T, H, D)
    assert torch.allclose(y_chunk, y_naive.real, atol=1e-4), (
        f"max diff = {(y_chunk - y_naive.real).abs().max().item()}"
    )
```

Reading it against this doc: `A = randn − 1.0` realizes the $\mathrm{Re}(A) \le 0$ init convention; `dt = 0` still exercises a real step size because $\operatorname{softplus}(0) = \ln 2$ (Section 4.2); `chunk_size = 4` with T = 16 gives a non-trivial 4-chunk decomposition; and `atol=1e-4` against `y_naive.real` is the honest allowance for *reordered summation* — the sequential loop accumulates in canonical order while the einsum chain contracts in a different order, so the two agree to float32 rounding, not bit-for-bit. The dtype assertions (`y_chunk` float32, `y_naive` complex64) pin the contract Section 6 flagged. Two edge cases complete coverage: `test_chunkwise_handles_uneven_T` (T = 20, not a multiple of 4 — exercises the padding branch and slice-back) and `test_chunkwise_handles_T_equal_to_chunk` (T = 4 — the single-chunk degenerate case). The deeper algebra of *why* chunkwise equals the scan is in [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md), and the API contract of both functions in [docs/references/ssd-reference.md](../references/ssd-reference.md). The full suite runs 28 tests on CPU with 5 GPU-gated skips; the doc map is [docs/README.md](../README.md).

---

## References

- [Mamba-3-Lite — SSD Theory](ssd-theory.md) — state-space duality, complex states, and the full chunkwise derivation.
- [Mamba-3-Lite — MIMO Head Mixing](mimo.md) — the cross-head mixer that follows the scan.
- [Mamba-3-Lite — Block Anatomy and Numerical Stability](block-and-stability.md) — the block's slice layout and the stability rules (`dt`/softplus, NaN guard).
- [Mamba-3-Lite — SSD Reference](../references/ssd-reference.md) — the API contracts of `models/ssd_complex.py:ssd_naive_complex` and `models/ssd_complex.py:ssd_complex_chunkwise`.
- S4 — Gu et al., 2021 (arXiv:2111.00396); Mamba — Gu & Dao, 2023 (arXiv:2312.00752); Mamba-2 / SSD — Dao & Gu, 2024 (arXiv:2405.21060).
