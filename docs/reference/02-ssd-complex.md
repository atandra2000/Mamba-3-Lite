# Complex SSD — The Naive Oracle and the Chunkwise Scan

Reference doc for `models/ssd_complex.py`: the two functions that implement the complex state-space-duality (SSD) scan — the O(T) sequential oracle and the production chunkwise path — with exact signatures, tensor-shape contracts, the corrected inter-chunk propagation math, and the tests that pin them down.

## 60-second summary

`models/ssd_complex.py` contains two public functions plus one helper. `models/ssd_complex.py:ssd_naive_complex` is an O(T) sequential scan over a complex64 state — the verification oracle that defines what the SSD must compute. `models/ssd_complex.py:ssd_complex_chunkwise` is the production path: it pads the sequence to a multiple of `chunk_size`, evaluates each chunk's scan in closed form as batched einsums, carries a state across chunks with a fixed propagation rule, and returns the **real** part of the output as float32. `models/ssd_complex.py:_discretise` is the shared discretization helper `exp(softplus(dt) · A)`. After reading this doc you will know each function's signature, every tensor shape, the exact `decay_chunk[z,c] = exp(CD[z-1] − CD[c])·1[z>c]` inter-chunk contract, and why the two functions deliberately return different dtypes.

## Why it exists

Every Mamba-3-Lite layer needs a sequence-mixing primitive that is (a) provably correct and (b) fast enough to train. The two functions split that job. The naive scan is the ground truth — a literal loop over time that cannot be wrong by construction — the yardstick every other implementation is measured against. The chunkwise scan is what the model runs: `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` calls `ssd_complex_chunkwise` per layer with `chunk_size=64`, because chunking turns the serial recurrence into batched GEMMs that use tensor cores. Keeping the two as separate functions with a machine-checked equivalence (the tests in `tests/test_ssd.py`) lets the fast path be aggressive without losing the reference semantics.

The two functions also encode a dtype contract that is easy to trip over: the recurrence is complex internally, but the model observes only the real projection of the scan — `ssd_naive_complex` returns complex64, `ssd_complex_chunkwise` float32. See [docs/theory/04-chunkwise-algorithm.md](../theory/04-chunkwise-algorithm.md) for the full derivation and [docs/theory/01-ssm-foundations.md](../theory/01-ssm-foundations.md) for the discretization theory.

## Intuition

The naive scan is a single worker walking the sequence left to right, carrying one complex state per head — serial and slow. The chunkwise scan splits the sequence into folders of `C` consecutive timesteps: **inside a folder, everything is a matmul** (output at position `l` is a causally-weighted sum over sources `s ≤ l`, i.e. a batched GEMM against a triangular decay matrix), and **across folders, you carry a state** (each chunk contributes one "state at its right edge"; earlier contributions arrive at chunk `z` decayed by a product of per-chunk decays that telescopes into `exp(CD[z-1] − CD[c])`). The state is complex because `A` is a complex per-head scalar: the real part controls how fast memory fades, the imaginary part makes it oscillate.

## Math: the recurrence both functions implement

With `A_bar_t = _discretise(dt_t, A) = exp(softplus(dt_t) · A) ∈ ℂ^H`, the recurrence over a state `h_t ∈ ℂ^{H×N×D}` is

$$h_t = \bar a_t \odot h_{t-1} + B_t \otimes x_t, \qquad y_t = \sum_{n=1}^{N} C_{t,n} \odot h_{t,n},$$

where `B_t ⊗ x_t` is the outer product into `(H, N, D)` and `C_t` contracts the `N` index. With zero initial state the closed form is

$$y_t = \sum_{s \le t} C_t^{\top} \left(\prod_{u=s+1}^{t} \bar a_u\right) B_s x_s,$$

with the empty product equal to 1. Both functions compute exactly this function; they differ only in *association* — the naive scan multiplies decay factors one timestep at a time, the chunkwise path groups them into chunk-closed forms and telescopes inter-chunk products through cumulative per-chunk decays (proof sketch in [docs/theory/04-chunkwise-algorithm.md](../theory/04-chunkwise-algorithm.md), section 5).

## Code walkthrough

### `_discretise(dt, A)` — the shared discretization

```python
def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.softplus(dt) * A)
```

`softplus` maps `dt` to a strictly positive step size; multiplying by the per-head complex scalar `A` gives a complex log-decay with real part `≤ 0` whenever `Re(A) ≤ 0` (the repo initializes `A = -1`, so every per-step factor has magnitude `≤ 1/2`). `dt` is float32, `A` complex64, so the multiply promotes to complex64 and the exponential is a rotation composed with a scale — this helper is what makes the naive `A_bar` and the chunkwise `A_log` agree on the *same* per-position decay.

### `ssd_naive_complex` — the O(T) oracle

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

Parameter by parameter:

- `x` — token content, `(B, T, H, D)`, complex64.
- `A` — per-head complex scalar, `(H,)`, complex64. Constant across time and batch, broadcast against `dt` in `_discretise`.
- `B_t` — input projection, `(B, T, H, N)`, complex64. The `_t` suffix means "per timestep".
- `C_t` — output projection, `(B, T, H, N)`, complex64.
- `dt` — per-timestep step size, `(B, T, H)`, **float32** (real). The only real input; it becomes complex only after multiplying by complex `A`.

Return contract: `(B, T, H, D)` complex64 — the *full* complex output, imaginary part included.

Walking the loop over state `s` of shape `(B, H, N, D)`: `A_bar[:, t]` unsqueezed to `(B, 1, H, 1)` broadcasts one complex scalar decay per head over the `N×D` state; `B_t[:, t].unsqueeze(-1)` `(B, H, N, 1)` times `x[:, t].unsqueeze(-2)` `(B, H, 1, D)` is the rank-1 update `B_t ⊗ x_t`; `C_t[:, t].unsqueeze(-1)` multiplies the state and sums over `dim=-2` (the `N` axis) to give `(B, H, D)`; `torch.stack(ys, dim=1)` collects the `T` readouts into `(B, T, H, D)`. O(T) sequential and memory-bound — it exists to be *correct*, not fast.

### `ssd_complex_chunkwise` — the production scan

```python
def ssd_complex_chunkwise(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
    chunk_size: int = 64,
    ssd_dispatch: str = "pytorch",
) -> torch.Tensor:
    """Complex chunkwise SSD.

    ssd_dispatch='pytorch': the original 5-einsum PyTorch chain.
    ssd_dispatch='triton': per-chunk Y_diag and state fused into a single
    Triton kernel; inter-chunk state propagation stays in PyTorch.
    """
```

The first five parameters are identical to `ssd_naive_complex`. The two extra parameters:

- `chunk_size` (`int`, default `64`) — the chunk length `C`; the sequence is padded to a multiple of `C`.
- `ssd_dispatch` (`"pytorch" | "triton"`, default `"pytorch"`) — which implementation computes the per-chunk work (see Dispatch semantics below).

The scan always starts from a fresh zero state — there is no `initial_states` parameter (see [docs/theory/01-ssm-foundations.md](../theory/01-ssm-foundations.md) §State-init zeros).

Return contract: `(B, T, H, D)` **float32 real** — `Y.real`, sliced back to `T`. The dtype difference from the oracle is part of the contract: the chunkwise path discards the imaginary part by design, because the block's residual branch must be real (see [docs/theory/06-block-anatomy.md](../theory/06-block-anatomy.md)).

**Padding rule.** Exactly as implemented:

```python
    pad = (C - (T % C)) % C
    if pad > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        B_t = F.pad(B_t, (0, 0, 0, 0, 0, pad))
        C_t = F.pad(C_t, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))

    T_padded = T + pad
    n_chunks = T_padded // C
```

The double-mod is defensive: when `T % C == 0` it yields 0, so the pad branch is skipped. Each `F.pad` appends `pad` zero-rows on the time axis; `A` is not padded (per-head constant broadcast over time). Padding never affects outputs at real positions `t < T` — it only aligns the chunk grid — and the return slices it back off.

**Chunking and the discretized log-decay.**

```python
    A_log = F.softplus(dt) * A

    def _chunk(t):
        return t.reshape(B_, n_chunks, C, *t.shape[2:])

    Xc, Bc, Cc, Ac = _chunk(x).to(torch.complex64), _chunk(B_t), _chunk(C_t), _chunk(A_log)

    A_cumsum = torch.cumsum(Ac, dim=2)
    decay_states = torch.exp(A_cumsum[:, :, -1:, :] - A_cumsum)
```

`_chunk` is a pure reshape (a view): the padded time axis has size exactly `n_chunks · C`, so global time `t = c·C + l` lands at chunk `c`, position `l`. `Ac` is complex64 (`softplus(dt) * A` promotes), and `A_cumsum` is its within-chunk cumsum over `dim=2`, so the product of decays from `s` to `l` telescopes into `exp(A_cumsum[l] − A_cumsum[s])`. `decay_states` specializes this to `l = C-1` (the chunk's *right edge*): `exp(A_cumsum[:, :, -1:, :] − A_cumsum)`, shape `(B, n_chunks, C, H)`. The `-1:` keepdim keeps a size-1 position axis that broadcasts against `(B, n_chunks, C, H)`.

**Intra-chunk outputs and per-chunk states** (the `"pytorch"` branch; the `"triton"` branch computes the same two tensors in one fused kernel):

```python
        Ac_perm = Ac.permute(0, 1, 3, 2).contiguous()
        T_c = Ac_perm.size(-1)
        Ac_cumsum = torch.cumsum(Ac_perm, dim=-1)
        Ac_seg = Ac_cumsum.unsqueeze(-1) - Ac_cumsum.unsqueeze(-2)
        mask = torch.tril(torch.ones(T_c, T_c, device=x.device, dtype=torch.bool))
        L = torch.exp(Ac_seg) * mask

        Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)
        states = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)
```

`L[c,l,s] = exp(A_cumsum[c,l] − A_cumsum[c,s])·1[l ≥ s]` is each chunk's causal decay matrix — lower-triangular (diagonal included), complex64, built from the pairwise cumsum difference. The `Y_diag` einsum contracts `s` and `n` to give the zero-carry-in output of every chunk, shape `(B, n_chunks, C, H, D)`. The per-chunk state einsum sums over positions `l` weighted by `decay_states` — each chunk's own contribution to the state at its right edge, shape `(B, n_chunks, H, D, N)`. (The current branch runs four einsums; the docstring's "5-einsum" label predates the current formulation, where `L` is built with `exp`/`tril` rather than an einsum.)

**Inter-chunk propagation — the fixed contract.** This is the load-bearing part of the function; read it exactly as implemented:

```python
    chunk_decay = A_cumsum[:, :, -1, :]
    cd_perm = chunk_decay.permute(0, 2, 1).contiguous()
    cd_cumsum = torch.cumsum(cd_perm, dim=-1)
    # cd_shift[z] = CD[z-1] with CD[-1] := 0: the total decay applied to the
    # initial state as it travels through chunks 0..z-1.
    cd_shift = torch.cat([torch.zeros_like(cd_cumsum[..., :1]), cd_cumsum[..., :-1]], dim=-1)
    # M[z, c] = exp(CD[z-1] - CD[c]) · 1[z > c]: decay applied to chunk c's
    # end-of-chunk state while it travels through chunks c+1..z-1.
    cd_seg = cd_shift.unsqueeze(-1) - cd_cumsum.unsqueeze(-2)
    decay_chunk = torch.exp(cd_seg) * torch.tril(
        torch.ones(n_chunks, n_chunks, device=x.device, dtype=torch.bool), diagonal=-1,
    )

    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    if initial_states is None:
        initial_states = torch.zeros(B_, H, D, N, device=x.device, dtype=torch.complex64)
    # The initial state predates chunk 0, so it decays through chunks 0..z-1.
    init_decay = torch.exp(cd_shift).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    states = states + init_decay * initial_states.unsqueeze(1)
```

Decoding: `chunk_decay[b,c,h] = Λ_c` is the total log-decay across chunk `c` (deliberately *squeezed*, unlike `decay_states`). `cd_perm` transposes to `(B, H, n_chunks)` and `cd_cumsum` is the running sum `CD[k] = Σ_{u≤k} Λ_u`. `cd_shift` prepends a zero and drops the last slot, so `cd_shift[z] = CD[z-1]` with `CD[-1] := 0`. The pairwise difference gives `cd_seg[z,c] = CD[z-1] − CD[c]`, and `torch.tril(..., diagonal=-1)` is **strictly** lower triangular, excluding the diagonal entry `exp(CD[z-1] − CD[z]) = exp(−Λ_z)` (a spurious artifact of the shift). The result is exactly the contract:

$$\text{decay\_chunk}[z, c] = \exp\!\big(CD[z-1] - CD[c]\big) \cdot \mathbf{1}[z > c], \qquad CD[-1] := 0.$$

The propagation einsum `"bhzc,bchpn->bzhpn"` contracts source-chunk `c`: the carry-in state of chunk `z` is the sum over all earlier chunks `c < z` of their end-of-chunk contributions, each decayed by `exp(CD[z-1] − CD[c])`.

The initial state is a **separate term**, not routed through `decay_chunk`: it predates chunk 0 and therefore decays through chunks `0, …, z-1` — a window one chunk longer than any `S_c`. Its factor is `init_decay = exp(cd_shift)`, reshaped to `(B, n_chunks, H, 1, 1)` and broadcast against `initial_states.unsqueeze(1)`:

$$G_z = \underbrace{\sum_{c < z} \exp\!\big(CD[z-1] - CD[c]\big)\, S_c}_{\text{decay\_chunk term}} + \underbrace{\exp\!\big(CD[z-1]\big)\, H_{-1}}_{\text{init term}}.$$

With the default `initial_states=None` the init term is exactly zero and the path still runs (it allocates the zero tensor). **Correctness note:** the propagation window must be `CD[z-1] − CD[c]`, not `CD[z] − CD[c]` — the latter carries each contribution one chunk too far and only agrees with the naive scan when all `Λ_c` are equal (e.g. `dt = 0` everywhere); this off-by-one is what the time-varying-`dt` regression test catches.

**Readout, real cast, slice-back.**

```python
    Y_off = torch.einsum("bclhn,bchpn,bclh->bclhp", Cc, states, torch.exp(A_cumsum))

    Y = Y_diag + Y_off
    Y = Y.real
    return Y.reshape(B_, T_padded, H, D)[:, :T, :, :]
```

`Y_off` reads the carry-in state of each chunk at each position: `C` at `(c,l)` times the carry-in state times `exp(A_cumsum[c,l])` — the decay from the chunk's left edge to position `l` — summed over `n`. `Y = Y_diag + Y_off` is the full output. Then `Y.real` drops the imaginary part (float32), the reshape inverts the chunk grid, and `[:, :T, :, :]` removes the padding — without this slice, an uneven sequence would return `T_padded` outputs.

**Dispatch semantics.**

- `ssd_dispatch="pytorch"` — the branch shown above in eager PyTorch. The default; the only path available on CPU/Mac (Triton is Linux+CUDA-only).
- `ssd_dispatch="triton"` — the same two per-chunk tensors computed by one fused Triton kernel:

```python
    if ssd_dispatch == "triton":
        from .ssd_triton import per_chunk_ssd_triton
        Y_diag, states = per_chunk_ssd_triton(Bc, Cc, Xc, Ac, decay_states)
```

`models/ssd_triton.py:per_chunk_ssd_triton` launches one program per `(B, c, H)` tile, splitting each complex tensor into contiguous float32 real/imag pairs and computing `Y_diag` and `states` with `tl.dot` (documented in [docs/reference/03-ssd-triton.md](../reference/03-ssd-triton.md)). Crucially, **only the per-chunk work is fused** — the inter-chunk `decay_chunk` construction, propagation einsum, init term, `Y_off`, `.real`, and slice-back run in PyTorch exactly as above, so the propagation contract is shared verbatim between dispatch modes. If Triton is missing, `per_chunk_ssd_triton` raises `ImportError`; `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` catches it and falls back to `"pytorch"` with a one-shot warning.

## Shape contract

| tensor | shape | dtype | role |
|---|---|---|---|
| `x` | `(B, T, H, D)` | complex64 (promoted) | token content per head |
| `A` | `(H,)` | complex64 | per-head decay scalar, constant over T |
| `B_t` | `(B, T, H, N)` | complex64 | input projection |
| `C_t` | `(B, T, H, N)` | complex64 | output projection |
| `dt` | `(B, T, H)` | float32 (real) | per-timestep step size |
| naive output | `(B, T, H, D)` | complex64 | full complex scan result |
| chunkwise output | `(B, T, H, D)` | float32 | `Y.real` sliced to T |
| `initial_states` | `(B, H, D, N)` | complex64 | carry-in predating chunk 0; defaults zeros |
| internal `states` | `(B, n_chunks, H, D, N)` | complex64 | carry-in state at each chunk's left edge |

Chunked intermediates are `(B, n_chunks, C, …)` with `n_chunks = ⌈T/C⌉`. The chunkwise state layout is `(…, H, D, N)` — head-dim before state — which is why `initial_states` is `(B, H, D, N)`, not `(B, H, N, D)`.

## Invariants

- **`T` may be any positive integer.** Uneven `T` is handled by right-padding to a multiple of `C` and slicing back; `T % C == 0` skips the pad branch entirely (no copy).
- **`chunk_size`, `state_dim` (N), and `head_dim` (D) are powers of two and at most 256 for the triton dispatch.** `models/ssd_triton.py:_check_block_dims` raises `ValueError` when head dim `P`, state dim `N`, or `chunk_size` exceeds the `_MAX_BLOCK = 256` cap, pointing back to `ssd_dispatch='pytorch'`; the kernel also passes these dims as Triton constexpr block sizes used by `tl.arange`, which requires power-of-two lengths (the repo uses 64/64/64). `[INFERENCE]` the power-of-two requirement follows from Triton's `tl.arange` semantics; the code-enforced check is only the 256 cap, with evenness of `N`/`D` a necessary consequence.
- **`A` has shape `(H,)` and is broadcast over `T`.** One complex decay scalar per head for the whole sequence; time-dependence comes *only* from `dt` via `softplus(dt_t) · A`. Both functions use the same `_discretise`, so oracle and fast path share per-position decays.
- **Return dtype is part of the contract.** Naive → complex64; chunkwise → float32 real. The tests assert both.

## Pitfalls

- **The two functions return different dtypes.** `ssd_naive_complex` returns complex64 (imaginary part intact); `ssd_complex_chunkwise` returns float32 (`Y.real`). Substituting one for the other without `.real` silently feeds complex numbers into a real residual branch, and PyTorch happily promotes downstream matmuls.
- **`initial_states` layout is `(B, H, D, N)`, not `(B, H, N, D)`.** It matches the chunkwise internal layout `(b, c, h, p, n)`, *not* the naive scan's internal `(B, H, N, D)`. With default zeros the discrepancy is invisible; a nonzero `initial_states` must be transposed before comparing against the naive oracle. The init term is also *not* routed through `decay_chunk` — it decays one chunk longer than any chunk contribution (`exp(CD[z-1])`, not `exp(CD[z-1] − CD[c])`).
- **`x` is silently promoted to complex64** by `_chunk(x).to(torch.complex64)`; `B_t`/`C_t` are already complex64 (packed from real/imag pairs in `models/mamba_block.py:Mamba3Block._forward_impl`), and `softplus(dt) * A` promotes `dt`. Verify dtypes at the boundary — silent promotions make dtype bugs cheap to introduce.
- **The `[:, :T]` slice-back is load-bearing.** Removing it returns `T_padded` outputs (including padding zeros), silently corrupting the residual add downstream.
- **`dt = 0` masks inter-chunk propagation bugs.** With constant per-position decay every inter-chunk factor collapses to products of identical `Λ`s, so an equivalence test with `dt = 0` cannot detect an off-by-one in the propagation window — the pre-fix code passed the constant-`dt` test while being wrong for input-dependent `dt`.
- **`decay_states` keepdim vs `chunk_decay` squeeze.** `A_cumsum[:, :, -1:, :]` keeps a size-1 position axis (shape `(B, n_chunks, 1, H)`) required for the `bclh` broadcast; `A_cumsum[:, :, -1, :]` deliberately squeezes to `(B, n_chunks, H)`. Dropping the colon in the first turns a broadcast into a shape error; adding it in the second feeds a 4-D tensor into the chunk-axis cumsum.
- **Triton dispatch is not silently degraded inside `ssd_complex_chunkwise`.** The lazy import raises `ImportError` if Triton is missing — the fallback lives one level up in `Mamba3Block._ssd_with_dispatch`.

## Tests

All four tests live in `tests/test_ssd.py`:

- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` — constant-`dt` equivalence: `B=2, T=16, H=2, D=4, N=4`, `chunk_size=4`, complex random `A, B_t, C_t, x` and `dt = 0`; asserts the dtype contract (`y_chunk` float32 vs `y_naive` complex64), equal shapes, and `allclose(y_chunk, y_naive.real, atol=1e-4)`.
- `tests/test_ssd.py::test_chunkwise_matches_naive_time_varying_dt` — the general regression test: same shapes but `dt = torch.randn(B, T, H)`, so per-chunk decay totals differ and the inter-chunk propagation is exercised in full. Fails on the pre-fix propagation (the `CD[z]`-vs-`CD[z-1]` off-by-one) and pins the corrected form documented above.
- `tests/test_ssd.py::test_chunkwise_handles_uneven_T` — `T=20`, `chunk_size=4`; asserts shape `(B, T, H, D)` and finiteness. Note `20 % 4 == 0`, so despite the name this test does *not* hit the `F.pad` branch — the padding path is currently untested.
- `tests/test_ssd.py::test_chunkwise_handles_T_equal_to_chunk` — `T=4`, `chunk_size=4`: a single chunk, so `n_chunks = 1` and there is no inter-chunk step; asserts shape and finiteness.

## Related

- [docs/theory/04-chunkwise-algorithm.md](../theory/04-chunkwise-algorithm.md) — the full derivation, einsum by einsum, with a worked example and equivalence proof sketch.
- [docs/theory/03-complex-ssd.md](../theory/03-complex-ssd.md) — why the state is complex; the discretization theory behind `_discretise`.
- [docs/theory/02-state-space-duality.md](../theory/02-state-space-duality.md) — the masked-linear-attention view of the chunkwise algorithm.
- [docs/theory/01-ssm-foundations.md](../theory/01-ssm-foundations.md) — the underlying SSM recurrence and softplus discretization.
- [docs/theory/06-block-anatomy.md](../theory/06-block-anatomy.md) — where the scan sits in the Mamba block; why its output must be real.
- [docs/reference/03-ssd-triton.md](../reference/03-ssd-triton.md) — the fused Triton kernel and its autograd `Function` contract.
- [docs/reference/05-mamba-block.md](../reference/05-mamba-block.md) — `Mamba3Block._ssd_with_dispatch`, the per-layer caller with its fallback semantics.

## Anchors cited

- `models/ssd_complex.py:ssd_complex_chunkwise`
- `models/ssd_complex.py:ssd_naive_complex`
- `models/ssd_complex.py:_discretise`
- `models/ssd_triton.py:per_chunk_ssd_triton`
- `models/ssd_triton.py:_check_block_dims`
- `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`
