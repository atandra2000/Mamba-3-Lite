# Mamba-3-Lite — SSD Reference: the complex scan API and kernel contract

Reference doc for `models/ssd_complex.py`: the two functions that implement the complex state-space-duality (SSD) scan — the O(T) sequential oracle and the production chunkwise path — with exact signatures, tensor-shape contracts, the corrected inter-chunk propagation math, and the tests that pin them down.

## 60-second summary

`models/ssd_complex.py` contains two public functions plus one helper. `models/ssd_complex.py:ssd_naive_complex` is an O(T) sequential scan over a complex64 state — the verification oracle that defines what the SSD must compute. `models/ssd_complex.py:ssd_complex_chunkwise` is the production path: it pads the sequence to a multiple of `chunk_size`, evaluates each chunk's scan in closed form as batched einsums, carries a state across chunks with a fixed propagation rule, and returns the **real** part of the output as float32. `models/ssd_complex.py:_discretise` is the shared discretization helper `exp(softplus(dt) · A)`. After reading this doc you will know each function's signature, every tensor shape, the exact `decay_chunk[z,c] = exp(CD[z-1] − CD[c])·1[z>c]` inter-chunk contract, and why the two functions deliberately return different dtypes.

## Why it exists

Every Mamba-3-Lite layer needs a sequence-mixing primitive that is (a) provably correct and (b) fast enough to train. The two functions split that job. The naive scan is the ground truth — a literal loop over time that cannot be wrong by construction — the yardstick every other implementation is measured against. The chunkwise scan is what the model runs: `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` calls `ssd_complex_chunkwise` per layer with `chunk_size=64`, because chunking turns the serial recurrence into batched GEMMs that use tensor cores. Keeping the two as separate functions with a machine-checked equivalence (the tests in `tests/test_ssd.py`) lets the fast path be aggressive without losing the reference semantics.

The two functions also encode a dtype contract that is easy to trip over: the recurrence is complex internally, but the model observes only the real projection of the scan — `ssd_naive_complex` returns complex64, `ssd_complex_chunkwise` float32. See [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md) for the full derivation and [docs/concepts/state-space-foundations.md](../concepts/state-space-foundations.md) for the discretization theory.

## Intuition

The naive scan is a single worker walking the sequence left to right, carrying one complex state per head — serial and slow. The chunkwise scan splits the sequence into folders of `C` consecutive timesteps: **inside a folder, everything is a matmul** (output at position `l` is a causally-weighted sum over sources `s ≤ l`, i.e. a batched GEMM against a triangular decay matrix), and **across folders, you carry a state** (each chunk contributes one "state at its right edge"; earlier contributions arrive at chunk `z` decayed by a product of per-chunk decays that telescopes into `exp(CD[z-1] − CD[c])`). The state is complex because `A` is a complex per-head scalar: the real part controls how fast memory fades, the imaginary part makes it oscillate.

## Math: the recurrence both functions implement

With `A_bar_t = _discretise(dt_t, A) = exp(softplus(dt_t) · A) ∈ ℂ^H`, the recurrence over a state `h_t ∈ ℂ^{H×N×D}` is

$$h_t = \bar a_t \odot h_{t-1} + B_t \otimes x_t, \qquad y_t = \sum_{n=1}^{N} C_{t,n} \odot h_{t,n},$$

where `B_t ⊗ x_t` is the outer product into `(H, N, D)` and `C_t` contracts the `N` index. With zero initial state the closed form is

$$y_t = \sum_{s \le t} C_t^{\top} \left(\prod_{u=s+1}^{t} \bar a_u\right) B_s x_s,$$

with the empty product equal to 1. Both functions compute exactly this function; they differ only in *association* — the naive scan multiplies decay factors one timestep at a time, the chunkwise path groups them into chunk-closed forms and telescopes inter-chunk products through cumulative per-chunk decays (proof sketch in [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md), section 5).

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

The scan always starts from a fresh zero state — there is no `initial_states` parameter (see [docs/concepts/state-space-foundations.md](../concepts/state-space-foundations.md) §State-init zeros).

Return contract: `(B, T, H, D)` **float32 real** — `Y.real`, sliced back to `T`. The dtype difference from the oracle is part of the contract: the chunkwise path discards the imaginary part by design, because the block's residual branch must be real (see [docs/concepts/block-and-stability.md](../concepts/block-and-stability.md)).

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

`models/ssd_triton.py:per_chunk_ssd_triton` launches one program per `(B, c, H)` tile, splitting each complex tensor into contiguous float32 real/imag pairs and computing `Y_diag` and `states` with `tl.dot` (documented in [Mamba-3-Lite — SSD Reference](ssd-reference.md)). Crucially, **only the per-chunk work is fused** — the inter-chunk `decay_chunk` construction, propagation einsum, init term, `Y_off`, `.real`, and slice-back run in PyTorch exactly as above, so the propagation contract is shared verbatim between dispatch modes. If Triton is missing, `per_chunk_ssd_triton` raises `ImportError`; `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` catches it and falls back to `"pytorch"` with a one-shot warning.

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

- [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md) — the full derivation, einsum by einsum, with a worked example and equivalence proof sketch.
- [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md) — why the state is complex; the discretization theory behind `_discretise`.
- [docs/concepts/ssd-theory.md](../concepts/ssd-theory.md) — the masked-linear-attention view of the chunkwise algorithm.
- [docs/concepts/state-space-foundations.md](../concepts/state-space-foundations.md) — the underlying SSM recurrence and softplus discretization.
- [docs/concepts/block-and-stability.md](../concepts/block-and-stability.md) — where the scan sits in the Mamba block; why its output must be real.
- [Mamba-3-Lite — SSD Reference](ssd-reference.md) — the fused Triton kernel and its autograd `Function` contract.
- [docs/references/model-reference.md](../references/model-reference.md) — `Mamba3Block._ssd_with_dispatch`, the per-layer caller with its fallback semantics.
---

#R3 — The Per-Chunk SSD Triton Kernel API

This reference doc is the migration and expansion of the retired `documentation/`-tree kernel design doc: it documents the sanctioned Triton kernel in `models/ssd_triton.py`, its host-wrapper API contract, the autograd backward contract, the numerical guarantees, and how it plugs into the production chunkwise path.

---

## 1. 60-second summary

After reading this doc you will understand: why Mamba-3-Lite ships one fused Triton kernel (the per-chunk `Y_diag` + `state` pass is the HBM-bandwidth hotspot of the chunkwise SSD), the exact host API — `models/ssd_triton.py:per_chunk_ssd_triton(Bc, Cc, Xc, A_log, decay_states) -> (Y_diag, state)` — plus its pure-PyTorch reference `models/ssd_complex.py:per_chunk_ssd_pytorch`, the tensor-shape and block-dim contracts (one program per `(B, n_chunks, H)`, `BLOCK_C/P/N = C/D/N`), the 256-cap hard-fail `models/ssd_triton.py:_check_block_dims`, the complex64→real/imag split `models/ssd_triton.py:_view_real_imag`, the recompute-based autograd backward of `models/ssd_triton.py:_PerChunkSSDTriton` (all five inputs get correct gradients, seeded with the true downstream `grad_outputs`), the env knobs (`TRITON_PER_CHUNK_NUM_STAGES=1`, `TRITON_PER_CHUNK_NUM_WARPS=4`), the numerical contract (fp32 accumulators, `atol=1e-3` fp32 / `1e-2` bf16 vs the reference), and the `tests/test_ssd_triton.py` verification surface.

## 2. Why this kernel exists

`models/ssd_complex.py:ssd_complex_chunkwise` is the per-step compute hotspot of Mamba-3-Lite. At the 434M config (`B=8, T=2048, C=64, n_chunks=32, H=16, D=64, N=64`) the pure-PyTorch chain materialises three large complex64 intermediates per chunk — the causal segment matrix `L` `(C, C)`, the intra-chunk term `Y_diag` `(C, P)`, and the per-chunk state contribution `(P, N)`, each ≈ 256 MB aggregate — writes them to HBM and reads them back for the subsequent einsums. The PyTorch path is therefore **HBM-bandwidth bound, not compute bound**. The kernel fuses the three into a single per-`(B, c, H)` pass: one Triton program reads `(Bc, Cc, Xc, A_log, decay_states)` once, computes `A_cumsum`, `L`, `Cb`, `Y_diag` and `state` in registers, and writes only the two outputs (`Y_diag`, `state`) to HBM. The inter-chunk propagation and the final `Y_off` application stay in PyTorch (see §8).

**Why not `torch.compile`?** `inductor` fuses elementwise + silu chains but cannot rewrite the cumsum → causal-mask → einsum structure into a single kernel, since `L[l, s]` is data-dependent on the cumsum of `A_log`.

**Why a GPU kernel at all?** The repo is a raw-PyTorch codebase with a strict "sanctioned paths" list; this is the single sanctioned Triton exception, gated behind a two-layered opt-in: `ssd_dispatch='triton'` in the config *and* `ENABLE_TRITON_KERNELS=1` at process start (§8). On CPU/Mac — where `triton` cannot be installed — the module imports cleanly and the triton path is simply unavailable, never broken.

**Performance estimate [INFERENCE]:** there is no `.benchmarks/` in the tree, so these are estimates from the traffic reduction, not measurements. Fusing the three intermediates cuts per-chunk HBM traffic by roughly 25% (two output writes instead of three full round-trips plus the segment matrix), an estimated ~20–30% wall-time speedup on the SSD call; at ~60% of a layer's compute, that is ~10–15% per training step. The A100-box checklist (§11) exists to confirm or refute this before the kernel ships in production.

## 3. Host wrapper API

`models/ssd_triton.py` is importable everywhere: the `triton` import is guarded, so `HAS_TRITON` is a bool and every host symbol exists even on a box without CUDA.

```python
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
```

### 3.1 `per_chunk_ssd_triton(Bc, Cc, Xc, A_log, decay_states) -> (Y_diag, state)` — the public entry point

```python
def per_chunk_ssd_triton(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public entry point. Returns (Y_diag, state) for the per-chunk pass."""
    if not HAS_TRITON:
        raise ImportError(
            "per_chunk_ssd_triton requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "For CPU/Mac, set ssd_dispatch='pytorch' in the model config."
        )
    return _PerChunkSSDTriton.apply(Bc, Cc, Xc, A_log, decay_states)
```

**Semantics.** Computes, for every `(b, chunk c, head h)`, the two per-chunk quantities of the chunkwise SSD:

- `Y_diag[b, c, :, h, :]` — the intra-chunk "attention" output `(C, P)`: the contribution to chunk `c`'s outputs from tokens *within* chunk `c` (the causal segment term),
- `state[b, c, h, :, :]` — the per-chunk end-of-chunk state `(P, N)` fed into the inter-chunk propagation.

Both are complex64 and match the einsums in the PyTorch path exactly (same `L`, same `decay_states` convention). The call is an autograd `Function` (see §6), so it is differentiable with respect to all five inputs.

**Shapes.** All tensors are complex64:

| Input | Shape | Meaning |
|---|---|---|
| `Bc` | `(B, n_chunks, C, H, N)` | chunked input projection `B_t` (from `in_proj`'s B-real/B-imag slices) |
| `Cc` | `(B, n_chunks, C, H, N)` | chunked output projection `C_t` |
| `Xc` | `(B, n_chunks, C, H, P)` | chunked SSM input `x` (here `P = D`, the head dim) |
| `A_log` | `(B, n_chunks, C, H)` | per-token complex log-decay `A_log = softplus(dt) * A` |
| `decay_states` | `(B, n_chunks, C, H)` | `exp(A_cumsum[:, :, -1:, :] - A_cumsum)` — the per-chunk decay applied to each token's state contribution |

Outputs: `Y_diag` is `(B, n_chunks, C, H, P)`, `state` is `(B, n_chunks, H, P, N)`.

**Failure modes.** Without `triton` installed, raises `ImportError` with an install hint; if any block dim exceeds 256, `_check_block_dims` raises `ValueError`. The parent dispatcher converts both into a per-instance one-shot fallback to the PyTorch path (§8).

### 3.2 `models/ssd_complex.py:per_chunk_ssd_pytorch` — the pure-PyTorch reference

```python
def per_chunk_ssd_pytorch(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for the per-chunk kernel.

    A_log is (B, n_chunks, C, H). L = exp(cumsum(A_log)[l] - cumsum(A_log)[s])
    matches the chunkwise linear-projection formula in ssd_complex_chunkwise.
    """
    C = Bc.shape[2]
    causal = torch.tril(torch.ones(C, C, device=Bc.device, dtype=torch.bool))
    A_log_h = A_log.permute(0, 1, 3, 2)
    A_cs = A_log_h.cumsum(dim=-1)
    seg = A_cs.unsqueeze(-1) - A_cs.unsqueeze(-2)
    L = torch.exp(seg) * causal
    Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)
    state = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)
    return Y_diag, state
```

**Roles.** (1) Correctness oracle: the GPU parity tests compare `per_chunk_ssd_triton` against it at `atol=1e-3` (fp32) / `1e-2` (bf16). (2) Recompute body of the backward pass (§6). (3) The single implementation of the per-chunk math: `ssd_complex_chunkwise`'s non-triton branch calls this function directly (a former inline copy of the same einsums lived there; consolidating removed the drift), which is why the dispatcher can swap implementations without changing outer numerics. Its shapes mirror the kernel contract exactly — `Y_diag (B, n_chunks, C, H, P)`, `state (B, n_chunks, H, P, N)` — verified by `test_reference_shape_and_finite` and `test_reference_matches_ssd_complex_chunkwise`.

### 3.3 `_check_block_dims(P, N, chunk_size)` — the 256-cap hard-fail

```python
def _check_block_dims(P: int, N: int, chunk_size: int) -> None:
    for name, dim in (("P", P), ("N", N), ("chunk_size", chunk_size)):
        if dim > _MAX_BLOCK:
            raise ValueError(
                f"per_chunk_ssd_triton: {name}={dim} exceeds the {_MAX_BLOCK}-cap. "
                f"Use ssd_dispatch='pytorch' for this config."
            )
```

Triton's constexpr block sizes are bounded; the kernel uses `BLOCK_C = chunk_size`, `BLOCK_P = D`, `BLOCK_N = N`, all constexpr. `_check_block_dims` turns an oversized dimension into a *named* `ValueError` — not a silent fallback and not a cryptic compiler error. The 434M config (`P=64, N=64, C=64`) passes; a config with, say, `state_dim=512` raises and the dispatcher falls back (§8). The error message names the offending dim and tells the user how to opt out (`ssd_dispatch='pytorch'`).

### 3.4 `_view_real_imag(z)` — the complex64 → float32 pair split

```python
def _view_real_imag(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split complex64 into two contiguous float32 tensors for the Triton kernel.

    view_as_real gives stride-2 on the inner dim; contiguous() copies to stride-1.
    """
    if z.dtype != torch.complex64:
        raise TypeError(
            f"per_chunk_ssd_triton: expected complex64, got {z.dtype}. "
            f"This kernel is specialised for the Mamba-3 complex SSD layout."
        )
    pair = torch.view_as_real(z.contiguous())
    return pair[..., 0].contiguous(), pair[..., 1].contiguous()
```

**Why this exists.** Triton ≥ 3.x has no native `complex64` pointer dtype, so the kernel sees only float32 tensors and performs complex arithmetic manually as real/imag pairs. `torch.view_as_real` yields a trailing stride-2 dimension; `.contiguous()` copies each half to stride-1 so the kernel can index them with a flat 1-D layout. The `TypeError` guard enforces the contract: the kernel is specialised for the Mamba-3 complex layout and refuses e.g. float32 inputs instead of silently misinterpreting them.

### 3.5 `_per_chunk_ssd_triton_forward(...)` — the launcher

```python
def _per_chunk_ssd_triton_forward(
    Bc: torch.Tensor, Cc: torch.Tensor, Xc: torch.Tensor,
    A_log: torch.Tensor, decay_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, n_chunks, C, H, N = Bc.shape
    P = Xc.shape[-1]
    _check_block_dims(P, N, C)
    ...
    T_padded = n_chunks * C
    num_stages = int(os.environ.get("TRITON_PER_CHUNK_NUM_STAGES", "1"))
    num_warps = int(os.environ.get("TRITON_PER_CHUNK_NUM_WARPS", "4"))
    _ssd_per_chunk_fwd_kernel[(B, n_chunks, H)](
        ...
        BLOCK_C=C, BLOCK_P=P, BLOCK_N=N,
        num_warps=num_warps, num_stages=num_stages,
    )

    Y_diag = torch.complex(Y_re, Y_im)
    state = torch.complex(S_re, S_im)
    return Y_diag, state
```

The launcher is the only host code that touches the JIT kernel (which itself must not be cited in docs — see §10). It allocates the float32 real/imag output buffers, splits all five complex inputs with `_view_real_imag`, launches the kernel on a **3-D grid** `(B, n_chunks, H)` — one program per `(batch, chunk, head)` triple — with constexpr block sizes `BLOCK_C=C`, `BLOCK_P=P`, `BLOCK_N=N`, then recombines the outputs with `torch.complex`. The env-knob plumbing (§7) lives here.

## 4. Algorithm and tensor-shape contract

The kernel computes, per program `(b, c, h)`, with `C` = chunk size, `P` = head dim, `N` = state dim:

1. Load `Bc, Cc` (each `(C, N)` complex), `Xc` (`(C, P)` complex), and `A_log, decay_states` (each `(C,)` complex) from HBM.
2. Compute `A_cumsum = cumsum(A_log)` in registers (length `C`).
3. Build the causal segment matrix `L[l, s] = exp(A_cs[l] - A_cs[s])` for `s ≤ l`, else 0 — on the fly in registers.
4. `Cb = Cc @ Bc^T` — one `tl.dot`, `(C, N) × (N, C) = (C, C)`.
5. `Y_diag = (L * Cb) @ Xc` — one `tl.dot`, `(C, C) × (C, P) = (C, P)`.
6. `state = Xc^T @ (decay_states[:, None] * Bc)` — one `tl.dot`, `(P, C) × (C, N) = (P, N)`.
7. Write `Y_diag` and `state` to HBM.

Total: **3 `tl.dot` calls per program**, all operands in registers, no HBM intermediate. The two output writes are the only HBM traffic from the kernel.

**Complex arithmetic as real/imag pairs.** All `tl.dot`s operate on float32 halves. For complex `a = a_re + i·a_im`, `b = b_re + i·b_im`, the products are the standard bilinear forms:

$$\begin{aligned}
Cb_\text{re} &= Cc_\text{re}\,Bc_\text{re}^\top - Cc_\text{im}\,Bc_\text{im}^\top, & Cb_\text{im} &= Cc_\text{re}\,Bc_\text{im}^\top + Cc_\text{im}\,Bc_\text{re}^\top, \\
Y_\text{re} &= M_\text{re}\,Xc_\text{re} - M_\text{im}\,Xc_\text{im}, & Y_\text{im} &= M_\text{re}\,Xc_\text{im} + M_\text{im}\,Xc_\text{re},
\end{aligned}$$

where `M = L * Cb` is the elementwise complex product of the segment matrix and the projection matrix, and `state` uses `w = decay_states * Bc` the same way. The segment matrix itself is built from the cumsum of the complex `A_log`:

$$L[l,s] = e^{\Re(\Sigma_l - \Sigma_s)}\bigl(\cos(\Im(\Sigma_l - \Sigma_s)) + i\sin(\Im(\Sigma_l - \Sigma_s))\bigr)\cdot \mathbb{1}[l \ge s], \qquad \Sigma_t = \sum_{u\le t} A\_log[u].$$

This is exactly the chunkwise linear-projection formula of `ssd_complex_chunkwise` (see `docs/concepts/ssd-theory.md` and `docs/concepts/ssd-theory.md` for the derivation); the kernel is a faithful re-implementation, not an approximation.

**Indexing contract.** All tensors are indexed with flat linear bases in the kernel: `Bc/Cc` at `(b·n_chunks·C·H·N + c·C·H·N + l·H·N + h·N + n)`, `Xc` the same with `P` in place of `N`, `A_log/decay_states` at `(b·T_padded·H + t·H + h)` with `t = c·C + l`, and `state` written at `(b·n_chunks·H·P·N + c·H·P·N + h·P·N + p·N + n)` — i.e. `(P, N)` per `(b, c, h)`, matching the einsum output `bchpn`. These layouts are the contract between `_per_chunk_ssd_triton_forward`'s buffer allocations and the kernel's pointer arithmetic; the parity tests (§9) pin them against the reference.

**Block sizes (constexpr, hard-coded):** `BLOCK_C = chunk_size`, `BLOCK_P = D`, `BLOCK_N = state_dim` — all 64 at the 434M config, all within the 256-cap enforced by `_check_block_dims`. The grid is `(B, n_chunks, H)` programs — `8 × 32 × 16 = 4096` at the 434M config — each doing three small `tl.dot`s entirely in registers.

## 5. Env knobs

| Env var | Default | Effect |
|---|---|---|
| `TRITON_PER_CHUNK_NUM_STAGES` | `"1"` | `num_stages` passed to the kernel launch (software pipelining depth). |
| `TRITON_PER_CHUNK_NUM_WARPS` | `"4"` | `num_warps` passed to the kernel launch. |

Both are read in `_per_chunk_ssd_triton_forward` with `os.environ.get(..., default)` and passed verbatim to the kernel. They are tuning knobs only — they do not change the math, shapes, or numerical contract — so a CUDA box can sweep occupancy/scheduling without a code edit (§11).

A third, unrelated env var participates in *dispatch* rather than the kernel: `ENABLE_TRITON_KERNELS=1`, enforced by `training/pretrain.py:_enforce_triton_env_var` (§8).

## 6. The autograd backward contract

```python
class _PerChunkSSDTriton(torch.autograd.Function):
    """Fused per-chunk forward; backward recomputes the same math in PyTorch
    and seeds it with the true downstream gradients (grad_outputs)."""

    @staticmethod
    def forward(ctx, Bc, Cc, Xc, A_log, decay_states):
        Y_diag, state = _per_chunk_ssd_triton_forward(Bc, Cc, Xc, A_log, decay_states)
        ctx.save_for_backward(Bc, Cc, Xc, A_log, decay_states)
        return Y_diag, state

    @staticmethod
    def backward(ctx, grad_y_diag, grad_state):
        Bc, Cc, Xc, A_log, decay_states = ctx.saved_tensors
        with torch.enable_grad():
            b = Bc.detach().requires_grad_(True)
            c = Cc.detach().requires_grad_(True)
            x = Xc.detach().requires_grad_(True)
            a = A_log.detach().requires_grad_(True)
            d = decay_states.detach().requires_grad_(True)
            Y_diag_ref, state_ref = per_chunk_ssd_pytorch(b, c, x, a, d)
        g_y = grad_y_diag if grad_y_diag is not None else torch.zeros_like(Y_diag_ref)
        g_s = grad_state if grad_state is not None else torch.zeros_like(state_ref)
        grads = torch.autograd.grad(
            (Y_diag_ref, state_ref), (b, c, x, a, d),
            grad_outputs=(g_y, g_s), allow_unused=True,
        )
        return grads
```

**Contract.** The backward MUST produce correct gradients for all five inputs — `Bc, Cc, Xc, A_log, decay_states` — and it does, by construction:

1. **Recompute.** The per-chunk math the kernel fused is re-run in PyTorch (`per_chunk_ssd_pytorch`) on detached copies of the saved inputs, inside `torch.enable_grad()`. The detach+`requires_grad_` dance re-roots the graph at the five inputs so `torch.autograd.grad` can differentiate the reference math exactly.
2. **True seed.** The engine's downstream gradients — `grad_y_diag` for `Y_diag` and `grad_state` for `state` — are passed as `grad_outputs` to `torch.autograd.grad`. A `None` incoming grad (a branch of the graph that did not consume the output) is replaced with `zeros_like`, so the contract holds even when `Y_diag` or `state` is partially unused.
3. **Exactness.** Because the recompute is the *same* math as the kernel implements (not a truncated or approximation scheme), the backward is exact w.r.t. the reference — verified by `torch.autograd.gradcheck` on CPU with the kernel forward substituted by the reference (§9).

**Why this shape.** The gradient of the *kernel* is never differentiated through — the forward is a black box. The two outputs are consumed by the outer PyTorch graph: `Y_diag` is added to `Y_off` to form `Y`, and `state` feeds the inter-chunk `decay_chunk` propagation einsum before `Y_off`. The outer graph's autograd computes `grad_y_diag`/`grad_state` from those consumers, and this recompute back-propagates them through the per-chunk math. This is the pattern that fixed the historical v1 stub bug: a stub that recomputed `y.sum()` (implicit all-ones downstream grad) and returned `None` for the token-content path silently zeroed the embedding/input-projection gradient. The current contract — true `grad_outputs` injection plus a non-zero-`Xc`-grad assertion (§9) — closes both holes.

**Cost.** Backward is exact but **slow**: it re-runs the per-chunk einsums in PyTorch, so the backward of the SSD call is dominated by the recompute rather than the kernel. A fused recompute backward (re-derive `L` in backward, compute `dY_diag`/`dstate` with `tl.dot`) is a known v2 plan with an estimated ~5–10× SSD-backward speedup; it is explicitly out of scope here.

## 7. Numerical contract

- **Accumulators:** all `tl.dot` accumulators are fp32 (Triton's default); inputs and outputs are complex64 in PyTorch and float32 halves inside the kernel. There is no reduced-precision accumulation anywhere in the kernel.
- **Agreement with the reference:** forward parity on CUDA holds at `atol=1e-3` for fp32 inputs (`test_forward_matches_pytorch_tiny`) and `atol=1e-2` for bf16 inputs (`test_forward_matches_pytorch_bf16`). The production-shaped call (`test_forward_production_shape`) is checked at `atol=1e-2`. The bf16 slack exists because *inputs* may be bf16-cast downstream; the kernel itself never computes in bf16.
- **Backward agreement:** end-to-end gradients through `ssd_complex_chunkwise` match the pytorch dispatch grad-for-grad on `x/A/B_t/C_t/dt` within `atol=1e-2, rtol=1e-2`.
- **Autograd plumbing:** `torch.autograd.gradcheck` passes on CPU (complex128 inputs) with the kernel forward monkeypatched to the reference — this verifies the Function plumbing, not the kernel arithmetic, which the CUDA forward-parity tests cover.
- **NaN safety:** decays stay ≤ 1 by construction (`softplus(dt) * A` has non-positive real part for negative-real `A`); the GPU tests assert finiteness of every output and every gradient.

## 8. Integration — how the kernel plugs into production

**Dispatch.** `models/ssd_complex.py:ssd_complex_chunkwise` takes `ssd_dispatch: str = "pytorch"`:

```python
    if ssd_dispatch == "triton":
        from .ssd_triton import per_chunk_ssd_triton
        Y_diag, states = per_chunk_ssd_triton(Bc, Cc, Xc, Ac, decay_states)
    else:
        Y_diag, states = per_chunk_ssd_pytorch(Bc, Cc, Xc, Ac, decay_states)
```

Note the input convention: both dispatch branches pass `Ac` (the *unpermuted* `(B, n_chunks, C, H)` log-decay) directly to `per_chunk_ssd_pytorch`, which permutes internally; the kernel's `tl.cumsum` runs over the `C` axis in the natural layout.

**What stays in PyTorch.** The kernel replaces only the two per-chunk einsums (`Y_diag` and `state`). Everything else in `ssd_complex_chunkwise` is untouched and runs on the same tensors regardless of dispatch:

- the `A_cumsum = torch.cumsum(Ac, dim=2)` and `decay_states = exp(A_cumsum[:, :, -1:, :] - A_cumsum)` prefix (inputs to the kernel),
- the inter-chunk propagation: `chunk_decay` cumsum, `cd_shift` (`CD[z-1]` with `CD[-1] := 0`), the causal `decay_chunk[z, c] = exp(CD[z-1] - CD[c]) · 1[z > c]` (strict tril, diagonal −1), the `states = einsum("bhzc,bchpn->bzhpn", decay_chunk, states)` propagation einsum, and the `init_decay * initial_states` term,
- the `Y_off = einsum("bclhn,bchpn,bclh->bclhp", Cc, states, exp(A_cumsum))` application and the final `Y = Y_diag + Y_off; Y.real` readout.

So the kernel's output contract is precisely `Y_diag` and per-chunk `state` in the exact shapes and semantics the outer einsums consume; the port is additive, and the inter-chunk `(c, c)` causal matrix (~1 MB at the 434M config) is not load-bearing to fuse.

**Block-level fallback.** `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` routes the SSD call:

```python
        if self.ssd_dispatch != "triton":
            return ssd_complex_chunkwise(
                x_ssm, self.A, B_t, C_t, dt, chunk_size=self.chunk_size,
            )
        try:
            return ssd_complex_chunkwise(
                x_ssm, self.A, B_t, C_t, dt,
                chunk_size=self.chunk_size, ssd_dispatch="triton",
            )
        except Exception as exc:
            if not self._triton_fallback_warned:
                print(
                    f"[Mamba3Block {self.layer_idx}] ssd_dispatch='triton' "
                    f"unavailable ({type(exc).__name__}: {exc}); "
                    f"falling back to 'pytorch' for this block."
                )
                self._triton_fallback_warned = True
            return ssd_complex_chunkwise(
                x_ssm, self.A, B_t, C_t, dt, chunk_size=self.chunk_size,
            )
```

Any `ImportError` (no triton) or `ValueError` (256-cap) is caught per block, converted into a one-shot warning (guarded by the `Mamba3Block._triton_fallback_warned` instance flag), and the block silently runs the production path. The run stays safe; the warning is the signal.

**Master env-var guard.** The config key alone is not enough. `training/pretrain.py:_enforce_triton_env_var` force-backs `ssd_dispatch` to `'pytorch'` at startup unless `ENABLE_TRITON_KERNELS=1`:

```python
    if (
        os.environ.get("ENABLE_TRITON_KERNELS", "0") != "1"
        and model_config.get("ssd_dispatch") == "triton"
    ):
        log(
            "[warn] ssd_dispatch='triton' requires ENABLE_TRITON_KERNELS=1; "
            "forcing ssd_dispatch='pytorch' for this run."
        )
        model_config["ssd_dispatch"] = "pytorch"
```

This is the two-layered opt-in contract: `ssd_dispatch='triton'` without the env var force-backs with a one-line startup warning; the env var set but a failing kernel prints the per-block one-shot warning above.

## 9. Verification surface — `tests/test_ssd_triton.py`

The test file is the executable form of every contract in this doc. CPU tests (run everywhere, including Mac):

- **`TestPerChunkSsdPytorchReference`** — `test_reference_shape_and_finite` pins the output shapes `Y_diag (B, n_chunks, C, H, P)` / `state (B, n_chunks, H, P, N)` and finiteness; `test_reference_matches_ssd_complex_chunkwise` builds real production tensors (`B_t`, `C_t`, `x`, `dt`, `A`) and checks the reference reproduces the chunkwise formula's per-chunk quantities.
- **`TestPerChunkSsdImportSurface`** — `test_module_imports_without_triton` asserts `HAS_TRITON` is a bool and both host entry points exist; `test_kernel_call_raises_clean_import_error_when_no_triton` asserts the `ImportError` contract; `test_check_block_dims_raises_value_error_on_too_large_dim` exercises all three cap axes (P, N, chunk_size); `test_check_block_dims_accepts_production_404m_shape` pins `P=N=C=64` passes.
- **`TestPerChunkSsdAutogradPlumbing`** — the CPU gradcheck: `_per_chunk_ssd_triton_forward` is monkeypatched to `per_chunk_ssd_pytorch`, then `torch.autograd.gradcheck` runs on complex128 inputs (verifying the Function plumbing, i.e. the `grad_outputs` injection contract, without CUDA); `test_backward_propagates_to_content_path_cpu` asserts all five inputs get finite grads and, critically, that `Xc.grad.abs().sum() > 0` — the token-content path is connected.
- **`TestPerChunkSsdDispatchWiring`** — `test_default_dispatch_is_pytorch`; the triton config runs end-to-end on CPU and falls back with exactly the `"ssd_dispatch='triton' unavailable"` / `"falling back to 'pytorch'"` messages; `test_triton_fallback_warning_is_one_shot_per_instance` asserts the warning appears exactly once across three forwards; `test_triton_path_output_matches_pytorch_path` asserts identical outputs (allclose `atol=1e-5`) between a pytorch-dispatch and a triton-dispatch model loaded with the same weights.
- **`TestEnableTritonKernelsForceBack`** — the `_enforce_triton_env_var` guard: missing env var forces `'pytorch'` with the warning, `ENABLE_TRITON_KERNELS=1` passes through, the pytorch dispatch is untouched by any env value.

GPU tests (`TestPerChunkSsdKernelGPU`, skipped unless `HAS_TRITON and torch.cuda.is_available()`):

- `test_forward_matches_pytorch_tiny` — fp32 parity at `atol=1e-3` for both outputs.
- `test_forward_matches_pytorch_bf16` — bf16 parity at `atol=1e-2`.
- `test_forward_production_shape` — `B=2, n_chunks=4, C=16, H=4, P=16, N=16` at `atol=1e-2`, plus the `Y_diag` shape check.
- `test_autograd_backward_runs` — real-kernel backward: all five grads finite and non-None, `Xc` grad non-zero.
- `test_backward_matches_pytorch_dispatch` — end-to-end `ssd_complex_chunkwise` backward on CUDA, triton vs pytorch dispatch, grad-for-grad on `x/A/B_t/C_t/dt` within `atol=1e-2, rtol=1e-2`, including the token-content path the v1 stub dropped.

## 10. Pitfalls

- **Do not cite the JIT kernel.** `_ssd_per_chunk_fwd_kernel` is defined under `if HAS_TRITON:` and does not exist on triton-less CI; docs and the link checker must cite only the always-defined host wrappers (`per_chunk_ssd_triton`, `_check_block_dims`, `_view_real_imag`, `_per_chunk_ssd_triton_forward`, `_PerChunkSSDTriton`); the pure-PyTorch reference lives in `models/ssd_complex.py:per_chunk_ssd_pytorch` (§3.2).
- **The 256-cap is a hard fail, not a clamp.** `_check_block_dims` raises `ValueError` naming the offending dim (`P`, `N`, or `chunk_size`). The kernel cannot be resized past 256; the correct response is the dispatcher's fallback to `ssd_dispatch='pytorch'` (one-shot warning per block instance), not a silent truncation. A config with `state_dim > 256` simply cannot use the triton path.
- **Backward is exact but slow.** `_PerChunkSSDTriton.backward` re-runs the per-chunk einsums in PyTorch on every backward step. It is a correctness contract, not a performance feature; do not measure training throughput assuming a fused backward.
- **complex64 or nothing.** `_view_real_imag` raises `TypeError` for non-complex64 input. The kernel is specialised for the Mamba-3 complex layout; do not feed it real tensors or complex128.
- **Inputs must be pre-chunked and pre-decayed.** The wrapper does *not* compute `A_cumsum`/`decay_states`; the caller (`ssd_complex_chunkwise`) must pass the exact `(B, n_chunks, C, H)` `A_log` and `decay_states = exp(A_cumsum[:, :, -1:, :] - A_cumsum)`. Passing raw `A_log` without that prefix silently changes the math (the `state` einsum multiplies by `decay_states`).
- **`None` grads are a trap.** In `backward`, a `None` `grad_y_diag`/`grad_state` (output not consumed downstream) is replaced with `zeros_like` *before* `torch.autograd.grad`; never "simplify" this to dropping the term.
- **Dispatch ≠ correctness gate.** A run with `ssd_dispatch='triton'` that falls back at startup (no `ENABLE_TRITON_KERNELS=1`) or per block (ImportError/ValueError) is numerically identical to the pytorch path, silently. The tests are the signal that the kernel itself is wrong; the fallback is a safety net, not a validation.

## 11. A100-box verification checklist (migrated from the legacy design doc)

The kernel ships and is tested on the Mac/CPU box. Before `ssd_dispatch='triton'` is enabled in production on a CUDA box, verify end-to-end:

1. **Triton install.** `pip install triton` (A100 supports Triton 3.0+); `python3 -c "import triton; print(triton.__version__)"` ≥ 3.0.
2. **CPU surface still green.** The full suite should show 37 tests collected (32 passed / 5 skipped) on the A100 box; the 5 GPU tests in `TestPerChunkSsdKernelGPU` now run.
3. **Forward parity.** `test_forward_matches_pytorch_tiny` (`atol=1e-3`) and `test_forward_matches_pytorch_bf16` (`atol=1e-2`).
4. **Production shape.** `test_forward_production_shape` (`atol=1e-2`).
5. **Autograd.** `test_autograd_backward_runs` (all five grads finite, non-zero `Xc` grad) and `test_backward_matches_pytorch_dispatch` (grad-for-grad vs pytorch, `atol=1e-2, rtol=1e-2`).
6. **Wall-time speedup.** Bench the 434M config (B=8, T=2048, BF16) on triton vs pytorch dispatch; the SSD-call speedup must clear the 1.5× bar before the kernel is enabled by default. Record the number once known.
7. **NaN guard.** A 50-step warmup sweep on the triton path must keep losses finite.

If any check fails, the kernel is not production-ready; the auto-fallback keeps training safe, and the failed test is the investigation signal.

## 12. What this kernel does not do

- **No inter-chunk or apply-pass port.** Inter-chunk state propagation and the `Y_off` application stay in PyTorch (§8); the inter-chunk `(c, c)` causal matrix is ~1 MB at the 434M config, so fusing it is not load-bearing.
- **No FFN or MIMO port.** Both are dominated by cuBLAS GEMMs at `d_model=1024, ffn_dim=2048`; a custom kernel cannot beat cuBLAS on those shapes, and `torch.compile` already fuses the silu.
- **No attention, MoE, or MTP additions.** This is a pure-SSM repo; the kernel exists only for the sanctioned SSD path.
- **No fallback beyond the contract.** The two-layered opt-in (`ssd_dispatch='triton'` + `ENABLE_TRITON_KERNELS=1`) plus the one-shot warnings are the contract: missing env var → startup force-back warning; kernel raises at runtime → per-block one-shot fallback warning. Nothing else degrades silently.

---

## References

- [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md) — the full derivation of the math both functions implement.
- [Mamba-3-Lite — SSD Foundations](../concepts/state-space-foundations.md) — the recurrence and discretization theory.
- [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) — where the scan sits in the block; why its output must be real.
- [Mamba-3-Lite — Model Reference](model-reference.md) — `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`, the per-layer caller with fallback semantics.
