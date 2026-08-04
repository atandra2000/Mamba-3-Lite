# R3 — The Per-Chunk SSD Triton Kernel API

This reference doc is the migration and expansion of the retired `documentation/`-tree kernel design doc: it documents the sanctioned Triton kernel in `models/ssd_triton.py`, its host-wrapper API contract, the autograd backward contract, the numerical guarantees, and how it plugs into the production chunkwise path.

---

## 1. 60-second summary

After reading this doc you will understand: why Mamba-3-Lite ships one fused Triton kernel (the per-chunk `Y_diag` + `state` pass is the HBM-bandwidth hotspot of the chunkwise SSD), the exact host API — `models/ssd_triton.py:per_chunk_ssd_triton(Bc, Cc, Xc, A_log, decay_states) -> (Y_diag, state)` — plus its pure-PyTorch reference `models/ssd_triton.py:per_chunk_ssd_pytorch`, the tensor-shape and block-dim contracts (one program per `(B, n_chunks, H)`, `BLOCK_C/P/N = C/D/N`), the 256-cap hard-fail `models/ssd_triton.py:_check_block_dims`, the complex64→real/imag split `models/ssd_triton.py:_view_real_imag`, the recompute-based autograd backward of `models/ssd_triton.py:_PerChunkSSDTriton` (all five inputs get correct gradients, seeded with the true downstream `grad_outputs`), the env knobs (`TRITON_PER_CHUNK_NUM_STAGES=1`, `TRITON_PER_CHUNK_NUM_WARPS=4`), the numerical contract (fp32 accumulators, `atol=1e-3` fp32 / `1e-2` bf16 vs the reference), and the `tests/test_ssd_triton.py` verification surface.

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

### 3.2 `per_chunk_ssd_pytorch(...)` — the pure-PyTorch reference

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

**Roles.** (1) Correctness oracle: the GPU parity tests compare `per_chunk_ssd_triton` against it at `atol=1e-3` (fp32) / `1e-2` (bf16). (2) Recompute body of the backward pass (§6). (3) The exact math `ssd_complex_chunkwise`'s non-triton branch computes inline (identical einsums modulo the `Ac.permute` layout), which is why the dispatcher can swap implementations without changing outer numerics. Its shapes mirror the kernel contract exactly — `Y_diag (B, n_chunks, C, H, P)`, `state (B, n_chunks, H, P, N)` — verified by `test_reference_shape_and_finite` and `test_reference_matches_ssd_complex_chunkwise`.

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

This is exactly the chunkwise linear-projection formula of `ssd_complex_chunkwise` (see `docs/theory/03-complex-ssd.md` and `docs/theory/04-chunkwise-algorithm.md` for the derivation); the kernel is a faithful re-implementation, not an approximation.

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
        Ac_perm = Ac.permute(0, 1, 3, 2).contiguous()
        ...
        states = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)
```

Note the input convention: the triton branch passes `Ac` (the *unpermuted* `(B, n_chunks, C, H)` log-decay) directly, because `per_chunk_ssd_pytorch` permutes internally; the pytorch branch permutes `Ac` to `(B, n_chunks, H, C)` first. Both produce the same `L`; the kernel's `tl.cumsum` runs over the `C` axis in the natural layout.

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

- **Do not cite the JIT kernel.** `_ssd_per_chunk_fwd_kernel` is defined under `if HAS_TRITON:` and does not exist on triton-less CI; docs and the link checker must cite only the always-defined host wrappers (`per_chunk_ssd_triton`, `per_chunk_ssd_pytorch`, `_check_block_dims`, `_view_real_imag`, `_per_chunk_ssd_triton_forward`, `_PerChunkSSDTriton`).
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

## Anchors cited in this doc

- `models/ssd_triton.py:per_chunk_ssd_triton`
- `models/ssd_triton.py:per_chunk_ssd_pytorch`
- `models/ssd_triton.py:_check_block_dims`
- `models/ssd_triton.py:_view_real_imag`
- `models/ssd_triton.py:_per_chunk_ssd_triton_forward`
- `models/ssd_triton.py:_PerChunkSSDTriton`
- `models/ssd_triton.py:HAS_TRITON`
- `models/ssd_complex.py:ssd_complex_chunkwise`
- `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`
- `models/mamba_block.py:Mamba3Block._triton_fallback_warned`
- `training/pretrain.py:_enforce_triton_env_var`
