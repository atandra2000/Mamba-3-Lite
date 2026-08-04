# Mamba3Block — the Mamba-3 residual block

Reference API doc for `models/mamba_block.py:Mamba3Block`: construction, instance state, the exact input-projection slice layout, dispatch, and the grad-checkpoint wrapper.

## 60-second summary

`Mamba3Block` is the composable unit of Mamba-3-Lite: one of the 28 residual layers stacked by `models/transformer.py:Mamba3Transformer.layers`. Each block applies **pre-norm → complex-SSD sequence mixing → MIMO head mixing → output projection → residual**, then a second **pre-norm → SwiGLU FFN → residual**. Construction is per-layer (`layer_idx`) and config-driven: every key the block reads is mandatory or has a documented default. The input projection fans `d_model` out to `H·(D+4N+1)` = 5,136 channels and `_forward_impl` splits that tensor into the token-content, complex B/C (real+imaginary pairs), and dt slices that feed `models/ssd_complex.py:ssd_complex_chunkwise`. A single complex parameter `A` per head (init −1.0) carries both decay and rotation.

## Why it exists

The block is the unit of reuse: the transformer's `forward` is just a loop over `self.layers` (`models/transformer.py:Mamba3Transformer.layers`), so everything that differs between "a 1-layer toy" and "the 28-layer 434M model" is expressed in this one class. It also isolates the three dirty details of the architecture — the flat-projection slice arithmetic, the Triton-vs-PyTorch dispatch with one-shot fallback, and the grad-checkpoint wrapper — behind a single `(B, T, d_model) -> (B, T, d_model)` interface.

## Intuition

Think of a block as a **learned stateful filter followed by a learned mixer**:

- `norm1` pre-normalizes (RMSNorm, no centering) so the recurrence sees unit-scale input.
- `in_proj` is a single wide linear map whose output is *interpreted*, not kept whole: the first `H·D` channels are token content, the next four `H·N` slabs are the real/imaginary parts of B and C, and the last `H` channels are per-head step sizes dt.
- The SSD primitive consumes those slices and produces `(B, T, H, D)` scan outputs — each head runs its own complex linear recurrence over the sequence.
- `MIMO` mixes the `H·D` channels *across heads* with an identity-initialized dense map, and `out_proj` maps them back to `d_model`; the residual add completes the sequence-mixing branch.
- `norm2` + the SwiGLU FFN add per-token nonlinear capacity (the "MLP" branch), again with a residual.

The complex `A` is the interesting part: the effective step is `exp(softplus(dt)·A)`, so the real part of `A` sets decay while the imaginary part sets rotation — two behaviors in one scalar per head (see `docs/theory/03-complex-ssd.md`).

## Construction — `Mamba3Block.__init__`

```python
def __init__(self, cfg: dict, layer_idx: int = 0):
    super().__init__()
    self.layer_idx = layer_idx
    self.d_model = cfg["d_model"]
    self.n_heads = cfg["n_heads"]
    self.head_dim = cfg["head_dim"]
    self.state_dim = cfg["state_dim"]
    self.chunk_size = cfg.get("chunk_size", 64)
    self.ssd_dispatch = cfg.get("ssd_dispatch", "pytorch")
    self._triton_fallback_warned = False
    self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
    self.grad_checkpoint = cfg.get("grad_checkpoint", False)

    in_dim = self.n_heads * (self.head_dim + 4 * self.state_dim + 1)
    self.in_proj = nn.Linear(self.d_model, in_dim, bias=False)
    self.mimo = MIMO(self.n_heads, self.head_dim)
    self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
    nn.init.constant_(self.A, -1.0)

    self.norm1 = nn.RMSNorm(self.d_model, eps=self.rms_norm_eps)
    self.norm2 = nn.RMSNorm(self.d_model, eps=self.rms_norm_eps)
    ffn_dim = cfg["ffn_dim"]
    self.ffn_gate_up = nn.Linear(self.d_model, 2 * ffn_dim, bias=False)
    self.ffn_down = nn.Linear(ffn_dim, self.d_model, bias=False)
```

Config keys consumed:

| cfg key | access | default | role |
|---|---|---|---|
| `d_model` | `cfg[...]` | mandatory | token width in/out; norm width |
| `n_heads` | `cfg[...]` | mandatory | `H`: number of parallel SSM heads |
| `head_dim` | `cfg[...]` | mandatory | `D`: per-head token-content width |
| `state_dim` | `cfg[...]` | mandatory | `N`: per-head state width |
| `chunk_size` | `cfg.get` | 64 | SSD chunk length `C` |
| `ssd_dispatch` | `cfg.get` | `"pytorch"` | `"pytorch"` or `"triton"` |
| `rms_norm_eps` | `cfg.get` | 1e-5 | both RMSNorm epsilons |
| `grad_checkpoint` | `cfg.get` | `False` | enable activation checkpointing |
| `ffn_dim` | `cfg[...]` | mandatory | SwiGLU hidden width |

`Mamba3Transformer` passes `cfg.__dict__` with `layer_idx=i` for `i in range(n_layers)` (`models/transformer.py:Mamba3Transformer.layers`), so the block is config-driven end to end; `TrainingConfig` injects `grad_checkpoint` before construction (see `tests/test_grad_checkpoint.py::test_grad_checkpoint_propagates_to_blocks`).

The projection width: `in_dim = H·(D + 4N + 1)` — `16·321 = 5,136` in production (`H=16, D=64, N=64`): the `+1` is the per-head dt slot, the `4N` the two complex pairs (real + imag) for B and C.

### Per-block parameters (derived from the shapes above)

| module | weight shape | params |
|---|---|---|
| `in_proj` | (5,136, 1,024) | 5,259,264 |
| `out_proj` | (1,024, 1,024) | 1,048,576 |
| `mimo.mix` | (1,024, 1,024) | 1,048,576 |
| `A` | (16,) complex64 | 16 |
| `norm1`, `norm2` | (1,024,) × 2 | 2,048 |
| `ffn_gate_up` | (4,096, 1,024) | 4,194,304 |
| `ffn_down` | (1,024, 2,048) | 2,097,152 |
| **total** | | **13,649,936** |

Sum: 5,259,264 + 1,048,576 + 1,048,576 + 16 + 2,048 + 4,194,304 + 2,097,152 = 13,649,936, matching the 28 × 13,649,936 + 51,463,168 (tied embedding) = 433,662,400 total in `docs/theory/08-scaling-efficiency.md`. Note `A` contributes 16 by `numel()` even though a complex64 element occupies 16 bytes in memory — this convention reconciles the count.

`A` is initialized to the constant complex value −1.0 + 0j for every head. All `nn.Linear` weights (but **not** `mimo.mix`) are then re-initialized to `N(0, init_std)` by `models/transformer.py:Mamba3Transformer._init_weights`; the MIMO map is exempted via its `_identity_init` flag (see `models/mimo.py:MIMO.mix` and `docs/theory/05-mimo-mixing.md`). `A` and the RMSNorm weights are untouched because `_init_weights` only visits `nn.Linear` and `nn.Embedding`.

## Instance attributes

All assigned directly in `Mamba3Block.__init__`; production shapes/dtypes shown.

| attribute | type | shape | dtype / value |
|---|---|---|---|
| `layer_idx` | int | scalar | position in the stack (0–27) |
| `d_model`, `n_heads`, `head_dim`, `state_dim`, `chunk_size` | int | scalar | 1024, 16, 64, 64, 64 |
| `ssd_dispatch` | str | — | `"pytorch"` / `"triton"` |
| `rms_norm_eps` | float | scalar | 1e-5 |
| `grad_checkpoint` | bool | — | `False` unless enabled |
| `_triton_fallback_warned` | bool | — | starts `False`; latches `True` after one warning |
| `in_proj` | `nn.Linear` | weight (5,136, 1,024) | float32, `N(0, init_std)` |
| `mimo` | `MIMO` | `mix` weight (1,024, 1,024) | float32, identity (`eye_`) |
| `out_proj` | `nn.Linear` | weight (1,024, 1,024) | float32 |
| `A` | `nn.Parameter` | (16,) | complex64, all −1.0+0j |
| `norm1`, `norm2` | `nn.RMSNorm` | weight (1,024,) | float32, ones |
| `ffn_gate_up` | `nn.Linear` | weight (4,096, 1,024) | float32 |
| `ffn_down` | `nn.Linear` | weight (1,024, 2,048) | float32 |

All `nn.Linear` weights are bias-free, so every block submodule is a pure matmul plus (for the norms) a gain vector.

## Math — the block-level recurrence

The sequence-mixing branch, per head and per token, is the complex diagonal SSM

$$s_t = \bar A_t \odot s_{t-1} + B_t x_t, \qquad y_t = C_t^\top s_t,$$

with effective transition (from `models/ssd_complex.py:_discretise`)

$$\bar A_t = \exp\big(\operatorname{softplus}(\mathrm{dt}_t) \odot A\big),$$

where `dt` is the projected step size and `A` the per-head complex parameter. With `A = a + ib` (initially `a = −1, b = 0`): `softplus(dt) > 0` always, so `softplus(dt)·a < 0` and `|Ā| = exp(softplus(dt)·a) ∈ (0, 1)` — every step is contractive — while the imaginary part contributes `exp(i·softplus(dt)·b)`, a per-step rotation. One complex scalar parameterizes both; `docs/theory/03-complex-ssd.md` derives the complex-state version and `docs/theory/04-chunkwise-algorithm.md` proves the chunkwise (attention-like) reorganization `_forward_impl` calls (`docs/theory/01-ssm-foundations.md` builds the discretization from scratch). The full block output is

$$x' = x + W_{\text{out}}\,\text{MIMO}\big(\text{SSD}(W_{\text{in}}\,h)\big), \qquad h = \text{RMSNorm}(x),$$

$$x'' = x' + W_{\text{down}}\big(\text{SiLU}(h' W_g) \odot (h' W_u)\big), \qquad h' = \text{RMSNorm}(x').$$

## `_forward_impl` — slice layout and data flow

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

The projection output `proj` has trailing width `H·(D+4N+1)`; the exact slice boundaries (production values in parentheses, `H·D = 1024`, `H·N = 1024`):

| slice | start index | width | end index | post-cast shape | dtype |
|---|---|---|---|---|---|
| `x_ssm` | 0 | `H·D` (1,024) | 1,024 | `(B, T, H, D)` | float32 |
| `B_real` | `H·D` (1,024) | `H·N` (1,024) | 2,048 | `(B, T, H·N)` | float32 |
| `B_imag` | `H·D + H·N` (2,048) | `H·N` (1,024) | 3,072 | `(B, T, H·N)` | float32 |
| `C_real` | `H·D + 2H·N` (3,072) | `H·N` (1,024) | 4,096 | `(B, T, H·N)` | float32 |
| `C_imag` | `H·D + 3H·N` (4,096) | `H·N` (1,024) | 5,120 | `(B, T, H·N)` | float32 |
| `dt` | `H·(D+4N)` (5,120) | `H` (16) | 5,136 | `(B, T, H)` | float32 |

Then `B_t = torch.complex(B_real, B_imag).reshape(B, T, H, N)` and likewise `C_t` — complex64 tensors. Note the ordering: B and C each carry real then imag as **two adjacent slabs**, not per-channel alternation, and dt is *not* contiguous with them — it is the final `H` columns, addressed as `proj[..., -H:]`.

Every slice is `.float()`-cast **before** complex assembly. This is load-bearing: `torch.complex` requires float32 real/imag inputs, and under BF16 autocast (the training default on CUDA, see `docs/reference/07-pretrain-cli.md`) the raw `in_proj` output is BF16. The casts also keep the SSD math in float32 — see `docs/theory/07-numerical-stability.md`.

The SSD call returns `Y.real` (the chunkwise scan is computed in complex64; the real part is taken inside `models/ssd_complex.py:ssd_complex_chunkwise`), so the rest of the path is real: `MIMO` mixes across heads (`models/mimo.py:MIMO.forward`, `docs/reference/06-mimo.md`), `out_proj` maps back to `d_model`, and both residuals are plain adds.

## `forward` — the grad-checkpoint wrapper

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """(B, T, d_model) -> (B, T, d_model)."""
    if self.grad_checkpoint and self.training:
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, use_reentrant=False
        )
    return self._forward_impl(x)
```

`forward` is a thin dispatch: when `grad_checkpoint` is enabled **and** the module is in training mode, `_forward_impl` runs under `torch.utils.checkpoint.checkpoint` with `use_reentrant=False` — the non-reentrant implementation, required because the Triton dispatch path contains a custom `torch.autograd.Function`. Checkpointing trades recompute-on-backward for activation memory: only the input `x` is saved; `proj`, `B_t`, `C_t`, `dt`, and the scan outputs are recomputed during backward. In eval mode the flag is ignored and the block runs eagerly (see Pitfalls).

## `_ssd_with_dispatch` — Triton dispatch and one-shot fallback

```python
def _ssd_with_dispatch(self, x_ssm, B_t, C_t, dt):
    """ssd_complex_chunkwise with optional triton dispatch and fallback."""
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

Semantics:

- Default `"pytorch"` calls `ssd_complex_chunkwise` directly (pure-PyTorch einsum path). With `"triton"`, the same function routes the per-chunk matmuls to the host wrapper `models/ssd_triton.py:per_chunk_ssd_triton`; the `try` catches **any** `Exception` — import errors on triton-less machines, kernel launch failures, shape/dtype mismatches — and falls back to the PyTorch path.
- The fallback warning is **one-shot per block instance**: `_triton_fallback_warned` latches after the first failure (which prints the block index + exception type); later failures are silent, so a broken Triton setup prints at most 28 lines total. Each block still retries Triton on every call and re-falls back silently.

At the training level, `ssd_dispatch="triton"` is additionally gated by the `ENABLE_TRITON_KERNELS=1` environment variable (`training/pretrain.py:_enforce_triton_env_var` force-backs to `"pytorch"` when unset); the block-level fallback above is the second line of defense, and the kernel itself is documented in `docs/reference/03-ssd-triton.md`.

## `_ffn` — SwiGLU

```python
def _ffn(self, x: torch.Tensor) -> torch.Tensor:
    gate, up = self.ffn_gate_up(x).chunk(2, dim=-1)
    return self.ffn_down(F.silu(gate) * up)
```

`ffn_gate_up` produces `2·ffn_dim` channels split evenly down the last axis: the first half is the gate (SiLU-activated, i.e. Swish), the second half is the linear "up" branch, and `ffn_down` projects the elementwise product back to `d_model`. With `ffn_dim = 2048` this is a 4,096→2,048→1,024 sandwich — standard SwiGLU:

$$\text{FFN}(x) = W_{\text{down}}\big(\text{SiLU}(xW_g) \odot (xW_u)\big).$$

## Pitfalls

1. **Grad checkpointing silently deactivates in eval mode.** The wrapper condition is `self.grad_checkpoint and self.training`. Inference always runs eager — intended, but "checkpointing" is a training-only property, and a model built with `grad_checkpoint=True` accidentally left in `eval()` will neither checkpoint nor warn.
2. **Slice-boundary off-by-ones shift heads silently.** All five quadrants are contiguous slabs of the same projection; a boundary error of `H·N` (1,024 columns) shifts B's real part into x's content or swaps B/C, and since every slice is a valid-shaped tensor, nothing raises — the model just trains garbage. Safe invariants: content width `H·D`, each of the four B/C slabs exactly `H·N`, dt is the **last** `H` columns (`proj[..., -H:]`), total width `H·D + 4·H·N + H`. `docs/theory/06-block-anatomy.md` derives this layout from scratch.
3. **Complex assembly requires float32.** `torch.complex(real, imag)` fails unless both inputs are float32 (float64 also works). Under BF16 autocast the projection output is BF16, so the `.float()` casts before `torch.complex` are not cosmetic — removing them breaks both paths on CUDA, and the casts also pin the recurrence math to FP32 regardless of autocast.
4. **`A` is complex64 and per-head, not per-head-per-dim.** It broadcasts against `(B, T, H)` dt as one complex scalar per head — no per-state-dimension decay. `D` and `N` are independently configurable, but the slice math and `MIMO` shapes assume the code's `H·D`/`H·N` widths — changing one without the others breaks the layout.
5. **Fallback failures are broad by design.** `_ssd_with_dispatch` catches `Exception` wholesale; a *logic* bug in the Triton path (wrong but non-raising results) will not be caught — `tests/test_ssd_triton.py::TestPerChunkSsdDispatchWiring::test_triton_path_output_matches_pytorch_path` exists precisely because silent divergence is the failure mode to watch for. And the first failure is the only diagnostic per block.
6. **The residual is taken before normalization.** Both branches are pre-norm, so residuals carry un-normalized activations; `x` entering the block may be BF16 while `y` is FP32 (cast by the SSD path), so the first residual add promotes to FP32. Expected, but it is where mixed-precision expectations break.

## Tests

- `tests/test_grad_checkpoint.py` — all three tests: `test_grad_checkpoint_propagates_to_blocks` (TrainingConfig → every block), `test_grad_checkpoint_explicit_false_disables`, and `test_grad_checkpoint_actually_triggers_training_mode` (a real backward through the checkpointed path yields finite grads).
- `tests/test_ssd_triton.py::TestPerChunkSsdDispatchWiring` — the dispatch contract: default `"pytorch"`, explicit `"pytorch"` runs the production path, `"triton"` on a triton-less box falls back cleanly with the warning text `falling back to 'pytorch'`, the warning is one-shot (`_triton_fallback_warned` latches `True`), and Triton output matches PyTorch output.
- `tests/test_mimo.py` — identity init (`test_mimo_identity_init`), that the identity survives `Mamba3Transformer._init_weights` via the `_identity_init` guard (`test_mimo_identity_survives_transformer_init`), and shape/finiteness (`test_mimo_shape_and_finite`).

See also: `docs/reference/01-model-config.md` (cfg key provenance), `docs/reference/02-ssd-complex.md` (the chunkwise primitive), `docs/reference/04-transformer.md` (stacking, weight tying), `docs/reference/06-mimo.md` (the mixer).
