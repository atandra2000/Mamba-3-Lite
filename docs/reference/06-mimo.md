# MIMO — The Inter-Head Mixer

Reference doc for `models/mimo.py`: the per-block linear layer that mixes information across the 16 SSD heads after the chunkwise scan.

## 60-second summary

After the chunkwise SSD scan, each token carries `H` independent head states stacked as a `(B, T, H, D)` tensor. `models/mimo.py:MIMO` is a single bias-free `nn.Linear(H*D, H*D)` that treats the flattened `H*D` vector as one mixing space and lets the network redistribute content between heads. At construction its weight is initialized to the identity matrix (`nn.init.eye_`), so the layer starts as a no-op; training moves the weight off identity, turning it into a learned inter-head attention surrogate. Because the identity init would otherwise be clobbered by the model-wide weight initialization, the `mix` submodule carries a plain `_identity_init = True` attribute that `models/transformer.py:Mamba3Transformer._init_weights` checks *before* its `isinstance(nn.Linear)` branch. The mixer adds 1,048,576 parameters per layer (~29.4M of the ~434M total, ≈ 6.8%).

## Why it exists

Linear-attention-style SSMs (this one included) keep heads *independent*: the state recurrence in `models/ssd_complex.py:ssd_complex_chunkwise` is per-head, and the output projection `out_proj` is the only place head outputs are combined — after which they are summed into a single `d_model` vector. A single linear mixing step between the scan and `out_proj` gives the network a cheap, learned way to exchange information across heads *before* projection, at quadratic cost in `H*D` per token rather than the quadratic-in-sequence cost of true attention. The identity initialization is what makes this addition safe: at initialization the model is *exactly* the no-mixer architecture, so the mixer's gradients start from a "no-op" operating point and the layer can decide how much mixing to learn.

## Intuition

Think of the mixer as a full square matrix over the flattened `H*D` coordinate space. A weight entry `W_{(h,d),(h',d')}` says "how much of head `h'`, dimension `d'` leaks into head `h`, dimension `d`". At init `W = I`, so each coordinate feeds only itself — each head stays sealed inside its own `D`-dimensional subspace. Training sculpts off-diagonal blocks: the `H×H` block structure of `W` (each block `D×D`) determines the inter-head mixing pattern, and non-identity diagonal blocks act as per-head linear transforms. Because there is no nonlinearity in the mixer, it is a pure linear map — its expressive role is to *permute/recombine* coordinates, never to gate them. The identity start is a *starting point*, not a constraint: nothing in the forward pass enforces `W = I` after construction.

## Math

Let `H` be the number of heads and `D` the head dimension. Flatten the per-token head stack `x ∈ R^{B×T×H×D}` into `x_flat ∈ R^{B×T×HD}` (row-major over `(h, d)`). The mixer is the bias-free linear map

$$ y = x_{\text{flat}} \, W^{\mathsf T}, \qquad W \in \mathbb{R}^{HD \times HD}, \quad b = 0, $$

which in coordinates is

$$ y_{(h,d)} = \sum_{h'=0}^{H-1} \sum_{d'=0}^{D-1} W_{(h,d),(h',d')} \, x_{(h',d')}. $$

At construction `nn.init.eye_(W)` sets `W = I_{HD}`, hence `y_{(h,d)} = x_{(h,d)}` — the layer is the identity map and the block output is bitwise (up to floating-point reshape order) the SSD output.

Parameter count: `W` has `H·D × H·D = 1,048,576` entries per layer with `d_model=1024` (`n_heads=16`, `head_dim=64`). Across 28 layers that is `28 × 1,048,576 = 29,360,128` parameters — about 6.8% of the 433,662,400 total, and ~7.7% of a single layer's 13,649,936. [DERIVED from `models/transformer.py:ModelConfig` block dims.]

## Code walkthrough

`models/mimo.py:MIMO` is a 25-line module with two members.

### Constructor — `models/mimo.py:MIMO.__init__`

```python
def __init__(self, n_heads: int, head_dim: int):
    super().__init__()
    self.n_heads = n_heads
    self.head_dim = head_dim
    self.mix = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
    nn.init.eye_(self.mix.weight)
    # Mamba3Transformer._init_weights skips Linears flagged this way.
    self.mix._identity_init = True
```

Three things happen:

1. `self.mix` is a plain `nn.Linear(H*D, H*D, bias=False)` — note the default Linear initialization is *not* left in place; `nn.init.eye_` overwrites it synchronously in the constructor, so an `MIMO` instance is born with `mix.weight == I`.
2. The flag `_identity_init = True` is set as a **plain Python attribute on the `mix` module** (not on `MIMO`, and not as a buffer or parameter — see Invariants).
3. `bias=False` keeps the mixer a pure homogeneous linear map; there is no per-coordinate offset to learn.

### Forward — `models/mimo.py:MIMO.forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """(B, T, H, D) -> (B, T, H, D)."""
    B, T, H, D = x.shape
    x_flat = x.reshape(B, T, H * D)
    out = self.mix(x_flat)
    return out.reshape(B, T, H, D)
```

Semantics: unpack the batch/seq dims, flatten the last two into the `H*D` mixing space, apply `x_flat @ Wᵀ` (the `nn.Linear` matmul, `bias=False`), then reshape back to `(B, T, H, D)` so the downstream `out_proj` in `models/mamba_block.py:Mamba3Block._forward_impl` can flatten again and project. Both reshapes are free *views* when the input is contiguous; the first one is a **copy** in this code path because the SSD output is not contiguous (see Pitfalls).

### The call site — `models/mamba_block.py:Mamba3Block._forward_impl`

```python
y = self._ssd_with_dispatch(x_ssm, B_t, C_t, dt)

y = self.mimo(y)
y = y.reshape(B, T, H * D)
y = self.out_proj(y)
x = residual + y
```

The mixer consumes the raw output of `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` — the real chunkwise scan result (PyTorch or Triton path, same `(B, T, H, D)` shape) — with no detach, no stop-gradient, and no intervening transformation. Gradients flow from `out_proj` through the mixer into the SSD's per-head outputs. This is the invariant "the mixer sees the real SSD output": the block does not feed the mixer a stale or copied input.

### Why the eye survives — `models/transformer.py:Mamba3Transformer._init_weights`

`Mamba3Transformer.__init__` ends with `self.apply(self._init_weights)`, which visits every submodule recursively — including each `MIMO.mix`. The guard is checked **before** the type dispatch:

```python
def _init_weights(self, module):
    if getattr(module, "_identity_init", False):
        return
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
```

The ordering matters: had the `isinstance(module, nn.Linear)` branch come first, every `mix` (a perfectly ordinary `nn.Linear`) would be re-initialized to `N(0, init_std)` and the eye would be destroyed. Because `getattr` is consulted first and the attribute is set on `mix` itself, `apply` hits the flag when it reaches the `mix` submodule and returns early. `MIMO` itself carries no flag, but it is neither `nn.Linear` nor `nn.Embedding`, so it is a no-op either way. This same pattern is what `models/mimo.py` relies on to keep its initialization; any *other* re-init pass that does not check the flag would still clobber it (see Pitfalls).

## Invariants

- **`mix.weight` is exactly `I_{HD}` at construction.** `nn.init.eye_` runs synchronously in `MIMO.__init__`; an instance created and never trained is an exact identity map. Verified: `torch.equal(m.mix.weight, torch.eye(128))` is `True` for `MIMO(4, 32)`.
- **The flag is not serialized.** `_identity_init` is a plain attribute, not a registered buffer or parameter, so `state_dict` contains exactly one key — `mix.weight` (verified: `list(m.state_dict().keys()) == ['mix.weight']`). Saving/loading a checkpoint therefore transports the trained `W` and *drops* the flag — which is the desired behavior: a loaded checkpoint's mixer is a trained matrix that must not be re-identity-initialized.
- **The mixer sees the real SSD output.** `_forward_impl` pipes `_ssd_with_dispatch`'s return directly into `self.mimo(y)`; gradients reach the scan's per-head outputs through the full `W` matrix.
- **The identity is a starting point, not a constraint.** Nothing in `forward` pins `W`; the moment training steps, `W` drifts off `I` and the mixer begins transferring content between heads.

## Pitfalls

- **Non-contiguous input makes the first `reshape` a copy.** `ssd_complex_chunkwise` returns `Y.real` sliced as `[..., :T]` (`models/ssd_complex.py:ssd_complex_chunkwise`), which is a non-contiguous view; the Triton path returns a fresh allocation. `x.reshape(B, T, H*D)` on a non-contiguous tensor cannot be a view, so MIMO's forward materializes a `(B*T, H*D)` copy per token block. This is a silent cost (one `H*D`-wide copy per layer per step — small next to the SSD), not a bug; a fused kernel would avoid the round trip. The *second* reshape (output → `(B, T, H, D)`) is always a free view because the Linear output is contiguous.
- **The flag protects only against `Mamba3Transformer._init_weights`.** It is a private convention, not a PyTorch mechanism. Any other code path that re-initializes linears — a manual `nn.init.normal_` on `mix.weight`, a future refactor that reorders the `getattr` check after the `isinstance` branch, or re-running a flag-less init pass after checkpoint load — silently destroys the eye (or, after load, the trained matrix). The eye guarantee holds only for a freshly constructed, never-trained model.
- **Do not read "identity" as "no-op forever".** The mixer has ~29.4M trainable parameters; after training it is a dense learned map. Debugging tools that assume `y == x` out of the mixer are only valid on a fresh instance.
- **The flag lives on `mix`, not on `MIMO`.** If you move the attribute to the `MIMO` module, `apply` will still visit `mix` (an unflagged `nn.Linear`) and clobber the eye. Keep the attribute on the exact submodule whose weight must survive.

## Tests

- `tests/test_mimo.py::test_mimo_identity_init` — builds `MIMO(4, 32)`, runs an eval-mode forward, and asserts `y ≈ x` (`atol=1e-6`): proves the construction-time identity behavior end to end.
- `tests/test_mimo.py::test_mimo_identity_survives_transformer_init` — constructs a tiny `Mamba3Transformer` (1 layer, `n_heads=4`, `head_dim=16`) and asserts `layers[0].mimo.mix.weight == eye(4*16)` *after* `self.apply(self._init_weights)` has run: proves the `getattr`-before-`isinstance` guard actually preserves the eye.
- `tests/test_mimo.py::test_mimo_shape_and_finite` — checks shape preservation `(2, 8, 4, 32)` and finiteness in train mode, guarding the reshape round trip.

## Related

The theory behind per-head independence and the motivation for inter-head mixing is developed in [docs/theory/05-mimo-mixing.md](../theory/05-mimo-mixing.md). The mixer's place in the block is covered by the Mamba block reference (R5) and its interaction with weight init by the transformer reference (R4).
