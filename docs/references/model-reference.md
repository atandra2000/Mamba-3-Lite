# Mamba-3-Lite — Model Reference: Transformer, Block, and MIMO

This reference documents `models/transformer.py:Mamba3Transformer`, the full 434M-parameter model, and its configuration dataclass `models/transformer.py:ModelConfig`: construction paths (dataclass or dict), weight tying, the `_init_weights` scheme, the forward contract, and the exact parameter accounting.

## 1. 60-second summary

After reading this doc you will know: how the model is assembled — token embedding, 28 identical Mamba-3 residual blocks, a final RMSNorm, and a vocabulary head that is **tied** to the embedding; the two accepted config forms (`ModelConfig` or a plain `dict`) and how each is validated; why `lm_head.weight` and `embed.weight` are literally the *same tensor object*, and what that implies for initialization, the optimizer, and checkpoints; the exact `_init_weights` rules (N(0, 0.02) for `Linear`/`Embedding`, a `_identity_init` skip that preserves the MIMO eye init, and nothing else touched — RMSNorm gains stay 1, the complex `A` parameter stays −1.0); the `(B, T) → (B, T, vocab)` forward contract returning float32 logits with no causal masking; and how the 433,662,400 parameters are accounted for, including why the tied embedding is counted exactly once.

## 2. Signature and semantics

```python
class Mamba3Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig | dict): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

| Aspect | Contract |
|---|---|
| Constructor argument | `ModelConfig` instance **or** a plain `dict` of field names → values; unknown dict keys raise `TypeError`, missing keys silently fall back to dataclass defaults (Section 5.1). |
| Input | `x` of shape `(B, T)` with integer token ids (int64), already on the target device. |
| Output | logits of shape `(B, T, vocab_size)` in **float32**, before any softmax; no masking is ever applied. |
| Weights | `embed.weight` and `lm_head.weight` are one tensor when `cfg.weight_tying` is True (default) — same `data_ptr`. |
| Init | `self.apply(self._init_weights)` runs at construction: N(0, `init_std`) on every `Linear`/`Embedding` except modules flagged `_identity_init` (the MIMO mixer). |

The model owns no parameters beyond the blocks' internals: the full parameter list is `embed.weight`, the 28 blocks' parameters, and `norm_f.weight` — `lm_head` contributes no separate storage under tying (verified: a full default build sums to exactly 433,662,400).

## 3. Why it exists

`Mamba3Transformer` is the composition point. Everything else in `models/` is a component: `models/mamba_block.py:Mamba3Block` is one residual layer, `models/ssd_complex.py:ssd_complex_chunkwise` is the sequence mixer inside it, `models/mimo.py:MIMO` is the head-mixing linear map. The transformer is the object the training loop actually constructs and optimizes (`training/pretrain.py` instantiates `Mamba3Transformer(config.model_config)` and calls `training/pretrain.py:count_parameters` on it), and the object checkpoints serialize. It also concentrates the two global design decisions that touch every parameter: **weight tying** (embedding and head share storage) and the **global init policy** (one `apply` pass over all 28 layers).

## 4. Intuition

Think of the model as a fixed pipeline of identical "factories", each refining a `(B, T, 1024)` tensor:

$$\text{Embed} \;\to\; \big[\text{RMSNorm} \to \text{in\_proj} \to \text{complex-SSD} \to \text{MIMO} \to \text{out\_proj} \to +\text{residual} \;\big|\; \text{RMSNorm} \to \text{SwiGLU} \to +\text{residual}\big]^{28} \;\to\; \text{RMSNorm} \;\to\; \text{head}$$

Each block mixes information *across time* (the chunkwise complex SSD recurrence, causal by construction) and *across the 16 heads* (MIMO), then a SwiGLU FFN mixes *within* each token's channel dimension. Pre-norm residual connections keep the path gradient-friendly. Because the sequence mixer is a causal linear recurrence rather than attention, the forward pass needs **no causal mask** — causality is baked into the state-space update, and the final head is a plain linear readout to 50,257 vocabulary logits.

Weight tying is the "shared vocabulary space" trick: the head is a linear map from the last-layer representation back into token-id space, and reusing the embedding matrix for it halves the vocabulary-sized storage and couples the gradient of the head with the gradient of the input embedding. The cost is that one tensor lives in two places.

## 5. Code walkthrough

### 5.1 `__init__` — config acceptance, module construction, tying, apply

```python
def __init__(self, cfg: ModelConfig | dict):
    super().__init__()
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    self.cfg = cfg

    self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

    self.layers = nn.ModuleList([
        Mamba3Block(cfg.__dict__, layer_idx=i)
        for i in range(cfg.n_layers)
    ])

    self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
    self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    if cfg.weight_tying:
        self.lm_head.weight = self.embed.weight

    self.apply(self._init_weights)
```

**Config acceptance.** A `dict` is converted with `ModelConfig(**cfg)`. Verified behaviors:

- An unknown key raises `TypeError` at construction (`ModelConfig.__init__() got an unexpected keyword argument 'bogus_key'`) — dataclasses do not ignore extra keys. A YAML config that smuggles training-only fields into the model sub-dict therefore fails loudly, which is the intended guardrail.
- A missing key is silently filled with the dataclass default (e.g. `{"vocab_size": 50}` yields `d_model=1024, n_layers=28, init_std=0.02, ssd_dispatch="pytorch"`). Per-field defaults are documented in [Mamba-3-Lite — Config Reference](config-reference.md).

**Construction order matters.** `embed` is created first, then the `ModuleList` of `Mamba3Block(cfg.__dict__, layer_idx=i)` — note the block receives the *dict* view (`cfg.__dict__`), and `models/mamba_block.py:Mamba3Block` reads keys like `d_model`, `n_heads`, `chunk_size`, `ssd_dispatch`, `rms_norm_eps`, `grad_checkpoint` out of it with `cfg.get(...)` defaults. Then `norm_f` (the final RMSNorm), then `lm_head` as a bias-free `Linear(d_model, vocab_size)`.

**Weight tying is object identity, not a copy.** `self.lm_head.weight = self.embed.weight` rebinds the head's parameter slot to the *same `Parameter` object*. `lm_head.weight.data_ptr() == embed.weight.data_ptr()` holds, so any in-place mutation of one is visible through the other, and PyTorch's optimizer deduplicates: the tensor appears once in `model.parameters()` (see Section 6). Checkpoints also collapse it — `utils/checkpoint.py:CheckpointManager` dedupes by `data_ptr` when saving.

**`self.apply(self._init_weights)` runs last.** `nn.Module.apply` visits every submodule depth-first (children before the module itself), so the recursion order is: `embed` → each block's internals (`in_proj`, `mimo.mix`, `out_proj`, `norm1`, `norm2`, `ffn_gate_up`, `ffn_down`, `A`) → `norm_f` → `lm_head` → finally `self`. Because initialization runs *after* tying, the single shared tensor is visited twice — once as an `Embedding`, once as a `Linear` — and written twice. Both writes draw from the same distribution, so the outcome is distributionally identical (Section 7 discusses why this is still worth knowing).

### 5.2 `_init_weights` — the exact rules

```python
def _init_weights(self, module):
    if getattr(module, "_identity_init", False):
        return
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
```

Three rules, in priority order:

1. **`_identity_init` skip.** Any module carrying a truthy `_identity_init` attribute is returned from immediately. The only module in the tree that sets this is the MIMO mixer's inner linear, `models/mimo.py:MIMO` (`self.mix._identity_init = True`), whose weight is `nn.init.eye_`-initialized to start as an exact identity map. The flag exists precisely so this `apply` pass cannot clobber that choice — verified in-model: after `Mamba3Transformer(cfg)` construction, `m.layers[0].mimo.mix.weight` still equals `torch.eye(64*16)` exactly.
2. **`Linear`/`Embedding` → N(0, `init_std`).** Every weight matrix in every block (`in_proj`, `out_proj`, `ffn_gate_up`, `ffn_down` — `mimo.mix` is exempted by rule 1), plus `embed.weight` and (via the tie) `lm_head.weight`, is drawn from $\mathcal{N}(0, \text{init\_std}^2)$ with the default `init_std = 0.02` — the standard GPT-style small-variance init. Biases are never touched because every linear in the model is constructed `bias=False`.
3. **Everything else untouched.** `nn.RMSNorm` is neither `Linear` nor `Embedding`, so its gain stays at the RMSNorm default of 1 (verified: `norm_f.weight` is all ones after construction). The per-head complex scalar `A` (`nn.Parameter(torch.empty(H, complex64))`, constant-initialized to −1.0 in `models/mamba_block.py:Mamba3Block.__init__`) is untouched, so every layer starts with decay $\mathrm{Re}(A) = -1$ per the convention described in `docs/concepts/block-and-stability.md`.

### 5.3 `forward` — the contract

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, T, vocab_size)."""
    x = self.embed(x)

    for layer in self.layers:
        x = layer(x)

    x = self.norm_f(x)
    logits = self.lm_head(x)

    return logits
```

- **Input**: `(B, T)` integer token ids. No `attention_mask`, no `position_ids`, no `past_key_values` — the SSM is inherently causal, so the model takes only the token sequence.
- **Dtype**: `nn.Embedding` parameters are float32, so activations enter the first block in float32. The model itself never casts. Under training, BF16 comes from *outside*: `training/pretrain.py` wraps the step in `torch.amp.autocast` on CUDA, and each block's mixer re-materializes float32 where it needs it (`models/mamba_block.py:Mamba3Block._forward_impl` slices `in_proj` output and calls `.float()` on the SSD inputs). Logits come out float32 for the cross-entropy loss (see `docs/concepts/block-and-stability.md` for why logits stay FP32).
- **Per-block flow** (delegated to `models/mamba_block.py:Mamba3Block._forward_impl`): RMSNorm → `in_proj` → slice into `x_ssm` (float32), complex `B_t`/`C_t` (complex64), `dt` → `ssd_complex_chunkwise` (or the Triton dispatch via `_ssd_with_dispatch`) → `MIMO` mixing across heads → `out_proj` → residual add; then RMSNorm → SwiGLU (`ffn_gate_up` chunked into gate/up, `F.silu(gate) * up`, `ffn_down`) → residual add.
- **Sequence length is unbounded at the model level.** The chunkwise SSD pads internally to a multiple of `chunk_size` and slices back (`models/ssd_complex.py:ssd_complex_chunkwise`), so `forward` accepts any `T`; `cfg.max_seq_len = 2048` is a *training* window used by the dataset, not an architectural cap enforced here.

## 6. Parameter accounting

Total with default config and tying: **433,662,400** (~434M) — verified by instantiating `Mamba3Transformer(ModelConfig())` and summing `p.numel()`. The breakdown:

| Component (per layer) | Shape | Params |
|---|---|---|
| `in_proj` | 1024 × 16·(64 + 4·64 + 1) = 1024 × 5136 | 5,259,264 |
| `out_proj` | 1024 × 1024 | 1,048,576 |
| `mimo.mix` | 1024 × 1024 | 1,048,576 |
| `ffn_gate_up` | 1024 × 4096 | 4,194,304 |
| `ffn_down` | 2048 × 1024 | 2,097,152 |
| `norm1.weight` + `norm2.weight` | 1024 + 1024 | 2,048 |
| `A` (complex64, 16 heads) | 16 | 16 |
| **Per-layer subtotal** | | **13,649,936** |
| `embed.weight` (tied, counted once) | 50,257 × 1024 | 51,463,168 |
| `norm_f.weight` | 1024 | 1,024 |
| **Total** | | **433,662,400** |

Two accounting subtleties:

1. **The tied embedding is counted once, and that is a PyTorch guarantee.** `nn.Module.parameters()` routes through `named_parameters(remove_duplicate=True)`, which skips any parameter object already yielded (by `id`). Verified on a toy model: naive iteration over `named_parameters(remove_duplicate=False)` double-counts the embed by exactly its own size (1,600 = 50×32), while `parameters()` returns 22,372 vs 23,972 naive — a difference of exactly the embed size. So `training/pretrain.py:count_parameters` (`sum(p.numel() for p in model.parameters())`) reports the honest 433,662,400 and, since nothing is frozen, `trainable == total` — the training log prints `Parameters: 433,662,400 total / 433,662,400 trainable`.
2. **The complex `A` counts as 16 parameters, not 32.** `torch.Tensor.numel()` counts elements, and a `(16,)` complex64 tensor has 16 elements (32 stored floats). This matches the per-layer subtotal above. A full derivation of the per-layer 13.65M figure also appears in `docs/concepts/block-and-stability.md`.

## 7. Pitfalls

1. **`embed.weight` and `lm_head.weight` are the same tensor — mutations are visible through both.** This is the point of tying, but it inverts the usual "copy" intuition: e.g. re-initializing `embed.weight` *after* construction also re-initializes the head, and any code that tries to zero or normalize one must know it affects the other. The optimizer sees a single parameter (one AdamW state), so `weight_decay` applies once — there is no separate head decay. In checkpoints the tensor is stored once (`utils/checkpoint.py:CheckpointManager` dedupes by `data_ptr`).
2. **Dict configs are strict on unknown keys but silent on missing ones.** `ModelConfig(**cfg)` raises `TypeError` for any key that is not a dataclass field — a YAML section that accidentally includes `batch_size` or `lr` will fail at model construction, not after a day of training. Conversely, a *missing* key never errors; it silently uses the dataclass default. If you intend a non-default value, a typo'd key raises (good) but an omitted key silently changes behavior (e.g. omitting `ssd_dispatch` silently selects the `"pytorch"` path). The authoritative field list and the annotated YAML live in [Mamba-3-Lite — Config Reference](config-reference.md).
3. **`init_std` applies *after* weight tying, so the shared tensor is initialized twice.** The `apply` pass visits `embed.weight` (Embedding branch) and then, through the tie, `lm_head.weight` (Linear branch). Both draws are i.i.d. $\mathcal{N}(0, \text{init\_std}^2)$, so the final value is still exactly N(0, `init_std`) — but do not rely on *which* branch ran last, and note that any custom per-module init must be registered after `Mamba3Transformer(...)` returns, or preserved via the `_identity_init` mechanism.
4. **`_init_weights` leaves RMSNorm gains and `A` alone — on purpose.** If you want non-unit gains or a different `A` init, there is no config hook; you must post-process after construction (`m.norm_f.weight.data.fill_(v)`), which also re-touches the tied head if you touch `embed.weight` (see pitfall 1).
5. **The model performs no masking and no padding logic.** Feeding a ragged batch requires the caller to pad and then (if needed) mask the loss with `ignore_index`; the forward pass itself is blind to it. Also, `forward` assumes the input tensor is already on the model's device — there is no internal `.to(device)`.
6. **`grad_checkpoint` is per-block, off by default.** When `cfg.grad_checkpoint` is True and the module is in training mode, each block routes through `torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)` (`models/mamba_block.py:Mamba3Block.forward`), trading recompute for activation memory. It is a training-time behavior; `eval()` disables it implicitly.

## 8. Tests

- `tests/test_transformer.py::test_mamba3_transformer_forward` — builds a small model (`vocab 100, d_model 64, n_layers 2`), asserts the forward output shape `(2, 16, 100)`, sanity-bounds the parameter count, and asserts **weight tying via `data_ptr` equality**: `m.embed.weight.data_ptr() == m.lm_head.weight.data_ptr()`.
- `tests/test_transformer.py::test_mamba3_transformer_accepts_dict_config` — constructs from a plain dict and asserts the same forward contract, pinning the `ModelConfig(**cfg)` path.
- `tests/test_mimo.py::test_mimo_identity_survives_transformer_init` — the `_identity_init` guard: after full `Mamba3Transformer` construction (which runs `_init_weights`), `m.layers[0].mimo.mix.weight` still equals `torch.eye(H*D)` exactly (`torch.equal`), proving the skip rule of Section 5.2.
- `tests/test_mimo.py::test_mimo_identity_init` — MIMO in isolation maps input to itself (identity pass-through), the behavior the flag protects.
- End-to-end: `tests/e2e_gpu_smoke.py` (CUDA + Triton) exercises the full model forward/backward; on a CPU-only box the suite runs 37 tests collected (32 passed / 5 GPU-skipped), including both `tests/test_transformer.py` tests.
---

#Mamba3Block — the Mamba-3 residual block

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

The complex `A` is the interesting part: the effective step is `exp(softplus(dt)·A)`, so the real part of `A` sets decay while the imaginary part sets rotation — two behaviors in one scalar per head (see `docs/concepts/ssd-theory.md`).

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

Sum: 5,259,264 + 1,048,576 + 1,048,576 + 16 + 2,048 + 4,194,304 + 2,097,152 = 13,649,936, matching the 28 × 13,649,936 + 51,463,168 (tied embedding) = 433,662,400 total in `docs/concepts/block-and-stability.md`. Note `A` contributes 16 by `numel()` even though a complex64 element occupies 16 bytes in memory — this convention reconciles the count.

`A` is initialized to the constant complex value −1.0 + 0j for every head. All `nn.Linear` weights (but **not** `mimo.mix`) are then re-initialized to `N(0, init_std)` by `models/transformer.py:Mamba3Transformer._init_weights`; the MIMO map is exempted via its `_identity_init` flag (see `models/mimo.py:MIMO.mix` and `docs/concepts/mimo.md`). `A` and the RMSNorm weights are untouched because `_init_weights` only visits `nn.Linear` and `nn.Embedding`.

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

where `dt` is the projected step size and `A` the per-head complex parameter. With `A = a + ib` (initially `a = −1, b = 0`): `softplus(dt) > 0` always, so `softplus(dt)·a < 0` and `|Ā| = exp(softplus(dt)·a) ∈ (0, 1)` — every step is contractive — while the imaginary part contributes `exp(i·softplus(dt)·b)`, a per-step rotation. One complex scalar parameterizes both; `docs/concepts/ssd-theory.md` derives the complex-state version and `docs/concepts/ssd-theory.md` proves the chunkwise (attention-like) reorganization `_forward_impl` calls (`docs/concepts/state-space-foundations.md` builds the discretization from scratch). The full block output is

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

Every slice is `.float()`-cast **before** complex assembly. This is load-bearing: `torch.complex` requires float32 real/imag inputs, and under BF16 autocast (the training default on CUDA, see `docs/guides/pretrain-cli.md`) the raw `in_proj` output is BF16. The casts also keep the SSD math in float32 — see `docs/concepts/block-and-stability.md`.

The SSD call returns `Y.real` (the chunkwise scan is computed in complex64; the real part is taken inside `models/ssd_complex.py:ssd_complex_chunkwise`), so the rest of the path is real: `MIMO` mixes across heads (`models/mimo.py:MIMO.forward`), `out_proj` maps back to `d_model`, and both residuals are plain adds.

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

At the training level, `ssd_dispatch="triton"` is additionally gated by the `ENABLE_TRITON_KERNELS=1` environment variable (`training/pretrain.py:_enforce_triton_env_var` force-backs to `"pytorch"` when unset); the block-level fallback above is the second line of defense, and the kernel itself is documented in `docs/references/ssd-reference.md`.

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
2. **Slice-boundary off-by-ones shift heads silently.** All five quadrants are contiguous slabs of the same projection; a boundary error of `H·N` (1,024 columns) shifts B's real part into x's content or swaps B/C, and since every slice is a valid-shaped tensor, nothing raises — the model just trains garbage. Safe invariants: content width `H·D`, each of the four B/C slabs exactly `H·N`, dt is the **last** `H` columns (`proj[..., -H:]`), total width `H·D + 4·H·N + H`. `docs/concepts/block-and-stability.md` derives this layout from scratch.
3. **Complex assembly requires float32.** `torch.complex(real, imag)` fails unless both inputs are float32 (float64 also works). Under BF16 autocast the projection output is BF16, so the `.float()` casts before `torch.complex` are not cosmetic — removing them breaks both paths on CUDA, and the casts also pin the recurrence math to FP32 regardless of autocast.
4. **`A` is complex64 and per-head, not per-head-per-dim.** It broadcasts against `(B, T, H)` dt as one complex scalar per head — no per-state-dimension decay. `D` and `N` are independently configurable, but the slice math and `MIMO` shapes assume the code's `H·D`/`H·N` widths — changing one without the others breaks the layout.
5. **Fallback failures are broad by design.** `_ssd_with_dispatch` catches `Exception` wholesale; a *logic* bug in the Triton path (wrong but non-raising results) will not be caught — `tests/test_ssd_triton.py::TestPerChunkSsdDispatchWiring::test_triton_path_output_matches_pytorch_path` exists precisely because silent divergence is the failure mode to watch for. And the first failure is the only diagnostic per block.
6. **The residual is taken before normalization.** Both branches are pre-norm, so residuals carry un-normalized activations; `x` entering the block may be BF16 while `y` is FP32 (cast by the SSD path), so the first residual add promotes to FP32. Expected, but it is where mixed-precision expectations break.

## Tests

- `tests/test_grad_checkpoint.py` — all three tests: `test_grad_checkpoint_propagates_to_blocks` (TrainingConfig → every block), `test_grad_checkpoint_explicit_false_disables`, and `test_grad_checkpoint_actually_triggers_training_mode` (a real backward through the checkpointed path yields finite grads).
- `tests/test_ssd_triton.py::TestPerChunkSsdDispatchWiring` — the dispatch contract: default `"pytorch"`, explicit `"pytorch"` runs the production path, `"triton"` on a triton-less box falls back cleanly with the warning text `falling back to 'pytorch'`, the warning is one-shot (`_triton_fallback_warned` latches `True`), and Triton output matches PyTorch output.
- `tests/test_mimo.py` — identity init (`test_mimo_identity_init`), that the identity survives `Mamba3Transformer._init_weights` via the `_identity_init` guard (`test_mimo_identity_survives_transformer_init`), and shape/finiteness (`test_mimo_shape_and_finite`).

See also: [Mamba-3-Lite — Config Reference](config-reference.md) (cfg key provenance), [Mamba-3-Lite — SSD Reference](ssd-reference.md) (the chunkwise primitive), and the theory docs [Mamba-3-Lite — MIMO Head Mixing](../concepts/mimo.md) and [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md).
---

#MIMO — The Inter-Head Mixer

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

The theory behind per-head independence and the motivation for inter-head mixing is developed in [docs/concepts/mimo.md](../concepts/mimo.md). The mixer's place in the block is covered by the Mamba block reference (R5) and its interaction with weight init by the transformer reference (R4).

---

## References

- [Mamba-3-Lite — Config Reference](config-reference.md) — every `models/transformer.py:ModelConfig` field the modules consume.
- [Mamba-3-Lite — SSD Reference](ssd-reference.md) — the scan primitive called by `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`.
- [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md) — the math behind the block's sequence-mixing lane.
- [Mamba-3-Lite — MIMO Head Mixing](../concepts/mimo.md) — the full derivation of the mixer and the identity warm start.
- [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) — slice layout, weight tying, init scheme.
