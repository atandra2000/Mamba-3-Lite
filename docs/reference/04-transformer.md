# Mamba3Transformer — the top-level model

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
- A missing key is silently filled with the dataclass default (e.g. `{"vocab_size": 50}` yields `d_model=1024, n_layers=28, init_std=0.02, ssd_dispatch="pytorch"`). Per-field defaults are documented in `docs/reference/01-model-config.md` (planned).

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
3. **Everything else untouched.** `nn.RMSNorm` is neither `Linear` nor `Embedding`, so its gain stays at the RMSNorm default of 1 (verified: `norm_f.weight` is all ones after construction). The per-head complex scalar `A` (`nn.Parameter(torch.empty(H, complex64))`, constant-initialized to −1.0 in `models/mamba_block.py:Mamba3Block.__init__`) is untouched, so every layer starts with decay $\mathrm{Re}(A) = -1$ per the convention described in `docs/theory/07-numerical-stability.md`.

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
- **Dtype**: `nn.Embedding` parameters are float32, so activations enter the first block in float32. The model itself never casts. Under training, BF16 comes from *outside*: `training/pretrain.py` wraps the step in `torch.amp.autocast` on CUDA, and each block's mixer re-materializes float32 where it needs it (`models/mamba_block.py:Mamba3Block._forward_impl` slices `in_proj` output and calls `.float()` on the SSD inputs). Logits come out float32 for the cross-entropy loss (see `docs/theory/07-numerical-stability.md` for why logits stay FP32).
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
2. **The complex `A` counts as 16 parameters, not 32.** `torch.Tensor.numel()` counts elements, and a `(16,)` complex64 tensor has 16 elements (32 stored floats). This matches the per-layer subtotal above. A full derivation of the per-layer 13.65M figure also appears in `docs/theory/08-scaling-efficiency.md`.

## 7. Pitfalls

1. **`embed.weight` and `lm_head.weight` are the same tensor — mutations are visible through both.** This is the point of tying, but it inverts the usual "copy" intuition: e.g. re-initializing `embed.weight` *after* construction also re-initializes the head, and any code that tries to zero or normalize one must know it affects the other. The optimizer sees a single parameter (one AdamW state), so `weight_decay` applies once — there is no separate head decay. In checkpoints the tensor is stored once (`utils/checkpoint.py:CheckpointManager` dedupes by `data_ptr`).
2. **Dict configs are strict on unknown keys but silent on missing ones.** `ModelConfig(**cfg)` raises `TypeError` for any key that is not a dataclass field — a YAML section that accidentally includes `batch_size` or `lr` will fail at model construction, not after a day of training. Conversely, a *missing* key never errors; it silently uses the dataclass default. If you intend a non-default value, a typo'd key raises (good) but an omitted key silently changes behavior (e.g. omitting `ssd_dispatch` silently selects the `"pytorch"` path). The authoritative field list is `docs/reference/01-model-config.md` (planned); the annotated YAML is `docs/reference/12-config-reference.md` (planned).
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
