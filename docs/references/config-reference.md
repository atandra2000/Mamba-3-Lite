# Mamba-3-Lite — Config Reference: ModelConfig and the Annotated YAML

Reference for `models/transformer.py:ModelConfig`: the single dataclass that defines the Mamba-3-Lite architecture, field by field, with defaults, consumers, and the dispatch/checkpointing contracts that hang off it.

## 60-second summary

After reading this doc you can answer, for every one of the 14 fields of `models/transformer.py:ModelConfig`: what it does, its default, and exactly which module consumes it. You will also know the three contracts that are *not* visible in the dataclass itself: the two-layer `ssd_dispatch` opt-in (config value **and** `ENABLE_TRITON_KERNELS=1`), the per-block one-shot triton fallback with its 256-cap, and the fact that `grad_checkpoint` is one global boolean applied uniformly to all 28 blocks. The production config (all defaults) yields a 433,662,400-parameter (~434M) model: 28 identical layers of 13,649,936 parameters each, plus a tied embedding/head of 51,463,168 and a final RMSNorm of 1,024.

## Why it exists

The architecture is fixed and compact, so a single dataclass is the whole configuration surface. There is no YAML schema, no `hydra`, no kwargs plumbing: `models/transformer.py:Mamba3Transformer.__init__` takes exactly one argument — a `ModelConfig` or an equivalent `dict` — and every submodule reads its numbers out of that one object. `training/pretrain.py` parses its YAML into a `TrainConfig` whose `model_config` dict is fed straight into the transformer, which means any field added to `ModelConfig` automatically becomes a knobs file knob. Knowing the dataclass is knowing the model.

## Intuition

Think of `ModelConfig` as a bill of materials. `vocab_size`, `d_model`, `n_layers`, `n_heads`, `head_dim`, `state_dim`, `ffn_dim`, `weight_tying`, `rms_norm_eps`, `init_std` fix the *shapes and sizes* of every tensor in the graph. `chunk_size` and `ssd_dispatch` fix the *algorithm* the sequence mixer runs (the chunked-vs-naive trade and the pytorch-vs-triton implementation), and `max_seq_len`/`grad_checkpoint` fix *training behavior* (window size and memory strategy) rather than the module graph. Two numbers deserve special attention because they are easy to confuse: `head_dim` (D = 64) is the per-head channel width of the *token content* `x`, while `state_dim` (N = 64) is the per-head width of the *hidden state* that B and C project into. The `in_proj` slice layout (`H*(D + 4N + 1)`) is where the two interact.

## Signature

```python
@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    head_dim: int = 64
    state_dim: int = 64
    chunk_size: int = 64
    ssd_dispatch: str = "pytorch"
    ffn_dim: int = 2048
    max_seq_len: int = 2048
    weight_tying: bool = True
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    grad_checkpoint: bool = False
```

## Per-field reference

| Field | Type | Default | What it controls | Consumer |
|---|---|---|---|---|
| `vocab_size` | `int` | 50257 | Embedding-table rows; `lm_head` output width. GPT-2 BPE vocab; EOS/PAD id 50,256. | `models/transformer.py:Mamba3Transformer.embed`, `models/transformer.py:Mamba3Transformer.lm_head` |
| `d_model` | `int` | 1024 | Hidden width everywhere: embedding output, residual stream, final norm, LM head input, FFN in/out. | `models/transformer.py:Mamba3Transformer.embed`, `models/transformer.py:Mamba3Transformer.norm_f`, `models/transformer.py:Mamba3Transformer.lm_head`, `models/mamba_block.py:Mamba3Block.in_proj`, `models/mamba_block.py:Mamba3Block.out_proj`, `models/mamba_block.py:Mamba3Block.ffn_gate_up`, `models/mamba_block.py:Mamba3Block.ffn_down` |
| `n_layers` | `int` | 28 | Number of stacked `Mamba3Block`s. | `models/transformer.py:Mamba3Transformer.layers` (ModuleList of `models/mamba_block.py:Mamba3Block`) |
| `n_heads` | `int` | 16 | Heads H: `in_proj` slice layout, `MIMO` head count, per-head complex decay `A`. | `models/mamba_block.py:Mamba3Block.in_proj`, `models/mamba_block.py:Mamba3Block.A`, `models/mimo.py:MIMO` |
| `head_dim` | `int` | 64 | D: per-head token-content width; `x_ssm` slice `H*D`; Triton `BLOCK_P`. | `models/mamba_block.py:Mamba3Block.in_proj`, `models/mamba_block.py:Mamba3Block.out_proj`, `models/ssd_complex.py:ssd_complex_chunkwise`, `models/ssd_triton.py:_check_block_dims` |
| `state_dim` | `int` | 64 | N: per-head complex state width; B/C slices `4N`; Triton `BLOCK_N`. | `models/mamba_block.py:Mamba3Block.in_proj`, `models/ssd_complex.py:ssd_complex_chunkwise` (B_t/C_t last dim), `models/ssd_triton.py:_check_block_dims` |
| `chunk_size` | `int` | 64 | C: chunk length; intra-chunk `L` is C×C; padding granularity; Triton `BLOCK_C`. | `models/mamba_block.py:Mamba3Block.chunk_size` → `models/ssd_complex.py:ssd_complex_chunkwise(chunk_size=…)` |
| `ssd_dispatch` | `str` | `"pytorch"` | `"pytorch"` (5-einsum chain) vs `"triton"` (fused per-chunk kernel, opt-in). | `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`, `training/pretrain.py:_enforce_triton_env_var`, `models/ssd_complex.py:ssd_complex_chunkwise(ssd_dispatch=…)` |
| `ffn_dim` | `int` | 2048 | SwiGLU hidden width: gate/up projects to `2*ffn_dim`, down projects from `ffn_dim`. | `models/mamba_block.py:Mamba3Block.ffn_gate_up`, `models/mamba_block.py:Mamba3Block.ffn_down` |
| `max_seq_len` | `int` | 2048 | **Not** consumed by the module. Training-window size: dataset windows of `max_seq_len+1` tokens, x=[:-1], y=[1:]; also the `seq_len` in the tps logger. | `training/pretrain.py:PretrainDataset`, `utils/logging.py:TrainingLogger` |
| `weight_tying` | `bool` | `True` | Shares one weight tensor between embedding and LM head: `lm_head.weight = embed.weight` (data_ptr identity). | `models/transformer.py:Mamba3Transformer.__init__` |
| `rms_norm_eps` | `float` | 1e-5 | `eps` of every RMSNorm: pre-SSD, pre-FFN, and final. | `models/transformer.py:Mamba3Transformer.norm_f`, `models/mamba_block.py:Mamba3Block.norm1`, `models/mamba_block.py:Mamba3Block.norm2` |
| `init_std` | `float` | 0.02 | Std of `N(0, init_std)` applied to every `nn.Linear`/`nn.Embedding` by the recursive init. | `models/transformer.py:Mamba3Transformer._init_weights` |
| `grad_checkpoint` | `bool` | `False` (model default; `TrainConfig` default is `True`) | One global boolean: when true and training, every block's forward runs under `torch.utils.checkpoint`. | `models/mamba_block.py:Mamba3Block.forward`; the trainer in `training/pretrain.py` injects it via `setdefault` |

The defaults compose into the production model: 16 heads × 64 head-dim = 1,024 = `d_model`; 64 state-dim per head; 64-token chunks; 2048-token windows. Changing any of these numbers changes parameter count — see [Shapes and invariants](#shapes-and-invariants).

## The `ssd_dispatch` contract in depth

`ssd_dispatch` is the only field that switches *implementation*, so it carries three nested layers of opt-in and fallback. Understanding them exactly prevents a classic silent-mode bug.

**Layer 1 — config value.** The dataclass default is `"pytorch"`, and `models/mamba_block.py:Mamba3Block` copies it at construction: `self.ssd_dispatch = cfg.get("ssd_dispatch", "pytorch")`. In `_ssd_with_dispatch`, any value other than the literal `"triton"` runs the pure-PyTorch 5-einsum path of `ssd_complex_chunkwise`:

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

**Layer 2 — the environment gate.** In the training harness, config value alone is not enough: `training/pretrain.py:_enforce_triton_env_var` is called before the model is built and force-rewrites `ssd_dispatch` back to `"pytorch"` (with one log warning) unless `ENABLE_TRITON_KERNELS=1`:

```python
def _enforce_triton_env_var(model_config: dict, log) -> None:
    """Force triton dispatch back to pytorch if ENABLE_TRITON_KERNELS != 1."""
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

So inside `pretrain.py` a `triton` config with the env var missing **never raises** — it silently becomes `pytorch` for the whole run, and the only trace is the warn line.

**Layer 3 — per-block one-shot fallback.** Outside the harness (notebooks, tests, the e2e script) there is no env gate; `_ssd_with_dispatch` simply attempts the triton path inside `try/except`. The failure modes that land here: `per_chunk_ssd_triton` raising `ImportError` when `triton` is not installed (CPU/Mac boxes), kernel compile errors on non-power-of-two block sizes, and the 256-cap:

```python
def _check_block_dims(P: int, N: int, chunk_size: int) -> None:
    for name, dim in (("P", P), ("N", N), ("chunk_size", chunk_size)):
        if dim > _MAX_BLOCK:
            raise ValueError(
                f"per_chunk_ssd_triton: {name}={dim} exceeds the {_MAX_BLOCK}-cap. "
                f"Use ssd_dispatch='pytorch' for this config."
            )
```

`_check_block_dims` runs inside `models/ssd_triton.py:_per_chunk_ssd_triton_forward` with `P = head_dim`, `N = state_dim`, `C = chunk_size`. Any of the three above 256 raises `ValueError`, which the block catches, warns **once per block instance** (guarded by `Mamba3Block._triton_fallback_warned`), and re-runs on the PyTorch path. So a config with `state_dim=512` and `ssd_dispatch="triton"` is not an error — it is a per-block fallback to pytorch with a single printed warning per layer. The output remains numerically correct; only the implementation changes.

What the triton dispatch actually replaces: inside `ssd_complex_chunkwise`, `ssd_dispatch="triton"` calls `models/ssd_triton.py:per_chunk_ssd_triton(Bc, Cc, Xc, Ac, decay_states)` to fuse the per-chunk `Y_diag` and end-of-chunk `states` into one kernel (`_PerChunkSSDTriton`, one program per `(B, chunk, H)`, complex64 split into contiguous float32 real/imag pairs by `_view_real_imag`). The inter-chunk state propagation einsums stay in PyTorch. The autograd `Function` recomputes the same per-chunk math with `models/ssd_triton.py:per_chunk_ssd_pytorch` in backward, seeded with the true downstream gradients. Env knobs `TRITON_PER_CHUNK_NUM_STAGES` (default 1) and `TRITON_PER_CHUNK_NUM_WARPS` (default 4) tune the kernel launch. For the full kernel anatomy see [Mamba-3-Lite — SSD Reference](../references/ssd-reference.md).

## `grad_checkpoint`: one global boolean

Verified against the source: there is **no per-4th-layer checkpointing** anywhere. `Mamba3Block` reads a single bool and wraps the whole block forward:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """(B, T, d_model) -> (B, T, d_model)."""
    if self.grad_checkpoint and self.training:
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, use_reentrant=False
        )
    return self._forward_impl(x)
```

The same flag is applied to every one of the `n_layers` blocks, so activation memory is cut by recomputing the full block (SSD included) in backward — a time/memory trade, uniform across depth. Two defaults meet here: `ModelConfig.grad_checkpoint = False` (so constructing the model directly, e.g. in tests, gets no checkpointing), while `TrainConfig.grad_checkpoint = True` and `pretrain.py` injects it via `config.model_config.setdefault("grad_checkpoint", config.grad_checkpoint)`. Consequence: a knobs YAML that *omits* `grad_checkpoint` trains with checkpointing on; one that explicitly writes `grad_checkpoint: false` wins, because `setdefault` only fills missing keys.

## Dict vs dataclass

`Mamba3Transformer.__init__` accepts both:

```python
def __init__(self, cfg: ModelConfig | dict):
    super().__init__()
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    self.cfg = cfg
```

A dict is normalized to a `ModelConfig` by keyword unpacking, so partial dicts are fine (missing keys fall back to dataclass defaults) but **unknown keys raise `TypeError`** — the dataclass rejects them at construction. Internally, the transformer hands `cfg.__dict__` (a plain dict) to each `Mamba3Block`, whose constructor reads with `cfg.get(...)` fallbacks (`chunk_size`, `ssd_dispatch`, `rms_norm_eps`, `grad_checkpoint` all default defensively) and with direct indexing for the fields that must exist (`d_model`, `n_heads`, `head_dim`, `state_dim`, `ffn_dim`). Because `ModelConfig(**dict)` fills defaults, the defensive `.get`s are belt-and-suspenders rather than reachable paths — except in tests that construct blocks directly with hand-rolled dicts.

## Shapes and invariants

The `in_proj` width derives from three fields: `H*(D + 4N + 1)`. In `Mamba3Block._forward_impl`, the projection is sliced as `x_ssm = proj[..., :H*D]` (token content), then `B_real, B_imag, C_real, C_imag` (each `H*N`, packed to complex64), then `dt = proj[..., -H:]`. The decay `A` is a per-head complex scalar, `torch.empty(H, complex64)` initialized to `-1.0` — shared across all N state slots of a head, not per-slot. `chunk_size` feeds `ssd_complex_chunkwise`'s padding: `pad = (C - (T % C)) % C`, zero-padding `x, B_t, C_t, dt` along time, computing on `T_padded`, and slicing back `[:, :T, :, :]` — so chunk_size does **not** need to divide T.

Parameter count (derived from the defaults; the 28 layers are identical):

| Component | Formula | Params |
|---|---|---|
| `in_proj` | `d_model × H(D+4N+1)` = 1024×16×321 | 5,259,264 |
| `out_proj` | `H·D × d_model` = 1024×1024 | 1,048,576 |
| `MIMO` | `H·D × H·D` | 1,048,576 |
| `ffn_gate_up` | `d_model × 2·ffn_dim` = 1024×4096 | 4,194,304 |
| `ffn_down` | `ffn_dim × d_model` | 2,097,152 |
| `norm1 + norm2` | `d_model + d_model` | 2,048 |
| `A` | `H` complex64 (numel 16) | 16 |
| **per layer** | | **13,649,936** |
| embed = lm_head (tied) | `vocab_size × d_model` = 50257×1024 | 51,463,168 |
| `norm_f` | `d_model` | 1,024 |
| **total** | 28 × 13,649,936 + 51,463,168 + 1,024 | **433,662,400** |

## Pitfalls

1. **`ssd_dispatch='triton'` can silently mean `'pytorch'`.** In `pretrain.py`, a missing `ENABLE_TRITON_KERNELS=1` is force-corrected with a single warn line, no error. In direct model use, a triton-less box prints one warning per block on the first forward and runs pytorch. If you set `triton` and see no speedup, check the warn line, not the config.
2. **There is no state_dim parity check.** The repo contains no "state_dim must be even" assertion; the genuine constraints on the triton path are (a) `head_dim`, `state_dim`, `chunk_size` must each be ≤ 256 (`_check_block_dims`), and (b) they must be powers of two, because they become Triton constexpr block sizes consumed by `tl.arange(0, BLOCK_*)`. A non-power-of-two `state_dim` fails inside kernel compilation and falls back. The pytorch path accepts any `N ≥ 1` and any `C ≥ 1`.
3. **`chunk_size` need not divide T**, but it is quadratic in the intra-chunk `L` matrix (`C×C` per head per chunk): doubling `chunk_size` quadruples that factor's memory. 64 is a deliberate balance.
4. **Changing `d_model` invalidates the weight-tying identity at load time.** `embed.weight` and `lm_head.weight` are one tensor (`lm_head.weight = embed.weight`, `Mamba3Transformer.__init__`); both are `(vocab_size, d_model)`. A checkpoint trained at `d_model=1024` reports shape mismatches on *both* names against any other `d_model`, and you cannot resize one side of the tie. `in_proj`/`out_proj`/FFN widths all scale with `d_model`, so a mid-stream resize is not a knob — it is a re-spawn.
5. **`n_heads`, `head_dim`, `state_dim` are coupled through `in_proj`.** Width `H*(D+4N+1)` means changing any one re-layouts the whole projection slice; keep all three consistent with the checkpoint. Watch the 256-cap on `D`, `N`, and `C` if you also want the triton path.
6. **`grad_checkpoint` default asymmetry.** Model default `False`, `TrainConfig` default `True`, injected with `setdefault`. A knobs file that forgets the key gets checkpointing; a `ModelConfig` constructed directly does not. If you benchmark a direct-construction model expecting training memory, you will see the un-checkpointed numbers.
7. **`max_seq_len` is not an architectural ceiling.** There are no positional embeddings; the SSD mixes positions causally at any T (chunk-padded). `max_seq_len=2048` only fixes the training window and the logger's `seq_len`; inference at longer T works, just with more chunks.

## Tests

- `tests/test_transformer.py::test_mamba3_transformer_accepts_dict_config` — dict → `ModelConfig` normalization and a full forward with a small dict config.
- `tests/test_ssd_triton.py::TestPerChunkSsdDispatchWiring::test_default_dispatch_is_pytorch` — the dataclass default is `"pytorch"`.
- `tests/test_ssd_triton.py::TestPerChunkSsdDispatchWiring::test_explicit_triton_dispatch_falls_back_cleanly_on_cpu` — per-block fallback warning and correct output on a triton-less box.
- `tests/test_ssd_triton.py::TestPerChunkSsdDispatchWiring::test_triton_fallback_warning_is_one_shot_per_instance` — the `_triton_fallback_warned` guard.
- `tests/test_ssd_triton.py::TestPerChunkSsdDispatchWiring::test_triton_path_output_matches_pytorch_path` — identical outputs across dispatches (same seed, same weights).
- `tests/test_ssd_triton.py::TestEnableTritonKernelsForceBack::test_triton_dispatch_forced_back_when_env_var_missing` and `::test_triton_dispatch_passes_through_when_env_var_set` — the `_enforce_triton_env_var` gate.
- `tests/test_ssd_triton.py::TestPerChunkSsdImportSurface::test_check_block_dims_raises_value_error_on_too_large_dim` and `::test_check_block_dims_accepts_production_404m_shape` — the 256-cap (note the stale test name; the accepted shape is the production 64/64/64).

See [Mamba-3-Lite — SSD Foundations](../concepts/state-space-foundations.md), [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md), [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md), [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md), [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) for the theory behind the fields, and [Mamba-3-Lite — Model Reference](../references/model-reference.md) / [Mamba-3-Lite — Model Reference](../references/model-reference.md) for the per-block consumers.
---

#R12 — Annotated Config Reference

Reference for `configs/pretrain_a100_400m.yaml`: every key in the canonical pre-training config, which dataclass consumes it, the default if it is absent, and the runtime effect — including the honest token-count arithmetic behind the "~8.0B tokens" claim.

## 60-second summary

After reading this doc you can, for every one of the 35 keys in `configs/pretrain_a100_400m.yaml`, name its consumer (`training/pretrain.py:TrainingConfig`, `models/transformer.py:ModelConfig`, or nothing), its fallback default, and its effect on the run. You will also know the key-name translations the YAML silently performs (`micro_batch_size` → `batch_size`, `total_steps` → `max_steps`, `grad_clip` → `max_grad_norm`, `save_interval` → `save_every`, `log_interval` → `log_every`, `save_dir` → `checkpoint_dir`, `compile` → `compile_model`, `train_data_path` → `data_path`), the fact that the whole `data:` section except `train_data_path` is *not consumed by any code*, and the arithmetic discrepancy: 256,000 steps × 2 micro-batches × 16 seqs × 2048 tokens = **16.78B token exposures**, not 8.0B.

## Why it exists

`training/pretrain.py:main` is the single entry point into pre-training, and its only structured input is this one YAML file. Unlike the model dataclass (`models/transformer.py:ModelConfig`), the YAML has **no schema and no validation**: it is parsed with `yaml.safe_load`, read field-by-field with `.get()` calls that each carry a silent default, and the keys it reads do not match the `TrainingConfig` field names (the YAML uses runbook-friendly names like `micro_batch_size`; the dataclass uses `batch_size`). Every key translation, every fallback, and every ignored key lives in one 40-line block of `main()`. This doc is the field-by-field map of that block, plus the arithmetic that makes the config's headline numbers check out — or fail to.

## The full config, verbatim

```yaml

---

## References

- [Mamba-3-Lite — Model Reference](model-reference.md) — the modules that consume `models/transformer.py:ModelConfig` fields.
- [Mamba-3-Lite — SSD Reference](ssd-reference.md) — the dispatch + 256-cap contract behind `ssd_dispatch`.
- [Mamba-3-Lite — Training Reference](training-reference.md) — `training/pretrain.py:TrainingConfig` consumers (dataset, checkpoint, logging).
- [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md) — the `main()` YAML→`TrainingConfig` mapping and every CLI flag.
- [Mamba-3-Lite — Tuning Guide](../guides/tuning.md) — how to measure the effect of each knob.
