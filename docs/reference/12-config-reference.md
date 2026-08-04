# R12 — Annotated Config Reference

Reference for `configs/pretrain_a100_400m.yaml`: every key in the canonical pre-training config, which dataclass consumes it, the default if it is absent, and the runtime effect — including the honest token-count arithmetic behind the "~8.0B tokens" claim.

## 60-second summary

After reading this doc you can, for every one of the 35 keys in `configs/pretrain_a100_400m.yaml`, name its consumer (`training/pretrain.py:TrainingConfig`, `models/transformer.py:ModelConfig`, or nothing), its fallback default, and its effect on the run. You will also know the key-name translations the YAML silently performs (`micro_batch_size` → `batch_size`, `total_steps` → `max_steps`, `grad_clip` → `max_grad_norm`, `save_interval` → `save_every`, `log_interval` → `log_every`, `save_dir` → `checkpoint_dir`, `compile` → `compile_model`, `train_data_path` → `data_path`), the fact that the whole `data:` section except `train_data_path` is *not consumed by any code*, and the arithmetic discrepancy: 256,000 steps × 2 micro-batches × 16 seqs × 2048 tokens = **16.78B token exposures**, not 8.0B.

## Why it exists

`training/pretrain.py:main` is the single entry point into pre-training, and its only structured input is this one YAML file. Unlike the model dataclass (`models/transformer.py:ModelConfig`), the YAML has **no schema and no validation**: it is parsed with `yaml.safe_load`, read field-by-field with `.get()` calls that each carry a silent default, and the keys it reads do not match the `TrainingConfig` field names (the YAML uses runbook-friendly names like `micro_batch_size`; the dataclass uses `batch_size`). Every key translation, every fallback, and every ignored key lives in one 40-line block of `main()`. This doc is the field-by-field map of that block, plus the arithmetic that makes the config's headline numbers check out — or fail to.

## The full config, verbatim

```yaml
# configs/pretrain_a100_400m.yaml
model:
  vocab_size:          50257          # GPT-2 tokenizer
  d_model:             1024
  n_layers:            28             # 28 blocks → ~434M params
  n_heads:             16             # SSM heads
  head_dim:            64             # D — per-head channel dim
  state_dim:           64            # N — SSM state expansion
  chunk_size:          64             # SSD chunk size (the tunable knob)
  ffn_dim:             2048           # SwiGLU intermediate (NOT 4096)
  max_seq_len:         2048
  weight_tying:        true
  rms_norm_eps:        1.0e-5
  init_std:            0.02

training:
  micro_batch_size:              16
  gradient_accumulation_steps:   2
  total_steps:                   256000          # ~8.0B tokens
  warmup_steps:                  2000
  lr:                            3.0e-4
  min_lr_ratio:                  0.05
  weight_decay:                  0.1
  beta1:                         0.9
  beta2:                         0.95
  grad_clip:                     1.0
  grad_checkpoint:               true
  compile:                       true
  compile_mode:                  "max-autotune"
  save_interval:                 4000
  log_interval:                  50
  nan_guard:                     true
  nan_guard_max_consecutive:     5
  save_dir:                      "checkpoints/pretrain_a100"

data:
  # Spec record (not read by code): tokenizer gpt2 · shard 50M tokens ·
  # 8.0B cap · mix 0.50/0.20/0.15/0.10/0.05 (fineweb-edu/fineweb/the-stack-
  # python/openmath-instruct-2/arxiv)
  train_data_path:      "data/pretrain_chinchilla"
```

## Model section → `ModelConfig`

The `model:` block is handed to the transformer whole: `main()` sets `model_config=yaml_cfg.get("model", yaml_cfg)` and `training/pretrain.py:Pretrainer.__init__` passes it to `models/transformer.py:Mamba3Transformer.__init__`, which normalizes the dict via `ModelConfig(**cfg)`. Consequences: missing keys fall back to dataclass defaults, but **unknown keys raise `TypeError`** at construction — a typo in the model section is a hard crash, not a silent no-op. All 12 keys map 1:1 to `ModelConfig` fields with identical names; none of the values here differ from the defaults, so the model built is exactly the default 433,662,400-parameter architecture (see [R1 — ModelConfig](01-model-config.md) for per-field detail).

| YAML key | `ModelConfig` field | Default if absent | Runtime effect |
|---|---|---|---|
| `vocab_size` | `vocab_size` | 50257 | Embedding rows and LM-head output width; also mirrored into `TrainingConfig.vocab_size` for the dataset. |
| `d_model` | `d_model` | 1024 | Hidden width everywhere (embed, residual, norms, head input, FFN in/out). |
| `n_layers` | `n_layers` | 28 | Number of `Mamba3Block`s; 28 × 13,649,936 params. |
| `n_heads` | `n_heads` | 16 | SSM heads H; `in_proj` slice layout and per-head complex `A`. |
| `head_dim` | `head_dim` | 64 | D, per-head token-content width; Triton `BLOCK_P`. |
| `state_dim` | `state_dim` | 64 | N, per-head complex state width; Triton `BLOCK_N`. |
| `chunk_size` | `chunk_size` | 64 | C, SSD chunk length; intra-chunk `L` is C×C per head; Triton `BLOCK_C`. |
| `ffn_dim` | `ffn_dim` | 2048 | SwiGLU intermediate width (gate/up project to `2·ffn_dim`). |
| `max_seq_len` | `max_seq_len` | 2048 | **Not** consumed by the module: dataset window size (`max_seq_len+1` tokens, x=[:-1], y=[1:]) and the logger's `seq_len`. |
| `weight_tying` | `weight_tying` | `True` | `lm_head.weight = embed.weight` (data_ptr identity) in `Mamba3Transformer.__init__`. |
| `rms_norm_eps` | `rms_norm_eps` | 1e-5 | `eps` of every RMSNorm (pre-SSD, pre-FFN, final). |
| `init_std` | `init_std` | 0.02 | Std of `N(0, init_std)` applied by `Mamba3Transformer._init_weights` to every Linear/Embedding. |

Not present, but legal: `ssd_dispatch` and `grad_checkpoint` are also `ModelConfig` fields. The canonical config sets neither under `model:` — `ssd_dispatch` therefore defaults to `"pytorch"` (the production run uses the pure-PyTorch chunkwise path unless you add the key **and** set `ENABLE_TRITON_KERNELS=1`; see `training/pretrain.py:_enforce_triton_env_var`), and `grad_checkpoint` is injected from the training section instead (below).

## Training section → `TrainingConfig`

This is the section with the translations. `main()` reads the `training:` dict `t` and constructs `TrainingConfig` field by field. Every key is consumed; the defaults in the table are the ones hard-coded in `main()`'s `.get()` calls (which differ from the dataclass defaults only in that they match the canonical config's values).

| YAML key | `TrainingConfig` field | Default if absent | Runtime effect |
|---|---|---|---|
| `micro_batch_size` | `batch_size` | 16 | Sequences per micro-batch: `DataLoader` batch and the `batch_size` fed to `TrainingLogger`. |
| `gradient_accumulation_steps` | `gradient_accumulation_steps` | 2 | Micro-batches per optimizer step; loss is divided by it in `training/pretrain.py:train_step`; `optimizer.step()` fires on the last micro-batch. |
| `total_steps` | `max_steps` | 256000 | Optimizer steps (see the arithmetic section — the `# ~8.0B tokens` comment is wrong). `--dry-run` forces 2. |
| `warmup_steps` | `warmup_steps` | 2000 | Linear warmup length; `LinearLR(start_factor=0.01)` then `SequentialLR` milestone. |
| `lr` | `lr` | 3e-4 | Peak AdamW LR after warmup. |
| `min_lr_ratio` | `min_lr_ratio` | 0.05 | `CosineAnnealingLR` floor = `lr × min_lr_ratio` = 1.5e-5. |
| `weight_decay` | `weight_decay` | 0.1 | AdamW decay for the decay group (all params with `dim() >= 2`); no-decay group gets 0.0. |
| `beta1` | `beta1` | 0.9 | AdamW first moment decay. |
| `beta2` | `beta2` | 0.95 | AdamW second moment decay. |
| `grad_clip` | `max_grad_norm` | 1.0 | `clip_grad_norm_` at each optimizer step. |
| `grad_checkpoint` | `grad_checkpoint` | `True` | Injected into the model dict via `setdefault("grad_checkpoint", …)`; `--no-checkpoint` forces False. One global boolean — **no per-4th-layer policy exists** (README's `grad_checkpoint_every 4` is stale). |
| `compile` | `compile_model` | `True` | Wraps the model in `torch.compile` (if available); `--no-compile` forces False. |
| `compile_mode` | `compile_mode` | `"max-autotune"` | **Wired**: `Pretrainer.__init__` reads `os.environ.get("TORCH_COMPILE_MODE", config.compile_mode)` — the env var wins over the YAML. |
| `save_interval` | `save_every` | 4000 | Checkpoint cadence (optimizer steps); also the worst-case NaN-rollback rewind. |
| `log_interval` | `log_every` | 50 | `TrainingLogger` window: loss/ppl/lr/tps every N steps. |
| `nan_guard` | `nan_guard` | `True` | NaN/Inf detection in `train_step`: skip backward, return `None`. |
| `nan_guard_max_consecutive` | `nan_guard_max_consecutive` | 5 | Consecutive NaN micro-batches before checkpoint rollback (or abort if none exists). |
| `save_dir` | `checkpoint_dir` | `"checkpoints/pretrain_a100"` | Where `CheckpointManager` writes the 3-file checkpoints; `--checkpoint-dir` overrides. |

## Data section → spec-only

| YAML key | Consumer | Default if absent | Runtime effect |
|---|---|---|---|
| `train_data_path` | `TrainingConfig.data_path` | `"data/pretrain_data.bin"` | Dataset root for `training/pretrain.py:PretrainDataset` — a directory of `shard_*.bin` (`sharded`), a single `torch.save` long tensor (`single`), or a missing path → `dummy` randint data with a warning. `--data-path` overrides. |

The former spec-only keys (`tokenizer`, `shard_size_tokens`, `max_tokens`, `data_mix`) were removed from the YAML in the cleanup — they had zero readers (the tokenizer is hard-wired to GPT-2 BPE; the pipeline, not `pretrain.py`, consumes shard metadata). Their values are preserved as comments above `train_data_path` so the 8.0B cap and the 0.50/0.20/0.15/0.10/0.05 mix record survive.

## The token arithmetic: 16.78B vs 8.39B vs 8.0B

`max_steps = 256000` counts **optimizer steps** — `opt_steps` increments only when `(micro_step + 1) % gradient_accumulation_steps == 0` in `training/pretrain.py:train_step`. Each optimizer step consumes:

$$256{,}000 \times \underbrace{2}_{\text{accum}} \times \underbrace{16 \times 2048}_{\text{micro-batch tokens}} = 16{,}777{,}216{,}000 \approx \textbf{16.78B token exposures.}$$

That is the honest consumption figure. The `# ~8.0B tokens` comment matches *neither* this nor the micro-batch-only reading:

$$256{,}000 \times (16 \times 2048) = 8{,}388{,}608{,}000 \approx 8.39\text{B}$$

The 8.0B figure lives only in the spec record comment (`max_tokens: 8000000000`) — no code reads it. Derived consequences: the dataset holds $(8\text{e}9 - 1) \mathbin{/} 2048 \approx 3.91\text{M}$ windows, while the run needs $256{,}000 \times 32 = 8.19\text{M}$ samples — so the data is re-read ~2.1× over the run (no shuffle: `DataLoader(..., num_workers=0, drop_last=True)`), and unique tokens seen (~8.0B) are far fewer than exposures (~16.8B). Chinchilla's 20-tokens-per-parameter rule for 433,662,400 params gives $20 \times 433{,}662{,}400 \approx 8.67\text{B}$ — so 8.0–8.4B is the intended *data size*, while the run actually spends 16.78B exposures on it. [INFERENCE] Whether the exposure/unique gap matters for convergence is an empirical question; the config's own comment is simply wrong arithmetic.

Also note the config header still says "~400M param" on its first line while the block comment below correctly says "~434M params: 28 layers × 13.65M/layer + 51.5M tied embed" — a stale comment inside the file itself; the code builds 433,662,400 parameters.

## Pitfalls

1. **The YAML is not a schema — it is a `.get()` script.** Every key has a silent default in `main()`. Removing a key never errors; it reverts to the canonical value, so "delete `compile_mode` to disable compile" does nothing (compile stays on via `compile: true`). The `model:` section is the one exception: `ModelConfig(**cfg)` raises `TypeError` on unknown keys, so *adding* a typo'd model key crashes the run.
2. **Key-name mismatches are pervasive** (`micro_batch_size` vs `batch_size`, `total_steps` vs `max_steps`, `grad_clip` vs `max_grad_norm`, `save_interval` vs `save_every`, `log_interval` vs `log_every`, `save_dir` vs `checkpoint_dir`, `compile` vs `compile_model`). A config written against `TrainingConfig` field names directly (as `tests/e2e_gpu_smoke.py:check_pretrainer_dry_run` does) bypasses the translations and still works — which makes the two spellings easy to mix up across docs and scripts.
3. **NaN rollback can rewind up to 4000 steps.** `save_interval=4000` means the newest checkpoint is at most 4000 optimizer steps old; after 5 consecutive NaN micro-batches, `Pretrainer.train` restores that checkpoint, discarding up to 4000 steps of progress (and their optimizer/scheduler state — restored from the checkpoint's `optim_step_N.pt`). If no checkpoint exists yet, it raises `RuntimeError` instead.
4. **`compile_mode` is consumed, and the env var overrides the YAML.** `TORCH_COMPILE_MODE` beats `compile_mode` in `Pretrainer.__init__`. A run script exporting `TORCH_COMPILE_MODE=default` silently ignores `"max-autotune"`.
5. **`ssd_dispatch` is absent from the canonical config.** The production run uses the PyTorch chunkwise path by default. To actually get the Triton kernel you must add `ssd_dispatch: triton` under `model:` **and** export `ENABLE_TRITON_KERNELS=1`, or `_enforce_triton_env_var` force-rewrites it back with one warn line.
6. **`vocab_size`/`max_seq_len` are duplicated.** They live in the model section but are re-read into `TrainingConfig` for the dataset. They come from the same YAML key, so they cannot diverge through the config file — but a `TrainingConfig` built in code (tests, notebooks) can carry values that disagree with its `model_config`, producing e.g. dummy data from the wrong vocab range. Keep the two in sync.
7. **The data section is documentation, not configuration.** Only `train_data_path` (and `--data-path`) reach the dataset; the other spec values (`max_tokens`, `data_mix`, …) live in comments and have zero effect on `pretrain.py`. The pipeline (see [R11](11-data-pipeline.md)) is the actual consumer of that metadata.

## Tests

No test parses `configs/pretrain_a100_400m.yaml` itself — the YAML-to-`TrainingConfig` mapping in `training/pretrain.py:main` is exercised only by the GPU-gated end-to-end smoke: `tests/e2e_gpu_smoke.py:check_pretrainer_dry_run` (check 8) drives a full `Pretrainer` for 2 steps with a hand-built `TrainingConfig` (tiny model, `max_steps=2`, `compile_model=False`) and asserts a final checkpoint exists. The nearest unit coverage of the config wiring:

- `tests/test_grad_checkpoint.py:test_grad_checkpoint_propagates_to_blocks` — the `TrainingConfig.grad_checkpoint` → `model_config` `setdefault` injection makes every block's flag True.
- `tests/test_grad_checkpoint.py:test_grad_checkpoint_explicit_false_disables` — an explicit False in the dict wins over the `setdefault`.
- `tests/test_grad_checkpoint.py:test_grad_checkpoint_actually_triggers_training_mode` — the injected flag reaches `Mamba3Block`'s `torch.utils.checkpoint` path in training mode.
- `tests/e2e_gpu_smoke.py:check_pretrainer_dry_run` — end-to-end `Pretrainer.train()` including save/find-latest checkpoint logic (GPU + triton required).

See also [R1 — ModelConfig](01-model-config.md) for the model-side fields, [R7 — Pre-training CLI](07-pretrain-cli.md) for `TrainingConfig` and the CLI flags, [R8 — Dataset](08-dataset.md) for `PretrainDataset` layouts, [R9 — Checkpoint](09-checkpoint.md) for rollback mechanics, [R10 — Logging](10-logging.md) for `log_every` semantics, [R11 — Data pipeline](11-data-pipeline.md) for the shard/mix spec, and [T8 — Scaling & efficiency](../theory/08-scaling-efficiency.md) for the memory/throughput reasoning behind these numbers.
