# Mamba-3-Lite — Pretrain CLI — TrainingConfig, flags, and the training loop

This reference documents the single entry point for Mamba-3-Lite pre-training: the `training/pretrain.py:TrainingConfig` dataclass that carries every hyper-parameter, the `training/pretrain.py:main` CLI that maps a YAML file plus command-line overrides onto it, and the `training/pretrain.py:Pretrainer` class that owns the optimizer, scheduler, `torch.compile` wiring, and the training loop.

## 1. 60-second summary

After reading this doc you will know: how a run is configured (a `configs/pretrain_a100_400m.yaml` file with `model:`, `training:`, and `data:` sections is loaded by `main()` and folded into a `TrainingConfig` dataclass, with every field overridable from the command line); how the 22 `TrainingConfig` fields map onto the YAML keys (several names differ — `batch_size` comes from `training.micro_batch_size`, `max_grad_norm` from `training.grad_clip`, `save_every` from `training.save_interval`); how `main()` flows through `Pretrainer` construction, optional `--resume <step>`, and `train()`; how the AdamW optimizer is split into decay (dim ≥ 2) and no-decay (dim < 2) parameter groups and wrapped in a `SequentialLR` warmup-then-cosine schedule (LinearLR from 0.01×lr to lr over `warmup_steps`, then CosineAnnealingLR down to `lr × min_lr_ratio`); and the three environment variables that change behavior: `ENABLE_TRITON_KERNELS`, `TORCH_COMPILE_MODE`, and `WANDB_PROJECT`/`WANDB_RUN_NAME`.

## 2. Why it exists

Everything else in this repo is a component: the model ([`Mamba3Transformer`](../references/model-reference.md)), the block ([`Mamba3Block`](../references/model-reference.md)), the SSD kernel ([`per_chunk_ssd_triton`](../references/ssd-reference.md)), the dataset ([`PretrainDataset`](../references/training-reference.md)), checkpoints ([`CheckpointManager`](../references/training-reference.md)), and logging ([`TrainingLogger`](../references/training-reference.md)). `training/pretrain.py` is where they are composed into an actual training run. It is also the only file that instantiates the full 434M-parameter model and the only CLI a user ever invokes. If you want to know *what a run does*, this file is the answer; the reference docs R8–R12 describe the objects it uses.

## 3. Signature and semantics

```python
def main() -> None: ...                                    # CLI entry point
def count_parameters(model: nn.Module) -> Tuple[int, int]: ...
@dataclass
class TrainingConfig: ...                                  # 22 hyper-parameter fields
class Pretrainer:                                          # owns model/optimizer/scheduler/loop
    def __init__(self, config: TrainingConfig): ...
    def train(self) -> None: ...
    def load_checkpoint(self, step: int) -> int: ...
    def save_checkpoint(self, step: int, tag: str = "") -> None: ...
def train_step(model, optimizer, scheduler, config, amp_context, log,
               opt_steps, tokens, targets, micro_step) -> Tuple[Optional[Dict[str, float]], int]: ...
```

| Aspect | Contract |
|---|---|
| Entry point | `python3 training/pretrain.py [--config …] [--data-path …] [--checkpoint-dir …] [--resume N] [--no-checkpoint] [--no-compile] [--dry-run]` |
| Config source | YAML file (default `configs/pretrain_a100_400m.yaml`), read by `main()`, merged with CLI overrides into a `TrainingConfig` |
| Model construction | `training/pretrain.py:Pretrainer.__init__` builds `models/transformer.py:Mamba3Transformer` on `_DEVICE` (cuda:0 if available, else CPU) |
| Counting | `training/pretrain.py:count_parameters` returns `(total, trainable)` and is logged at construction — the default build reports 433,662,400 total |
| Device | CUDA if present; otherwise a warning is printed and the run executes on CPU ("smoke-testing only") |
| Outputs | `checkpoint_dir/model_step_N.safetensors` + `optim_step_N.pt` + `meta_step_N.json` every `save_every` steps (see [`CheckpointManager`](../references/training-reference.md)) |

## 4. TrainingConfig — every field

All 22 fields, with defaults as declared in `training/pretrain.py:TrainingConfig`. The **effective default** column is what a run launched via `main()` actually gets when the key is absent from the YAML (they differ for `data_path` and `checkpoint_dir`, because `main()` supplies its own fallbacks).

| Field | Dataclass default | Effective default via `main()` | Effect |
|---|---|---|---|
| `model_config: dict` | `{}` | whole `model:` YAML section | passed to `Mamba3Transformer`; the `grad_checkpoint` key is filled in from the training flag if missing |
| `data_path: str` | `"data/pretrain_data.bin"` | `data.train_data_path` (yaml: `data/pretrain_chinchilla`) | token tensor for `PretrainDataset` (single file, sharded dir, or missing → dummy) |
| `checkpoint_dir: str` | `"checkpoints/pretrain"` | `training.save_dir` (yaml: `checkpoints/pretrain_a100`) | where `CheckpointManager` writes/reads step triplets |
| `vocab_size: int` | `50257` | `model.vocab_size` | GPT-2 BPE vocabulary size (EOS/PAD id 50,256) |
| `max_seq_len: int` | `2048` | `model.max_seq_len` | packed window length; the model sees `max_seq_len+1` tokens per window, x = `[:-1]`, y = `[1:]` |
| `batch_size: int` | `16` | `training.micro_batch_size` | micro-batch size (per optimizer step before accumulation) |
| `gradient_accumulation_steps: int` | `2` | same key | micro-batches per optimizer step; loss is divided by this before backward |
| `max_steps: int` | `256000` | `training.total_steps` (or `2` with `--dry-run`) | number of optimizer steps; the loop runs `while global_step < max_steps` |
| `warmup_steps: int` | `2000` | `training.warmup_steps` | LinearLR warmup length in optimizer steps |
| `lr: float` | `3e-4` | `training.lr` | base learning rate |
| `min_lr_ratio: float` | `0.05` | `training.min_lr_ratio` | cosine floor = `lr × min_lr_ratio` |
| `weight_decay: float` | `0.1` | `training.weight_decay` | AdamW weight decay for the decay group only |
| `beta1: float` | `0.9` | `training.beta1` | AdamW beta1 |
| `beta2: float` | `0.95` | `training.beta2` | AdamW beta2 |
| `max_grad_norm: float` | `1.0` | `training.grad_clip` | global gradient clip, applied only on optimizer steps |
| `grad_checkpoint: bool` | `True` | `training.grad_checkpoint` AND NOT `--no-checkpoint` | gradient checkpointing; propagated into `model_config` via `setdefault` |
| `compile_model: bool` | `True` | `training.compile` AND NOT `--no-compile` | whether to wrap the model in `torch.compile` |
| `compile_mode: str` | `"max-autotune"` | `training.compile_mode` (overridden by `TORCH_COMPILE_MODE` env) | `torch.compile` mode; `fullgraph=False` |
| `save_every: int` | `4000` | `training.save_interval` | checkpoint every N optimizer steps (plus a `tag="final"` one at the end) |
| `log_every: int` | `50` | `training.log_interval` | `TrainingLogger` window length |
| `nan_guard: bool` | `True` | `training.nan_guard` | skip backward on NaN/Inf loss; after `nan_guard_max_consecutive` strikes, restore latest checkpoint |
| `nan_guard_max_consecutive: int` | `5` | same key | consecutive NaN/Inf count that triggers checkpoint rollback |

## 5. CLI flags and the YAML → config mapping

```python
parser.add_argument("--config", type=str, default="configs/pretrain_a100_400m.yaml")
parser.add_argument("--data-path", type=str, default=None)
parser.add_argument("--checkpoint-dir", type=str, default=None)
parser.add_argument("--resume", type=str, default=None, help="Checkpoint step number to resume from")
parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing")
parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
parser.add_argument("--dry-run", action="store_true", help="Run 2 steps to verify wiring")
```

| Flag | Semantics |
|---|---|
| `--config PATH` | YAML file to load; default `configs/pretrain_a100_400m.yaml` |
| `--data-path PATH` | overrides `data.train_data_path` |
| `--checkpoint-dir DIR` | overrides `training.save_dir` |
| `--resume N` | `main()` calls `Pretrainer.load_checkpoint(int(N))` *before* `train()`; note it is `--resume`, **not** `--resume-from` (older README text used the wrong flag) |
| `--no-checkpoint` | forces `grad_checkpoint=False` — it disables **gradient** checkpointing, not disk checkpoints |
| `--no-compile` | forces `compile_model=False` |
| `--dry-run` | sets `max_steps=2`: two micro-batches, i.e. one optimizer step at the default accumulation of 2, then the loop ends |

Mapping table (`t` = `yaml_cfg["training"]`, `d` = `yaml_cfg["data"]`, `m` = `yaml_cfg.get("model", yaml_cfg)` — note the fallback: if there is no `model:` section, the *whole YAML dict* becomes the model config):

| YAML key | TrainingConfig field |
|---|---|
| `model:` (entire section) | `model_config` |
| `model.max_seq_len` | `max_seq_len` |
| `model.vocab_size` | `vocab_size` |
| `data.train_data_path` | `data_path` |
| `training.save_dir` | `checkpoint_dir` |
| `training.micro_batch_size` | `batch_size` |
| `training.gradient_accumulation_steps` | `gradient_accumulation_steps` |
| `training.total_steps` | `max_steps` |
| `training.warmup_steps` | `warmup_steps` |
| `training.lr` | `lr` |
| `training.min_lr_ratio` | `min_lr_ratio` |
| `training.weight_decay` | `weight_decay` |
| `training.beta1` / `training.beta2` | `beta1` / `beta2` |
| `training.grad_clip` | `max_grad_norm` |
| `training.grad_checkpoint` | `grad_checkpoint` |
| `training.compile` | `compile_model` |
| `training.compile_mode` | `compile_mode` |
| `training.save_interval` | `save_every` |
| `training.log_interval` | `log_every` |
| `training.nan_guard` | `nan_guard` |
| `training.nan_guard_max_consecutive` | `nan_guard_max_consecutive` |

Keys that are **not** read by `main()`: everything else under `data:` (`tokenizer`, `shard_size_tokens`, `max_tokens`, `data_mix`) — the data-mix proportions (fineweb-edu 0.50 / fineweb 0.20 / the-stack-python 0.15 / openmath-instruct-2 0.10 / arxiv 0.05) are documentation only, applied by the workspace-level `LLM/shared_data/` pipeline that produces the token shards (see `data/prepare_data.py`).

## 6. The `main()` flow

```
parse args → yaml.safe_load(args.config)
           → t = yaml["training"], d = yaml["data"]
           → TrainingConfig(model_config=…, …every field…)
           → Pretrainer(config)          # builds model, optimizer, scheduler, compile
           → if --resume N: load_checkpoint(N)
           → train()
```

The construction call in `main()`:

```python
config = TrainingConfig(
    model_config=yaml_cfg.get("model", yaml_cfg),
    data_path=args.data_path or d.get("train_data_path", "data/pretrain_data.bin"),
    checkpoint_dir=args.checkpoint_dir or t.get("save_dir", "checkpoints/pretrain_a100"),
    max_seq_len=yaml_cfg.get("model", yaml_cfg).get("max_seq_len", 2048),
    vocab_size=yaml_cfg.get("model", yaml_cfg).get("vocab_size", 50257),
    batch_size=t.get("micro_batch_size", 16),
    ...
    max_steps=2 if args.dry_run else t.get("total_steps", 256000),
    grad_checkpoint=t.get("grad_checkpoint", True) and not args.no_checkpoint,
    compile_model=t.get("compile", True) and not args.no_compile,
    ...
)
```

The loop lives in `training/pretrain.py:Pretrainer.train`: it builds the `PretrainDataset` + `DataLoader` (batch size `config.batch_size`, `num_workers=0`, `drop_last=True`), then **auto-resumes**: `_find_latest_checkpoint()` queries `CheckpointManager.latest_step()` and loads it if one exists. It then iterates `while global_step < max_steps`, calling `train_step` per micro-batch: forward under BF16 autocast, cross-entropy with `ignore_index=-100`, loss divided by `gradient_accumulation_steps`, NaN guard check, backward, and — only every `gradient_accumulation_steps`-th micro-batch — `clip_grad_norm_`, `optimizer.step()`, `scheduler.step()`. Logging goes to `TrainingLogger` every `log_every` steps and checkpoints are written every `save_every` steps; a final checkpoint with `tag="final"` is saved when the loop exits.

## 7. Optimizer and scheduler construction

`training/pretrain.py:Pretrainer.__init__` splits parameters into two AdamW groups:

```python
decay_params = [p for p in all_params if p.dim() >= 2]
no_decay_params = [p for p in all_params if p.dim() < 2]
self.optimizer = AdamW([
    {"params": decay_params, "weight_decay": config.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=config.lr, betas=(config.beta1, config.beta2), fused=False)
```

Parameters are first deduplicated by `id(p)` — under weight tying, `lm_head.weight` *is* `embed.weight` (same `data_ptr`), so the shared tensor is optimized exactly once. Dim ≥ 2 (matrices, embeddings) gets `weight_decay=0.1`; dim < 2 (biases, norms, the complex per-head `A` scalar) gets none. `fused=False` keeps the optimizer portable (CPU and non-fused-CUDA paths).

The schedule is a `SequentialLR`:

```python
warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
cosine = CosineAnnealingLR(self.optimizer, T_max=config.max_steps - config.warmup_steps,
                           eta_min=config.lr * config.min_lr_ratio)
self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps])
```

The learning rate over optimizer step $s$ is

$$\eta(s) = \begin{cases} \eta_{\max}\left(0.01 + 0.99\,\dfrac{s}{w_{\text{warm}}}\right) & 0 \le s \le w_{\text{warm}} \\[6pt] \eta_{\min} + \left(\eta_{\max}-\eta_{\min}\right)\dfrac{1+\cos\left(\pi\,\dfrac{s-w_{\text{warm}}}{S_{\max}-w_{\text{warm}}}\right)}{2} & w_{\text{warm}} < s \le S_{\max} \end{cases}$$

with $\eta_{\max}=\text{lr}=3\times10^{-4}$, $\eta_{\min}=\text{lr}\times\text{min\_lr\_ratio}=1.5\times10^{-5}$, $w_{\text{warm}}=2000$, $S_{\max}=256000$. Verified empirically against torch 2.12 with `warmup_steps=3`: LR at steps 0/1/2/3 is $0.01\times$, $0.34\times$, $0.67\times$, $1.0\times\eta_{\max}$ — linear interpolation hitting exactly $\eta_{\max}$ at the milestone — and the cosine is then re-initialized to its epoch-0 value (also $\eta_{\max}$), so the transition is continuous with no spike. The scheduler steps **only on optimizer steps** (every `gradient_accumulation_steps` micro-batches), so "step" above means optimizer step.

## 8. `torch.compile` wiring and environment variables

```python
if config.compile_model and hasattr(torch, "compile"):
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", config.compile_mode)
    training_model = torch.compile(training_model, mode=compile_mode, fullgraph=False)
```

- `compile_model` defaults to `True`; the YAML `training.compile` and `--no-compile` both gate it. `fullgraph=False` is hard-coded. `self.model` is the compiled module; `self.raw_model` is kept for checkpoints (state dicts come from the raw model, so compiled artifacts never leak into serialized weights).
- **`ENABLE_TRITON_KERNELS`** (default `"0"`): consumed by `training/pretrain.py:_enforce_triton_env_var` — if the model config requests `ssd_dispatch="triton"` but the env var is not `"1"`, dispatch is force-backed to `"pytorch"` with a warning. The default dispatch is `"pytorch"` anyway; set `ENABLE_TRITON_KERNELS=1` and `ssd_dispatch: triton` in the model config to use the Triton per-chunk kernel (see [03-ssd-triton.md](../references/ssd-reference.md)).
- **`TORCH_COMPILE_MODE`**: overrides `training.compile_mode` (default `"max-autotune"`).
- **`WANDB_PROJECT`** / **`WANDB_RUN_NAME`**: consumed by `utils/logging.py:TrainingLogger` — if `WANDB_PROJECT` is set, `wandb.init(project=…, name=WANDB_RUN_NAME, reinit=True)` runs (silently skipped if `wandb` is not installed).

## 9. Pitfalls

1. **`warmup_steps=0` does not raise, it silently starves the run.** Verified empirically: with `milestones=[0]`, `SequentialLR` hands control to the cosine on the very first optimizer step, but the LinearLR's epoch-0 branch has already pinned the optimizer LR to `start_factor × lr` (3e-6). The cosine's first update then runs its iterative formula from that state, and the LR converges *upward* to $\eta_{\min}$ over the whole run — the model trains at roughly 1%→5% of the configured LR and never reaches $\eta_{\max}$. Keep `warmup_steps ≥ 1`.
2. **`--resume` is `--resume`, not `--resume-from`.** The README historically documented the wrong flag. Additionally, `--resume N` is partially redundant: `train()` *also* auto-resumes from `CheckpointManager.latest_step()`. If the checkpoint directory contains a newer complete triplet than step N, the explicit `load_checkpoint(N)` in `main()` is overwritten by the auto-resume. To resume a specific older step, point `--checkpoint-dir` at a directory whose latest complete checkpoint *is* that step.
3. **`--no-checkpoint` disables gradient checkpointing, not disk checkpoints.** There is no CLI switch to suppress disk checkpoints; `save_every` (or a huge value) is the only lever.
4. **Name mismatches between YAML and config.** `batch_size` ← `micro_batch_size`; `max_grad_norm` ← `grad_clip`; `save_every` ← `save_interval`; `log_every` ← `log_interval`; `max_steps` ← `total_steps`. Copying a config between projects with the wrong key names silently falls back to defaults.
5. **Token-count arithmetic vs. the YAML comment.** With the default config, tokens per optimizer step are $16 \times 2 \times 2048 = 65{,}536$, and $256{,}000$ steps × 65,536 = **≈16.8B tokens** — not the "~8.0B Chinchilla-optimal tokens" claimed in the YAML header comment (which matches `micro_batch_size=8`). The YAML comment is stale; the code arithmetic is authoritative.
6. **Loss scaling.** The loss is divided by `gradient_accumulation_steps` before backward, but `ce_loss_val` is logged *before* the division, so logged loss is the per-micro-batch CE, not the accumulated mean.
7. **NaN guard rewinds.** On `nan_guard_max_consecutive` strikes, the latest checkpoint is reloaded, which resets `global_step`, the optimizer, *and* the scheduler state — the LR schedule restarts from the checkpointed step (scheduler state is saved in the meta). With no checkpoint available it raises `RuntimeError`.
8. **CPU runs.** Without CUDA, `amp_dtype` falls back to float32 and autocast is disabled; `torch.compile` still applies (mode `max-autotune`) unless `--no-compile`. CPU is for smoke-testing only.

## 10. Tests

- `tests/test_train_step.py::test_train_step_on_tiny_model` — drives the module-level `training/pretrain.py:train_step` on a tiny model: asserts a finite loss and that parameters changed (guards the accumulation/step plumbing and the NaN guard's `None` return).
- `tests/test_ssd_triton.py::TestEnableTritonKernelsForceBack` — `test_triton_dispatch_forced_back_when_env_var_missing` asserts `_enforce_triton_env_var` rewrites `ssd_dispatch` to `"pytorch"` when `ENABLE_TRITON_KERNELS` is unset, and `test_triton_dispatch_passes_through_when_env_var_set` asserts it stays `"triton"` when the env var is `"1"`.
- End-to-end (CUDA + Triton only): `tests/e2e_gpu_smoke.py` runs a short real training loop; on a CPU box the suite is 37 tests collected (32 passed / 5 GPU-skipped).

## References

- [Mamba-3-Lite — Config Reference](../references/config-reference.md) — the annotated YAML behind `TrainingConfig`.
- [Mamba-3-Lite — Training Reference](../references/training-reference.md) — `training/pretrain.py:PretrainDataset`, `utils/checkpoint.py:CheckpointManager`, `utils/logging.py:TrainingLogger`.
- [Mamba-3-Lite — Training Runbook](training-runbook.md) — operational launch, monitoring, and recovery.
- [Mamba-3-Lite — Tuning Guide](tuning.md) — measuring the schedule and batch knobs.
- [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) — the NaN guard and precision placement.
