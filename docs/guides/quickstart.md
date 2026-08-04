# Mamba-3-Lite — Quickstart — run Mamba-3-Lite in 10 minutes

This guide walks a fresh clone from zero to a running training loop: install, verify the math, smoke-test the model, dry-run the trainer, and launch a real pre-training run with resume support.

## 1. 60-second summary

After this guide you will have:

- A working Python environment with Mamba-3-Lite installed and its 37-test suite passing (32 passed, 5 GPU-gated skips on CPU).
- Proof that the complex chunkwise SSD matches the naive scan oracle, and that the model forward pass produces the expected shape.
- A verified `--dry-run` training pass that writes real checkpoint files.
- The exact commands and environment variables for a full pre-training run and for resuming from a saved step.

Mamba-3-Lite is a from-scratch PyTorch reproduction of Mamba-3 at ~434M parameters: 28 Mamba blocks built from a **complex-valued chunkwise SSD** (state `N=64`, `complex64`, chunk 64) plus a **MIMO inter-head mixer**, with no causal convolution and no `mamba-ssm` dependency. The recurrence it computes per layer is

$$h_t = \exp\!\big(\text{softplus}(dt_t)\, A\big)\, h_{t-1} + B_t\, x_t, \qquad y_t = C_t\, h_t,$$

with $A, B, C, h \in \mathbb{C}$. Everything runs on CPU; a GPU only speeds it up.

## 2. Prerequisites

- **Python 3.10+** (`python3 --version`).
- **PyTorch ≥ 2.1** with `torch.compile` available. Check:

```bash
python3 -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

- **A working CPU.** Every command in this guide runs on CPU/Mac. A CUDA GPU is optional and only used for real training (section 8) — the code paths are identical, just faster.
- ~3 GB free disk for the repo plus whatever `pip` installs.

No CUDA toolchain, no Triton, and no `mamba-ssm` are required. The default SSD dispatch is pure PyTorch (`models/ssd_complex.py:ssd_complex_chunkwise`); the opt-in Triton kernel (`models/ssd_triton.py:per_chunk_ssd_triton`) is only compiled when you explicitly enable it.

## 3. Install

```bash
git clone https://github.com/atandra2000/Mamba-3-Lite.git
cd Mamba-3-Lite
pip install -r requirements.txt
```

`requirements.txt` installs exactly:

| Package | Pin | Why |
|---|---|---|
| `torch` | `>=2.1` | the only deep-learning dependency; CPU, MPS, and CUDA builds all work |
| `safetensors` | `>=0.4` | checkpoint format (no pickle) |
| `pyyaml` | `>=6.0` | config files |
| `tqdm` | `>=4.65` | training progress bars |
| `wandb` | `>=0.16` | optional experiment tracking (only used if `WANDB_PROJECT` is set) |
| `pytest` | `>=7.0` | the test suite |

The `mamba-ssm` / `causal-conv1d` lines are commented out — this repo deliberately does not use them. If you want WandB later, `pip install wandb` (it is already in the file, so the install above covers it).

## 4. Verify the math: run the test suite

```bash
python3 -m pytest tests/ -v
```

Expected output on a CPU box: **37 tests collected, 32 passed, 5 skipped**. `pytest.ini` already adds `-v` and `--tb=short`, and restricts collection to `tests/`. The 5 skips are the `TestPerChunkSsdKernelGPU` class in `tests/test_ssd_triton.py`, gated by `skipif(not (HAS_TRITON and torch.cuda.is_available()))` — they need a CUDA GPU plus Triton and are not runnable here. The four tests that carry the correctness story:

- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` — the headline check. It feeds identical random complex inputs to `models/ssd_complex.py:ssd_complex_chunkwise` (chunked, parallel) and `models/ssd_complex.py:ssd_naive_complex` (the O(T) sequential scan oracle) and asserts `allclose` to `atol=1e-4`. This is the proof that the chunkwise algorithm computes the same recurrence as the naive scan. `tests/test_ssd.py::test_chunkwise_matches_naive_time_varying_dt` is a regression companion that uses input-dependent `dt`, so the inter-chunk decay factors actually differ between chunks.
- `tests/test_mimo.py::test_mimo_identity_survives_transformer_init` — verifies the MIMO mixer's eye initialization survives `models/transformer.py:Mamba3Transformer._init_weights`: the mixer weight in a full model is still exactly the identity, so at init the MIMO layer is a no-op.
- `tests/test_grad_checkpoint.py::test_grad_checkpoint_actually_triggers_training_mode` — builds a model with gradient checkpointing enabled, runs a real backward pass through the checkpointed blocks, and asserts every parameter received a finite gradient.
- `tests/test_train_step.py::test_train_step_on_tiny_model` — runs one real forward + backward + optimizer step through `training/pretrain.py:train_step` on a tiny model with dummy tokens, and asserts the loss is finite and parameters actually changed.

Other groups worth knowing: the chunkwise edge cases (uneven `T`, `T == chunk_size`), the Triton host-wrapper reference and dispatch guards, the two CPU autograd-gradcheck tests for the kernel's backward plumbing, and `tests/test_doc_refs.py::test_doc_refs_all_anchors_resolve` which machine-checks that every code citation in `docs/` resolves to a real symbol.

## 5. Check docs↔code alignment

```bash
python3 tests/test_doc_refs.py
```

Expected output ends with `[doc-refs] resolution: PASS` and exit code 0. The script scans every markdown file under `docs/` (excluding the plan document itself), extracts code citations of the form `path/to/file.py` followed by a colon and a symbol name, imports the target modules, and verifies each symbol exists. This is the harness's way of keeping the docs honest — if a doc cites a renamed function, this command fails. Add `--coverage` to also require that every public function in the coverage modules is cited somewhere in `docs/`.

## 6. Quick model smoke

This snippet mirrors `tests/test_transformer.py::test_mamba3_transformer_forward` with a tiny config — no GPU, no data, a few seconds on any laptop. It builds a `models/transformer.py:Mamba3Transformer` from a `models/transformer.py:ModelConfig` and runs one forward pass:

```bash
python3 -c "
import torch
from models.transformer import Mamba3Transformer, ModelConfig

cfg = ModelConfig(
    vocab_size=100, d_model=64, n_layers=2, n_heads=4,
    head_dim=16, state_dim=8, chunk_size=4, ffn_dim=128,
    max_seq_len=32, weight_tying=True,
)
m = Mamba3Transformer(cfg)
x = torch.randint(0, 100, (2, 16))
y = m(x)
assert y.shape == (2, 16, 100), y.shape
assert m.embed.weight.data_ptr() == m.lm_head.weight.data_ptr(), 'weight tying not applied'
print('forward OK:', tuple(y.shape))
"
```

Expected output: `forward OK: (2, 16, 100)` — batch 2, sequence 16, vocab logits 100. The second assert confirms the embedding and the output head are the same tensor (weight tying), one of the two claims that make the parameter count 433,662,400 (~434M) rather than ~485M. `Mamba3Transformer` accepts either a `ModelConfig` or a plain dict, so you can also pass `model:` sections straight from a YAML config.

## 7. Dry-run the trainer

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_400m.yaml \
    --dry-run --no-compile
```

What this does, step by step:

1. `training/pretrain.py:main` parses the flags, loads the YAML, and builds a `TrainingConfig` with `max_steps=2` (that is the entire meaning of `--dry-run`) and compilation disabled by `--no-compile`.
2. `training/pretrain.py:Pretrainer.__init__` prints the parameter count (`Parameters: 433,662,400 total / 433,662,400 trainable`), warns that CUDA is not available if you are on CPU, and sets up the AdamW weight-decay groups and the warmup→cosine scheduler.
3. `PretrainDataset` looks for the data path (`data/pretrain_chinchilla`). In a fresh clone that directory does not exist, so it prints `[warn] Pre-training data not found: ... Using dummy data for testing.` and falls back to random token windows (`training/pretrain.py:PretrainDataset` layout `dummy`). No data download is needed.
4. The loop runs **two forward/backward micro-steps** at the real `micro_batch_size` (16) with gradient accumulation (so exactly one optimizer step), gradient clipping, and the NaN guard all active, then the final `save_checkpoint` writes the "final" checkpoint tagged step 2.

Expected end state: `checkpoints/pretrain_a100/` contains the three files every checkpoint consists of — `model_step_2.safetensors`, `optim_step_2.pt`, `meta_step_2.json` — written by `utils/checkpoint.py:CheckpointManager` (the "atomic" docstring notwithstanding, the files are written directly; crash tolerance comes from the all-three-files completeness check — see the honesty note in [Mamba-3-Lite — Training Reference](../references/training-reference.md)). That directory is the contract for `--resume`.

One honest caveat: the dry-run builds the **full 434M-parameter model**, so on a laptop CPU the two steps take a few minutes; on an A100 it is seconds. That is expected and fine — it is the only command in this guide that instantiates the production-scale model.

## 8. Launch a real training run + resume

On a CUDA GPU with the prepared dataset:

```bash
python3 training/pretrain.py --config configs/pretrain_a100_400m.yaml
```

This trains for the full 256,000 optimizer steps (65,536 tokens per step — 16 micro-batch × 2 accumulation × 2048 sequence — so ~16.8B token exposures over the run; the 8.0B figure in the yaml header comment is the corpus size, not the exposure count, see [Mamba-3-Lite — Config Reference](../references/config-reference.md)), saving a checkpoint every 4,000 steps to `checkpoints/pretrain_a100/`. If `data/pretrain_chinchilla` is missing you will get the same dummy-data warning as in the dry run — point the trainer at real data with `--data-path <path>` (see [Mamba-3-Lite — Training](../training.md) for the shard layout it expects).

To resume from an existing checkpoint (e.g. step 80,000):

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_400m.yaml \
    --resume 80000
```

`--resume <step>` calls `utils/checkpoint.py:CheckpointManager.load` with `strict=False`; it requires all three files for that step (`model_step_80000.safetensors`, `optim_step_80000.pt`, `meta_step_80000.json`), restores model, optimizer, and scheduler state, and continues from that step. If no `--resume` is given, the trainer instead auto-resumes the latest **complete** checkpoint found in the checkpoint directory.

Other useful flags: `--no-checkpoint` disables **gradient** checkpointing (it is on by default as one global boolean covering all 28 blocks — there is no `grad_checkpoint_every` cadence), and `--data-path` / `--checkpoint-dir` override the YAML's data and save locations.

### Environment variables

| Variable | Effect |
|---|---|
| `ENABLE_TRITON_KERNELS=1` | Required to actually use the Triton dispatch (`ssd_dispatch='triton'`). If it is not `"1"`, `training/pretrain.py:_enforce_triton_env_var` force-backs the config to `pytorch` with a warning. Leave it unset unless you are on CUDA with Triton installed — the PyTorch path is the default and is numerically equivalent (verified by the parity tests). |
| `TORCH_COMPILE_MODE` | Overrides the YAML's `compile_mode` (default `max-autotune`) for `torch.compile`. Example: `TORCH_COMPILE_MODE=reduce-overhead python3 training/pretrain.py ...`. |
| `WANDB_PROJECT` | Enables WandB logging. If set, `utils/logging.py:TrainingLogger` calls `wandb.init(project=...)` (with optional `WANDB_RUN_NAME`) and logs loss/ppl/lr/tokens-per-sec every `log_interval` steps. Unset, it prints to console only. |

## 9. Where to go next

- **`docs/README.md`** — the full docs map; every concept, reference, and guide file indexed by topic.
- **`docs/concepts/ssd-theory.md`** — the math behind `ssd_complex_chunkwise`: the intra-chunk recurrence matrix, per-chunk state accumulation, and inter-chunk decay propagation. Read this before touching the SSD code.
- **`docs/guides/training-runbook.md`** — the complete runbook for a production pre-training run: data preparation, launch scripts, monitoring, and failure recovery.
- **`docs/guides/pretrain-cli.md`** — every CLI flag of `training/pretrain.py:main` with defaults and examples.
- **`docs/references/training-reference.md`** — the checkpoint file format and the resume/rollback semantics in detail.

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` (or pytest cannot collect any tests) | `pip install -r requirements.txt` ran in a different environment than `python3` | Check `which python3` and `python3 -m pip --version`; reinstall into the same interpreter (`python3 -m pip install -r requirements.txt`). Verify with the section-2 version check. |
| `ImportError` about `triton` when building a model | You set `ssd_dispatch='triton'` on a machine without Triton (standard on macOS / Windows) | Don't. The default is `pytorch`; `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` also falls back to the PyTorch path with a one-shot warning per model instance, so a stray `ssd_dispatch` never hard-crashes training. The Triton kernel is opt-in and GPU-only (`docs/references/ssd-reference.md`). |
| `CUDA out of memory` during real training | Batch × sequence × model footprint exceeds VRAM | Lower `micro_batch_size` in the YAML (the accumulation step count compensates), or reduce `max_seq_len`. Keep gradient checkpointing on (`--no-checkpoint` is a foot-gun). Disable `torch.compile` with `--no-compile` if the compiler's memory overhead is the issue. |
| `[warn] Pre-training data not found` and loss is garbage | Expected in a fresh clone — dummy random tokens carry no signal | That warning is by design for smoke tests. Point `--data-path` at real data (or prepare it via `docs/guides/training-runbook.md`) before judging loss curves. |
| `--resume 80000` errors about a missing checkpoint | One of the three step files is missing or the step was never saved | List `checkpoints/pretrain_a100/`; `utils/checkpoint.py:CheckpointManager.latest_step` only considers steps where all three files exist. Resume a step that does. |
| Tests unexpectedly slow | The 434M dry-run model on CPU is inherently minutes; unit tests use tiny configs | Confirm you are running `pytest tests/` (tiny models, ~seconds), not the dry-run. The dry-run's two steps are the only heavyweight commands. |

## References

- [Mamba-3-Lite — SSD Foundations](../concepts/state-space-foundations.md) — the recurrence the quickstart's model smoke exercises.
- [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md) — the math behind `models/ssd_complex.py:ssd_complex_chunkwise`; read before touching SSD code.
- [Mamba-3-Lite — Pretrain CLI](pretrain-cli.md) — every `training/pretrain.py:main` flag with defaults.
- [Mamba-3-Lite — Training Reference](../references/training-reference.md) — dataset layouts, checkpoint format, logger semantics.
- [Mamba-3-Lite — Training](../training.md) — preparing the real data the runbook needs.
