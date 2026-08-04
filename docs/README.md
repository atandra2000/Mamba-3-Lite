# Mamba-3-Lite Docs

> The full documentation tree for the ~434M-param, pure-PyTorch Mamba-3
> reproduction with complex-valued SSD state spaces. Everything is
> **symbol-anchored to the code** and machine-checked by
> `tests/test_doc_refs.py` (stale docs fail CI).

## Reading paths

| You want to… | Start here |
|---|---|
| Understand the SSD math from zero | `theory/01-ssm-foundations.md` → `02-state-space-duality.md` → `03-complex-ssd.md` → `04-chunkwise-algorithm.md` |
| Understand one model component | `theory/05-mimo-mixing.md`, `06-block-anatomy.md`, `07-numerical-stability.md`, `08-scaling-efficiency.md` |
| Look up a function/class contract | `reference/01`–`12` (one doc per module) |
| Launch or operate a training run | `guides/01-quickstart.md`, `guides/02-training-runbook.md` |
| Tune or extend the repo | `guides/03-tuning.md`, `guides/04-extending.md` |

## Theory (`theory/` — from-scratch, concept-building)

| Doc | Topic |
|---|---|
| [`01-ssm-foundations.md`](theory/01-ssm-foundations.md) | From RNNs to state space models: the linear recurrence, ZOH discretization, diagonal A, S4 → S6 → Mamba-2 → Mamba-3 arc |
| [`02-state-space-duality.md`](theory/02-state-space-duality.md) | The SSD theorem: chunking, the causal segment matrix L, the linear-attention link |
| [`03-complex-ssd.md`](theory/03-complex-ssd.md) | Mamba-3's complex states: decay + rotation, the N-halving packing argument, complex gradients |
| [`04-chunkwise-algorithm.md`](theory/04-chunkwise-algorithm.md) | **The authoritative derivation**: the chunkwise complex SSD, einsum-by-einsum, with a worked 2×2 example |
| [`05-mimo-mixing.md`](theory/05-mimo-mixing.md) | SISO → MIMO head mixing: why, the math, identity init |
| [`06-block-anatomy.md`](theory/06-block-anatomy.md) | The residual block: in_proj packing, A parameterization, no causal conv, RMSNorm, SwiGLU, grad checkpointing |
| [`07-numerical-stability.md`](theory/07-numerical-stability.md) | Why every dtype/precision choice: BF16, TF32, complex64 state, FP32 gating, the NaN guard |
| [`08-scaling-efficiency.md`](theory/08-scaling-efficiency.md) | Param count derived (433,662,400), Chinchilla budget, memory, throughput levers |

## Reference (`reference/` — symbol-anchored API docs)

| Doc | Module |
|---|---|
| [`01-model-config.md`](reference/01-model-config.md) | `models/transformer.py:ModelConfig` — every field |
| [`02-ssd-complex.md`](reference/02-ssd-complex.md) | `models/ssd_complex.py` — `ssd_complex_chunkwise`, `ssd_naive_complex` |
| [`03-ssd-triton.md`](reference/03-ssd-triton.md) | `models/ssd_triton.py` — kernel API, autograd contract, env knobs |
| [`04-transformer.md`](reference/04-transformer.md) | `models/transformer.py` — `Mamba3Transformer` |
| [`05-mamba-block.md`](reference/05-mamba-block.md) | `models/mamba_block.py` — `Mamba3Block` + dispatch |
| [`06-mimo.md`](reference/06-mimo.md) | `models/mimo.py` — `MIMO` |
| [`07-pretrain-cli.md`](reference/07-pretrain-cli.md) | `training/pretrain.py` — `TrainingConfig`, CLI, `main()` |
| [`08-dataset.md`](reference/08-dataset.md) | `training/pretrain.py:PretrainDataset` — layouts |
| [`09-checkpoint.md`](reference/09-checkpoint.md) | `utils/checkpoint.py` — `CheckpointManager` |
| [`10-logging.md`](reference/10-logging.md) | `utils/logging.py` — `TrainingLogger` |
| [`11-data-pipeline.md`](reference/11-data-pipeline.md) | `data/prepare_data.py` — the shim + pipeline |
| [`12-config-reference.md`](reference/12-config-reference.md) | `configs/pretrain_a100_400m.yaml` — annotated, field by field |

## Guides (`guides/` — task-oriented)

| Doc | Task |
|---|---|
| [`01-quickstart.md`](guides/01-quickstart.md) | Install → verify → dry-run → train → resume |
| [`02-training-runbook.md`](guides/02-training-runbook.md) | The A100 run: pre-flight, monitoring, NaN recovery, resume |
| [`03-tuning.md`](guides/03-tuning.md) | chunk_size, lr, batch geometry, compile, Triton dispatch — measure, don't guess |
| [`04-extending.md`](guides/04-extending.md) | Add an SSM variant; add a sanctioned Triton kernel |

## Doc rules

- **Anchor style:** docs cite code only as `file.py:Symbol` / `file.py:Class.method` — never line numbers.
- **Alignment gate:** `python3 tests/test_doc_refs.py` parses every anchor and fails on unknown files/symbols; `python3 tests/test_doc_refs.py --coverage` also requires every public symbol to be cited. Run both before committing doc changes.
- **Code map:** `python3 scripts/generate_code_map.py` regenerates `CODE_MAP.md` from the docs' own citations.
- **Expansion plan:** the original audit + tree + writing contract live in [`docs_expansion_plan.md`](docs_expansion_plan.md).
