# Mamba-3-Lite — Documentation Index

> The full documentation tree for the ~434M-param, pure-PyTorch Mamba-3
> reproduction with complex-valued SSD state spaces. Everything is
> **symbol-anchored to the code** and machine-checked by
> `tests/test_doc_refs.py` (`--coverage --links` in CI).

## Reading paths

| You want to… | Start here |
|---|---|
| Understand the SSD math from zero | `concepts/state-space-foundations.md` → `concepts/ssd-theory.md` |
| Understand one model component | `concepts/mimo.md`, `concepts/block-and-stability.md` |
| Look up a function/class contract | `references/` (consolidated API docs, below) |
| Launch or operate a training run | `guides/quickstart.md`, `guides/training-runbook.md` |
| Tune or extend the repo | `guides/tuning.md`, `guides/extending.md` |
| The data path end to end | `training.md` |

## Concepts (`concepts/` — from-scratch, concept-building)

| Doc | Topic |
|---|---|
| [`state-space-foundations.md`](concepts/state-space-foundations.md) | From RNNs to state space models: the linear recurrence, ZOH discretization, diagonal A, S4 → S6 → Mamba-2 → Mamba-3 arc, and `models/ssd_complex.py:ssd_naive_complex` as the O(T) oracle |
| [`ssd-theory.md`](concepts/ssd-theory.md) | **The authoritative derivation cluster**: the SSD theorem (chunking, the causal segment matrix L, the linear-attention link), Mamba-3's complex states (decay + rotation, the N-halving packing argument, complex gradients), and the chunkwise complex SSD einsum-by-einsum with a worked 2×2 example |
| [`mimo.md`](concepts/mimo.md) | SISO → MIMO head mixing: why, the math, identity init |
| [`block-and-stability.md`](concepts/block-and-stability.md) | The residual block (in_proj packing, A parameterization, no causal conv, RMSNorm, SwiGLU, grad checkpointing), why every dtype/precision choice exists (BF16, TF32, complex64, FP32 gating, the NaN guard), and the derived scaling/memory/throughput numbers (433,662,400 params, Chinchilla budget, chunk_size memory) |

## References (`references/` — symbol-anchored API docs)

| Doc | Module |
|---|---|
| [`config-reference.md`](references/config-reference.md) | `models/transformer.py:ModelConfig` — every field — plus the annotated `configs/pretrain_a100_400m.yaml`, key-name translations, and the 16.78B-vs-8.0B token arithmetic |
| [`ssd-reference.md`](references/ssd-reference.md) | `models/ssd_complex.py` — `ssd_complex_chunkwise`, `ssd_naive_complex` — and `models/ssd_triton.py` — kernel API, autograd contract, env knobs |
| [`model-reference.md`](references/model-reference.md) | `models/transformer.py:Mamba3Transformer`, `models/mamba_block.py:Mamba3Block` + dispatch, `models/mimo.py:MIMO` |
| [`training-reference.md`](references/training-reference.md) | `training/pretrain.py:PretrainDataset`, `utils/checkpoint.py:CheckpointManager`, `utils/logging.py:TrainingLogger`, `data/prepare_data.py` |

## Guides (`guides/` — task-oriented)

| Doc | Task |
|---|---|
| [`quickstart.md`](guides/quickstart.md) | Install → verify → dry-run → train → resume |
| [`training-runbook.md`](guides/training-runbook.md) | The A100 run: pre-flight, monitoring, NaN recovery, resume |
| [`tuning.md`](guides/tuning.md) | chunk_size, lr, batch geometry, compile, Triton dispatch — measure, don't guess |
| [`extending.md`](guides/extending.md) | Add an SSM variant; add a sanctioned Triton kernel |
| [`pretrain-cli.md`](guides/pretrain-cli.md) | `training/pretrain.py:TrainingConfig`, CLI flags, `main()`, optimizer/scheduler wiring |

## Training pipeline

| Doc | Topic |
|---|---|
| [`training.md`](training.md) | The data path end to end: corpus mix, shard format, `data/prepare_data.py` shim, `training/pretrain.py:PretrainDataset` layouts, and how the training loop consumes them |

## Code → doc map

Every public symbol in the coverage modules is cited somewhere in this tree
(enforced by `tests/test_doc_refs.py --coverage`). The map below is the
reverse index: which doc documents which file.

| Source file | Documented in |
|---|---|
| `models/transformer.py` | [`config-reference.md`](references/config-reference.md), [`model-reference.md`](references/model-reference.md), [`block-and-stability.md`](concepts/block-and-stability.md) |
| `models/mamba_block.py` | [`model-reference.md`](references/model-reference.md), [`block-and-stability.md`](concepts/block-and-stability.md) |
| `models/ssd_complex.py` | [`ssd-reference.md`](references/ssd-reference.md), [`ssd-theory.md`](concepts/ssd-theory.md), [`state-space-foundations.md`](concepts/state-space-foundations.md) |
| `models/ssd_triton.py` | [`ssd-reference.md`](references/ssd-reference.md), [`block-and-stability.md`](concepts/block-and-stability.md) |
| `models/mimo.py` | [`model-reference.md`](references/model-reference.md), [`mimo.md`](concepts/mimo.md) |
| `training/pretrain.py` | [`pretrain-cli.md`](guides/pretrain-cli.md), [`training-reference.md`](references/training-reference.md), [`training.md`](training.md), [`block-and-stability.md`](concepts/block-and-stability.md) |
| `utils/checkpoint.py` | [`training-reference.md`](references/training-reference.md), [`block-and-stability.md`](concepts/block-and-stability.md) |
| `utils/logging.py` | [`training-reference.md`](references/training-reference.md) |
| `data/prepare_data.py` | [`training-reference.md`](references/training-reference.md), [`training.md`](training.md) |

## Doc rules

- **Anchor style:** docs cite code only as `file.py:Symbol` / `file.py:Class.method` — never line numbers.
- **Alignment gate:** `python3 tests/test_doc_refs.py` parses every anchor and fails on unknown files/symbols; `--coverage` also requires every public symbol in `models/`, `training/`, `utils/`, `data/` to be cited; `--links` validates every intra-repo `.md` link. Run `python3 tests/test_doc_refs.py --coverage --links` before committing doc changes.
- **Layout:** only `README.md`, `AGENTS.md`, `SKILLS.md` at the root plus this `docs/` tree — no other markdown anywhere in the repo.
