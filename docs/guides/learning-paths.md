# Mamba-3-Lite — Learning Paths

Three routes through the documentation, by goal. Every row links a real doc; every
path ends with a **Landmark check** — a concrete question you should be able to
answer (or a task you should be able to do) before moving on. The full doc map is
[`docs/README.md`](../README.md); notation and one-line definitions live in the
[glossary](glossary.md).

## Beginner — "run it, and follow a conversation about SSD"

Goal: launch the repo on CPU, understand what a state-space model is, and read the
config without guessing.

| # | Read | Why |
|---|---|---|
| 1 | [`../../README.md`](../../README.md) | What Mamba-3 is, the three breakthroughs (complex SSD, MIMO mixing, no causal conv), repo layout, test-suite status |
| 2 | [`../concepts/state-space-foundations.md`](../concepts/state-space-foundations.md) | From RNNs to SSMs: the linear recurrence, zero-order hold, diagonal A, and the S4 → S6 → Mamba-2 → Mamba-3 arc |
| 3 | [`../references/config-reference.md`](../references/config-reference.md) | Every `models/transformer.py:ModelConfig` field, the annotated yaml, and the 16.78B-vs-8.0B token arithmetic |
| 4 | [`quickstart.md`](quickstart.md) | Install → verify → dry-run → train → resume on CPU |
| 5 | [`glossary.md`](glossary.md) | Notation table (N, D, H, chunk, param counts) and one-line definitions |

**Landmark check.** You can run `python3 -m pytest tests/ -v` and explain what
"32 pass on CPU; 5 GPU-gated skip" means; and you can say why
`in_proj` outputs 5,136 features (`H(D + 4N + 1)` at H=16, D=N=64) without
looking it up.

## Intermediate — "understand the algorithm and operate a run"

Goal: derive the chunkwise complex-SSD pass, know why every dtype choice exists,
and operate training end to end.

| # | Read | Why |
|---|---|---|
| 1 | [`../concepts/ssd-theory.md`](../concepts/ssd-theory.md) | **The authoritative derivation cluster**: SSD decomposition, the causal segment matrix L, complex states (why N halves in ℂ), the chunkwise einsums with a worked 2×2 example |
| 2 | [`../concepts/mimo.md`](../concepts/mimo.md) | SISO → MIMO head mixing: the math, the identity init, and the `_identity_init` escape hatch |
| 3 | [`../concepts/block-and-stability.md`](../concepts/block-and-stability.md) | Block anatomy (slice boundaries, A parameterization, no causal conv) + the precision landscape (BF16/TF32/complex64/FP32, the NaN guard) |
| 4 | [`../references/ssd-reference.md`](../references/ssd-reference.md), [`../references/model-reference.md`](../references/model-reference.md) | API contracts: `models/ssd_complex.py:ssd_complex_chunkwise`, `models/ssd_complex.py:ssd_naive_complex` (the O(T) oracle), `models/mamba_block.py:Mamba3Block`, `models/mimo.py:MIMO` |
| 5 | [`../training.md`](../training.md) | The data path: the shared pipeline shim, shard format, `training/pretrain.py:PretrainDataset` layouts |
| 6 | [`pretrain-cli.md`](pretrain-cli.md), [`training-runbook.md`](training-runbook.md) | `training/pretrain.py:TrainingConfig` + CLI flags; operating the A100 run (pre-flight, monitoring, NaN recovery, resume) |

**Landmark check.** You can write down the three terms of the chunkwise
decomposition (intra-chunk `Y_diag`, per-chunk states, inter-chunk `decay_chunk`
propagation) and say which matrix plays the role attention's T×T score matrix
plays; and you can explain why the NaN guard in
`training/pretrain.py:train_step` checks *before* `backward()` and why five
consecutive failures trigger a checkpoint rollback instead of a clip.

## Expert — "modify it, tune it, extend it"

Goal: change the model safely, tune throughput, and add sanctioned kernels
without breaking the gates.

| # | Read | Why |
|---|---|---|
| 1 | [`tuning.md`](tuning.md) | `chunk_size` (the one tunable knob), lr, batch geometry, `torch.compile`, Triton dispatch — measure, don't guess |
| 2 | [`extending.md`](extending.md) | Add an SSM variant; add a sanctioned Triton kernel per the contract (`HAS_TRITON` gate, autograd wrapper, CPU-runnable reference test) |
| 3 | [`../references/training-reference.md`](../references/training-reference.md) | `utils/checkpoint.py:CheckpointManager` (three-file format, torn-save guard, weight-tying dedup), `utils/logging.py:TrainingLogger` |
| 4 | [`../../tests/`](../../tests/) | `tests/test_ssd.py::test_chunkwise_matches_naive_complex` (the oracle), `tests/test_ssd_triton.py` (dispatch guards + GPU parity), `tests/test_doc_refs.py` (the doc gate itself) |
| 5 | [`../../AGENTS.md`](../../AGENTS.md), [`../AUDIT.md`](../AUDIT.md) | Hard rules (Triton contract, complex-packing constraint, comment budgets) and the docs audit trail |

**Landmark check.** You can double `state_dim` and correctly re-derive every
`in_proj` slice offset (and say why N must stay even); and you can describe the
two-layer Triton opt-in — `ssd_dispatch='triton'` **and**
`ENABLE_TRITON_KERNELS=1` — plus what happens when either is missing.

## After any path

Before committing doc or code changes, run the gates:

```bash
python3 scripts/check_docs.py          # anchors + coverage + links (+ line/triton checks)
python3 -m pytest tests/ -v            # CPU suite; triton/GPU tests auto-skip
```
