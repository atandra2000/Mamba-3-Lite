# Mamba-3-Lite visual guide

Open the [interactive guide](../diagrams/mamba3_interactive_guide.html), or the standalone [architecture](../diagrams/mamba3_architecture.html), [data flow](../diagrams/mamba3_dataflow.html), and [training workflow](../diagrams/mamba3_training_workflow.html). The guide is a scrolling, offline HTML document with native section navigation and a labeled SVG recurrence experiment.

## Source contract

The visual package describes current source, including differences from older summaries:

- `models/transformer.py:Mamba3Transformer.forward` materializes full logits; `training/pretrain.py:train_step` uses ordinary cross-entropy. Chunked CE is not implemented on this path.
- `training/pretrain.py:Pretrainer.train` uses an unshuffled DataLoader with zero workers and no pinned memory. Its counter tracks successful micro-steps, not optimizer updates. The default 256,000 iterations imply 8.388608B token positions and normally 128,000 updates. Recovery can replay work. The scheduler advances on updates but is configured using the micro-step horizon.
- `data/prepare_data.py:main` delegates raw uint32 shard production to shared_data. `training/pretrain.py:PretrainDataset._init_sharded` expects torch.load-compatible serialized tensors. A conversion bridge is absent.
- `models/ssd_complex.py:ssd_complex_chunkwise` materializes a cross-chunk decay matrix; the chunk-pair dimension is quadratic. A fixed-state recurrence is mathematical potential, not an incremental model API.
- N=64 complex64 and N=128 float32 occupy equal bytes. Neither tensor-count reduction nor numerical oracle parity proves trained-model perplexity parity.
- `utils/checkpoint.py:CheckpointManager` uses direct writes and path-completeness checks, not atomic rename or RNG/data-cursor persistence.
- `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` catches kernel errors and warns before falling back. This behavior differs from the documented fail-loud policy.

The canonical corpus recipe is workspace shared_data/config/mixture.yaml: FineWeb-Edu 40%, DCLM 15%, Python 15%, Jupyter 5%, OpenMath 10%, arXiv 10%, Cosmopedia 5%. The prepared corpus target is 8B tokens.

## Verification

Artifact hashes, showcase delivery checks, automated Chrome measurements, and visual-review scope are recorded in [mamba3_receipts.json](../diagrams/mamba3_receipts.json). Browser evidence is separate from static validation and from image-based perceptual review. The guide's simulator is checked against the closed-form homogeneous recurrence; it intentionally excludes input forcing.

## Remaining runtime work

This completes the visual documentation scope, not pretraining readiness. The storage bridge, schedule counter alignment, checkpoint semantics, and Triton policy mismatch require separate runtime changes and validation. No model code, data preparation code, or hyperparameters were changed here. GPU throughput, memory peaks, kernel performance and a full pretraining run remain unverified.
