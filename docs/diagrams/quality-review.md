# Diagram quality review

Scope: three specs, their existing HTML outputs, guide and evidence only. No model code changes.

## Pinned source

Revision: `11f8b82405aacb316267b68bd246e7a897c7fe66`. Configuration: `configs/pretrain_a100_400m.yaml`.

Config SHA-256: `be6a59ffdae5978cc849a3fd6e705bbfe73ac85b7f1c4f54d0eb1bd6b4113054`.

Shared data is not a Git repository. Its relevant on-disk files are pinned by SHA-256 below. Dtype is verified from the packing code, not a production manifest or corpus inventory.

## Source and semantic checks

- `models/transformer.py:Mamba3Transformer.forward` defines output contracts.
- `training/pretrain.py:Pretrainer.train` defines iteration, counters, recovery and stop transitions.
- `utils/checkpoint.py:CheckpointManager.save` writes directly, without RNG or data-cursor persistence.
- The bounded repeated unit excludes final norm, head and loss. Both residual additions are explicit.
- GPT-2 packing selects uint16; HiLS copies to a uint32 host buffer. Mamba expects serialized tensor shards, so the raw-shard bridge remains absent.
- Mamba counts successful micro-steps separately from optimizer updates. Cross-shard assembly uses tolist, list.extend and a new torch.long tensor, not torch.cat. Same-shard windows are views.

## Acceptance and limitations

All three specs must pass nine showcase checks before delivery. Browser checks run only after successful delivery. Exact hashes and fresh status are in the adjacent combined receipt. No stale visual approval is inherited.

Perceptual review: unavailable. The image reader returned a provider/model image-input limitation. No claim of visual polish is made. The proposed 12px essential / 11px secondary first-screen target is not met. Automated 6px floor is a different gate. GPU benchmarks, production corpus inventory and full pretraining are not run.

Unlabelled solid model edges denote computation whose endpoints imply the operation. Workflow labels state conditions. Dashed workflow edges are returns, not asynchronous work.

## Source hashes

| Path | SHA-256 |
|---|---|
| `models/transformer.py` | `90ce60e4c63e62568d76c225b769e885d37ceb81a1fad248a15dcba57176e91c` |
| `training/pretrain.py` | `4e542dca584300ea7309af0faf66c95be1a2fc43f8479ab3f7df0f1128bbbf7f` |
| `utils/checkpoint.py` | `fc600ef87df1952c47ce84f01405a3c55216a04b18bbbe2160e905b1016b4058` |
| `data/prepare_data.py` | `0f23dce52f969a4538b8df74a22542c518f7d39751b5f6d746eef42f9c60335f` |
| `models/mamba_block.py` | `4208d2d07df9f7550bef9e1dd6ad9ec12198f72ad71f45d6ce088379cd409e78` |
| `models/ssd_complex.py` | `6cdb834149bed235fffbf7fe6d787d7fcad2a6e25684381e424cc15f0930c010` |
| `../shared_data/shard_writer.py` | `35a81ffc5d268af96009b68c6ea17cf82c6ea5a2b5a83c225399ee9e78c5b3a8` |
| `../shared_data/scripts/pack_shards.py` | `0f90101375437b5155260897ce7478f02f71bc0476f635465bb98e7fe21d8f31` |
| `../shared_data/loader.py` | `18fcdd882c84eafecb4d27c719d9c80df6fed7ec279e5199f61a8da485168347` |
| `../shared_data/dataset.py` | `74a3fe12b76138e78b8fc2115441d88b139bceb9fd4bcdcb132bec20d352ceab` |
| `../shared_data/config/mixture.yaml` | `fefd48c045af197cb18da1e0d710dee4aaf7aebf63af544633f89848364820e0` |
