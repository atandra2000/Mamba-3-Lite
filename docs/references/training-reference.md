# Mamba-3-Lite — Training Reference: dataset, checkpoint, logging, data pipeline

Reference for `training/pretrain.py:PretrainDataset`: the packed-token dataset that turns flat, pre-tokenized token streams into `(inputs, targets)` windows, covering its three storage layouts (single file, sharded directory, dummy), the window math, shard-spanning reads, and its `DataLoader` contract inside the training loop.

## 60-second summary

`training/pretrain.py:PretrainDataset` is a thin `torch.utils.data.Dataset` over one or more flat `torch.long` token tensors. It picks a layout by probing `data_path`: `dummy` when the path is missing, `sharded` when it is a directory, `single` otherwise. Every sample is a window of exactly `max_seq_len + 1` tokens sliced as `x = chunk[:-1]`, `y = chunk[1:]` — perfect next-token alignment, no EOS handling, no padding. `training/pretrain.py:Pretrainer.train` consumes it through a `DataLoader` with `batch_size`, `num_workers=0`, `drop_last=True`, and no shuffling. Shards are memory-mapped; windows straddling a shard boundary are stitched through a Python list.

## Why it exists

Pre-training a ~434M next-token model needs a data path that streams flat token streams with zero per-sample tokenization cost, scales past a single file via sharding, and runs end-to-end on a box with no data at all (CPU smoke tests, `--dry-run`). `PretrainDataset` satisfies all three, which is why the layout is a runtime `if` on the *shape* of `data_path`, not a configuration flag. Tokenization and document packing happen upstream in the data pipeline (`data/prepare_data.py` shim → the workspace-level `LLM/shared_data/` pipeline, see [Mamba-3-Lite — Training](../references/training-reference.md)); this class only sees already-packed token ids.

## Intuition

Think of the dataset as a **windowed view over one long token string**. The corpus is a single sequence of `T` token ids (one tensor or many shards; logically one sequence). Sample `i` is the slice `tokens[i·L : i·L + L + 1]` with `L = max_seq_len`: the first `L` tokens are the model input, the last `L` are the next-token targets, because `y[j] = x[j+1]`. Consecutive windows therefore **overlap by exactly one token**. Nothing re-segments documents; EOS ids packed between documents (id 50,256 for the GPT-2 BPE tokenizer) are ordinary tokens in the stream. The sharded layout is the same mental model with an address book: a bisect over shard start offsets maps any global token index to `(shard, offset)`, and a crossing window is stitched from adjacent shards.

## Signature

```python
class PretrainDataset(Dataset):
    """Packed pre-training dataset backed by flat token tensors (single-file or sharded)."""

    def __init__(self, data_path: str, max_seq_len: int, vocab_size: int):
        ...
    def _init_single(self, data_path: str) -> None: ...
    def _init_sharded(self, data_dir: str) -> None: ...
    def _get_window_single(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def _get_window_sharded(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def _locate(self, global_idx: int) -> Tuple[int, int]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
```

`max_seq_len` is the window length `L` (2048 in production; see [Mamba-3-Lite — Config Reference](../references/config-reference.md), which notes this field is *not* consumed by the model architecture, only by the dataset and the logger). `vocab_size` is used **only** in dummy mode as the upper bound for random ids; it never validates real data. It holds no device state and always returns CPU tensors.

## The layout decision (`training/pretrain.py:PretrainDataset.__init__`)

The entire layout selection is a two-branch `if`:

```python
def __init__(self, data_path: str, max_seq_len: int, vocab_size: int):
    self.max_seq_len = max_seq_len
    self.vocab_size = vocab_size
    if not os.path.exists(data_path):
        print(f"[warn] Pre-training data not found: {data_path}. Using dummy data for testing.")
        self.layout = "dummy"
        self._n_samples = 1000
        return
    self._init_sharded(data_path) if os.path.isdir(data_path) else self._init_single(data_path)
```

Three outcomes, decided by filesystem probing alone:

1. **Path missing** → `layout = "dummy"`, `_n_samples = 1000`, and a single `[warn]` line: `Pre-training data not found: {data_path}. Using dummy data for testing.` This is the *only* visible signal that you are not training on real data.
2. **Path is a directory** → `sharded` layout via `_init_sharded`.
3. **Path is a regular file** → `single` layout via `_init_single`.

The production default `data_path` in `training/pretrain.py:TrainingConfig` is `"data/pretrain_data.bin"` — a *file* path — and `main()` resolves it as `args.data_path or yaml data.train_data_path or "data/pretrain_data.bin"` (see [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md)). A default run therefore expects the single-file layout; sharded mode is opted into by pointing `--data-path` at a directory of `shard_*.bin` files.

## Layouts

### single — one big tensor

```python
def _init_single(self, data_path: str) -> None:
    self.layout = "single"
    self.data = torch.load(data_path, weights_only=True)
    self._n_samples = (len(self.data) - 1) // self.max_seq_len
```

The whole corpus is loaded into RAM as one 1-D `torch.long` tensor (`weights_only=True` — the file must deserialize to plain tensors). Unlike the sharded path, the single path does **not** pass `mmap=True`; the tensor is fully materialized.

```python
def _get_window_single(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    start = idx * self.max_seq_len
    chunk = self.data[start: start + self.max_seq_len + 1]
    return chunk[:-1], chunk[1:]
```

`training/pretrain.py:PretrainDataset._get_window_single` slices `L+1` tokens starting at `i·L`; `chunk[:-1]` and `chunk[1:]` are views of the same storage (zero copies) shifted by one position.

**Sample count.** For `T` tokens, `_n_samples = (T − 1) // L`. The `−1` is not a fencepost accident; it *guarantees every window is complete*. The last window starts at `(n−1)·L` and needs tokens through `n·L` inclusive; since `n = ⌊(T−1)/L⌋`, we have `n·L ≤ T−1`, so the last index is always `< T`. Example: `T = 4096`, `L = 32` → `n = 127`, and the last window covers tokens `4032..4064` — exactly 33 tokens. The leftover tail shorter than `L+1` tokens is never used. Consecutive windows overlap by one token (window `i` spans `[i·L, (i+1)·L]`), so each epoch presents `n·L` input tokens and every interior token appears in ~2 windows.

### sharded — many `shard_*.bin` files, memory-mapped

```python
def _init_sharded(self, data_dir: str) -> None:
    shard_paths = sorted(Path(data_dir).glob("shard_*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No `shard_*.bin` files in {data_dir}")
    self.layout = "sharded"
    self.shards = [torch.load(p, weights_only=True, mmap=True) for p in shard_paths]
    self.shard_sizes = [s.numel() for s in self.shards]
    self.shard_offsets = [0] + [sum(self.shard_sizes[:i+1]) for i in range(len(self.shard_sizes)-1)]
    self._total_tokens = sum(self.shard_sizes)
    self._n_samples = (self._total_tokens - 1) // self.max_seq_len
```

Directory mode globs `shard_*.bin` (sorted lexicographically — zero-padded names like `shard_0000.bin` matter for order) and **raises `FileNotFoundError` if no shards exist** — the one layout that fails loudly instead of falling back to dummy. Each shard is loaded with `torch.load(..., weights_only=True, mmap=True)`: the file stays memory-mapped for the dataset's lifetime. `self.shard_offsets` is the global start index of each shard (`[0, s₀, s₀+s₁, …]`, length = number of shards). `_n_samples` uses the same `(total − 1) // L` formula over the **whole corpus**, so shard boundaries are invisible to indexing.

**Address translation** (`training/pretrain.py:PretrainDataset._locate`):

```python
def _locate(self, global_idx: int) -> Tuple[int, int]:
    lo = bisect.bisect_right(self.shard_offsets, global_idx) - 1
    return lo, global_idx - self.shard_offsets[lo]
```

`bisect.bisect_right` returns the insertion point *after* any value equal to `global_idx`, so a `global_idx` exactly on a shard boundary resolves to the *next* shard with offset 0 — correct, since boundary index `s₀` is the first token of shard 1. Otherwise `lo` is the last shard with start `≤ global_idx`, and the offset is the difference.

**Window fetch** (`training/pretrain.py:PretrainDataset._get_window_sharded`) has a fast path and a stitching path:

```python
def _get_window_sharded(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    start = idx * self.max_seq_len
    shard_idx, offset_in_shard = self._locate(start)
    if offset_in_shard + (self.max_seq_len + 1) <= self.shard_sizes[shard_idx]:
        chunk = self.shards[shard_idx][offset_in_shard: offset_in_shard + self.max_seq_len + 1]
        return chunk[:-1], chunk[1:]
    needed = self.max_seq_len + 1
    collected = []
    cursor = start
    while len(collected) < needed:
        s_idx, off = self._locate(cursor)
        take = min(needed - len(collected), self.shard_sizes[s_idx] - off)
        collected.extend(self.shards[s_idx][off: off + take].tolist())
        cursor += take
    chunk = torch.tensor(collected[:needed], dtype=torch.long)
    return chunk[:-1], chunk[1:]
```

**Fast path:** if the whole `L+1`-token window fits inside the starting shard, it is a single slice — two views, zero copies. **Spanning path:** when the window crosses boundaries, the code walks shards with `_locate(cursor)`, taking `min(remaining, tokens-left-in-shard)` tokens per step, accumulating them in a Python list via `.tolist()`, then builds a fresh `torch.tensor(..., dtype=torch.long)`. Windows may span arbitrarily many shards; the loop terminates because `cursor` strictly increases and `_total_tokens ≥ needed` for every valid `idx`.

### dummy — random data, used when the data path is missing

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.layout == "dummy":
        return torch.randint(0, self.vocab_size, (self.max_seq_len,)), torch.randint(0, self.vocab_size, (self.max_seq_len,))
    return self._get_window_single(idx) if self.layout == "single" else self._get_window_sharded(idx)
```

Dummy mode ignores `idx` and returns two **independent** `torch.randint(0, vocab_size, (L,))` draws — the only use of `vocab_size`. Two honesty notes: (1) it is **not deterministic across processes**: `torch.randint` draws from the global RNG and `__getitem__` never seeds it, so runs see different data unless `torch.manual_seed` is set externally; (2) `y` is *not* `x` shifted — the next-token relationship is destroyed, so this mode only exercises plumbing, never measures real loss. `_n_samples` is hard-coded to 1000.

## Window semantics

For every layout, `__getitem__(i)` returns a pair of 1-D `torch.long` tensors of length exactly `L`:

- `x = chunk[:-1]` — model input: `L` tokens.
- `y = chunk[1:]` — targets: the same span shifted by one, so `y[j] == x[j+1]`.

There is **no EOS special handling, no padding, no masking at this level.** `training/pretrain.py:train_step`'s `cross_entropy` consumes every position of `y`; the model's causal structure (see [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md), [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md)) enforces "don't look ahead". If the upstream pipeline inserted EOS/PAD (id 50,256) between documents, those ids are trained on like any other token — document semantics are the pipeline's job ([Mamba-3-Lite — Training](../references/training-reference.md)); packing is this class's job.

## DataLoader integration

```python
def train(self) -> None:
    dataset = PretrainDataset(self.config.data_path, self.config.max_seq_len, self.config.vocab_size)
    loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=0, drop_last=True)
```

`training/pretrain.py:Pretrainer.train` constructs the dataset once per `train()` call and wraps it in a `DataLoader` with three settings that shape training:

- **`batch_size=self.config.batch_size`** — in production this is the YAML `micro_batch_size` (default 16), *not* the total batch; with `gradient_accumulation_steps = 2` the effective optimizer step consumes `2 × 16 × 2048 = 65,536` tokens.
- **`num_workers=0`** — windows are sliced in the main process: mmap'd shard tensors never need cross-process pickling and there is no fork overhead; the cost is that data loading blocks the training loop.
- **`drop_last=True`** — a final partial batch is silently discarded (with the dummy layout's 1000 samples and batch 16, 62 full batches are seen and 8 samples dropped).

The loader is created **without `shuffle=True`**, so epoch order is the deterministic index order `0, 1, 2, …` every epoch. Batches move to the device with `tokens.to(self.device, non_blocking=True)` before `train_step` (see [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md) for the loop contract, [Mamba-3-Lite — Training Reference](../references/training-reference.md) for the resume preceding the loop, and [Mamba-3-Lite — Training Reference](../references/training-reference.md) for how `max_seq_len` and `batch_size` feed the `tps` metric).

## Shapes

| Quantity | Shape / value |
|---|---|
| `__getitem__` return, any layout | `(L,)` `torch.long` each for `x` and `y` |
| Batched by DataLoader | `(B, L)` `torch.long` for inputs and targets |
| `single` layout `self.data` | `(T,)` 1-D integer tensor, fully in RAM |
| `sharded` layout, per shard | `(s_k,)` 1-D integer tensor, mmap'd |
| `shard_offsets` | `(n_shards,)` int list: `[0, s₀, s₀+s₁, …]` |
| `_n_samples` | `⌊(T−1)/L⌋` (single) or `⌊(Σs_k − 1)/L⌋` (sharded); fixed `1000` (dummy) |

## Invariants

1. **Every window is exactly `L+1` tokens.** The `(T−1)//L` count plus slicing at `idx·L` guarantees the last window ends at index `n·L ≤ T−1 < T`; no truncated, wrapped, or padded windows ever reach the model.
2. **`y[j] = x[j+1]` for all `j`.** Perfect next-token alignment, preserved identically across all three layouts, including spanning windows (the stitched chunk is sliced the same way).
3. **`len(dataset) == _n_samples`** for every layout; index `i` maps to global token position `i·L`, and shard boundaries never affect sample numbering.
4. **Windows overlap by exactly one token** (shift `L`, width `L+1`).
5. **`layout ∈ {"single", "sharded", "dummy"}`**; exactly one of `self.data` / `self.shards` / dummy path is populated.
6. **All returned tensors are CPU `torch.long`**; device transfer happens only in `Pretrainer.train`.

## Pitfalls

1. **mmap keeps file handles and page-cache references.** The sharded layout calls `torch.load(..., mmap=True)`, so `shard_*.bin` files stay open and memory-mapped for the dataset's lifetime; deleting or re-saving a shard mid-run corrupts the view, and on memory pressure the OS may page shard pages out and fault them back in. The single layout is the opposite extreme: everything is loaded eagerly, so a multi-billion-token corpus is a multi-GB resident tensor.
2. **Spanning windows are slow.** The stitching path does `collected.extend(… .tolist())` — every token becomes a Python `int`, then the whole window is rebuilt with `torch.tensor(collected, dtype=torch.long)`. For `L = 2048` that is ~2048 Python-level boxing/unboxing operations per crossing window, and it breaks the zero-copy view property of the fast path. If shards are small relative to `max_seq_len`, most windows span and this dominates data-loading time.
3. **Dummy mode silently masks a missing dataset.** A typo'd `--data-path`, an un-downloaded corpus, or an un-packed default all produce one `[warn]` line and 1000 samples of random garbage. Training "succeeds" — loss converges toward `ln(vocab) ≈ 10.82` for 50,257 ids — while the model learns noise. Grep the run log for the warn line before trusting any run that did not use a synthetic shard.
4. **Shards must be `torch.save`'d tensors.** `weights_only=True` rejects arbitrary pickled objects, and `_locate`/slicing assumes 1-D integer tensors. The canonical workspace pipeline's raw on-disk format (EOS-separated `uint32` records, per `LLM/shared_data/` docs) is *not* directly loadable here — it must be converted to packed `torch.long` tensors first; `tests/e2e_gpu_smoke.py:_build_synthetic_shard` shows the exact expected format.
5. **The `(T−1)` in `_n_samples` is load-bearing.** Changing it to `T // L` lets the last window run past the end of the corpus (silent truncation in `single`; an out-of-bounds walk in `sharded`).
6. **No shuffling.** `Pretrainer.train`'s `DataLoader` has no `shuffle=True`: every epoch scans windows in the same order — fine for convergence, but inter-epoch order variation must be added at the loader.
7. **`drop_last=True` hides the tail batch.** If `_n_samples` is not divisible by `batch_size`, the remainder is dropped each epoch without warning.
8. **Dummy mode is not seeded.** `torch.randint` in `__getitem__` draws from the global RNG; reproducibility across processes requires an external `torch.manual_seed`.

## Tests

- `tests/e2e_gpu_smoke.py:check_data_pipeline` — check 2 of the 8-check GPU smoke suite: writes a synthetic shard via `tests/e2e_gpu_smoke.py:_build_synthetic_shard` (4096 tokens, vocab 128), constructs `PretrainDataset(..., max_seq_len=32)`, asserts `len(ds) > 0` and `(32,)` sample shapes, batches with `DataLoader(..., batch_size=4, drop_last=True)`, and asserts a `(4, 32)` CUDA batch — end-to-end proof of the single-file layout + loader contract.
- `tests/e2e_gpu_smoke.py:check_pretrainer_dry_run` — check 8: full `training/pretrain.py:Pretrainer` dry-run (2 steps, triton dispatch, tiny model) fed by a synthetic shard; asserts a final checkpoint exists.
- `training/pretrain.py:Pretrainer.train` dry-run — the CLI path: `main()` maps `--dry-run` to `max_steps=2`, so `python3 training/pretrain.py --dry-run` is the fastest way to validate dataset + loop wiring on any box (dummy layout if no data exists — watch for the warn line).
- `tests/test_train_step.py::test_train_step_on_tiny_model` — CPU-side exercise of the exact `(tokens, targets)` shape the dataset produces (random `(2, 16)` long tensors), asserting the loss is finite and parameters move.

See [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md) / [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) for what happens to each window inside the model, [Mamba-3-Lite — Config Reference](../references/config-reference.md) for `max_seq_len`'s role, [Mamba-3-Lite — Training](../references/training-reference.md) for how the token stream is produced, and [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md) / [Mamba-3-Lite — Training Reference](../references/training-reference.md) / [Mamba-3-Lite — Training Reference](../references/training-reference.md) for the surrounding training loop.
---

#Checkpointing — `CheckpointManager` and the 3-File Format

Reference doc for `utils/checkpoint.py:CheckpointManager`: how Mamba-3-Lite persists model weights, optimizer state, and training metadata, and how the training loop discovers and restores them.

## 60-second summary

Every saved step writes **three files** into the checkpoint directory: `model_step_N.safetensors` (weights, via `safetensors.torch.save_file`), `optim_step_N.pt` (optimizer state, via `torch.save`), and `meta_step_N.json` (step, LR-scheduler state, optimizer-step count, tag, and the full `TrainingConfig`). `utils/checkpoint.py:CheckpointManager.save` first walks the model state dict and materializes **aliased tensors into independent copies** by tracking `data_ptr()` — this is mandatory, not an optimization: `safetensors.save_file` refuses dicts whose entries share storage, and Mamba-3-Lite's weight tying makes `lm_head.weight` the *same tensor* as `embed.weight`. `utils/checkpoint.py:CheckpointManager.load` restores weights with `load_state_dict(strict=False)` (reporting missing/unexpected keys), loads optimizer state only if the file exists, and returns the meta dict. `utils/checkpoint.py:CheckpointManager.latest_step` returns the highest step for which **all three** files exist, which is also the mechanism `training/pretrain.py:Pretrainer.train` uses to auto-resume. **Honesty note:** the module docstring says "Atomic safetensors checkpoint manager", but the code writes each file *directly* — there is no `tmp → os.rename` anywhere. Crash safety comes from the write order (weights → optimizer → meta) plus the all-three-files completeness check, not from atomic rename.

## Why it exists

Pre-training runs for 256,000 steps and is interrupted routinely (preemption, OOM, NaN excursions). The checkpoint layer has three jobs:

1. **Resume**: restart from the last good step without losing optimizer momentum or LR-schedule position — momentum alone is often worth more than the weights themselves for the first few hundred steps after a restart.
2. **Rollback**: the NaN guard in `training/pretrain.py:train_step` skips backward on NaN/Inf losses; after `nan_guard_max_consecutive = 5` consecutive bad micro-steps, `training/pretrain.py:Pretrainer.train` restores the latest checkpoint and continues from that step instead of aborting.
3. **Weight-tying correctness**: because `models/transformer.py:Mamba3Transformer` sets `self.lm_head.weight = self.embed.weight` (`weight_tying=True`), a naive `save_file` of `model.state_dict()` would feed safetensors two dict entries sharing one storage — which `save_file` rejects with `RuntimeError: Some tensors share memory`. The data_ptr dedup pass exists to make saving *possible*.

## Intuition

Think of a checkpoint as a **transaction with three ledger entries** that must all be present to count. The weights are the product, the optimizer `.pt` is the momentum ledger, and the JSON is the receipt (step number, schedule position, config snapshot). Because a crash can interrupt the transaction at any point, the manager never trusts a single file: a step "exists" only when all three files are on disk, and discovery always picks the *largest complete* step.

The dedup pass is easiest to misread. "Dedup" here does **not** mean "store the tied weight once and save space" — the saved file contains **two independent copies** of the 51.46M-parameter embedding/head tensor (the second alias is cloned). What the pass guarantees is that no two entries in the dict handed to `save_file` share storage, which is a hard requirement of the safetensors format. Measured on the dev box (2026-08-04): passing an aliased dict raises; passing the deduped dict saves fine, with both copies present in the file.

## The 3-file format

| File | Writer | Contents | Size (full 434M model, float32) |
|---|---|---|---|
| `model_step_N.safetensors` | `safetensors.torch.save_file` | every `state_dict()` entry, deduped to independent storage | ~1.94 GB [DERIVED: 433,662,400 + 51,463,168 (tied pair counted twice) entries × 4 B] |
| `optim_step_N.pt` | `torch.save` | `optimizer.state_dict()` — per-parameter `exp_avg` / `exp_avg_sq` plus the two param-group configs | depends on optimizer moments |
| `meta_step_N.json` | `json.dump(..., default=str)` | `{"step": N, "scheduler": {...}, "opt_steps": ..., "tag": ..., "config": {...}}` | a few KB |

Note the **asymmetry of formats**: weights are safetensors (self-describing, memory-mappable, no pickle), optimizer state is `torch.save` (pickle-based but loaded with `weights_only=True`), and metadata is plain JSON. Only the model file participates in step discovery (`_list_steps` globs `model_step_*`); the other two are companions that gate completeness.

## Code walkthrough

### `utils/checkpoint.py:CheckpointManager.save` — write the triple

```python
def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
         extra_meta: Optional[dict] = None, state_dict: Optional[dict] = None) -> None:
    state = state_dict if state_dict is not None else model.state_dict()
    seen_ptrs: set = set()
    deduped: dict = {}
    for k, v in state.items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            deduped[k] = v.contiguous().clone()
        else:
            seen_ptrs.add(ptr)
            deduped[k] = v.contiguous()
    save_file(deduped, self.save_dir / f"model_step_{step}.safetensors")
    torch.save(optimizer.state_dict(), self.save_dir / f"optim_step_{step}.pt")
    meta: dict = {"step": step}
    if extra_meta:
        meta.update({k: v for k, v in extra_meta.items() if k != "step"})
    self._write_json(self.save_dir / f"meta_step_{step}.json", meta)
```

Semantics, in order:

1. **Dedup pass** — iterate `state_dict()` in key order; the first tensor at a given `data_ptr()` is kept as `v.contiguous()` (a no-op view for an already-contiguous tensor, so it *retains the original storage*), every later key sharing that pointer becomes `v.contiguous().clone()`. For the tied pair, `embed.weight` (registered first) keeps the live storage and `lm_head.weight` is cloned, so the two file entries are byte-identical but storage-independent. The pass also quietly fixes non-contiguous views by materializing them.
2. **Weights** — `save_file(deduped, ...)` writes directly to the final path (see Honesty check).
3. **Optimizer** — `optimizer.state_dict()` is pickled with `torch.save`. For the tied pair this contains **one** moment pair, because `training/pretrain.py:Pretrainer.__init__` builds its optimizer from a deduplicated parameter list (`id(p)` set), so `lm_head.weight` never appears twice.
4. **Meta** — the caller's `step` key is filtered out (`k != "step"`); the manager's `step` argument always wins. `_write_json` serializes with `indent=2` and `default=str` (a stringification fallback for non-JSON values — see Pitfalls).

`training/pretrain.py:Pretrainer.save_checkpoint` is the production caller:

```python
def save_checkpoint(self, step: int, tag: str = "") -> None:
    state = self.raw_model.state_dict()
    extra_meta = {"scheduler": self.scheduler.state_dict(), "opt_steps": self._opt_steps,
                  "tag": tag or f"step_{step}", "config": asdict(self.config)}
    self.ckpt_manager.save(self.raw_model, self.optimizer, step, extra_meta=extra_meta, state_dict=state)
```

It passes `self.raw_model` (the uncompiled module) rather than `self.model`, so `torch.compile` wrapper parameters never leak into the file, and it records the `SequentialLR` scheduler state, the optimizer-step counter, a human tag, and the full `TrainingConfig` snapshot (which is what makes the JSON a self-contained run record).

### `utils/checkpoint.py:CheckpointManager.load` — restore the triple

```python
def load(self, model: torch.nn.Module, step: int, device: str = "cuda",
         optimizer: Optional[torch.optim.Optimizer] = None, strict: bool = True) -> dict:
    weight_path = self.save_dir / f"model_step_{step}.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}\nAvailable steps: {self._list_steps()}")
    weights = load_file(str(weight_path), device=device)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        msg = f"[checkpoint] {len(missing)} missing key(s): {missing[:5]}{'…' if len(missing) > 5 else ''}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    ...
    if optimizer is not None:
        optim_path = self.save_dir / f"optim_step_{step}.pt"
        if optim_path.exists():
            optimizer.load_state_dict(torch.load(optim_path, map_location=device, weights_only=True))
        else:
            logger.warning("[checkpoint] no optimiser state at %s — optimizer will start from scratch", optim_path)
    meta_path = self.save_dir / f"meta_step_{step}.json"
    meta: dict = json.load(open(meta_path)) if meta_path.exists() else {"step": step}
    return meta
```

Key behaviors:

- **Weights load with `strict=False` always**; the `strict` parameter only controls whether a mismatch *raises* or *warns*. With `strict=True` (the manager default) any missing/unexpected key is a `RuntimeError`; `training/pretrain.py:Pretrainer.load_checkpoint` passes `strict=False`, so a config change that alters the module list degrades to warnings instead of a crash.
- **Weight tying survives the round trip.** `load_state_dict` is called with the default `assign=False`, which copies values *in place* into the existing parameters via `param.copy_(...)`. Because the live model still has `lm_head.weight is embed.weight` (one object, one storage), the two file copies (equal values) are copied into the same storage twice — the model stays tied and consistent. Verified empirically: a freshly constructed tied model with corrupted weights loaded from a deduped file ends with `lm_head.weight is embed.weight` and equal values. If the code ever switched to `assign=True`, the second key would replace the shared parameter with a fresh unaliased tensor and silently break tying — it does not.
- **Optimizer state is optional**: if `optimizer is None` or the `.pt` file is missing, loading proceeds and logs a warning ("optimizer will start from scratch"). `weights_only=True` blocks arbitrary pickle gadgets.
- **The return value is the meta dict** — `{"step": N, "scheduler": ..., "opt_steps": ..., "tag": ..., "config": ...}` — with `{"step": step}` as the fallback when the JSON is absent.

`training/pretrain.py:Pretrainer.load_checkpoint` turns the meta dict back into training state:

```python
def load_checkpoint(self, step: int) -> int:
    meta = self.ckpt_manager.load(self.raw_model, step, device=str(self.device), optimizer=self.optimizer, strict=False)
    if "scheduler" in meta:
        self.scheduler.load_state_dict(meta["scheduler"])
    if "opt_steps" in meta:
        self._opt_steps = meta["opt_steps"]
    resumed_step = meta.get("step", step)
    return resumed_step
```

The LR schedule resumes exactly where it left off (`SequentialLR` state carries the epoch counters and base LRs), the optimizer-step counter is restored, and the returned step becomes the loop's `global_step`.

### `utils/checkpoint.py:CheckpointManager.latest_step` — completeness-gated discovery

```python
def latest_step(self) -> Optional[int]:
    steps = self._list_steps()
    return next((s for s in sorted(steps, reverse=True) if self._checkpoint_complete(s)), None)

def _list_steps(self) -> list:
    return [int(p.stem.removeprefix("model_step_"))
            for p in self.save_dir.glob("model_step_[0-9]*.safetensors")]

def _checkpoint_complete(self, step: int) -> bool:
    return all((self.save_dir / n).exists() for n in [
        f"model_step_{step}.safetensors", f"optim_step_{step}.pt", f"meta_step_{step}.json"])
```

Steps are parsed from the weight-file names only (`model_step_[0-9]*.safetensors` — non-numeric suffixes never match), sorted descending, and the first step whose three companion files all exist wins. A step whose write was interrupted is simply invisible to discovery.

## Honesty check: "Atomic" is not what the code does

The module docstring reads:

```python
"""Atomic safetensors checkpoint manager with shared-tensor dedup and step discovery."""
```

and the class docstring says "Save/load model checkpoints. Files: model_step_N.safetensors, optim_step_N.pt, meta_step_N.json." The retired expansion-plan doc (R9) likewise listed "atomicity (tmp→rename)" as a topic. **The code does not do that.** Verbatim from `utils/checkpoint.py:CheckpointManager.save`:

- `save_file(deduped, self.save_dir / f"model_step_{step}.safetensors")` — direct write to the final path;
- `torch.save(optimizer.state_dict(), self.save_dir / f"optim_step_{step}.pt")` — direct write;
- `self._write_json(self.save_dir / f"meta_step_{step}.json", meta)` — and `_write_json` opens the target path directly (`with open(tmp, "w")`; the parameter is *named* `tmp` but no temporary file is ever created — the `tempfile` import at the top of the module is dead code).

There is no `tmp → os.rename` sequence anywhere in the file. The resilience properties the code *actually* has are:

1. **Write order**: weights → optimizer → meta, with the JSON written *last*. A crash mid-save leaves the meta file missing, so the step fails `_checkpoint_complete` and `latest_step` skips it.
2. **Completeness-gated discovery**: a step counts only when all three files exist, so partially-written triples are never resumed.

That is a crash-*tolerant* design (torn saves are skipped) but not an *atomic* one: each individual file can be observed in a half-written state by a concurrent reader, and the window between files is unprotected. If the docstring's claim is ever needed for real (e.g., multi-process readers), the fix is the rename pattern the docstring implies. [VERIFIED against `utils/checkpoint.py`; the discrepancy is deliberate — this reference documents the code as written.]

## Resume flow in the training loop

`training/pretrain.py:Pretrainer.train` auto-discovers and restores at startup:

```python
global_step = 0
latest = self._find_latest_checkpoint()
if latest is not None:
    try:
        global_step = self.load_checkpoint(latest)
    except Exception as exc:
        self._log(f"[warn] Could not load checkpoint: {exc}")
```

(`_find_latest_checkpoint` is a one-line delegate to `utils/checkpoint.py:CheckpointManager.latest_step`.) A failed load — e.g. a torn-but-complete triple — degrades to a warning and training restarts from scratch rather than aborting. The CLI path in `training/pretrain.py:main` additionally supports `--resume <step>` (an explicit `trainer.load_checkpoint(int(args.resume))` before `train()`); note that `train()` then re-runs auto-discovery, so an explicit `--resume` to an *older* step is overridden by the latest complete step (see Pitfalls). The same restore path is used by the NaN guard: after 5 consecutive NaN/Inf micro-steps, `Pretrainer.train` calls `load_checkpoint(latest)` and resets `global_step` to the resumed step, discarding the poisoned run's optimizer momentum.

## Pitfalls

1. **Torn-but-complete triples are undetectable.** The completeness check is existence-only. If a crash truncates `model_step_N.safetensors` *after* the meta file was written, all three files exist, `latest_step` returns `N`, and `load` fails at deserialization (safetensors raises on truncated data). `Pretrainer.train` catches this and warns, but the checkpoint is unrecoverable — the only defense is disk-level atomicity, which the code does not implement.
2. **The dedup clone does not save space.** The tied pair is written twice (~411.8 MB of the ~1.94 GB model file). Calling this "dedup" is accurate only in the sense of *storage independence*: `safetensors.save_file` hard-rejects shared-storage dicts, so the clone is what makes saving a tied model possible at all. Do not "optimize" the pass into actually skipping the second key — `load_state_dict(strict=False)` would then report `lm_head.weight` as missing on every load.
3. **In-memory tying is preserved only because `assign=False`.** `load_state_dict` copies into the existing aliased parameter, so tying survives. Switching to `assign=True` (or calling `load_state_dict` on a model built with `weight_tying=False`) would silently un-tie or misassign. If you load into a non-tied model, the two copies are still written to the two distinct parameters — correct, just redundant.
4. **Optimizer state is `torch.save`, not safetensors.** It is loaded with `weights_only=True`, so version-skew or a missing file means "optimizer starts from scratch" (warned, not fatal). Also, the optimizer's moment tensors are keyed by *parameter index* over the deduplicated parameter list — a run whose model config changes the parameter order will restore moments onto the wrong parameters without any error.
5. **The meta JSON round-trips through `default=str`.** The scheduler state that `Pretrainer.save_checkpoint` stores contains only JSON-native numbers and lists today, so `json.load` → `scheduler.load_state_dict` round-trips. If a tensor ever lands in `extra_meta`, it will be stringified and the load will fail (or silently corrupt). And a *torn* meta JSON raises `json.JSONDecodeError` inside `load` — there is no fallback for a corrupt-but-present file.
6. **`--resume` an older step? The loop will re-discover.** `main()`'s explicit `--resume N` is followed by `train()`'s unconditional auto-discovery of `latest_step()`, so resuming to a step older than the latest complete checkpoint does not stick. The practical resume is: point `--checkpoint-dir` at the directory and let auto-discovery pick the newest triple.
7. **Save is synchronous and single-threaded.** The three writes happen inline in `save()`; there is no background flush, so checkpoint I/O stalls the training loop, and two concurrent `save` calls to the same directory would interleave files.

## Tests

- `tests/e2e_gpu_smoke.py:check_checkpoint` (check 7 of 8, CUDA + triton-gated): saves step 1 with `CheckpointManager`, asserts `model_step_1.safetensors` exists, builds a *fresh* tied `Mamba3Transformer`, loads with `strict=False`, and asserts round-trip max-abs drift `< 1e-4`. This exercises the dedup path end to end (fresh model with `weight_tying=True` + aliased dict → dedup → save → load → in-place copy).
- `tests/e2e_gpu_smoke.py:check_pretrainer_dry_run` (check 8 of 8): runs a 2-step `Pretrainer` dry run with a synthetic shard and asserts `_find_latest_checkpoint()` returns a real step after training — the auto-save side of the loop.
- There is **no CPU unit test** for `CheckpointManager` (the CPU suite's 28 tests cover model/SSD behavior); on a CPU-only box both e2e checks are skipped, which is why the dedup/tying behavior verified here was confirmed with an ad-hoc round-trip script rather than a committed test.

## Cross-links

- Weight tying and the model's parameter layout: [Mamba-3-Lite — Model Reference](../references/model-reference.md) and [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) (weight-tying param math, NaN-guard rationale).
- The full training loop, LR schedule, and NaN guard: [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md) (concurrent), [Mamba-3-Lite — Config Reference](../references/config-reference.md) (concurrent, every `TrainingConfig` field), [Mamba-3-Lite — Training Reference](../references/training-reference.md) (concurrent, data loader).
- Operational practice — when checkpoints are written, how to resume, NaN recovery: [Mamba-3-Lite — Training Runbook](../guides/training-runbook.md) and [Mamba-3-Lite — Quickstart](../guides/quickstart.md) .
---

#TrainingLogger — windowed metrics, throughput, and optional WandB

Reference for `utils/logging.py:TrainingLogger`: how the training loop turns raw per-step losses into console summaries and optional WandB records.

## 60-second summary

`TrainingLogger` is the step-driven logger for pre-training: every `log_every` steps it prints one line with the step number, average loss, perplexity (exp of the average loss), learning rate, and tokens-per-second, and optionally mirrors those values to WandB under `train/…` keys. The throughput number is `tps = log_every × seq_len × batch_size / elapsed`, where `elapsed` is the wall time between two logged windows. WandB is enabled purely by environment: setting `WANDB_PROJECT` triggers `wandb.init` at construction (`WANDB_RUN_NAME` sets the run name), and a missing `wandb` package is tolerated with a printed warning. `Pretrainer` builds one logger from `TrainingConfig` fields and calls `TrainingLogger.log` from inside `training/pretrain.py:Pretrainer.train` on steps divisible by `log_every`.

## Why it exists

Pre-training runs for up to `max_steps = 256,000` steps ([Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md)); a line per step would drown a terminal, and per-step WandB points would bury a dashboard. Nothing else in the repo records training progress — checkpoints only capture state every `save_every = 4000` steps — so the logger is the only place a human (or an experiment tracker) sees loss, perplexity, and throughput over time. Its jobs: a greppable **console heartbeat** once per window, **perplexity** from a window average (caveat in [Pitfalls](#pitfalls)), and **throughput telemetry** — `tps` catches stalls and regressions in effective tokens/sec, the number that matters for scaling ([Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md)).

## Intuition

Think of the logger as a **windowed accumulator with a clock**. Every call appends the loss to a Python list (`_loss_window`). When the incoming step is a multiple of `log_every`, the logger averages the window, divides the window's token count by the wall time since the previous flush (`elapsed`) to get `tps`, prints one line, optionally records one WandB entry keyed by the global step, then empties the window and restarts the clock. So each printed line summarizes exactly one window, and consecutive lines are independent — state is fully reset at every flush.

**Verified nuance about the "window":** the class docstring promises a "rolling-window summary every `log_every` steps", which implies the caller invokes `log` every step and the logger buffers. The actual integration does the opposite: `Pretrainer.train` only calls `TrainingLogger.log` on steps where `global_step % log_every == 0`, and `log` itself flushes on the same condition. So **every call flushes immediately and the window holds exactly one loss** in the shipped training loop; the averaging machinery only engages if a caller invokes `log` every step. The math below covers the general (windowed) case, with the integration-specific simplification called out.

## Math

**Throughput.** In the integration each training step consumes one DataLoader batch of `batch_size` sequences of `seq_len = max_seq_len` tokens (see [Mamba-3-Lite — Training Reference](../references/training-reference.md) for how windows are packed). Between two flushes there are exactly `log_every` such steps, so the tokens attributable to a window are

$$N_{\text{win}} = \text{log\_every} \times \text{seq\_len} \times \text{batch\_size}.$$

With $T_{\text{win}}$ the wall time between flushes (computed as `elapsed = max(time.time() - self._step_start, 1e-6)`), the logged throughput is

$$\text{tps} = \frac{N_{\text{win}}}{T_{\text{win}}} = \frac{\text{log\_every} \times \text{seq\_len} \times \text{batch\_size}}{\text{elapsed}}.$$

With the defaults (`log_every=50`, `seq_len=2048`, `batch_size=16`, all from `training/pretrain.py:TrainingConfig`), one window covers $50 \times 2048 \times 16 = 1{,}638{,}400$ tokens. The `1e-6` floor on `elapsed` only matters for degenerate zero-duration windows; it makes `tps` finite rather than a `ZeroDivisionError`.

**Perplexity.** The window loss is the arithmetic mean $\bar{l} = \frac{1}{W}\sum_{i=1}^{W} l_i$ over the $W$ losses in the window, and perplexity is

$$\text{ppl} = \exp\!\left(\bar{l}\right).$$

Because $\exp$ is convex, Jensen's inequality gives

$$\frac{1}{W}\sum_{i=1}^{W} \exp(l_i) \;\ge\; \exp\!\left(\frac{1}{W}\sum_{i=1}^{W} l_i\right),$$

so the logged ppl is **not** the mean of the per-step perplexities — it is at or below it, with equality only when all losses in the window are identical. Concretely: `ppl = torch.tensor(avg_loss).exp().item()` — exp is applied *after* averaging, never per step. In the current integration $W = 1$, so ppl degenerates to the single-step value $\exp(l)$.

## Code walkthrough

### Signature and semantics

`utils/logging.py:TrainingLogger` has two public methods.

- `TrainingLogger.__init__(self, log_every: int = 10, seq_len: int = 1024, batch_size: int = 1)` — records the three scalars, starts two clocks (`_start`, `_step_start`), initializes an empty `_loss_window: list[float]`, and conditionally initializes WandB.
- `TrainingLogger.log(self, step: int, loss: float, lr: float = 0.0) -> None` — appends `loss` to the window and flushes when `step` is a multiple of `log_every`.

### Construction and the WandB gate

```python
def __init__(self, log_every: int = 10, seq_len: int = 1024, batch_size: int = 1):
    self.log_every = log_every
    self.seq_len = seq_len
    self.batch_size = batch_size
    self._start = time.time()
    self._step_start = time.time()
    self._loss_window: list[float] = []
    self._wandb = None
    wandb_project = os.environ.get("WANDB_PROJECT")
    if wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project, name=os.environ.get("WANDB_RUN_NAME"), reinit=True)
            self._wandb = wandb
        except ImportError:
            print("[logging] wandb not installed -- skipping WandB integration")
```

Semantics worth stating precisely:

- WandB is enabled **entirely by environment**, never by a constructor flag: `WANDB_PROJECT` set → `wandb.init` runs at construction; unset → `self._wandb` stays `None` and `log` prints only.
- `WANDB_RUN_NAME` optionally names the run; `reinit=True` permits repeated `wandb.init` in one process (e.g., multiple logger constructions in tests).
- Only `ImportError` is caught — a `wandb` install that fails for another reason (bad API key, no network) raises out of `__init__` and aborts training.
- `_start` is recorded but never read — dead state, harmless.

### `log`: append, gate, flush

```python
def log(self, step: int, loss: float, lr: float = 0.0) -> None:
    self._loss_window.append(loss)
    if step % self.log_every != 0 or not self._loss_window:
        return
    avg_loss = sum(self._loss_window) / len(self._loss_window)
    elapsed = max(time.time() - self._step_start, 1e-6)
    tokens_per_sec = (self.log_every * self.seq_len * self.batch_size) / elapsed
    ppl = torch.tensor(avg_loss).exp().item()
    parts = [f"step={step:>7}", f"loss={avg_loss:.4f}", f"ppl={ppl:.2f}", f"lr={lr:.2e}", f"tps={tokens_per_sec:,.0f}"]
    print(" | ".join(parts))
    if self._wandb is not None:
        log_dict = {"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr, "train/tokens_per_sec": tokens_per_sec}
        self._wandb.log(log_dict, step=step)
    self._loss_window = []
    self._step_start = time.time()
```

Step by step:

1. **Append unconditionally**, then gate: `step % log_every != 0` or an empty window → early return. The window can only be empty if `loss` was never appended, which cannot happen — the guard is defensive.
2. **`avg_loss`** is a plain Python mean over `_loss_window`; `loss` arrives as a Python `float`, so no tensor arithmetic happens until ppl.
3. **`tokens_per_sec`** implements the formula above; `lr` is whatever the caller passed (default `0.0`).
4. **Print format**: fields joined by `" | "` — `step` right-aligned 7 wide, `loss` 4 decimals, `ppl` 2 decimals, `lr` scientific 2 decimals, `tps` thousands-separated with no decimals.
5. **WandB**: exactly one `log_dict` per flush with the four `train/` keys, recorded at `step=step` — the global training step, so WandB's x-axis is training progress even across resume runs.
6. **Reset**: window emptied and `_step_start` restarted, so the next `elapsed` measures the next window only.

### Integration in the training loop

`training/pretrain.py:Pretrainer.__init__` builds the logger once:

```python
self.logger = TrainingLogger(
    log_every=config.log_every, seq_len=config.max_seq_len, batch_size=config.batch_size,
)
```

i.e. `seq_len` is the training-window size `max_seq_len` (2048), not the model's context length per se (the module does not consume `max_seq_len` — see [Mamba-3-Lite — Config Reference](../references/config-reference.md)), and `batch_size` is the DataLoader batch size (16). The call site inside `training/pretrain.py:Pretrainer.train`:

```python
metrics = self.train_step(tokens, targets, global_step)
if metrics is None:
    nan_guard_streak += 1
    if nan_guard_streak >= self.config.nan_guard_max_consecutive:
        latest = self._find_latest_checkpoint()
        if latest is not None:
            self._log(f"[nan-guard] {nan_guard_streak} consecutive NaN/Inf — restoring checkpoint step {latest}.")
            global_step = self.load_checkpoint(latest)
        else:
            self._log("[nan-guard] No checkpoint to restore from. Aborting.")
            raise RuntimeError("NaN/Inf with no checkpoint to restore from")
        nan_guard_streak = 0
    continue
nan_guard_streak = 0
if global_step % self.config.log_every == 0:
    lr = self.scheduler.get_last_lr()[0]
    self.logger.log(global_step, metrics["loss"], lr=lr)
```

Precise facts about this call pattern:

- **The guard is only `global_step % self.config.log_every == 0`.** The "metrics non-None" condition is enforced *upstream*, not at the call site: `training/pretrain.py:train_step` returns `None` when `nan_guard` detects NaN/Inf (after zeroing grads and skipping backward), the loop `continue`s before reaching the logger, and a NaN step does not advance `global_step`. So `log` is never invoked with a NaN loss — but nothing in `TrainingLogger.log` itself would reject one.
- **`metrics["loss"]`** is the cross-entropy value from `train_step` (computed with `ignore_index=-100`), passed positionally as `loss`; **`lr`** is `scheduler.get_last_lr()[0]`, the current effective LR of the `SequentialLR` (linear warmup → cosine).
- **`metrics` is not passed.** The former `metrics` parameter was dropped because `train_step` returns only `{"loss": …}`; its dict is consumed via the `["loss"]` lookup, never passed whole.
- **Step 0 is logged**: `global_step` starts at 0 and `0 % log_every == 0`, so the first line appears immediately. Its `elapsed` covers logger construction through the first step, including dataset build, checkpoint scan, and any `load_checkpoint` ([Mamba-3-Lite — Training Reference](../references/training-reference.md)) — discard this line for steady-state tps.
- **After resume**, `global_step` is the restored step (`Pretrainer.load_checkpoint`), so the first flush occurs at the next multiple of `log_every`, with a one-loss window.

## Invariants

- A flush occurs **iff** the caller passes `step % log_every == 0` and the window is non-empty; every flush empties `_loss_window` and restarts `_step_start`.
- Exactly one console line and at most one WandB record per flush, keyed by the global `step`; WandB keys are `train/loss`, `train/ppl`, `train/lr`, `train/tokens_per_sec`.
- `tps` is only as accurate as its numerator assumption: `log_every × seq_len × batch_size` tokens must actually have been processed between flushes.

## Pitfalls

1. **The window is size 1 in the shipped integration.** Because the caller and the logger share the `% log_every == 0` gate, `avg_loss` is a single-step loss and ppl is `exp` of that step — there is no smoothing, despite the "rolling-window" docstring. To get true windowing, call `log` every step and let the internal gate do the buffering; the current `Pretrainer.train` does not.
2. **`tps` counts tokens including the batch factor, but only over logged windows.** It is correct *only because* the caller fires exactly once per window, so `elapsed` spans exactly `log_every` steps. Two ways it silently breaks: (a) NaN-guard skips consume wall time without advancing `global_step`, so a window's clock includes dead forward passes and `tps` reads low; (b) if the caller switched to calling `log` every step, `elapsed` would span one step while the numerator still counts `log_every × seq_len × batch_size` — inflating `tps` by a factor of `log_every`. Keep the two gates aligned.
3. **`ppl` is exp of the window-average loss, not the mean of per-step ppls** (Jensen; see [Math](#math)). Comparing the logged ppl against a mean of per-step perplexities will show a small, systematic gap.
4. **WandB initializes at construction, not at first log.** With `WANDB_PROJECT` set, `wandb.init` runs even if training crashes before the first log — you still get an empty experiment. Only `ImportError` is tolerated; auth/network failures propagate and abort training.
5. **First-line contamination.** The step-0 window's `elapsed` includes startup (dataset construction, checkpoint scan/load), so its `tps` is not representative; ignore it for steady-state readings.
6. **Cadence is config, not CLI.** `log_every` (default 50) comes from `TrainingConfig`/the yaml ([Mamba-3-Lite — Config Reference](../references/config-reference.md)); there is no `--log-every` flag ([Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md)). A full 256k-step run emits ~5,120 console lines and WandB records.

## Tests

There are **no dedicated tests** for the logger: no `tests/test_utils.py` exists in the tree (the test files are `test_doc_refs.py`, `test_ssd.py`, `test_ssd_triton.py`, `test_mimo.py`, `test_train_step.py`, `test_transformer.py`, `test_grad_checkpoint.py`, and the GPU-gated `e2e_gpu_smoke.py`). The only automated check that touches `utils/logging.py` is `tests/test_doc_refs.py`, which verifies that doc citations like `` `utils/logging.py:TrainingLogger` `` resolve to real symbols — it checks anchors, not behavior. `tests/test_train_step.py` exercises `training/pretrain.py:train_step`, the function that produces the `{"loss": …}` dict the logger consumes, but never instantiates `TrainingLogger`. Behavioral coverage is indirect: `python training/pretrain.py --dry-run` (2 steps, per [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md)) logs step 0 — one line with `step=0` — and the [G2 runbook](../guides/training-runbook.md) covers reading the loss/ppl/tps columns and `WANDB_PROJECT` setup.
---

#R11 — Data pipeline

Reference for `data/prepare_data.py`: the thin shim that adapts the shared 8.0B-token CoreProjects pipeline (GPT-2 BPE, vocab 50,257) to Mamba-3-Lite, plus the shard format, the data mix, and the vendored-vs-workspace lookup.

## 60-second summary

`data/prepare_data.py` does not tokenize, download, or clean anything itself: it is a ~110-line adapter that (1) materialises a project-local `data/data_config.yaml` with GPT-2's vocab/EOS/PAD ids, (2) delegates to `shared_data.prepare_data.run_pipeline(...)`, which runs `download_raw → clean → tokenize → pack_shards` as separate subprocesses for crash isolation. The `shared_data` package is **not vendored in this clone** — `data/shared_data/` is absent — but the workspace-level `LLM/shared_data/` **is** present, so `_require_shared_data()` resolves the workspace fallback and the CLI proceeds. The E2E test bypasses the whole shim and feeds `PretrainDataset` a synthetic `torch.save`'d token shard. After reading this doc you can state exactly what the shim does, why the guard passes on this clone, what the shard format is, and how to re-enable the full 8.0B-token pipeline.

## Why it exists

Mamba-3-Lite shares the universal 8.0B-token pretraining corpus with its four sibling LLM projects (GPT-2, LLaMA-3-Lite, DeepSeek-v3-Lite, GPT-OSS-Lite, and HyMo in the workspace listing). The pipeline — download, quality filter, dedup, tokenize, pack — lives once, at the workspace level (`LLM/shared_data/`), and each project's clone is expected to vendor a bit-identical copy at `data/shared_data/` so the repo is self-contained. Mamba-3-Lite's clone ships **no vendored copy**, but the workspace-level `LLM/shared_data/` is present on this machine, so the shim's guard resolves the workspace fallback. What the clone does ship is the adapter, `data/prepare_data.py`, whose only Mamba-specific responsibilities are:

1. Pin the tokenizer contract: GPT-2 BPE, vocab 50,257, EOS/PAD 50,256 (every other project tokenizes with a different vocab, so this is a real per-project decision).
2. Produce `data/data_config.yaml` from the universal config plus those overrides.
3. Call the orchestrator with the right paths and `--skip-*` flags.

The pretrainer never talks to the pipeline: it consumes pre-tokenized shards through `training/pretrain.py:PretrainDataset`, so the pipeline is an offline, run-once-then-forget step.

## Intuition

Think of the shim as a **config adapter + import shim** rather than a data tool. Three layers, outermost first:

- **CLI layer** (`main`): argument parsing and stage gating — the only user-visible surface.
- **Config layer** (`_ensure_mamba_data_config`): loads the universal `data_config.yaml` from the shared package, overwrites four tokenizer fields, stamps provenance keys, writes the project-local copy.
- **Import layer** (`_require_shared_data` + module-level `sys.path` inserts): decides *which* `shared_data` package (vendored or workspace) is importable when the config layer's lazy `from shared_data.config import ...` runs.

The pipeline itself is a black box to this repo: `run_pipeline` spawns one subprocess per stage so a crash in stage 2 leaves stage 1's outputs reusable. "Thin shim" is the correct model — the only Mamba-3-specific logic is the four tokenizer constants and the config override.

## Data mix and shard math

### The documented mix (in-clone source: `docs/training.md`)

The corpus target is 8.0B tokens, Chinchilla-optimal for a ~434M-param model (`configs/pretrain_a100_400m.yaml` trains for 256,000 steps × 16 micro-batch × 2 grad-accum × 2,048 seq = $2^{33} \approx 8.0\times 10^9$ tokens). Per-source token budget is the weight times the total:

$$T_i = w_i \times 8.0 \times 10^9$$

| Source | Weight | Tokens |
|---|---:|---:|
| FineWeb-Edu | 0.50 | $4.00 \times 10^9$ |
| FineWeb | 0.20 | $1.60 \times 10^9$ |
| the-stack-python | 0.15 | $1.20 \times 10^9$ |
| OpenMathInstruct-2 | 0.10 | $0.80 \times 10^9$ |
| arxiv | 0.05 | $0.40 \times 10^9$ |
| **Total** | **1.00** | **$8.00 \times 10^9$** |

**Honesty note — the workspace recipe differs.** The mix the pipeline would actually run is whatever `mixture.yaml` the imported `shared_data` package carries: `main()` passes `UNIVERSAL_MIXTURE_PATH` (`shared_data/config/mixture.yaml`) unless `--mixture` overrides it. The current workspace `LLM/shared_data/config/mixture.yaml` specifies a **7-source** recipe — fineweb-edu 0.40 / dclm-baseline 0.15 / the-stack-v2-python 0.15 / the-stack-v2-jupyter 0.05 / openmath 0.10 / arxiv 0.10 / cosmopedia 0.05 — which contradicts the 5-source table above (verified by reading the workspace file, 2026-08-04). `README.md`'s knobs table lists a third variant (0.6/0.2/0.1/0.1) and is stale. Treat the retired DATA_PIPELINE.md table as documented intent and the workspace `mixture.yaml` as the live recipe — reconcile the two before a real run.

### Shard arithmetic

Shards are 50,000,000 tokens each (the `sharding.shard_size_tokens` field of the materialised `data/data_config.yaml`), so the corpus packs to

$$N_{\text{shards}} = \frac{8.0\times 10^9}{5.0\times 10^7} = 160$$

At 4 bytes per token (uint32), the corpus is $8.0\times 10^9 \times 4\ \text{B} = 32\ \text{GB}$ of token data, ≈190 MiB per shard.

**Correcting a loose claim in `docs/training.md`.** The doc says GPT-2's 50,257-vocab shards are "larger ... since each token takes more bytes ... when the vocabulary is smaller". Per *stored* token every project is identical: uint32, 4 bytes. A smaller vocabulary changes tokenization of text, not storage: fewer BPE merges mean tokens cover *fewer* characters, so the same source text yields *more* tokens. At a fixed 8.0B-token budget the physical layout is identical across projects (160 shards, 32 GB); the smaller vocab only means *less* raw text is needed to hit the budget.

## Code walkthrough

### Module-level contract: constants and paths

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent.parent  # .../CoreProjects/

MAMBA_TOKENIZER_NAME = "gpt2"
MAMBA_VOCAB_SIZE = 50_257
MAMBA_EOS_TOKEN_ID = 50_256
MAMBA_PAD_TOKEN_ID = 50_256
```

`data/prepare_data.py:MAMBA_TOKENIZER_NAME` is the one project decision: GPT-2 BPE. `data/prepare_data.py:MAMBA_VOCAB_SIZE` = 50,257 (the GPT-2 vocab — 50,000 merges + 256 byte-level + 1 `<|endoftext|>`), and `data/prepare_data.py:MAMBA_EOS_TOKEN_ID` = `data/prepare_data.py:MAMBA_PAD_TOKEN_ID` = 50,256: **EOS and PAD share one id**, which matters for how the packed corpus reads (see Pitfalls). `_PROJECT_ROOT` is the repo root; `_LLM_ROOT` is `CoreProjects/LLM/` — the workspace level where `shared_data/` actually lives (see Vendored-vs-workspace lookup).

### The guard: `_require_shared_data`

```python
def _require_shared_data() -> None:
    """Raise a clean, actionable error if the vendored pipeline is missing."""
    vendored = _PROJECT_ROOT / "data" / "shared_data"
    workspace = _LLM_ROOT / "shared_data" if _LLM_ROOT.exists() else None
    if vendored.exists() or (workspace is not None and workspace.exists()):
        return
    raise FileNotFoundError(
        "Mamba-3-Lite data prep requires the `shared_data` package. "
        f"Neither {vendored} nor {workspace} was found on this clone. "
        "Vendor `shared_data/` from a sibling CoreProjects repo, or use a "
        "pre-tokenized uint32 shard with `PretrainDataset` (see tests/e2e_gpu_smoke.py)."
    )
```

`data/prepare_data.py:_require_shared_data` is called first thing in `main()`, before argparse. On this clone the vendored copy is absent but the workspace fallback `LLM/shared_data/` exists, so the guard returns normally and the CLI proceeds to parse args and delegate to the workspace pipeline. Verified behaviour on this clone:

```
$ python3 data/prepare_data.py
[data/mamba3] universal corpus: 8,000,000,000 tokens
[data/mamba3] tokenizer: gpt2 (vocab=50,257, EOS=50,256)
[data/mamba3] shard size: 50,000,000 tokens (uint32)
```

The `FileNotFoundError` branch fires only when *neither* `data/shared_data/` nor `LLM/shared_data/` exists (e.g. a standalone clone outside the workspace).

### The `sys.path` dance

```python
for _p in (_PROJECT_ROOT, _LLM_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

At module import, the repo root and `CoreProjects/LLM/` are prepended to `sys.path`. The `shared_data` imports are deliberately **lazy** (inside `_ensure_mamba_data_config` and `main`), so importing the shim never fails even when the package is absent — only running the CLI does. Two precision points:

- The vendored import resolves through `sys.path[0]`, not through these inserts: invoked as `python3 data/prepare_data.py`, Python puts the *script's* directory — `data/` — at `sys.path[0]`, so `import shared_data` finds `data/shared_data/`. The `_PROJECT_ROOT` insert (`<repo root>` on the path) would only serve a package at `<repo root>/shared_data`, which nobody ships.
- `docs/training.md` documents this dance as inserting `<project_root>/data` then `<workspace_root>/LLM`. The code inserts `<project_root>` and `CoreProjects/LLM` — the workspace root itself, which is exactly where the fallback `shared_data/` lives. The doc's `<project_root>/data` is the script-directory effect (`sys.path[0]`), not an explicit insert; see the lookup box under Vendored-vs-workspace lookup.

### `_ensure_mamba_data_config`

```python
def _ensure_mamba_data_config(project_root: Path) -> Path:
    """Materialise a project-local data_config.yaml with Mamba-3-Lite's vocab."""
    from shared_data.config import UNIVERSAL_DATA_CONFIG_PATH
    from shared_data.common import load_yaml

    out_path = project_root / "data" / "data_config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(UNIVERSAL_DATA_CONFIG_PATH)
    cfg["pipeline"]["tokenizer"]["name"] = MAMBA_TOKENIZER_NAME
    cfg["pipeline"]["tokenizer"]["vocab_size"] = MAMBA_VOCAB_SIZE
    cfg["pipeline"]["tokenizer"]["eos_token_id"] = MAMBA_EOS_TOKEN_ID
    cfg["pipeline"]["tokenizer"]["pad_token_id"] = MAMBA_PAD_TOKEN_ID
    cfg["_generator"] = "Mamba-3-Lite/data/prepare_data.py"
    cfg["_tokenizer_family"] = "gpt2"

    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
```

`data/prepare_data.py:_ensure_mamba_data_config` loads the universal config from the shared package, overwrites exactly four tokenizer fields, stamps `_generator`/`_tokenizer_family` provenance keys, and writes `data/data_config.yaml`. **It touches nothing else** — mixture, shard size, quality thresholds, dedup, seed all come from the shared package. The clone ships a materialised `data/data_config.yaml` (whose `_generator` confirms provenance): `dtype: uint32`, `shard_size_tokens: 50000000`, `target_total_tokens: 8000000000`, `add_eos: true`, `cross_document_boundary_ok: false`, `verify_after_pack: true`.

### `main()`: CLI and delegation

```python
def main() -> int:
    _require_shared_data()

    parser = argparse.ArgumentParser(
        description="Mamba-3-Lite data prep (delegates to universal pipeline)"
    )
    parser.add_argument("--mixture", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    args = parser.parse_args()

    project_data_config = _apply_mamba_defaults()

    from shared_data.config import UNIVERSAL_MIXTURE_PATH
    from shared_data.prepare_data import run_pipeline

    return run_pipeline(
        mixture_path=Path(args.mixture) if args.mixture else UNIVERSAL_MIXTURE_PATH,
        data_config_path=Path(args.data_config) if args.data_config else project_data_config,
        source=args.source,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenize=args.skip_tokenize,
        skip_pack=args.skip_pack,
        data_root=Path(args.data_root) if args.data_root else None,
    )
```

`data/prepare_data.py:main`'s flags map 1:1 onto `run_pipeline`'s keyword-only parameters (verified against the workspace orchestrator): `mixture_path`, `data_config_path`, `source`, `skip_download`, `skip_clean`, `skip_tokenize`, `skip_pack`, `data_root`, plus a `skip_train_tokenizer=True` default the shim does not expose — sensible, because GPT-2 BPE is a pretrained HF tokenizer that needs no training. The return value is the orchestrator's process exit code (0 success; 2 = a required config file missing). Before delegating, `_apply_mamba_defaults` prints the contract:

```
[data/mamba3] universal corpus: 8,000,000,000 tokens
[data/mamba3] tokenizer: gpt2 (vocab=50,257, EOS=50,256)
[data/mamba3] shard size: 50,000,000 tokens (uint32)
```

The `--skip-*` flags implement the documented quick-start ladder: full run, re-use corpus (`--skip-download`), re-pack only (`--skip-download --skip-clean --skip-tokenize`).

### Downstream consumption: `PretrainDataset`

The pretrainer reads shards through `training/pretrain.py:PretrainDataset` (fully covered in [Mamba-3-Lite — Training Reference](../references/training-reference.md)); the three facts this doc needs:

- **Layout dispatch** (`training/pretrain.py:PretrainDataset.__init__`): a missing path → `layout = "dummy"` (1,000 random windows, warn only — see Pitfalls); a directory → `_init_sharded` (glob `shard_*.bin`, `torch.load(..., mmap=True)`, bisect via `_locate`); a file → `_init_single` (`torch.load(weights_only=True)`).
- **Windows**: `max_seq_len+1` tokens per sample, `x = chunk[:-1]`, `y = chunk[1:]` (`training/pretrain.py:PretrainDataset._get_window_single`); `n_samples = (len(data) - 1) // max_seq_len`.
- The config's `data:` section is consulted for exactly one key — `train_data_path` (with fallback default `data/pretrain_data.bin`); `data_mix`, `shard_size_tokens`, `max_tokens`, and `tokenizer` are **never read** by `training/pretrain.py:main` (verified by repo-wide search: `data_mix` appears only in `README.md` and the YAML, in zero code files). The YAML's `data_mix: "mamba2-default"` plus the comment line is a spec annotation, not a runtime knob — the real mix lives in the shared package's `mixture.yaml`.

## Vendored-vs-workspace lookup

The lookup order, as *documented* in `docs/training.md`: vendored `data/shared_data/` first, then workspace `LLM/shared_data/`, with the workspace copy intended as the fallback for standalone clones that still sit inside `CoreProjects`. The code implements exactly that:

- `_LLM_ROOT = _PROJECT_ROOT.parent` resolves to **`CoreProjects/LLM/`** — the workspace root — so the guard checks `LLM/shared_data` and the path insert exposes `LLM/` on `sys.path`, which is precisely where the workspace copy lives. **Measured consequence on this machine**: `LLM/shared_data/` exists, so the guard's workspace branch succeeds and the CLI runs against the workspace copy. The vendored branch is the only one that can fail here, because `data/shared_data/` is absent.
- Because the vendored branch works only via `sys.path[0]` (script directory = `data/`), the recommended invocation is exactly `python3 data/prepare_data.py` from the repo root. Running the module as `python3 -m data.prepare_data` puts the repo root at `sys.path[0]` and would not resolve `data/shared_data/` — though the workspace fallback (via the `_LLM_ROOT` insert) would still serve `LLM/shared_data/`.

**To make the repo fully self-contained**: vendor the package (`rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' LLM/shared_data/ data/shared_data/` from the workspace, per `docs/training.md`), or copy it from a sibling project that already vendors it (LLaMA-3-Lite's `data/shared_data/` exists on this machine, though its `config/` tree is incomplete there — verify before relying on it). With `data/shared_data/` present, the vendored branch wins and the clone no longer depends on the workspace copy. If the workspace copy is absent (standalone clone), `_require_shared_data` raises the clean `FileNotFoundError` naming both candidate locations.

## Shard format

In-clone authoritative statement (`docs/training.md`): shards are **uint32, 50,000,000 tokens, EOS-separated** — every document boundary carries the configured EOS id, and documents are never split across shards (`cross_document_boundary_ok: false` in `data/data_config.yaml`). The workspace canonical `LLM/shared_data/README.md` (§7) and `shard_writer.py` pin the physical format: a **flat raw uint32 little-endian buffer**, ≈190 MiB, written atomically via `.tmp` + rename, re-read and SHA-256-verified into the manifest, with sources round-robin-interleaved so every shard sees every source. Readers there use `np.memmap(..., dtype=np.uint32)`.

**The E2E bypass is a different format.** `tests/e2e_gpu_smoke.py::_build_synthetic_shard` writes

```python
tokens = torch.randint(0, vocab, (n_tokens,), dtype=torch.long)
torch.save(tokens, path)
```

i.e. a `torch.save`'d **int64** tensor, not a raw uint32 buffer. `PretrainDataset`'s `torch.load`-based loaders match the bypass, not the raw format; `_init_sharded` does `torch.load(p, weights_only=True, mmap=True)` on `shard_*.bin`, which cannot parse a raw uint32 dump. `[INFERENCE]` Consuming real workspace shards through `PretrainDataset` therefore needs a conversion shim (`torch.from_numpy(np.memmap(...))` or a re-pack), unless the vendored copy's pack stage writes torch archives. Fine for pipeline testing — `PretrainDataset` only needs token ids — but a genuine integration gap before a production run.

## Pitfalls

1. **The guard passes on this clone via the workspace fallback.** `_require_shared_data`'s workspace branch checks `LLM/shared_data` (via `_LLM_ROOT = _PROJECT_ROOT.parent`), which **exists** on this machine, so the guard returns normally and the CLI does **not** fail fast — it runs against the workspace copy. The `FileNotFoundError` fires only on a standalone clone with neither `data/shared_data/` nor `LLM/shared_data/` present. The practical consequence of the absent vendored copy is workspace coupling, not a hard failure.
2. **Silent dummy-data training.** If `train_data_path` is missing, `training/pretrain.py:PretrainDataset` switches to `layout = "dummy"` and the trainer happily runs on `torch.randint` garbage with only a warn line. The default path `data/pretrain_data.bin` does not exist on this clone — an unconfigured run will "train" on noise and look plausible in the loss curve.
3. **`data_mix`, `shard_size_tokens`, `max_tokens`, `tokenizer` in `configs/pretrain_a100_400m.yaml` are not read by the pretrainer.** They are spec annotations. Changing them changes nothing until the pipeline is re-run; the real knobs are `--mixture`/`--data-config` on the shim and the shared package's `mixture.yaml`.
4. **EOS == PAD == 50,256.** One id serves as document separator and padding. With `add_eos: true` and `cross_document_boundary_ok: false`, every window boundary in the packed corpus is a real document boundary — the model never sees padding-as-content.
5. **The "bigger GPT-2 shards" claim is imprecise:** all projects store uint32 (4 B/token) and 160 shards. GPT-2's 50,257 vocab yields *more, shorter* tokens per byte of source text — less raw text needed, not bigger shards.
6. **Invocation form matters.** The vendored import resolves via `sys.path[0]`; run `python3 data/prepare_data.py` from the repo root, not `-m data.prepare_data` from elsewhere.
7. **`--stage` accepts only `pretrain`.** The flag exists for forward-compatibility; there is no other stage today.

## Tests

- `tests/e2e_gpu_smoke.py::check_data_pipeline` (check 2/8, GPU-gated): writes a synthetic 4,096-token int64 shard (`tests/e2e_gpu_smoke.py::_build_synthetic_shard`), builds `PretrainDataset(..., max_seq_len=32, vocab_size=128)` → 127 windows `(4096-1)//32`, asserts `(32,)` sample shapes, `(4, 32)` CUDA batches via `DataLoader`, and `inp_b.is_cuda`. Exercises the full shim-free path: synthetic shard → dataset → loader → GPU.
- `tests/e2e_gpu_smoke.py::check_pretrainer_dry_run` (check 8/8) reuses the same bypass (2,048 tokens) for a 2-step `Pretrainer` dry-run.
- `tests/test_doc_refs.py` (CPU) is the gate for this document: every `data/prepare_data.py:*` anchor cited here must resolve on a triton-less box, and the module must import without `shared_data` (its imports are lazy, so it does).

See [Mamba-3-Lite — Training Reference](../references/training-reference.md) for `PretrainDataset` internals, [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md) for how `--data-path` overrides `train_data_path`, [Mamba-3-Lite — Config Reference](../references/config-reference.md) for the annotated YAML, [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) for the Chinchilla-optimal token budget, and [Mamba-3-Lite — Training Runbook](../guides/training-runbook.md) for the end-to-end recipe.

---

## References

- [Mamba-3-Lite — Training](../training.md) — the end-to-end data path: corpus mix, shard format, and how `training/pretrain.py:PretrainDataset` consumes it.
- [Mamba-3-Lite — Pretrain CLI](../guides/pretrain-cli.md) — `training/pretrain.py:TrainingConfig`, CLI flags, and the loop that consumes all four modules.
- [Mamba-3-Lite — Config Reference](config-reference.md) — the `training:` YAML keys mapped to `TrainingConfig` fields.
- [Mamba-3-Lite — Training Runbook](../guides/training-runbook.md) — operational checkpoint/NaN recovery.
- [Mamba-3-Lite — Block Anatomy and Numerical Stability](../concepts/block-and-stability.md) — the NaN-guard rationale behind checkpoint rollback.
