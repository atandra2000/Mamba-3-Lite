# R8 — PretrainDataset

Reference for `training/pretrain.py:PretrainDataset`: the packed-token dataset that turns flat, pre-tokenized token streams into `(inputs, targets)` windows, covering its three storage layouts (single file, sharded directory, dummy), the window math, shard-spanning reads, and its `DataLoader` contract inside the training loop.

## 60-second summary

`training/pretrain.py:PretrainDataset` is a thin `torch.utils.data.Dataset` over one or more flat `torch.long` token tensors. It picks a layout by probing `data_path`: `dummy` when the path is missing, `sharded` when it is a directory, `single` otherwise. Every sample is a window of exactly `max_seq_len + 1` tokens sliced as `x = chunk[:-1]`, `y = chunk[1:]` — perfect next-token alignment, no EOS handling, no padding. `training/pretrain.py:Pretrainer.train` consumes it through a `DataLoader` with `batch_size`, `num_workers=0`, `drop_last=True`, and no shuffling. Shards are memory-mapped; windows straddling a shard boundary are stitched through a Python list.

## Why it exists

Pre-training a ~434M next-token model needs a data path that streams flat token streams with zero per-sample tokenization cost, scales past a single file via sharding, and runs end-to-end on a box with no data at all (CPU smoke tests, `--dry-run`). `PretrainDataset` satisfies all three, which is why the layout is a runtime `if` on the *shape* of `data_path`, not a configuration flag. Tokenization and document packing happen upstream in the data pipeline (`data/prepare_data.py` shim → the workspace-level `LLM/shared_data/` pipeline, see [R11 — data pipeline](11-data-pipeline.md)); this class only sees already-packed token ids.

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

`max_seq_len` is the window length `L` (2048 in production; see [R1 — ModelConfig](01-model-config.md), which notes this field is *not* consumed by the model architecture, only by the dataset and the logger). `vocab_size` is used **only** in dummy mode as the upper bound for random ids; it never validates real data. It holds no device state and always returns CPU tensors.

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

The production default `data_path` in `training/pretrain.py:TrainingConfig` is `"data/pretrain_data.bin"` — a *file* path — and `main()` resolves it as `args.data_path or yaml data.train_data_path or "data/pretrain_data.bin"` (see [R7 — pretrain CLI](07-pretrain-cli.md)). A default run therefore expects the single-file layout; sharded mode is opted into by pointing `--data-path` at a directory of `shard_*.bin` files.

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

There is **no EOS special handling, no padding, no masking at this level.** `training/pretrain.py:train_step`'s `cross_entropy` consumes every position of `y`; the model's causal structure (see [T4 — chunkwise algorithm](../theory/04-chunkwise-algorithm.md), [T6 — block anatomy](../theory/06-block-anatomy.md)) enforces "don't look ahead". If the upstream pipeline inserted EOS/PAD (id 50,256) between documents, those ids are trained on like any other token — document semantics are the pipeline's job ([R11 — data pipeline](11-data-pipeline.md)); packing is this class's job.

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

The loader is created **without `shuffle=True`**, so epoch order is the deterministic index order `0, 1, 2, …` every epoch. Batches move to the device with `tokens.to(self.device, non_blocking=True)` before `train_step` (see [R7 — pretrain CLI](07-pretrain-cli.md) for the loop contract, [R9 — checkpoint](09-checkpoint.md) for the resume preceding the loop, and [R10 — logging](10-logging.md) for how `max_seq_len` and `batch_size` feed the `tps` metric).

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

See [T4 — chunkwise algorithm](../theory/04-chunkwise-algorithm.md) / [T6 — block anatomy](../theory/06-block-anatomy.md) for what happens to each window inside the model, [R1 — ModelConfig](01-model-config.md) for `max_seq_len`'s role, [R11 — data pipeline](11-data-pipeline.md) for how the token stream is produced, and [R7 — pretrain CLI](07-pretrain-cli.md) / [R9 — checkpoint](09-checkpoint.md) / [R10 — logging](10-logging.md) for the surrounding training loop.
