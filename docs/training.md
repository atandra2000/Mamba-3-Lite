# Mamba-3-Lite — Training: The Data Pipeline and Dataset Path

This doc covers the data path end to end: the 8.0B-token corpus mix, the `data/prepare_data.py:main` shim that adapts the shared CoreProjects pipeline to Mamba-3-Lite's tokenizer, the on-disk shard format, and how `training/pretrain.py:PretrainDataset` consumes the shards inside the training loop. The full API-level details of the dataset, checkpointing, and logging live in [Mamba-3-Lite — Training Reference](references/training-reference.md); this doc is the pipeline narrative.

## Quick start

```bash
# Full pipeline (download → clean → tokenize → pack)
python3 data/prepare_data.py

# Skip download (re-use an existing corpus)
python3 data/prepare_data.py --skip-download

# Re-pack only (after a config change)
python3 data/prepare_data.py --skip-download --skip-clean --skip-tokenize
```

`data/prepare_data.py:main` delegates to `shared_data.prepare_data.run_pipeline(...)`, which runs `download_raw → clean → tokenize → pack_shards` as separate subprocesses (crash isolation: a failure in one stage leaves the previous stages' outputs reusable). The `--skip-*` flags map 1:1 onto the orchestrator's keyword arguments, plus a `skip_train_tokenizer=True` default the shim does not expose — GPT-2 BPE is a pretrained HuggingFace tokenizer that needs no training.

## Tokenizer used by Mamba-3-Lite

| Field | Value |
|---|---|
| Family | GPT-2 BPE (HuggingFace) |
| Vocab size | **50,257** |
| EOS id | **50,256** |
| PAD id | 50,256 |

`data/prepare_data.py:MAMBA_TOKENIZER_NAME` is the one project decision; `data/prepare_data.py:MAMBA_VOCAB_SIZE` = 50,257 (50,000 merges + 256 byte-level + 1 `<|endoftext|>`), and `data/prepare_data.py:MAMBA_EOS_TOKEN_ID` = `data/prepare_data.py:MAMBA_PAD_TOKEN_ID` = 50,256 — **EOS and PAD share one id**, which matters for how the packed corpus reads: with `add_eos: true` and `cross_document_boundary_ok: false`, every window boundary in the packed corpus is a real document boundary, and the model never sees padding-as-content.

## What the shim does

`data/prepare_data.py` does not tokenize, download, or clean anything itself: it is a ~110-line adapter that

1. Materialises a project-local `data/data_config.yaml` with GPT-2's vocab/EOS/PAD ids (overwriting exactly four tokenizer fields of the universal config, stamping `_generator`/`_tokenizer_family` provenance keys, touching nothing else — mixture, shard size, quality thresholds, dedup, and seed all come from the shared package).
2. Delegates to `shared_data.prepare_data.run_pipeline(...)` with the right paths and `--skip-*` flags.

The pretrainer never talks to the pipeline: it consumes pre-tokenized shards through `training/pretrain.py:PretrainDataset`, so the pipeline is an offline, run-once-then-forget step. The clone ships a materialised `data/data_config.yaml` (whose `_generator` confirms provenance): `dtype: uint32`, `shard_size_tokens: 50000000`, `target_total_tokens: 8000000000`, `add_eos: true`, `cross_document_boundary_ok: false`, `verify_after_pack: true`.

## Data mix (8.0B tokens, Chinchilla-optimal for ~434M-param models)

Per-source token budget is the weight times the total: $T_i = w_i \times 8.0 \times 10^9$.

| Source | Weight | Tokens |
|---|---:|---:|
| FineWeb-Edu (HuggingFaceFW) | 0.50 | 4.00 B |
| FineWeb (HuggingFaceFW) | 0.20 | 1.60 B |
| the-stack-python (bigcode) | 0.15 | 1.20 B |
| OpenMathInstruct-2 (nvidia) | 0.10 | 0.80 B |
| arxiv (cdv) | 0.05 | 0.40 B |
| **Total** | **1.00** | **8.00 B** |

**Honesty note — the workspace recipe differs.** The mix the pipeline would actually run is whatever `mixture.yaml` the imported `shared_data` package carries: `main()` passes `UNIVERSAL_MIXTURE_PATH` (`shared_data/config/mixture.yaml`) unless `--mixture` overrides it. The current workspace `LLM/shared_data/config/mixture.yaml` specifies a **7-source** recipe — fineweb-edu 0.40 / dclm-baseline 0.15 / the-stack-v2-python 0.15 / the-stack-v2-jupyter 0.05 / openmath 0.10 / arxiv 0.10 / cosmopedia 0.05 — which contradicts the 5-source table above (verified by reading the workspace file, 2026-08-04). A stale third variant (0.6/0.2/0.1/0.1) also appears in the root `README.md`'s knobs table. Treat the table above as documented intent and the workspace `mixture.yaml` as the live recipe — reconcile the two before a real run.

## Shard arithmetic and format

Shards are 50,000,000 tokens each (`sharding.shard_size_tokens` in the materialised config), so the corpus packs to

$$N_{\text{shards}} = \frac{8.0\times 10^9}{5.0\times 10^7} = 160$$

At 4 bytes per token (uint32), the corpus is $8.0\times 10^9 \times 4\ \text{B} = 32\ \text{GB}$ of token data, ≈190 MiB per shard.

**Correcting a loose claim in the retired `data/DATA_PIPELINE.md`.** That doc said GPT-2's 50,257-vocab shards are "larger ... since each token takes more bytes ... when the vocabulary is smaller". Per *stored* token every project is identical: uint32, 4 bytes. A smaller vocabulary changes tokenization of text, not storage: fewer BPE merges mean tokens cover *fewer* characters, so the same source text yields *more* tokens. At a fixed 8.0B-token budget the physical layout is identical across projects (160 shards, 32 GB); the smaller vocab only means *less* raw text is needed to hit the budget.

**The canonical workspace format** (`LLM/shared_data/README.md` §7 and `shard_writer.py`): a **flat raw uint32 little-endian buffer**, ≈190 MiB, written atomically via `.tmp` + rename, re-read and SHA-256-verified into the manifest, with sources round-robin-interleaved so every shard sees every source. Readers there use `np.memmap(..., dtype=np.uint32)`.

**The E2E bypass is a different format.** `tests/e2e_gpu_smoke.py::_build_synthetic_shard` writes a `torch.save`'d **int64** tensor, not a raw uint32 buffer. `PretrainDataset`'s `torch.load`-based loaders match the bypass, not the raw format; `_init_sharded` does `torch.load(p, weights_only=True, mmap=True)` on `shard_*.bin`, which cannot parse a raw uint32 dump. `[INFERENCE]` Consuming real workspace shards through `PretrainDataset` therefore needs a conversion shim (`torch.from_numpy(np.memmap(...))` or a re-pack), unless the vendored copy's pack stage writes torch archives. Fine for pipeline testing — `PretrainDataset` only needs token ids — but a genuine integration gap before a production run.

## The shim's internals

### Module-level contract: constants and paths

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent  # .../CoreProjects/LLM/ (workspace shared_data lives here)

MAMBA_TOKENIZER_NAME = "gpt2"
MAMBA_VOCAB_SIZE = 50_257
MAMBA_EOS_TOKEN_ID = 50_256
MAMBA_PAD_TOKEN_ID = 50_256
```

`_PROJECT_ROOT` is the repo root; `_LLM_ROOT` is `CoreProjects/LLM/` — the workspace level where `shared_data/` actually lives.

### The guard: `data/prepare_data.py:_require_shared_data`

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

Called first thing in `main()`, before argparse. On this clone the vendored copy is absent but the workspace fallback `LLM/shared_data/` exists, so the guard returns normally and the CLI proceeds to delegate to the workspace pipeline. The `FileNotFoundError` branch fires only when *neither* `data/shared_data/` nor `LLM/shared_data/` exists (e.g. a standalone clone outside the workspace). The `shared_data` imports are deliberately **lazy** (inside `_ensure_mamba_data_config` and `main`), so importing the shim never fails even when the package is absent — only running the CLI does.

### The `sys.path` dance

```python
for _p in (_PROJECT_ROOT, _LLM_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

At module import, the repo root and `CoreProjects/LLM/` are prepended to `sys.path`. Two precision points:

- The vendored import resolves through `sys.path[0]`, not through these inserts: invoked as `python3 data/prepare_data.py`, Python puts the *script's* directory — `data/` — at `sys.path[0]`, so `import shared_data` finds `data/shared_data/`. The `_PROJECT_ROOT` insert (`<repo root>` on the path) would only serve a package at `<repo root>/shared_data`, which nobody ships.
- The retired `data/DATA_PIPELINE.md` documented this dance as inserting `<project_root>/data` then `<workspace_root>/LLM`. The code inserts `<project_root>` and `CoreProjects/LLM` — the workspace root itself, which is exactly where the fallback `shared_data/` lives. The doc's `<project_root>/data` was the script-directory effect (`sys.path[0]`), not an explicit insert.

### `data/prepare_data.py:_ensure_mamba_data_config`

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

Loads the universal config, overwrites exactly four tokenizer fields, stamps provenance keys, writes `data/data_config.yaml`.

### `data/prepare_data.py:main` — CLI and delegation

`main`'s flags map 1:1 onto `run_pipeline`'s keyword-only parameters (verified against the workspace orchestrator): `mixture_path`, `data_config_path`, `source`, `skip_download`, `skip_clean`, `skip_tokenize`, `skip_pack`, `data_root`, plus a `skip_train_tokenizer=True` default the shim does not expose. The return value is the orchestrator's process exit code (0 success; 2 = a required config file missing). Before delegating, `_apply_mamba_defaults` prints the contract:

```
[data/mamba3] universal corpus: 8,000,000,000 tokens
[data/mamba3] tokenizer: gpt2 (vocab=50,257, EOS=50,256)
[data/mamba3] shard size: 50,000,000 tokens (uint32)
```

## Vendored-vs-workspace lookup

The lookup order: vendored `data/shared_data/` first, then workspace `LLM/shared_data/`, with the workspace copy intended as the fallback for standalone clones that still sit inside `CoreProjects`. **Measured consequence on this machine**: `LLM/shared_data/` exists, so the guard's workspace branch succeeds and the CLI runs against the workspace copy. The vendored branch is the only one that can fail here, because `data/shared_data/` is absent — the repo is therefore **not** fully self-contained on a fresh clone.

Because the vendored branch works only via `sys.path[0]` (script directory = `data/`), the recommended invocation is exactly `python3 data/prepare_data.py` from the repo root. Running the module as `python3 -m data.prepare_data` puts the repo root at `sys.path[0]` and would not resolve `data/shared_data/` — though the workspace fallback (via the `_LLM_ROOT` insert) would still serve `LLM/shared_data/`.

**To make the repo fully self-contained**, vendor the package from the workspace root:

```bash
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
    LLM/shared_data/  LLM/Mamba-3-Lite/data/shared_data/
```

With `data/shared_data/` present, the vendored branch wins and the clone no longer depends on the workspace copy. The workspace-level pipeline may receive updates; the 5 vendored copies across the portfolio are kept **bit-identical**.

## Downstream consumption: `PretrainDataset`

The pretrainer reads shards through `training/pretrain.py:PretrainDataset` (full API in [Mamba-3-Lite — Training Reference](references/training-reference.md)); the facts that matter at the pipeline level:

- **Layout dispatch** (`training/pretrain.py:PretrainDataset.__init__`): a missing path → `layout = "dummy"` (1,000 random windows, warn only); a directory → `_init_sharded` (glob `shard_*.bin`, `torch.load(..., mmap=True)`, bisect via `_locate`); a file → `_init_single` (`torch.load(weights_only=True)`).
- **Windows**: `max_seq_len+1` tokens per sample, `x = chunk[:-1]`, `y = chunk[1:]`; `n_samples = (len(data) - 1) // max_seq_len`. Consecutive windows overlap by exactly one token; nothing re-segments documents; EOS ids between documents are ordinary tokens in the stream.
- The config's `data:` section is consulted for exactly one key — `train_data_path` (with fallback default `data/pretrain_data.bin`); `data_mix`, `shard_size_tokens`, `max_tokens`, and `tokenizer` are **never read** by `training/pretrain.py:main` (verified by repo-wide search: `data_mix` appears only in `README.md` and the YAML, in zero code files). The YAML's `data_mix` plus its comment line is a spec annotation, not a runtime knob — the real mix lives in the shared package's `mixture.yaml`.

## Pitfalls

1. **The guard passes on this clone via the workspace fallback** — the CLI runs against `LLM/shared_data/`, coupling the repo to the workspace. The `FileNotFoundError` fires only on a standalone clone with neither copy present.
2. **Silent dummy-data training.** If `train_data_path` is missing, `training/pretrain.py:PretrainDataset` switches to `layout = "dummy"` and the trainer happily runs on `torch.randint` garbage with only a `[warn]` line. The default path `data/pretrain_data.bin` does not exist on this clone — an unconfigured run will "train" on noise and look plausible in the loss curve (loss converges toward $\ln(\text{vocab}) \approx 10.82$). Grep the run log for the warn line before trusting any run that did not use a synthetic shard.
3. **`data_mix`, `shard_size_tokens`, `max_tokens`, `tokenizer` in `configs/pretrain_a100_400m.yaml` are not read by the pretrainer.** They are spec annotations; changing them changes nothing until the pipeline is re-run. The real knobs are `--mixture`/`--data-config` on the shim and the shared package's `mixture.yaml`.
4. **EOS == PAD == 50,256.** One id serves as document separator and padding; with `add_eos: true` and `cross_document_boundary_ok: false`, the model never sees padding-as-content.
5. **Invocation form matters.** The vendored import resolves via `sys.path[0]`; run `python3 data/prepare_data.py` from the repo root, not `-m data.prepare_data` from elsewhere.
6. **mmap keeps file handles and page-cache references.** The sharded layout calls `torch.load(..., mmap=True)`, so `shard_*.bin` files stay open and memory-mapped for the dataset's lifetime; deleting or re-saving a shard mid-run corrupts the view. The single layout is the opposite extreme: everything is loaded eagerly, so a multi-billion-token corpus is a multi-GB resident tensor.
7. **Spanning windows are slow.** The sharded stitching path does `collected.extend(… .tolist())` — every token becomes a Python `int`, then the whole window is rebuilt. If shards are small relative to `max_seq_len`, most windows span and this dominates data-loading time.
8. **Shards must be `torch.save`'d tensors.** `weights_only=True` rejects arbitrary pickled objects, and `_locate`/slicing assumes 1-D integer tensors. The workspace's raw on-disk format is *not* directly loadable — convert it first; `tests/e2e_gpu_smoke.py:_build_synthetic_shard` shows the exact expected format.
9. **Dummy mode is not seeded and not next-token-aligned.** `torch.randint` in `__getitem__` draws from the global RNG (no seed), and `y` is *not* `x` shifted — the mode only exercises plumbing, never measures real loss.
10. **No shuffling, `drop_last=True`.** The `DataLoader` has no `shuffle=True` (every epoch scans windows in the same order) and silently discards a final partial batch.

## Tests

- `tests/e2e_gpu_smoke.py::check_data_pipeline` (check 2/8, GPU-gated): writes a synthetic 4,096-token int64 shard (`tests/e2e_gpu_smoke.py::_build_synthetic_shard`), builds `PretrainDataset(..., max_seq_len=32, vocab_size=128)` → 127 windows `(4096-1)//32`, asserts `(32,)` sample shapes, `(4, 32)` CUDA batches via `DataLoader`, and `inp_b.is_cuda`. Exercises the full shim-free path: synthetic shard → dataset → loader → GPU.
- `tests/e2e_gpu_smoke.py::check_pretrainer_dry_run` (check 8/8) reuses the same bypass (2,048 tokens) for a 2-step `Pretrainer` dry-run.
- `tests/test_doc_refs.py` (CPU) is the gate for this document: every `data/prepare_data.py:*` anchor cited here must resolve on a triton-less box, and the module must import without `shared_data` (its imports are lazy, so it does).

## References

- [Mamba-3-Lite — Training Reference](references/training-reference.md) — the full `training/pretrain.py:PretrainDataset` API, `utils/checkpoint.py:CheckpointManager`, `utils/logging.py:TrainingLogger`.
- [Mamba-3-Lite — Pretrain CLI](guides/pretrain-cli.md) — how `--data-path` overrides `train_data_path`, and the loop that consumes the dataset.
- [Mamba-3-Lite — Config Reference](references/config-reference.md) — the annotated YAML and the 16.78B-vs-8.0B token arithmetic.
- [Mamba-3-Lite — Block Anatomy and Numerical Stability](concepts/block-and-stability.md) — the Chinchilla-optimal token budget the corpus size is derived from.
- Workspace-level canonical docs: `LLM/shared_data/README.md`, `LLM/shared_data/config/mixture.yaml`, `LLM/shared_data/config/data_config.yaml`, and the per-module deep-dives in `LLM/shared_data/documentation/`.
