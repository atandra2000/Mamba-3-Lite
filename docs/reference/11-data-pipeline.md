# R11 — Data pipeline

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

### The documented mix (in-clone source: `data/DATA_PIPELINE.md`)

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

**Honesty note — the workspace recipe differs.** The mix the pipeline would actually run is whatever `mixture.yaml` the imported `shared_data` package carries: `main()` passes `UNIVERSAL_MIXTURE_PATH` (`shared_data/config/mixture.yaml`) unless `--mixture` overrides it. The current workspace `LLM/shared_data/config/mixture.yaml` specifies a **7-source** recipe — fineweb-edu 0.40 / dclm-baseline 0.15 / the-stack-v2-python 0.15 / the-stack-v2-jupyter 0.05 / openmath 0.10 / arxiv 0.10 / cosmopedia 0.05 — which contradicts the 5-source table above (verified by reading the workspace file, 2026-08-04). `README.md`'s knobs table lists a third variant (0.6/0.2/0.1/0.1) and is stale. Treat the DATA_PIPELINE.md table as documented intent and the workspace `mixture.yaml` as the live recipe — reconcile the two before a real run.

### Shard arithmetic

Shards are 50,000,000 tokens each (the `sharding.shard_size_tokens` field of the materialised `data/data_config.yaml`), so the corpus packs to

$$N_{\text{shards}} = \frac{8.0\times 10^9}{5.0\times 10^7} = 160$$

At 4 bytes per token (uint32), the corpus is $8.0\times 10^9 \times 4\ \text{B} = 32\ \text{GB}$ of token data, ≈190 MiB per shard.

**Correcting a loose claim in `data/DATA_PIPELINE.md`.** The doc says GPT-2's 50,257-vocab shards are "larger ... since each token takes more bytes ... when the vocabulary is smaller". Per *stored* token every project is identical: uint32, 4 bytes. A smaller vocabulary changes tokenization of text, not storage: fewer BPE merges mean tokens cover *fewer* characters, so the same source text yields *more* tokens. At a fixed 8.0B-token budget the physical layout is identical across projects (160 shards, 32 GB); the smaller vocab only means *less* raw text is needed to hit the budget.

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
$ python3 data/prepare_data.py --stage pretrain
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
- `data/DATA_PIPELINE.md` documents this dance as inserting `<project_root>/data` then `<workspace_root>/LLM`. The code inserts `<project_root>` and `CoreProjects/LLM` — the workspace root itself, which is exactly where the fallback `shared_data/` lives. The doc's `<project_root>/data` is the script-directory effect (`sys.path[0]`), not an explicit insert; see the lookup box under Vendored-vs-workspace lookup.

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
    parser.add_argument("--stage", choices=["pretrain"], default="pretrain")
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

`data/prepare_data.py:main`'s flags map 1:1 onto `run_pipeline`'s keyword-only parameters (verified against the workspace orchestrator): `mixture_path`, `data_config_path`, `source`, `skip_download`, `skip_clean`, `skip_tokenize`, `skip_pack`, `data_root`, plus a `skip_train_tokenizer=True` default the shim does not expose — sensible, because GPT-2 BPE is a pretrained HF tokenizer that needs no training. `--stage` currently accepts only `pretrain`. The return value is the orchestrator's process exit code (0 success; 2 = a required config file missing). Before delegating, `_apply_mamba_defaults` prints the contract:

```
[data/mamba3] universal corpus: 8,000,000,000 tokens
[data/mamba3] tokenizer: gpt2 (vocab=50,257, EOS=50,256)
[data/mamba3] shard size: 50,000,000 tokens (uint32)
```

The `--skip-*` flags implement the documented quick-start ladder: full run, re-use corpus (`--skip-download`), re-pack only (`--skip-download --skip-clean --skip-tokenize`).

### Downstream consumption: `PretrainDataset`

The pretrainer reads shards through `training/pretrain.py:PretrainDataset` (fully covered in [R8 — dataset](08-dataset.md)); the three facts this doc needs:

- **Layout dispatch** (`training/pretrain.py:PretrainDataset.__init__`): a missing path → `layout = "dummy"` (1,000 random windows, warn only — see Pitfalls); a directory → `_init_sharded` (glob `shard_*.bin`, `torch.load(..., mmap=True)`, bisect via `_locate`); a file → `_init_single` (`torch.load(weights_only=True)`).
- **Windows**: `max_seq_len+1` tokens per sample, `x = chunk[:-1]`, `y = chunk[1:]` (`training/pretrain.py:PretrainDataset._get_window_single`); `n_samples = (len(data) - 1) // max_seq_len`.
- The config's `data:` section is consulted for exactly one key — `train_data_path` (with fallback default `data/pretrain_data.bin`); `data_mix`, `shard_size_tokens`, `max_tokens`, and `tokenizer` are **never read** by `training/pretrain.py:main` (verified by repo-wide search: `data_mix` appears only in `README.md` and the YAML, in zero code files). The YAML's `data_mix: "mamba2-default"` plus the comment line is a spec annotation, not a runtime knob — the real mix lives in the shared package's `mixture.yaml`.

## Vendored-vs-workspace lookup

The lookup order, as *documented* in `data/DATA_PIPELINE.md`: vendored `data/shared_data/` first, then workspace `LLM/shared_data/`, with the workspace copy intended as the fallback for standalone clones that still sit inside `CoreProjects`. The code implements exactly that:

- `_LLM_ROOT = _PROJECT_ROOT.parent` resolves to **`CoreProjects/LLM/`** — the workspace root — so the guard checks `LLM/shared_data` and the path insert exposes `LLM/` on `sys.path`, which is precisely where the workspace copy lives. **Measured consequence on this machine**: `LLM/shared_data/` exists, so the guard's workspace branch succeeds and the CLI runs against the workspace copy. The vendored branch is the only one that can fail here, because `data/shared_data/` is absent.
- Because the vendored branch works only via `sys.path[0]` (script directory = `data/`), the recommended invocation is exactly `python3 data/prepare_data.py` from the repo root. Running the module as `python3 -m data.prepare_data` puts the repo root at `sys.path[0]` and would not resolve `data/shared_data/` — though the workspace fallback (via the `_LLM_ROOT` insert) would still serve `LLM/shared_data/`.

**To make the repo fully self-contained**: vendor the package (`rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' LLM/shared_data/ data/shared_data/` from the workspace, per `data/DATA_PIPELINE.md`), or copy it from a sibling project that already vendors it (LLaMA-3-Lite's `data/shared_data/` exists on this machine, though its `config/` tree is incomplete there — verify before relying on it). With `data/shared_data/` present, the vendored branch wins and the clone no longer depends on the workspace copy. If the workspace copy is absent (standalone clone), `_require_shared_data` raises the clean `FileNotFoundError` naming both candidate locations.

## Shard format

In-clone authoritative statement (`data/DATA_PIPELINE.md`): shards are **uint32, 50,000,000 tokens, EOS-separated** — every document boundary carries the configured EOS id, and documents are never split across shards (`cross_document_boundary_ok: false` in `data/data_config.yaml`). The workspace canonical `LLM/shared_data/README.md` (§7) and `shard_writer.py` pin the physical format: a **flat raw uint32 little-endian buffer**, ≈190 MiB, written atomically via `.tmp` + rename, re-read and SHA-256-verified into the manifest, with sources round-robin-interleaved so every shard sees every source. Readers there use `np.memmap(..., dtype=np.uint32)`.

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

See [R8 — dataset](08-dataset.md) for `PretrainDataset` internals, [R7 — pretrain CLI](07-pretrain-cli.md) for how `--data-path` overrides `train_data_path`, [R12 — config reference](12-config-reference.md) for the annotated YAML, [T8 — scaling efficiency](../theory/08-scaling-efficiency.md) for the Chinchilla-optimal token budget, and [G2 — training runbook](../guides/02-training-runbook.md) for the end-to-end recipe.
