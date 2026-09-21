# Mamba-3-Lite — Documentation Audit

Verification record for the Wave-1 docs upgrade (2026-08-20). Everything
measured, nothing estimated: each number below comes from the gate runs and the
test suite recorded under "Verification runs".

## Verification runs

```bash
python3 scripts/check_docs.py          # == tests/test_doc_refs.py --coverage --links
python3 -m pytest tests/ --tb=no -q
```

- **Doc gate:** 23 docs scanned, 463 anchors — resolution PASS,
  coverage PASS (9 modules), line anchors PASS, Triton citations PASS, links PASS.
- **Test suite:** 32 passed, 5 skipped, 0 warnings (1.6–2.5 s wall) — skips are the GPU-gated Triton parity tests
  (auto-skip on this macOS/CPU box by design; `pytest.ini` marker `slow`, AGENTS.md
  §2 rule 7). No warnings surfaced.
- **Stale-count sweep:** grepped README.md and docs/ for test-count claims —
  README.md:162 and README.md:296 both say "37 tests, 32 pass on CPU, 5 GPU-gated",
  which matches the measured run. No stale counts found.

## Findings

| ID | Finding | Disposition |
|---|---|---|
| F1 | `AGENTS.md` §7 (Obsidian rule) instructs running `bash scripts/sync_to_vault.sh` — the script does not exist in the repo (verified `ls scripts/`). Vault syncing must be done manually. | **ACCEPTED** — agent-tooling gap, outside docs scope. |
| F2 | `tests/test_doc_refs.py` `SKIP_DOCS` references `docs/docs_expansion_plan.md`, which no longer exists — a dead skip-list entry (the plan file was deleted after the docs it seeded landed). Harmless: the set filter never matches. | **ACCEPTED** — one-line cleanup for a future tests/ pass; left untouched to keep this wave docs-only. |
| F3 | The concept docs are multi-part files: `concepts/ssd-theory.md` concatenates three H1 documents (state-space duality, complex-SSD, chunkwise algorithm) and `concepts/block-and-stability.md` two (block anatomy, numerical stability). Cross-references target the file, not mid-file anchors, so links resolve — but heading slugs are non-unique and section numbering restarts per part. | **ACCEPTED** — deliberate "derivation cluster" design, described as such in `docs/README.md`. |
| F4 | Two different corpus mixes are in circulation: the 5-source "mamba2-default" table (yaml comment, `docs/training.md`, README knobs table) vs the live 7-source workspace `shared_data/config/mixture.yaml`. `docs/training.md` already carries the honesty note naming the live recipe. | **ACCEPTED** — pre-existing, already documented; reconcile before the real 8B run. |
| F5 | Headline performance figures (12–15 h, 35–40% MFU, ~434M) are design targets; the 8.0B-token run has not started and there is no `.benchmarks/` tree. Parameter count (433,662,400) *is* measured (config accounting + parameter-count window test). | **ACCEPTED** — unmeasured claims stay tagged `[INFERENCE]` in the docs. |
| F6 | Coverage gap check before this wave: none. `--coverage` already passed on the pre-existing corpus (20 docs / 436 anchors, 9 modules fully cited), so no citation push was needed — the new docs only add anchors. | **CLOSED** |

## Codebase map (what the gate covers)

`tests/test_doc_refs.py` `COVERAGE_MODULES` — every public symbol must be cited:

| Module | Public symbols |
|---|---|
| `models/transformer.py` | `Mamba3Transformer`, `ModelConfig` |
| `models/mamba_block.py` | `Mamba3Block` |
| `models/ssd_complex.py` | `per_chunk_ssd_pytorch`, `ssd_complex_chunkwise`, `ssd_naive_complex` |
| `models/ssd_triton.py` | `per_chunk_ssd_triton` |
| `models/mimo.py` | `MIMO` |
| `training/pretrain.py` | `PretrainDataset`, `Pretrainer`, `TrainingConfig`, `count_parameters`, `main`, `train_step` |
| `utils/checkpoint.py` | `CheckpointManager` |
| `utils/logging.py` | `TrainingLogger` |
| `data/prepare_data.py` | `main` |

Empty-dir note: none — all coverage modules are real and shipped (contrast with
other portfolio repos whose unused scaffold dirs had to be pruned from the list).

Doc corpus layout: `concepts/` (4 files, incl. the two multi-part clusters),
`references/` (4), `guides/` (8 after this wave: extending, glossary,
learning-paths, pretrain-cli, quickstart, training-runbook, tuning,
visual-guide), `training.md`, `docs/README.md`, plus `docs/diagrams/quality-review.md`
(scanned by the gate like any other doc). `SKIP_DOCS` is empty in effect (F2).

## Gate dialect (this repo)

- Anchors: `file.py:Symbol` / `file.py:Class.method`, resolved by import +
  `hasattr` (instance attrs via `self.name =` source scan). `file.py` itself is a
  skipped metasyntax placeholder.
- **Line anchors are banned** (`check_line_anchors`) with an L1/L2 math allowlist —
  unlike HyMo, where line ranges are legal.
- **Triton citation rule** (`check_triton_citations`): symbols defined inside
  `if HAS_TRITON:` blocks may not be cited; docs cite the module-level wrapper.
- `--coverage` and `--links` are opt-in flags; `scripts/check_docs.py` is a thin
  shim that defaults both on and delegates to `tests/test_doc_refs.py` (single
  source of truth — the pytest file *is* the gate here).
- Pytest side: `tests/test_doc_refs.py` runs resolution, line-anchor, and
  Triton-citation checks on every `pytest tests/` run; coverage/links stay
  CLI-only flags.

## Modification plan (executed this wave)

1. **Gate shim** — `scripts/check_docs.py` added (delegates to
   `tests/test_doc_refs.py`; no logic duplicated). Commit `e28f89c`.
2. **Learning paths + glossary** — `docs/guides/learning-paths.md` (three tiers
   with landmark checks), `docs/guides/glossary.md` (notation, canonical numbers,
   terms). No citation push needed (F6).
3. **Audit** — this file, with measured runs.
4. **Nav wiring** — `docs/README.md`: AUDIT link in the intro, layout rows for the
   new guides, reading-paths pointer.

## Acceptance criteria

- [x] `scripts/check_docs.py` exists and exits 0 with `--coverage --links`.
- [x] Coverage PASS over all 9 `COVERAGE_MODULES`.
- [x] `docs/guides/learning-paths.md` — 3 tiers, landmark checks, links resolve.
- [x] `docs/guides/glossary.md` — notation + canonical numbers sourced from the
      config and the derived accounting; unmeasured numbers tagged `[INFERENCE]`.
- [x] `docs/AUDIT.md` — verification runs measured post-write (this file's counts
      were filled in from the final gate run, not estimated).
- [x] `docs/README.md` wired (intro AUDIT link, new rows, reading-path pointer).
- [x] `python3 -m pytest tests/` green (skips = GPU-gated only).
- [ ] 8.0B-token A100 run — not started; the acceptance item that remains open,
      per the runbook in [`guides/training-runbook.md`](guides/training-runbook.md).
