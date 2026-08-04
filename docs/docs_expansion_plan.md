# Docs Expansion Plan — Mamba-3-Lite

> Goal: turn ~4,700 words of thin/stale docs into a comprehensive,
> from-scratch, concept-building documentation set (target ≥ 25,000 words)
> covering every module with detailed theory, **strictly aligned with the
> codebase** (symbol-anchored, machine-verified).
>
> Status: EXECUTED (2026-08-04). Execution was phased (Phase 0 → 5);
> writers dispatched in parallel batches per Phase. This file is the single
> source of truth for the tree, the per-doc outlines, and the acceptance gates.

---

## 1. Current-state audit (measured 2026-08-04)

### 1.1 Inventory

> **Pre-expansion baseline (measured 2026-08-04).** The table below is the
> snapshot taken *before* execution: `README.md` (1,969 words), `SSD.md`
> (169), and `documentation/ssd_triton.md` (1,462) were all deleted or
> retired during execution. Current corpus total (2026-08-04): **73,625
> words** in the `docs/` subtree (incl. this plan) / **77,725** incl. the
> top-level markdown (`README.md`, `AGENTS.md`, `SKILLS.md`,
> `data/DATA_PIPELINE.md`).

| Doc | Words | Grade | Verdict |
|---|---|---|---|
| `README.md` | 1,969 | C+ | Good landing page; asserts claims (parity, params, speedup) without deriving them |
| `documentation/ssd_triton.md` | 1,462 | B+ | Best doc; design-focused; assumes SSD knowledge; no from-scratch build-up |
| `AGENTS.md` | 945 | A (for its purpose) | Agent rules, not educational |
| `data/DATA_PIPELINE.md` | 597 | B− | Honest about absent vendored package; references code not in this tree |
| `SKILLS.md` | 427 | B− | Workflow cheat sheet, thin theory |
| `SSD.md` | **169** | **D** | **The "authoritative algorithm reference" is a stub.** README/AGENTS call it "full mathematical derivation" / "equivalence proof" — it contains two short sections and zero derivation |
| `docs/superpowers/*` | 6,667 | — | Historical planning artifacts (plans/specs of past work), not docs |
| **Total educational** | **~4,700** | — | Thin for a 434M-param from-scratch SSM + custom Triton kernel |

### 1.2 Coverage gaps (module symbols vs docs)

| Module (public surface) | Doc coverage today | Gap |
|---|---|---|
| `models/ssd_complex.py` — `ssd_complex_chunkwise`, `ssd_naive_complex` | **None.** README says "see SSD.md for the derivation" → SSD.md has none | **Critical.** The heart of the repo is undocumented: the 5-einsum chain, L/Y_diag/Y_off math, padding, dispatch |
| `training/pretrain.py` — `Pretrainer`, `TrainingConfig`, `PretrainDataset`, `train_step`, `_enforce_triton_env_var` | **None** | Training loop, NaN guard, LR schedule, resume, BF16 autocast, torch.compile, env-var guard — all unexplained |
| `utils/checkpoint.py` — `CheckpointManager` | **None** | Atomicity, dedup, 3-file format, restore semantics |
| `utils/logging.py` — `TrainingLogger` | **None** | tps metric, WandB hook |
| `models/mamba_block.py` — `Mamba3Block` | README block diagram only | in_proj packing (x/B_real/B_imag/C_real/C_imag/dt layout), A parameterization, dispatch/fallback — no walkthrough |
| `models/transformer.py` — `Mamba3Transformer`, `ModelConfig` | README table only | No per-field reference; init scheme (incl. `_identity_init` skip) unexplained |
| `models/mimo.py` — `MIMO` | README snippet only | No theory: why fully-connected mixer, identity-init rationale, cost |
| `models/ssd_triton.py` — `per_chunk_ssd_triton`, `per_chunk_ssd_pytorch` | Good (design doc) | Needs migration into reference/ + from-scratch Triton tutorial |
| `data/prepare_data.py` — shim | DATA_PIPELINE.md | Decent; needs API reference for the shim |

### 1.3 Theory depth gaps (from-scratch / concept-building)

- **SSM foundations** (continuous→discrete, diagonal state spaces, S4→S6→Mamba-2): zero self-contained coverage; README only links arXiv papers.
- **Complex-valued SSD math**: the N-halving / parity-perplexity claim and the decay+rotation property of `exp(α + iβ)` are asserted, never derived. No complex-gradient (Wirtinger) discussion.
- **Chunkwise algorithm**: no derivation of `L`, `Y_diag`, inter-chunk `decay_chunk`, `Y_off`; the einsum strings are unlabeled magic.
- **Discretization**: `A_bar = exp(softplus(dt) · A)` rationale (why softplus, why per-head scalar complex A) unexplained.
- **Numerical stability**: AGENTS.md lists rules; no doc explains *why* (FP32 gating, complex64 state, TF32, NaN rollback design).
- **Scaling/efficiency**: "Chinchilla-optimal 8B tokens", "~434M params", "12–15 h", "35-40% MFU" all asserted; no derivations (param breakdown 28×13.65M + 51.5M tied embed, memory analysis of chunkwise vs scan, chunk_size tradeoffs).
- **Training theory**: AdamW decay grouping, warmup+cosine, grad accumulation, checkpoint rollback — none documented.

### 1.4 Alignment status

Doc↔code alignment was audited and fixed in the previous round (11 files: Triton backward, MIMO init, param count 404M→434M, README/AGENTS/config stale claims, `--resume` flag, tokenizer, data mix). Current docs cite symbols accurately where they cite anything. Remaining risks are *coverage* (above) and *new* drift introduced by this expansion — mitigated by the machine checker (Phase 0).

Known residual fixes to fold into expansion:
- `data/DATA_PIPELINE.md` heading updated from the stale ~400M claim to ~434M (done during execution).
- `SSD.md` will be **deleted** (replaced by `docs/theory/04-chunkwise-algorithm.md` + `03-complex-ssd.md`); all links updated.
- `documentation/ssd_triton.md` will be **migrated** to `docs/reference/03-ssd-triton.md`; AGENTS.md rule-1 text updated to point at `docs/reference/`.

---

## 2. Target tree

```
docs/
├── README.md                       # doc map + reading paths (researcher / engineer / beginner)
├── docs_expansion_plan.md          # this file
├── theory/                         # from-scratch concept building, math-first
│   ├── 01-ssm-foundations.md       # T1
│   ├── 02-state-space-duality.md   # T2
│   ├── 03-complex-ssd.md           # T3
│   ├── 04-chunkwise-algorithm.md   # T4  (successor to SSD.md)
│   ├── 05-mimo-mixing.md           # T5
│   ├── 06-block-anatomy.md         # T6
│   ├── 07-numerical-stability.md   # T7
│   └── 08-scaling-efficiency.md    # T8
├── reference/                      # symbol-anchored API docs (1:1 with code)
│   ├── 01-model-config.md          # R1
│   ├── 02-ssd-complex.md           # R2
│   ├── 03-ssd-triton.md            # R3  (migrated from documentation/ssd_triton.md)
│   ├── 04-transformer.md           # R4
│   ├── 05-mamba-block.md           # R5
│   ├── 06-mimo.md                  # R6
│   ├── 07-pretrain-cli.md          # R7
│   ├── 08-dataset.md               # R8
│   ├── 09-checkpoint.md            # R9
│   ├── 10-logging.md               # R10
│   ├── 11-data-pipeline.md         # R11 (expands data/DATA_PIPELINE.md)
│   └── 12-config-reference.md      # R12 (annotated yaml, every field)
└── guides/
    ├── 01-quickstart.md            # G1 (expands README Quick start)
    ├── 02-training-runbook.md      # G2
    ├── 03-tuning.md                # G3 (expands SKILLS.md Skill 3)
    └── 04-extending.md             # G4 (expands SKILLS.md Skills 4 + AGENTS.md rule 1)
```

Retired on landing: `SSD.md`, `documentation/ssd_triton.md` (content migrated). `README.md`, `AGENTS.md`, `SKILLS.md`, `data/DATA_PIPELINE.md` stay, updated with links to `docs/`.

---

## 3. Per-doc outlines

### Theory (T1–T8) — concept-building, from scratch

**T1 `01-ssm-foundations.md`** — From RNNs to state space models.
- Why sequence models need hidden state; the linear recurrence `h_t = A h_{t-1} + B x_t, y_t = C h_t` from first principles.
- Continuous→discrete: ZOH discretization of `ḣ = A h + B x`; where `dt` enters; the "selective" (input-dependent) parameterization of S6.
- Diagonal state spaces: why diagonal `A` makes the recurrence parallelizable (associative scan).
- Historical arc S4 → S6 → Mamba-2 → Mamba-3; what each step changed and why.
- Code anchors: `models/ssd_complex.py:ssd_naive_complex` (the O(T) oracle as the ground truth).
- Pitfalls: why naive sequential scan is memory-bound; why `softplus(dt)` keeps dt positive.

**T2 `02-state-space-duality.md`** — The SSD theorem (Mamba-2).
- From sequential scan to chunkwise matrix multiplication; the connection to linear attention (`(C B) ⊙ L`).
- The `L` (causal segment) matrix: `L[l,s] = exp(A_cs[l] − A_cs[s]) · 1[l ≥ s]`.
- Why chunking: O(T) scan → O(T/C) chunk-level scan + O(C²) intra-chunk matmuls; Tensor-Core friendliness.
- Intuition: each chunk is a "mini attention block" with a structured causal mask.
- Code anchors: `models/ssd_complex.py:ssd_complex_chunkwise` (chunk-level view only).

**T3 `03-complex-ssd.md`** — Mamba-3's complex-valued state spaces.
- Complex numbers as 2D rotations+scales: `exp(α + iβ) = exp(α)(cos β + i sin β)`; decay (α) and oscillation (β) in one parameter.
- Why real SSMs need N=128 for the same capacity that N=64 complex achieves; the packing argument (two real sub-states per complex state) and the parity claim's derivation.
- Complex recurrence, complex B/C, output projection; `torch.complex` mechanics (`torch.complex(real, imag)`, `view_as_real`, stride-2 constraint).
- Gradients through complex tensors: Wirtinger calculus in PyTorch autograd (what `.grad` on complex64 means); why the state stays `complex64` while logits are FP32.
- Code anchors: `models/ssd_complex.py:_discretise`, `ssd_naive_complex`, `models/mamba_block.py:Mamba3Block._forward_impl` (B_real/B_imag/C_real/C_imag split).
- Pitfalls: N must be even (real-pair packing); silent stride bugs in `view_as_complex`.

**T4 `04-chunkwise-algorithm.md`** — The chunkwise complex SSD, derived end to end. **Successor to `SSD.md`.**
- Step 1: pad T to a multiple of C; reshape into `(B, n_chunks, C, …)`.
- Step 2: `A_log = softplus(dt)·A`; `A_cumsum`; the intra-chunk `L` matrix (complex): real/imag split of `exp(seg)`.
- Step 3: `Y_diag = einsum("bclhn,bcshn,bchls,bcshp→bclhp")` — the intra-chunk term, einsum by einsum, with tensor-shape table.
- Step 4: per-chunk state `states = einsum("bclhn,bclh,bclhp→bchpn")`; `decay_states` derivation.
- Step 5: inter-chunk propagation `decay_chunk` (`chunk_decay` cumsum, causal tril) and `Y_off = einsum("bclhn,bchpn,bclh→bclhp")`.
- Step 6: `Y = Y_diag + Y_off`; why only `.real` is returned (output projection is real).
- Equivalence proof sketch: why chunkwise == naive O(T) scan (associativity of the semiring); reference to `tests/test_ssd.py::test_chunkwise_matches_naive_complex` as the machine proof.
- Code anchors: every einsum maps to a named line in `models/ssd_complex.py:ssd_complex_chunkwise`; padding branch; `initial_states` contract.
- Pitfalls: uneven T padding must be sliced back; `decay_states` shape `(B, n_chunks, C, H)` vs `A_log`; dtype promotion to complex64.

**T5 `05-mimo-mixing.md`** — MIMO head mixing.
- SISO constraint in classical SSMs: head i sees only its own channel; why that limits cross-head communication.
- MIMO mixer as a fully-connected linear map `(H·D) → (H·D)` after the scan; the README reshape dance, explained.
- Relation to attention's cross-head role; cost analysis (one GEMM per token, no sequence cost).
- Identity init: why start at identity (stable warm start, gradient flow), how `_identity_init` is preserved by `Mamba3Transformer._init_weights` (`models/transformer.py:Mamba3Transformer._init_weights`).
- Code anchors: `models/mimo.py:MIMO`, `models/mamba_block.py:Mamba3Block._forward_impl`.

**T6 `06-block-anatomy.md`** — The Mamba-3 residual block.
- Full data flow with tensor shapes at every stage: RMSNorm → in_proj → SSD → MIMO → out_proj → residual; RMSNorm → SwiGLU → residual.
- in_proj layout derived: `H·(D + 4N + 1)`; the exact slice boundaries for x / B_real / B_imag / C_real / C_imag / dt.
- `A` parameterization: per-head complex scalar, constant-init −1.0; why no per-state A (vs Mamba-2's per-state).
- Zero causal convolution: what `causal_conv1d` did in Mamba-1/2, why Mamba-3 replaces it with the chunked linear projection; the inductive-bias tradeoff.
- SwiGLU FFN: gate/up split, `F.silu(gate) * up`, why ffn_dim=2048 not 4096.
- Grad checkpointing (`torch.utils.checkpoint.checkpoint`) wiring; per-4th-layer policy in the config.
- Code anchors: `models/mamba_block.py:Mamba3Block` (every method), `models/transformer.py:Mamba3Transformer.forward`.

**T7 `07-numerical-stability.md`** — Why every dtype/norm choice.
- BF16 + `torch.compile` + TF32: where each precision lives (activations BF16, SSD complex64, gating FP32); `torch.set_float32_matmul_precision("high")`.
- The NaN guard: detection, skip-backward, consecutive-streak rollback to last checkpoint (`training/pretrain.py:train_step`, `Pretrainer.train`); why rollback beats clipping for complex recurrences.
- Weight tying (embed ↔ head): param savings math, `data_ptr` identity.
- Init scheme: `N(0, 0.02)` for Linear/Embedding, eye for MIMO, constant A; the `_identity_init` escape hatch.
- Logits in FP32; cross-entropy on FP32 logits; `ignore_index=-100`.
- Code anchors: `models/transformer.py:Mamba3Transformer._init_weights`, `training/pretrain.py:train_step`, `utils/checkpoint.py:CheckpointManager`.

**T8 `08-scaling-efficiency.md`** — Scale, memory, and throughput.
- Param count derived: `28 × 13.65M + 51.5M tied embed = 433.7M`; per-layer breakdown table (in_proj/out_proj/MIMO/FFN/A/norms).
- Chinchilla-optimality: why 8.0B tokens for ~434M params; the tokens/params ratio.
- Memory analysis: chunkwise vs sequential scan activations; complex64 = 2× element bandwidth, offset by N halving; chunk_size tradeoff table (32/64/128/256).
- Throughput levers: `torch.compile` modes, grad checkpointing every 4th layer, batch×accumulation math (`16 × 2 × 2048` tokens/step), the 12–15 h / 35-40% MFU target as a *measured-goal*, not a claim.
- Triton path economics: what `per_chunk_ssd_triton` saves (L/Y_diag/state fusion), backward recompute cost (reference-backed), v2 plan.
- Code anchors: `training/pretrain.py:Pretrainer.__init__` (decay/no-decay split), `models/ssd_triton.py:per_chunk_ssd_triton`.

### Reference (R1–R12) — symbol-anchored, 1:1 with code

Every reference doc follows: **signature → semantics → shapes → invariants → pitfalls → test links**. Writers cite symbols only (`file.py:Class.method`), never line numbers.

- **R1 `01-model-config.md`** — every `ModelConfig` field (`models/transformer.py:ModelConfig`): default, allowed values, effect, validation; `ssd_dispatch` two-layer opt-in contract.
- **R2 `02-ssd-complex.md`** — `ssd_naive_complex`, `ssd_complex_chunkwise` (`models/ssd_complex.py`): full signature, tensor-shape contract (B/T/H/D/N/C), padding rule, `initial_states`, dispatch semantics; test links (`tests/test_ssd.py`).
- **R3 `03-ssd-triton.md`** — migrate `documentation/ssd_triton.md`; add: host-wrapper API (`models/ssd_triton.py:per_chunk_ssd_triton`, `per_chunk_ssd_pytorch`), the 256-cap, autograd Function contract (forward/backward, grad_outputs seeding), env knobs (`TRITON_PER_CHUNK_NUM_STAGES/WARPS`).
- **R4 `04-transformer.md`** — `Mamba3Transformer` (`models/transformer.py`): init, dict-vs-dataclass config, weight tying, `_init_weights`, forward contract `(B,T) → (B,T,V)`.
- **R5 `05-mamba-block.md`** — `Mamba3Block` (`models/mamba_block.py`): cfg keys consumed, in_proj slice layout table, `_ssd_with_dispatch` fallback semantics (one-shot warning, which exceptions).
- **R6 `06-mimo.md`** — `MIMO` (`models/mimo.py`): shapes, eye init, `_identity_init` flag.
- **R7 `07-pretrain-cli.md`** — `TrainingConfig` fields (`training/pretrain.py:TrainingConfig`), CLI flags (`--config/--data-path/--checkpoint-dir/--resume/--no-checkpoint/--no-compile/--dry-run`), `main()` flow, yaml→config mapping table.
- **R8 `08-dataset.md`** — `PretrainDataset` (`training/pretrain.py:PretrainDataset`): three layouts (single/sharded/dummy), shard format (torch.save long tensor), windowing + shard-spanning windows, `_locate` bisect.
- **R9 `09-checkpoint.md`** — `CheckpointManager` (`utils/checkpoint.py`): 3-file format (safetensors/pt/json), shared-tensor dedup (data_ptr), atomicity (tmp→rename), `latest_step` completeness check, strict/loose load semantics.
- **R10 `10-logging.md`** — `TrainingLogger` (`utils/logging.py`): windowed metrics, tps formula (`log_every × seq_len × batch_size / elapsed`), WandB env hooks.
- **R11 `11-data-pipeline.md`** — expand `data/DATA_PIPELINE.md`: shim API (`data/prepare_data.py`), vendored-vs-workspace lookup order, `_require_shared_data` guard, data mix table (0.50/0.20/0.15/0.10/0.05), E2E bypass.
- **R12 `12-config-reference.md`** — the full annotated `configs/pretrain_a100_400m.yaml`: every field with its `TrainingConfig`/`ModelConfig` target and effect; the 434M comment math.

### Guides (G1–G4) — task-oriented

- **G1 `01-quickstart.md`** — install → verify (pytest) → dry-run → train → resume; expands README Quick start with expected outputs.
- **G2 `02-training-runbook.md`** — launching the A100 run, monitoring loss/ppl/tps, NaN recovery (rollback semantics), checkpoint hygiene, `WANDB_PROJECT` setup.
- **G3 `03-tuning.md`** — chunk_size, lr/warmup/cosine, batch×accumulation, grad-checkpoint cadence, compile modes, Triton env knobs; each knob → expected effect → how to measure.
- **G4 `04-extending.md`** — adding a new SSM variant (SKILLS.md Skill 4 procedure, expanded); adding a sanctioned Triton kernel (AGENTS.md rule-1 contract: file placement, `HAS_TRITON` gate, autograd Function, CPU test, doc requirement).

---

## 4. Writing contract (`local://doc_contract.md`)

All writers MUST follow `local://doc_contract.md` (created in Phase 0, not a repo file). Mandatory rules:

1. **Style template** per doc: `60-second summary → why it exists → intuition → math/proof → code walkthrough (symbol-anchored) → pitfalls → tests`.
2. **Symbol anchors only**: `file.py:Class.method` or `file.py:function`. NEVER `file.py:123` / `L123` — line numbers rot.
3. **Snippet policy**: real code must be verbatim from the repo; pseudo-code marked `# illustrative`.
4. **Cross-links**: only to docs in this plan's tree (`docs/theory/04-…`), never to files that don't exist yet; no dead anchors.
5. **Scope**: write ONLY your assigned file. No tests, no linters, no git, no edits to README/AGENTS/SKILLS (coordinator does link updates).
6. **Citation rule**: cite only always-defined symbols. Never cite JIT kernels under `if HAS_TRITON:` (e.g. `_ssd_per_chunk_fwd_kernel`) — cite the host wrappers (`per_chunk_ssd_triton`).
7. **Honesty**: mark measured vs derived vs `[INFERENCE]`; no `.benchmarks/` exists, so all perf numbers are estimates unless stated.
8. **LaTeX**: use `$…$`/`$$…$$` for math (terminal renders it).

## 5. Alignment checker (Phase 0, before any writer)

`tests/test_doc_refs.py` — machine-enforced doc↔code alignment, added to the pytest suite (the repo's de facto CI gate; also runnable standalone).

- Parse all `docs/**/*.md` for anchors matching `([A-Za-z_][A-Za-z0-9_./-]*\.py):([A-Za-z_][A-Za-z0-9_.]*)`.
- Resolve each `file.py` relative to repo root via `importlib.util.spec_from_file_location` (add repo root to `sys.path`); `hasattr`-chain the symbol (`Mamba3Block._forward_impl`).
- Fail on: unknown file, unknown top-level symbol, unknown attribute; warn on symbols defined only under `if HAS_TRITON:`.
- Line-anchor regex must exclude math terms (`L2`, `L1` etc. — negative lookahead) so theory math doesn't false-positive.
- Gate: the suite stays green (37 tests collected: 32 passed, 5 GPU-skipped) + checker 0 failures.

## 6. Phased execution

| Phase | Content | Dispatch (parallel batch) | Exit gate |
|---|---|---|---|
| **0** | Checker, contract, code map scaffold, README/doc-map skeleton, retire plan for SSD.md | coordinator (no writers) | `tests/test_doc_refs.py` passes on existing docs; pytest green |
| **1** | T1–T4 (SSM foundations, SSD duality, complex SSD, chunkwise algorithm) — the math core | 4 writers | checker 0 failures; each doc has derivation + code walkthrough |
| **2** | T5–T8 (MIMO, block anatomy, numerical stability, scaling) | 4 writers | same |
| **3** | R1–R6 (model-side reference) then R7–R12 (training/utils/data reference) | 6 + 6 writers | every public symbol in `models/`, `training/`, `utils/` cited ≥1× |
| **4** | G1–G4 + `docs/README.md` + README/AGENTS/SKILLS/DATA_PIPELINE link updates | 4 writers + coordinator | link audit clean; AGENTS.md rule "docs ship with code; stale docs fail CI" added |
| **5** | Verify + land: checker, markdown-link audit (dead `#anchor` fragments), full pytest; `git rm SSD.md documentation/ssd_triton.md`; `scripts/generate_code_map.py`; vault sync `bash ~/Desktop/CoreProjects/scripts/sync_to_vault.sh` | coordinator | all gates green |

Concurrency: batches ≤ 6 writers (well under the 32 cap); retries cheap.

## 7. Acceptance metrics

1. **Coverage**: every public symbol in `models/` (7 modules), `training/pretrain.py`, `utils/` (2 modules) appears as a citation in ≥1 doc — enforced by a coverage section in the checker (inventory vs citations).
2. **Alignment**: `tests/test_doc_refs.py` 0 failures; 0 dead internal markdown links; 0 stale `#anchor` fragments.
3. **Depth**: every theory doc (T1–T8) contains all six contract sections (summary/why/intuition/math/walkthrough/pitfalls); every reference doc (R1–R12) contains signature/shapes/invariants/pitfalls.
4. **Scale**: docs total ≥ 25,000 words (from ~4,700), with `SSD.md`'s missing derivation restored in T3+T4.
5. **Code health**: full pytest suite green (37 tests collected: 32 passed, 5 GPU-skipped) at every phase exit.
6. **Repo consistency**: README/AGENTS/SKILLS link only to existing docs; retired files removed; vault mirror synced.

## 8. Risks / watch-for

- **Latent code bugs surface during writing** (per past experience: docs audits found the Triton backward bug, MIMO init override). If a writer finds a code/doc contradiction: writer flags it, coordinator fixes code + docs + adds regression test in the same phase.
- **Writer drift** (double path prefixes, bare `loader.py:` in mermaid blocks, JIT-kernel citations) — fixed centrally at phase exits via the checker.
- **Perf numbers**: no `.benchmarks/` in tree → all speedup/MFU numbers are `[INFERENCE]` until the A100 run; T8 must say so explicitly.
- **Vendored `shared_data` absent**: R11/G2 must not cite `data/shared_data/*` symbols (they don't exist in this clone); cite the shim + workspace path only.
