# Mamba-3-Lite — Extending the Repo — SSM variants and Triton kernels

Task-oriented guide for adding new sequence-mixing variants and new sanctioned Triton kernels to Mamba-3-Lite, exactly as the repo's rules require.

## 60-second summary

After reading this guide you can ship a new SSM variant (Path A) or a new Triton kernel (Path B) through the repo's two sanctioned extension paths. Path A is pure PyTorch: write the O(T) naive reference **first**, add a flag to `models/transformer.py:ModelConfig`, gate the dispatch in `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`, prove chunkwise≡naive with an equivalence test that includes time-varying dt, and keep the doc checker green. Path B is Triton: a new `models/<name>_triton.py` with a `try/except ImportError` → `HAS_TRITON` gate, a `torch.autograd.Function` whose backward **recomputes the reference math and seeds it with the true `grad_outputs`** (the pattern in `models/ssd_triton.py:_PerChunkSSDTriton`), a CPU-runnable pure-PyTorch reference test, GPU parity tests auto-skipped on CPU, and an update to the sanctioned list in AGENTS.md.

The governing rules (restated from AGENTS.md): raw PyTorch by default, custom Triton kernels first-party only for sanctioned hot paths, no HF Trainer / Lightning / mamba-ssm / causal_conv1d (rule 1); know the chunkwise algorithm before touching SSD code — rule 2, successor reference `docs/concepts/ssd-theory.md`; the chunkwise linear projection must match the naive O(T) scan oracle exactly (rule 3); N must stay even so complex packing is exact (rule 4); never add MoE, MTP, or attention layers — pure SSM repo (rule 5); new Triton kernels need a CPU-runnable unit test with GPU-only behaviour gated so it auto-skips on CPU (rule 7); comments stay concise (rule 8).

## Why it exists

The repo deliberately has a small surface: 28 identical Mamba-3 blocks and exactly one sanctioned Triton kernel. That discipline is what makes the codebase auditable — every extension touches a well-defined seam (config → block dispatch → scan function → equivalence test). This guide turns the two thin rules (`SKILLS.md` Skill 4 and AGENTS.md rule 1) into a complete procedure, because the failure modes are subtle: a backward that returns `None` for one input silently zeroes that input's gradient, a stale doc anchor fails CI, and an unguarded kernel falls back to PyTorch without telling anyone. Follow the steps below and each failure mode is caught by the repo's own gates.

---

## Path A — add a new SSM variant in pure PyTorch

This is the boring, preferred path. You are changing *math*, not *plumbing*: the new recurrence plugs into the existing `in_proj → scan → MIMO → out_proj` pipeline.

### Step 1: write the O(T) reference FIRST

Implement the recurrence in `models/ssd_complex.py` (or a new module if the variant is unrelated to the complex recurrence). The non-negotiable first move is the naive sequential scan, because it is the ground truth every fast path must match. The existing oracle is `models/ssd_complex.py:ssd_naive_complex`:

```python
def ssd_naive_complex(
    x: torch.Tensor, A: torch.Tensor, B_t: torch.Tensor, C_t: torch.Tensor, dt: torch.Tensor,
) -> torch.Tensor:
    """O(T) sequential complex SSM scan — reference oracle for ssd_complex_chunkwise."""
    B_, T, H, D = x.shape
    N = B_t.shape[-1]
    A_bar = _discretise(dt, A)
    s = torch.zeros(B_, H, N, D, dtype=torch.complex64, device=x.device)
    ys = []
    for t in range(T):
        s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s             + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
        ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
    return torch.stack(ys, dim=1)
```

Model your `my_naive_variant(x, A, B_t, C_t, dt)` on this shape contract: state `(B, H, N, D)` in complex64, output `(B, T, H, D)` complex64. Keep it dumb — a Python loop is fine; it is a reference, not a product. If your variant changes the state dimension, keep N even (rule 4) and keep the parameterization per-head-complex so the `A_bar = exp(softplus(dt)·A)` discretisation (`models/ssd_complex.py:_discretise`) carries over.

### Step 2: add a config flag and gate the block

Add a boolean (or enum) to `models/transformer.py:ModelConfig`:

```python
@dataclass
class ModelConfig:
    # ... existing fields ...
    my_variant: bool = False   # illustrative — Path A gate
```

`models/transformer.py:Mamba3Transformer.__init__` normalises dicts with `ModelConfig(**cfg)`, so any new field becomes a knobs-file key automatically; unknown keys raise `TypeError`, so there is no silent misspelling.

Then read the flag defensively in `models/mamba_block.py:Mamba3Block.__init__` — mirror the existing `self.ssd_dispatch = cfg.get("ssd_dispatch", "pytorch")` pattern — and branch in the dispatch seam, `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` (or in `models/mamba_block.py:Mamba3Block._forward_impl` if the variant also changes the `in_proj` slice layout — e.g. a different N or a real-only state). `_forward_impl` currently slices `proj[..., :H*D]`, four `H*N` blocks, and a trailing `H*1` dt; if your variant re-layouts those slices, `_forward_impl` must change with them, and `in_proj`'s width `H*(D + 4N + 1)` follows N. Keep the triton branch out of Path A entirely — `_ssd_with_dispatch`'s existing `ssd_dispatch="triton"` branch is the only kernel route.

### Step 3: add the equivalence test

In `tests/test_ssd.py`, add a chunkwise-vs-naive parity test. Follow the existing regression pattern — note that `dt = 0` masks off-by-one decay errors because every chunk then has an identical decay total; the regression test `tests/test_ssd.py::test_chunkwise_matches_naive_time_varying_dt` exists precisely because a dt=0-only test passed while the inter-chunk propagation was wrong. Your test MUST use random, time-varying dt and assert `allclose(y_chunk, y_naive.real, atol=1e-4)` (chunkwise returns `.real`, the oracle returns complex). Also cover uneven T (`T % chunk_size != 0`) and `T == chunk_size`.

### Step 4: run the suite, update docs

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/test_doc_refs.py -v
```

Expected on a CPU/Mac box: 37 tests collected (32 passed, 5 GPU-skipped), doc checker 0 failures. The second command is the repo's doc rule (section below): new public functions must be cited in a reference doc, or the coverage check flags them. Update `docs/references/` (and cross-link `docs/concepts/ssd-theory.md` if you touched the chunkwise derivation), then report the anchors you cited.

---

## Path B — add a sanctioned Triton kernel

The contract comes from AGENTS.md rule 1 and its §1 kernel clause. Read `docs/references/ssd-reference.md` first — it is the migrated anatomy of the one existing kernel and the template for yours.

### Step 1: file placement and the import gate

Place the kernel in `models/<name>_triton.py`. The module MUST import on machines without triton:

```python
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
```

Everything the kernel needs is defined under `if HAS_TRITON:`; the always-defined pieces (reference math, block-dims guard, host wrapper, autograd Function) live at module top level.

### Step 2: write the pure-PyTorch reference

Define `per_<name>_pytorch(...)` at top level — no triton required. This is both the CPU test oracle and the backward's recompute engine. The existing one is `models/ssd_complex.py:per_chunk_ssd_pytorch`, which materialises `L = exp(cumsum(A_log)[l] − cumsum(A_log)[s]) · tril` and computes `Y_diag` and per-chunk `state` with einsums.

### Step 3: the kernel and the autograd Function — correct backward, always

The forward helper (kernel launch) goes under `if HAS_TRITON:`. Two things to copy from the sanctioned path:

- **Block-dims guard.** Constexpr block sizes must not blow up: `models/ssd_triton.py:_check_block_dims` raises a clean `ValueError` when P, N, or chunk_size exceed the 256-cap (`_MAX_BLOCK`), instead of surfacing a compiler error. Replicate it for your kernel.
- **complex64 handling.** Triton ≥ 3.x has no complex pointer dtype; `models/ssd_triton.py:_view_real_imag` splits complex64 into contiguous float32 real/imag pairs (`torch.view_as_real` gives stride-2 on the inner dim; `.contiguous()` fixes it). Copy this — a wrong stride fails silently in the kernel.

The autograd wrapper is where correctness lives. Study `models/ssd_triton.py:_PerChunkSSDTriton` end to end:

```python
class _PerChunkSSDTriton(torch.autograd.Function):
    """Fused per-chunk forward; backward recomputes the same math in PyTorch
    and seeds it with the true downstream gradients (grad_outputs)."""

    @staticmethod
    def forward(ctx, Bc, Cc, Xc, A_log, decay_states):
        Y_diag, state = _per_chunk_ssd_triton_forward(Bc, Cc, Xc, A_log, decay_states)
        ctx.save_for_backward(Bc, Cc, Xc, A_log, decay_states)
        return Y_diag, state

    @staticmethod
    def backward(ctx, grad_y_diag, grad_state):
        Bc, Cc, Xc, A_log, decay_states = ctx.saved_tensors
        with torch.enable_grad():
            b = Bc.detach().requires_grad_(True)
            c = Cc.detach().requires_grad_(True)
            x = Xc.detach().requires_grad_(True)
            a = A_log.detach().requires_grad_(True)
            d = decay_states.detach().requires_grad_(True)
            Y_diag_ref, state_ref = per_chunk_ssd_pytorch(b, c, x, a, d)
        g_y = grad_y_diag if grad_y_diag is not None else torch.zeros_like(Y_diag_ref)
        g_s = grad_state if grad_state is not None else torch.zeros_like(state_ref)
        grads = torch.autograd.grad(
            (Y_diag_ref, state_ref), (b, c, x, a, d),
            grad_outputs=(g_y, g_s), allow_unused=True,
        )
        return grads
```

This is the **recompute + grad_outputs injection** pattern, and it exists because the kernel's predecessor backward was a broken stub: it recomputed `y.sum()` as the loss and returned `None` for the token-content input — so gradients were only correct when downstream grads happened to be all-ones, and the content path got zero gradient. The two rules that prevent a repeat: (1) `torch.autograd.grad` must be seeded with the *true* downstream gradients (`grad_outputs=(grad_y_diag, grad_state)`, with `zeros_like` fallbacks for `None`), and (2) never return `None` for a differentiated input — in a `torch.autograd.Function`, `None` means "zero gradient", not "no-op". Also detach and re-`requires_grad_` the saved tensors directly: a `tensor * 0.1` that carries `requires_grad=True` is a NON-LEAF, and its `.grad` never populates.

### Step 4: host wrapper + two-layer opt-in

Expose a top-level `per_<name>_triton(...)` that raises a clean `ImportError` when `HAS_TRITON` is False (mirror `models/ssd_triton.py:per_chunk_ssd_triton`) and otherwise calls `_Per<Name>Triton.apply(...)`. Then wire the opt-in exactly like the sanctioned path:

1. **Config value.** A `ssd_dispatch`-style flag on `models/transformer.py:ModelConfig`, consumed by the block.
2. **Environment gate.** `ENABLE_TRITON_KERNELS=1` must be required too. The trainer enforces this in `training/pretrain.py:_enforce_triton_env_var`, which currently force-rewrites only `ssd_dispatch` back to `"pytorch"` with one warn line when the env var is missing — extend that function to your new flag as well, so a default-config training run never silently takes the kernel path (AGENTS rule 6).

### Step 5: tests, AGENTS.md, docs

Add `tests/test_<name>_triton.py` with three layers, copying `tests/test_ssd_triton.py`:

1. **CPU-runnable reference tests** (no triton installed): shapes, finiteness, and reference-vs-chunkwise parity.
2. **Autograd plumbing on CPU**: monkeypatch the kernel forward helper with the pure-PyTorch reference, then `torch.autograd.gradcheck` the Function and assert every input's grad is non-finite-free **and the content-path input's grad is non-zero** (that last assertion is the regression guard against the `None`-grad bug — see `tests/test_ssd_triton.py::TestPerChunkSsdAutogradPlumbing`).
3. **GPU parity** gated by the actual marker in `tests/test_ssd_triton.py` (AGENTS rule 7 says `@pytest.mark.gpu`; the concrete implementation is a module-level skipif applied to the test class):

```python
gpu_required = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires triton + CUDA",
)
```

GPU-only tests decorated with `@gpu_required` auto-skip on CPU/Mac. Finally, add the new path to the sanctioned list in AGENTS.md §1 (rule 1 — no new kernel ships without this) and add a `docs/references/` doc per the doc rule below.

---

## The doc rule

Docs ship with code. `tests/test_doc_refs.py::test_doc_refs_all_anchors_resolve` parses every `docs/**/*.md`, extracts `file.py:Symbol` anchors, resolves each against the working tree (importlib + `hasattr`-chain), and fails on unknown files or symbols; its coverage mode additionally requires every public symbol in `models/`, `training/`, `utils/` to be cited somewhere. The canonical doc style is visible throughout `docs/` (see `docs/README.md` for the layout): a one-paragraph intro, backticked `file.py:Symbol` citations, no line numbers, `## References` last. Anchor rules: cite only `file.py:Symbol` / `file.py:Class.method`, never line numbers; never cite symbols defined under `if HAS_TRITON:` (cite host wrappers like `models/ssd_triton.py:per_chunk_ssd_triton` instead); never cite `data/shared_data/*` (absent in this clone); mark measured vs derived vs `[INFERENCE]` — there is no `.benchmarks/`, so all throughput numbers are estimates; the parameter count is 433,662,400 (~434M), never 404M.

## Path C — non-goals

Extensions that are off the table by rule: attention layers, MoE, and MTP (AGENTS rule 5 — this is a pure SSM repo, deliberately separated from the portfolio's HyMo hybrid project); the `mamba-ssm` / `causal_conv1d` dependencies and any HF Trainer / Lightning wrapper (rule 1); odd N state sizes (rule 4 — complex packing needs even N); and new kernels that skip the sanctioned-list + doc requirement (rule 1). If a proposal needs any of these, it is a new project, not an extension.

## Pitfalls

1. **Backward correctness** — the `y.sum()`-style stub and `None`-grad bugs described above are the highest-risk failure. Always inject true `grad_outputs` and never return `None` for a differentiated input.
2. **Block dims > 256** — oversized constexpr blocks fail inside kernel compilation with an unreadable error; the `_check_block_dims`-style guard converts that into a clean `ValueError` suggesting the pytorch dispatch.
3. **Citing triton-gated symbols in docs** — `tests/test_doc_refs.py` warns (and coverage/CI can fail) if you cite `_ssd_per_chunk_fwd_kernel`-style JIT symbols. Cite the always-defined host wrappers.
4. **Forgetting the env-var guard** — `ssd_dispatch='triton'` without `ENABLE_TRITON_KERNELS=1` is silently force-backed to `'pytorch'` by `_enforce_triton_env_var`; if you see no speedup, check the warn line, not the config. A new kernel flag must be added to that guard.
5. **Complex strides** — `torch.view_as_complex` / `view_as_real` require stride-2-contiguous inner dims; `.contiguous()` before splitting or packing, or the kernel reads garbage without an error.
6. **dt=0-only tests** — they cannot see inter-chunk decay bugs; equivalence tests must use random time-varying dt.

## Verification checklist

- `python3 -m pytest tests/ -v` → 37 tests collected (32 passed, 5 GPU-skipped on CPU).
- `python3 -m pytest tests/test_doc_refs.py -v` → 0 anchor failures.
- New public symbols cited in `docs/references/` (coverage gate).
- New kernel listed in AGENTS.md §1 sanctioned paths.
- GPU class under `@gpu_required` skipif; CPU reference class runs without triton.

Cross-links: [Mamba-3-Lite — Config Reference](../references/config-reference.md), [Mamba-3-Lite — SSD Reference](../references/ssd-reference.md), [Mamba-3-Lite — SSD Reference](../references/ssd-reference.md), [Mamba-3-Lite — Model Reference](../references/model-reference.md), [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md), [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md), [Mamba-3-Lite — Quickstart](quickstart.md), [Mamba-3-Lite — Tuning Guide](tuning.md), and the doc map in `docs/README.md`.

## References

- [Mamba-3-Lite — Config Reference](../references/config-reference.md) — `models/transformer.py:ModelConfig` field addition pattern.
- [Mamba-3-Lite — SSD Reference](../references/ssd-reference.md) — the migrated anatomy of the one existing kernel and the template for new ones.
- [Mamba-3-Lite — Model Reference](../references/model-reference.md) — the block dispatch seam.
- [Mamba-3-Lite — SSD Theory](../concepts/ssd-theory.md) — the chunkwise algorithm any new variant must match.
- [Mamba-3-Lite — MIMO Head Mixing](../concepts/mimo.md) — the `_identity_init` escape-hatch pattern.
- [Mamba-3-Lite — Quickstart](quickstart.md), [Mamba-3-Lite — Tuning Guide](tuning.md), and the doc map in `docs/README.md`.
