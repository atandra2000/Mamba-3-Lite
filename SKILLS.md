# SKILLS.md — Mamba-3-Lite

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific workflows.

Companion to `AGENTS.md` (in this folder) — that file holds the
architecture summary. This file holds the day-to-day developer workflows.

## Skill 1: Run the CPU-friendly smoke-test suite

```bash
cd LLM/Mamba-3-Lite
python3 -m pytest tests/ -v
```

Tests cover complex-SSD recurrence, MIMO head mixing equivalence, and
naive-vs-chunkwise parity. Must pass before any architectural change.

## Skill 2: Verify the complex-SSD math matches the naive scan

The chunkwise linear projection is the regression oracle. To re-verify:

```python
import torch
from models.ssd import ComplexSSDBlock, ModelConfig

cfg = ModelConfig()
m = ComplexSSDBlock(cfg).eval()
x = torch.randn(2, 128, cfg.d_inner, dtype=torch.complex64)
y_chunkwise = m(x)

# Compare against the naive O(T) scan
y_naive = m.naive_scan(x)
assert torch.allclose(y_chunkwise, y_naive, atol=1e-4)
```

If the assertion fails, **do not** ship — `SSD.md §` chunkwise-vs-naive is
the breakdown region.

## Skill 3: Tune chunk_size for throughput

`chunk_size` lives in `models/ssd.py`. Default 64.

| chunk_size | Memory | Throughput | Notes |
|------------|--------|------------|-------|
| 32         | low    | ~baseline  | best for short seqs (<2K) |
| 64         | mid    | baseline   | production default |
| 128        | higher | +5–10%     | only if FA2-equivalent throughput target met |
| 256        | high   | +10–15%    | risk of OOM at seq_len 8K, batch ≥ 32 |

Sweep via `scripts/chunk_sweep.py` (writes a CSV).

## Skill 4: Add a new SSM variant to the block

To add e.g. a gated-output variant:

1. Implement in `models/ssd.py` extending `ComplexSSDBlock`.
2. Add a config flag in `ModelConfig` and gate the new path on it.
3. Add an equivalence test in `tests/test_ssd.py`.
4. Run `python3 -m pytest tests/ -v` — must stay green.

**Pitfall:** changing the gating strategy invalidates the
`naive_scan` reference unless you re-derive it.

## Skill 5: Run a microbenchmark for the headline metric

```bash
python3 scripts/microbench_a100.py --chunk-sizes 32,64,128,256 --seq-lens 2048,4096,8192
```

Outputs `tokens/sec` vs. chunk_size vs. seq_len. The portfolio headline
(2.1× vs Mamba-1 scan) comes from chunk_size=64 at seq_len=4096.

## Pitfalls

- **Pure SSM purity:** never add `mamba-ssm` package, custom CUDA, or
  Triton — those are explicitly excluded.
- **FA2 is disabled:** don't add `with sdpa_kernel(FLASH_ATTENTION)` here;
  the chunkwise projection replaces attention.
- **Complex stride:** `torch.view_as_complex` requires the last dim to be
  stride-2 contiguous. If you change tensor layouts, double-check.
- **State size is even:** N must remain even to pack into complex pairs.
  Odd N silently breaks the complex recurrence.
- **NaN guard:** `nan_guard_max_consecutive=5` — after 5 consecutive NaN
  steps the run auto-rolls back to the last good checkpoint.
