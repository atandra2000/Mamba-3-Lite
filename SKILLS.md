# SKILLS.md — Mamba-3-Lite

> Companion to `AGENTS.md` (architecture summary). This file holds
> day-to-day developer workflows.

## Skill 1: Run the CPU-friendly smoke-test suite

```bash
cd LLM/Mamba-3-Lite
python3 -m pytest tests/ -v
```

Tests cover complex-SSD recurrence, MIMO head mixing, chunkwise-vs-naive
parity, the Triton kernel reference (CPU), and the env-var guards. Must
pass before any architectural change.

## Skill 2: Verify the complex-SSD math matches the naive scan

The chunkwise linear projection is the regression oracle. To re-verify:

```python
import torch
from models.ssd_complex import ssd_complex_chunkwise, ssd_naive_complex

torch.manual_seed(0)
B, T, H, D, N = 2, 16, 2, 4, 4
x = torch.randn(B, T, H, D, dtype=torch.complex64)
A = torch.randn(H, dtype=torch.complex64) - 1.0
B_t = torch.randn(B, T, H, N, dtype=torch.complex64)
C_t = torch.randn(B, T, H, N, dtype=torch.complex64)
dt = torch.zeros(B, T, H)

y_chunk = ssd_complex_chunkwise(x, A, B_t, C_t, dt, chunk_size=4)
y_naive = ssd_naive_complex(x, A, B_t, C_t, dt)
assert torch.allclose(y_chunk, y_naive.real, atol=1e-4)
```

If the assertion fails, **do not** ship — `docs/theory/04-chunkwise-algorithm.md`
is the breakdown reference for the chunkwise-vs-naive derivation.

## Skill 3: Tune chunk_size for throughput

`chunk_size` lives in `ModelConfig` (default 64).

| chunk_size | Memory | Throughput | Notes |
|------------|--------|------------|-------|
| 32         | low    | ~baseline  | best for short seqs (<2K) |
| 64         | mid    | baseline   | production default |
| 128        | higher | +5–10%     | only if FA2-equivalent throughput target met |
| 256        | high   | +10–15%    | risk of OOM at seq_len 8K, batch ≥ 32 |

## Skill 4: Add a new SSM variant to the block

1. Implement in `models/ssd_complex.py` or `models/mamba_block.py`.
2. Add a config flag in `ModelConfig` and gate the new path on it.
3. Add an equivalence test in `tests/test_ssd.py`.
4. Run `python3 -m pytest tests/ -v` — must stay green.

## Skill 5: GPU E2E smoke test

```bash
ENABLE_TRITON_KERNELS=1 python3 tests/e2e_gpu_smoke.py
```

Runs 8 checks: environment, data pipeline, model forward (pytorch +
triton dispatch), triton-vs-pytorch parity, training step, checkpoint
round-trip, and a full `Pretrainer` dry-run. Requires CUDA + triton.

## Pitfalls

- **Triton is sanctioned for one hot path:** `per_chunk_ssd_triton` in
  `models/ssd_triton.py`. No new kernel without updating `AGENTS.md` §1
  and adding `docs/reference/<name>.md`.
- **FA2 is disabled:** don't add `with sdpa_kernel(FLASH_ATTENTION)`;
  the chunkwise projection replaces attention.
- **Complex stride:** `torch.view_as_complex` requires the last dim to
  be stride-2 contiguous. If you change tensor layouts, double-check.
- **State size is even:** N must remain even to pack into complex pairs.
  Odd N silently breaks the complex recurrence.
- **NaN guard:** `nan_guard_max_consecutive=5` — after 5 consecutive
  NaN steps the run auto-rolls back to the last good checkpoint.