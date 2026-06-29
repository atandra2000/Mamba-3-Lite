# State Space Duality (SSD) and Complex Extension

## The Mamba-2 Baseline (Real SSD)
State Space Duality (SSD) connects structured state space models (SSMs) to self-attention.
In the original Mamba-2 formulation, the hidden state update follows:
`h_t = exp(A * dt) * h_{t-1} + B * x_t`
The output is then projected via `C`:
`y_t = C * h_t`

Because `A` is diagonal, the sequential recurrence can be parallelized into a chunkwise matrix multiplication, significantly boosting throughput on Tensor Cores.

## The Mamba-3 Complex SSD Extension
Mamba-3 expands on this by moving the state representation from real numbers into the complex plane `ℂ^{H×N}`.

```
h_t = exp((A_real + i * A_imag) * dt) * h_{t-1} + (B_real + i * B_imag) * x_t
```

This simple promotion doubles the expressive capability per parameter dimension, allowing Mamba-3 to halve the state size `N=128 -> N=64` without incurring a perplexity penalty.
The exponential of the complex eigenvalue models both the decay (scaling) and the oscillation (rotation) natively during the sequence scan.
