# Mamba-3-Lite — SSD Theory: State-Space Duality, Complex States, and the Chunkwise Algorithm

This document proves the central structural result behind Mamba-2-style models — that a linear, diagonal state-space recurrence can be reorganized into block matrix multiplications whose intra-chunk term is an attention-like expression — and shows how that reorganization appears in `models/ssd_complex.py:ssd_complex_chunkwise`.

## 60-second summary

After reading this document you will understand:

- **Why the sequential scan is slow**: it is a serial chain of $T$ tiny steps that underutilizes a GPU's Tensor Cores.
- **The SSD decomposition**: split the sequence into chunks of length $C$; outputs inside a chunk are computed by matrix multiplications against a structured causal "segment" matrix $L$, and only a cheap $T/C$-step recurrence remains for the state carried between chunks.
- **The attention link**: each chunk is a mini self-attention block with score matrix $C B^\top$, values $X$, and the decay $L[l,s]=\exp(A_{cs}[l]-A_{cs}[s])\cdot\mathbf{1}[l\ge s]$ in place of softmax scores — a *linear attention* with a data-dependent kernel.
- **The code mapping**: the chunked view is literally what `models/ssd_complex.py:ssd_complex_chunkwise` computes, and the equivalence is machine-checked by `tests/test_ssd.py::test_chunkwise_matches_naive_complex`.

## 2. Why it exists: the sequential scan is the bottleneck

A state space model (SSM) is the linear recurrence

$$h_t = \bar A_t\, h_{t-1} + B_t x_t, \qquad y_t = C_t h_t,$$

with $\bar A_t = \exp\big(\mathrm{softplus}(dt_t)\, A\big)$ (see `models/ssd_complex.py:_discretise` for the discretization; the foundations, including why $A$ is diagonal and why $dt$ is exponentiated through softplus, are developed in [docs/concepts/state-space-foundations.md](docs/concepts/state-space-foundations.md)). The reference implementation `models/ssd_complex.py:ssd_naive_complex` evaluates this literally:

```python
for t in range(T):
    s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
    ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
```

Each iteration updates the state $h_t\in\mathbb{C}^{N\times D}$ per head, then reads it out through $C_t$. The loop has two problems, both of which are about *hardware*, not flops:

1. **Serial dependency.** Step $t$ cannot start until step $t-1$ has produced $h_{t-1}$. The critical path is $T$ sequential steps, and each step is a handful of small elementwise/outer-product operations. A GPU's strength is parallel work; a serial chain of tiny steps runs at a small fraction of peak.
2. **Memory-bound, low arithmetic intensity.** Per step, per head, the kernel reads and writes the full state $h$ ($N\cdot D$ complex numbers) and $x_t$, performs $O(ND)$ flops, and moves on. There is no reuse of loaded data across steps, so the scan is limited by memory bandwidth, not compute. Nothing here is shaped like a matrix multiply, so Tensor Cores — the units that deliver most of a modern GPU's FLOPs — sit idle.

The key observation that breaks both problems is that the recurrence is **linear**. A linear recurrence is a *matrix multiplication in disguise*: expand it and the products of the $\bar A$'s telescope. Because $\bar A_t$ is diagonal (indeed scalar-per-head in this repo), the product $\bar A_t \bar A_{t-1}\cdots \bar A_{s+1}$ is just $\exp\big(\sum_{u=s+1}^{t} A_{log,u}\big)$, a single exponential of a cumulative sum. That collapses the sequential chain into per-position factors that can be rearranged freely — in particular, *blocked* into chunks. This blocking is the State Space Duality (SSD) theorem of Dao & Gu (2024, arXiv:2405.21060).

## 3. Intuition: each chunk is a mini self-attention block

Unroll the recurrence once. The output at time $t$ is a sum over all past inputs:

$$y_t = \sum_{s \le t} C_t\, \underbrace{\bar A_t \bar A_{t-1}\cdots \bar A_{s+1}}_{\text{decay from } s \text{ to } t}\, B_s x_s.$$

Compare this with self-attention, $y_t = \sum_{s\le t} \langle q_t, k_s\rangle\, v_s$:

| Self-attention | SSD recurrence |
|---|---|
| query $q_t$ | $C_t$ (a complex $N$-vector per position) |
| key $k_s$ | $B_s$ (a complex $N$-vector per position) |
| score $\langle q_t, k_s\rangle$ | $C_t B_s^\top$, a bilinear form over $N$ |
| causal mask $\mathbf{1}[s\le t]$ | $\mathbf{1}[s\le t]$ — identical |
| softmax-normalized scores | $\exp(\text{cumulative }A_{log})$ decay — *not* normalized |

So the SSD is attention without the softmax, where the score matrix is the outer structure $C B^\top$ and the causal mask is multiplied by a position-dependent decay. Softmax attention normalizes scores so that $\sum_s \text{score} = 1$ regardless of context; the SSM instead *decays* old positions multiplicatively, which is what lets it keep a fixed-size state $h$ instead of a growing list of keys/values.

Now the intuition: split the sequence into chunks of $C$ positions. For positions *inside the same chunk*, the decay factors between them are known in advance (they depend only on the cumulative $A_{log}$ within the chunk), so the whole intra-chunk sum is one batched matrix product — the "mini attention block." For positions in *different chunks*, the decay factorizes into a product of (a) decay from the source position to the end of its chunk, (b) decay across whole intermediate chunks, and (c) decay from the start of the destination chunk to the target position. Term (a) is folded into a per-chunk state; term (b) is a *chunk-level* recurrence over only $T/C$ steps; term (c) is another per-position factor. The serial scan of length $T$ becomes a scan of length $T/C$ whose steps are big matrix multiplications.

## 4. The SSD decomposition, derived

### 4.1 Setup and the unrolled form

Adopt the code's convention: `A_log = softplus(dt) * A` (complex, shape $(B,T,H)$; `A` is a per-head complex scalar, so `A_log` is diagonal in the state index — this is what makes everything below work). The discrete transition is $\bar A_t = \exp(A_{log,t})$, and the recurrence is

$$h_t = e^{A_{log,t}} h_{t-1} + B_t x_t, \qquad y_t = C_t h_t.$$

Unrolling from the initial state $h_0=0$:

$$h_t = \sum_{s=1}^{t} \left(\prod_{u=s+1}^{t} e^{A_{log,u}}\right) B_s x_s, \qquad
y_t = C_t h_t = \sum_{s\le t} e^{\sum_{u=s+1}^{t} A_{log,u}}\, C_t B_s x_s.$$

The product of exponentials collapses into the exponential of a sum — the single algebraic fact the entire chunking scheme rests on. Define the **inclusive cumulative sum**

$$\Lambda(t) = \sum_{u=1}^{t} A_{log,u},$$

so that $\sum_{u=s+1}^{t} A_{log,u} = \Lambda(t) - \Lambda(s)$ and

$$y_t = \sum_{s\le t} e^{\Lambda(t)-\Lambda(s)}\, C_t B_s x_s .$$

### 4.2 Chunking

Let $C$ be the chunk size, $n = \lceil T/C\rceil$ the number of chunks, and write the global time $t = cC + l$ for chunk $c\in\{0,\dots,n-1\}$ and intra-chunk position $l\in\{0,\dots,C-1\}$. Similarly $s = c'C + j$. Split the sum over $s$ into two groups: **intra-chunk** ($c'=c$, so $j\le l$) and **inter-chunk** ($c'<c$). The global output is the sum of the two contributions:

$$Y_{c,l} = Y^{\mathrm{diag}}_{c,l} + Y^{\mathrm{off}}_{c,l}.$$

### 4.3 The causal segment matrix $L$ and the intra-chunk term $Y^{\mathrm{diag}}$

For two positions in the same chunk, define the within-chunk cumulative sum

$$A_{cs}[l] = \sum_{j=0}^{l} A_{log, cC+j},$$

i.e., the code's `A_cumsum = torch.cumsum(Ac, dim=2)` (cumulative over the intra-chunk axis). The decay from position $(c,j)$ to position $(c,l)$ is

$$e^{\Lambda(cC+l)-\Lambda(cC+j)} = e^{A_{cs}[l]-A_{cs}[j]},$$

because both global cumulants contain the identical prefix $\sum_{u< cC} A_{log,u}$, which cancels in the difference. Causality requires $j\le l$. Packing the causal indicator into a matrix:

$$\boxed{\,L[l,j] = e^{A_{cs}[l]-A_{cs}[j]}\,\mathbf{1}[l\ge j]\,}$$

This is the **causal segment matrix**: lower-triangular (inclusive diagonal), with entries that are exponentials of cumulative-$\log$-decay differences. Note the diagonal is exactly $1$ (a position's own input enters its output undecayed, matching the recurrence), and $L$ is the same for every batch element and every head only in the special case of position-independent `A_log` — in general it is per-$(b,h)$ because `dt` is input-dependent.

The intra-chunk output is the sum over earlier positions inside the chunk:

$$Y^{\mathrm{diag}}_{c,l} = \sum_{j\le l} C_{c,l}\, e^{A_{cs}[l]-A_{cs}[j]}\, B_{c,j}\, x_{c,j}.$$

In index form this is a contraction over the source position $s$ and the feature index $n$:

$$Y^{\mathrm{diag}}[b,c,l,h,p] = \sum_{s,n} Cc[b,c,l,h,n]\; L[b,c,h,l,s]\; Bc[b,c,s,h,n]\; Xc[b,c,s,h,p],$$

which is exactly the code's `Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)`. Written as matrices per $(b,c,h)$ — with $C$ the $(C\times N)$ matrix of readout vectors, $B$ the $(C\times N)$ matrix of input couplings, $L$ the $(C\times C)$ segment matrix, and $X$ the $(C\times D)$ chunk of inputs:

$$\boxed{\,Y^{\mathrm{diag}} = \big(C B^{\top} \odot L\big)\, X\,}$$

The score matrix $C B^{\top}$ has entries $\sum_n C_{l,n} B_{s,n}$; it is masked (multiplied elementwise) by the decay $L$; the result acts on the values $X$ by an ordinary matrix product over positions.

### 4.4 The linear-attention link: $\exp(\cdot)$ as a kernel between positions

Write the intra-chunk term as a sum over positions:

$$Y^{\mathrm{diag}}_{c,l} = \sum_{j\le l} \underbrace{\Big(\sum_n C_{c,l,n} B_{c,j,n}\Big)}_{\text{score } \langle C_{c,l}, B_{c,j}\rangle}\;\; \underbrace{e^{A_{cs}[l]-A_{cs}[j]}}_{\text{decay kernel}}\; x_{c,j}.$$

This is **linear attention** (Katharopoulos et al., 2020): outputs are a sum over past positions of a *bilinear score* between a query vector ($C_{c,l}$) and a key vector ($B_{c,j}$), applied to a value ($x_{c,j}$) — with no softmax normalization, hence "linear." The decay factor is the twist: it is a **kernel between positions**,

$$K(l,j) = e^{A_{cs}[l]-A_{cs}[j]} = e^{A_{cs}[l]}\, e^{-A_{cs}[j]},$$

which *factorizes* into a product of a function of the query position and a function of the key position. This rank-1 factorization is the duality. Because of it, the sum over *all* past positions can be reorganized as a running state:

$$\sum_{j \le l} e^{A_{cs}[l]-A_{cs}[j]}\, B_{c,j}\, x_{c,j} = e^{A_{cs}[l]}\, \underbrace{\sum_{j\le l} e^{-A_{cs}[j]}\, B_{c,j}\, x_{c,j}}_{\text{accumulated state}}.$$

The same computation is simultaneously (i) an attention-like sum over positions inside a chunk (the $L$-masked matrix product above) and (ii) a linear recurrence whose state is the accumulated sum on the right. *Both views are exact* — this equivalence between the attention view and the recurrence view is precisely the "duality" in State Space Duality. It is also why the state size never grows with $T$: unlike attention's key/value lists, the accumulated state is a fixed-shape tensor.

### 4.5 The inter-chunk term: per-chunk states and the chunk-level scan

Now let the source position $(c',j)$ and the destination $(c,l)$ lie in different chunks, $c' < c$. Split the decay sum into three segments: within the source chunk, across whole intermediate chunks, and within the destination chunk:

$$\sum_{u=c'C+j+1}^{cC+l} A_{log,u}
= \underbrace{\big(A_{cs}[c'][C{-}1] - A_{cs}[c'][j]\big)}_{\text{to end of source chunk}}
+ \underbrace{\sum_{c''=c'+1}^{c-1} \mathrm{ct}[c'']}_{\text{whole chunks in between}}
+ \underbrace{A_{cs}[c][l]}_{\text{into destination chunk}},$$

where $\mathrm{ct}[c] = A_{cs}[c][C{-}1] = \sum_{j} A_{log,cC+j}$ is the **chunk total** (the code's `chunk_decay = A_cumsum[:, :, -1, :]`). Define the chunk-level cumulative sum $\mathrm{CD}[c] = \sum_{c''\le c} \mathrm{ct}[c'']$ (the code's `cd_cumsum`), so that $\sum_{c''=c'+1}^{c-1}\mathrm{ct}[c''] = \mathrm{CD}[c{-}1] - \mathrm{CD}[c']$.

First, each chunk's contribution to the state *at the end of its own chunk* — the source segment folded together:

$$S_c = \sum_{j=0}^{C-1} e^{A_{cs}[c][C{-}1] - A_{cs}[c][j]}\, B_{c,j}\, x_{c,j},$$

which is the code's `states = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)` with `decay_states = exp(A_cumsum[:, :, -1:, :] - A_cumsum)`. Second, the state entering chunk $c$ is the sum of all earlier chunks' end-states, decayed across the intermediate chunks:

$$H_c = \sum_{c'<c} e^{\mathrm{CD}[c-1] - \mathrm{CD}[c']}\, S_{c'}.$$

This is the **chunk-level scan**: it is a recurrence over only $n = T/C$ steps (the code's `decay_chunk` cumsum + causal tril, contracted over chunks). Finally, read the accumulated state out at position $l$ by decaying from the start of the chunk to $l$:

$$Y^{\mathrm{off}}_{c,l} = C_{c,l}\; e^{A_{cs}[c][l]}\, H_c,$$

the code's `Y_off = torch.einsum("bclhn,bchpn,bclh->bclhp", Cc, states, torch.exp(A_cumsum))`. Multiplying the three exponentials reproduces the global decay $\exp(\Lambda(cC{+}l)-\Lambda(c'C{+}j))$ term by term, so

$$Y_{c,l} = Y^{\mathrm{diag}}_{c,l} + Y^{\mathrm{off}}_{c,l} = \sum_{s\le cC+l} e^{\Lambda(cC+l)-\Lambda(s)}\, C_{c,l} B_s x_s = y_{cC+l},$$

which is exactly the unrolled recurrence. This is the **SSD decomposition**: the sequential scan is replaced by (a) per-chunk matrix products ($Y^{\mathrm{diag}}$, the per-chunk states $S_c$, the readout $Y^{\mathrm{off}}$) and (b) one short scan over $n = T/C$ chunks ($H_c$). A full einsum-by-einsum account of the implementation, including shapes and the padding contract, is in [docs/concepts/ssd-theory.md](docs/concepts/ssd-theory.md).
Multiplying the three exponentials reproduces the global decay $\exp(\Lambda(cC{+}l)-\Lambda(c'C{+}j))$ term by term, so

$$Y_{c,l} = Y^{\mathrm{diag}}_{c,l} + Y^{\mathrm{off}}_{c,l} = \sum_{s\le cC+l} e^{\Lambda(cC+l)-\Lambda(s)}\, C_{c,l} B_s x_s = y_{cC+l},$$

which is exactly the unrolled recurrence.

### 4.6 Visual Tensor Contraction Map of Chunkwise Einsums

```
Intra-Chunk Path:
  Cc (B, n_c, C, H, N) --\
  Bc (B, n_c, C, H, N) ---> [ einsum: bclhn,bcshn,bchls,bcshp -> bclhp ] ---> Y_diag (B, n_c, C, H, D)
  L  (B, n_c, H, C, C) --/
  Xc (B, n_c, C, H, D) --/

End-of-Chunk State & Inter-Chunk Scan:
  Bc, decay_states, Xc -> [ einsum: bclhn,bclh,bclhp -> bchpn ] -> states (B, n_c, H, D, N)
  states, decay_chunk  -> [ einsum: bhzc,bchpn -> bzhpn ]       -> states (decayed across chunks)

Inter-Chunk Output Path:
  Cc, states, exp(A_cumsum) -> [ einsum: bclhn,bchpn,bclh -> bclhp ] -> Y_off (B, n_c, C, H, D)

Final Output:
  Y = Y_diag + Y_off  ===> reshape back to (B, T, H, D)
```

### 4.7 Complex Autograd and Wirtinger Derivatives

Because $B_t, C_t, X_c, A$ are complex tensors in `torch.complex64`, PyTorch autograd handles backpropagation using **Wirtinger derivatives** ($\frac{\partial \mathcal{L}}{\partial z} = \frac{1}{2}(\frac{\partial \mathcal{L}}{\partial x} - i \frac{\partial \mathcal{L}}{\partial y})$ for $z = x + iy$).

When writing custom autograd functions (such as `_PerChunkSSDTriton` in `models/ssd_triton.py:per_chunk_ssd_triton`), PyTorch tracks complex gradients through real and imaginary tensor channels. The backward pass must compute downstream gradients with respect to both real and imaginary inputs, taking care not to detach tensor graphs prematurely or pass un-instantiated zero gradients.
## 5. Complexity: what chunking buys

**Sequential scan (naive).** $T$ serial steps; per step and per $(b,h)$: $O(ND)$ flops (state scaling, outer product, readout) and $O(ND)$ complex numbers of state traffic; total $O(B\,H\,T\,N\,D)$ flops with a critical path of length $T$ and negligible arithmetic intensity. The scan is bandwidth-bound and latency-bound.

**Chunkwise.** Counting per $(b,h)$:

- Intra-chunk term $Y^{\mathrm{diag}}$: the score product is $C^2 N$ flops (a $(C\times N)\cdot(N\times C)$ multiply) and the masked application to $X$ is $C^2 D$ (a $(C\times C)\cdot(C\times D)$ multiply), per chunk. Over $n=T/C$ chunks: $O\big(T\,C\,(N{+}D)\big)$ — **linear in $T$**, with everything expressed as batched GEMMs.
- Per-chunk states $S_c$ and readout $Y^{\mathrm{off}}$: $O(TC\,ND) = O(TND)$ each, again einsum-shaped matmuls.
- Inter-chunk scan $H_c$: the decay matrix is $n\times n$, so $O(n^2 ND) = O\big((T/C)^2 ND\big)$ flops, with a serial dependency of length $n = T/C$ instead of $T$.

For the repo's dimensions ($T=2048$, $C=64$, $N=D=64$, $n=32$) the serial depth drops from 2048 to 32 steps, and the dominant work is dense, batched GEMMs of shape $(64\times 64)\cdot(64\times 64)$-style tiles — exactly what Tensor Cores execute. The flop count is *not* lower than the scan (chunking adds the $C^2$ and $n^2$ factors); the win is that the flops are now dense matmuls with high arithmetic intensity instead of a serial chain of tiny elementwise ops. The Triton path (`per_chunk_ssd_triton` in `models/ssd_triton.py`, behind `ssd_dispatch='triton'`) fuses the `L` construction, $Y^{\mathrm{diag}}$, and $S_c$ into one kernel, avoiding materializing $L$ in global memory; the inter-chunk scan stays in PyTorch.

**Why $C$ is a tunable knob.** Raising $C$: larger GEMM tiles (better Tensor-Core efficiency), fewer serial chunk steps, but the intra-chunk work grows as $C$ and, if $L$ is materialized, its storage grows as $C^2$ — $(B\cdot \tfrac{T}{C}\cdot H\cdot C^2)$ complex entries, i.e. $8\,B\,T\,H\,C$ bytes (for $B{=}16$, $T{=}2048$, $H{=}16$, $C{=}64$: $\approx 268$ MB in complex64). Lowering $C$: less memory, more serial steps, smaller tiles. $C=64$ matches $N=D=64$, giving square power-of-two tiles and a comfortable $T/C = 32$ chunk depth. All throughput figures here are `[INFERENCE]` — there is no `.benchmarks/` in the tree; the memory arithmetic above is derived from the shapes.

## 6. Code walkthrough: the chunked view in `models/ssd_complex.py:ssd_complex_chunkwise`

The function `models/ssd_complex.py:ssd_complex_chunkwise` is the theorem above, line by line. (The sibling doc [docs/concepts/ssd-theory.md](docs/concepts/ssd-theory.md) walks the full einsum chain; here we stay at the theorem level.)

**Step 1 — pad and reshape into chunks.** The sequence is padded to a multiple of the chunk size, then every tensor is reshaped from $(B,T,\ldots)$ to $(B, n_{\text{chunks}}, C, \ldots)$:

```python
pad = (C - (T % C)) % C
if pad > 0:
    x = F.pad(x, (0, 0, 0, 0, 0, pad))
    B_t = F.pad(B_t, (0, 0, 0, 0, 0, pad))
    C_t = F.pad(C_t, (0, 0, 0, 0, 0, pad))
    dt = F.pad(dt, (0, 0, 0, pad))

T_padded = T + pad
n_chunks = T_padded // C

A_log = F.softplus(dt) * A

def _chunk(t):
    return t.reshape(B_, n_chunks, C, *t.shape[2:])
```

The $C$-axis of `Xc, Bc, Cc` is the intra-chunk position axis of the theorem; the second axis is the chunk index.

**Step 2 — the log-decay and its cumulative sum.** `A_log = F.softplus(dt) * A` (softplus keeps the discretized decay contract: $dt\ge 0$), then

```python
A_cumsum = torch.cumsum(Ac, dim=2)
decay_states = torch.exp(A_cumsum[:, :, -1:, :] - A_cumsum)
```

`A_cumsum` is $A_{cs}[l]$ (Section 4.3); `decay_states` is the factor $e^{A_{cs}[C-1]-A_{cs}[l]}$ used to build the per-chunk end-states $S_c$.

**Step 3 — the causal segment matrix and the intra-chunk term** (PyTorch branch):

```python
Ac_perm = Ac.permute(0, 1, 3, 2).contiguous()
T_c = Ac_perm.size(-1)
Ac_cumsum = torch.cumsum(Ac_perm, dim=-1)
Ac_seg = Ac_cumsum.unsqueeze(-1) - Ac_cumsum.unsqueeze(-2)
mask = torch.tril(torch.ones(T_c, T_c, device=x.device, dtype=torch.bool))
L = torch.exp(Ac_seg) * mask
```

`Ac_seg[l, s] = A_{cs}[l] - A_{cs}[s]`, the `torch.tril` mask enforces $l \ge s$, and the exponential produces exactly $L[l,s] = e^{A_{cs}[l]-A_{cs}[s]}\mathbf{1}[l\ge s]$. Then

```python
Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)
```

is the index form of $Y^{\mathrm{diag}} = (C B^{\top}\odot L) X$ from Section 4.3. This is the "mini attention block" — the only place the model ever materializes an attention-like score matrix.

**Step 4 — per-chunk states and the chunk-level scan.**

```python
states = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)
```

computes $S_c$ (Section 4.5). Then the chunk-level recurrence:

```python
chunk_decay = A_cumsum[:, :, -1, :]
cd_perm = chunk_decay.permute(0, 2, 1).contiguous()
cd_cumsum = torch.cumsum(cd_perm, dim=-1)
cd_seg = cd_cumsum.unsqueeze(-1) - cd_cumsum.unsqueeze(-2)
decay_chunk = torch.exp(cd_seg) * torch.tril(torch.ones(n_chunks, n_chunks, device=x.device, dtype=torch.bool))
states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states[:, :-1])
```

`chunk_decay` is $\mathrm{ct}[c]$; `cd_cumsum` is $\mathrm{CD}[c]$; `decay_chunk` is the causal chunk-level decay matrix (cumsum-difference + `tril`, the same construction as $L$ but over the $n \times n$ chunk grid). The einsum is $H_c = \sum_{c'} \text{decay\_chunk}[c,c']\, S_{c'}$ — the scan over $T/C$ steps. (A known discrepancy between this propagation and the theorem is documented in Pitfalls; see also the coordinator's bug report.)

**Step 5 — readout and reassembly.**

```python
Y_off = torch.einsum("bclhn,bchpn,bclh->bclhp", Cc, states, torch.exp(A_cumsum))
Y = Y_diag + Y_off
Y = Y.real
return Y.reshape(B_, T_padded, H, D)[:, :T, :, :]
```

`torch.exp(A_cumsum)` is $e^{A_{cs}[l]}$, the decay from the start of the chunk to position $l$; the einsum is $Y^{\mathrm{off}}_{c,l} = C_{c,l} e^{A_{cs}[c][l]} H_c$. The sum $Y^{\mathrm{diag}} + Y^{\mathrm{off}}$ is the full SSD decomposition. Only the real part is returned: the model's output projection is real-valued, and the imaginary part of the state is an internal representation (see [docs/concepts/ssd-theory.md](docs/concepts/ssd-theory.md) for why the state is complex at all). Finally the padding added in Step 1 is sliced away (`[:, :T]`).

## 7. Pitfalls

**Causality is only enforced by the mask.** Nothing in `torch.einsum` or `torch.cumsum` knows about causality; the only things making the computation causal are the `torch.tril` masks on $L$ and on `decay_chunk`. An upper-triangular or missing mask silently lets future positions leak into the present — and since the score matrix $C B^{\top}$ is not normalized, the leak is not bounded by any attention-style sum-to-one constraint. Note also that $L$'s diagonal is intentionally inclusive ($l\ge s$): the self-contribution $s=l$ has decay $e^0 = 1$ and *must* be included, matching the recurrence's own-input term. A strictly-lower-triangular mask (excluding $l=s$) would drop each position's own input.

**`cumsum` vs `segsum` conventions.** The decay from $s$ to $l$ is $e^{\Lambda(l)-\Lambda(s)}$ with an *inclusive* cumulative sum. If the cumsum is shifted (exclusive — summing positions strictly before the current one), every off-diagonal entry of $L$ picks up an extra factor $e^{A_{log}}$ and the chunkwise result silently disagrees with the naive scan. The code is consistent: `A_cumsum = cumsum(Ac, dim=2)` is inclusive, `chunk_decay = A_cumsum[:, :, -1, :]` is the inclusive chunk total, and the same inclusive convention feeds $L$, `decay_states`, and `decay_chunk`. Any change to one of the three must change all three.

**Numerical stability of `exp` of large-magnitude arguments.** $L$ is built as $\exp$ of differences of cumulative sums, not as a product of $T$ per-step decays; this is deliberate — products of many near-one factors accumulate error and underflow, while the log-space difference keeps the computation in a range where $\exp$ is well-conditioned. With the initial $A=-1$ and $dt\ge 0$, $A_{log}$ has a non-positive real part, so all differences $A_{cs}[l]-A_{cs}[j]$ for $l\ge j$ are $\le 0$ and $L\in(0,1]$ — no overflow, and underflow to 0 is the *intended* long-range forgetting. But `A` is learned and complex: if its real part drifts positive, `A_cumsum` grows, and `torch.exp(A_cumsum)` in the $Y^{\mathrm{off}}$ readout can overflow to `inf` (then `NaN` after the additions). This is the riskiest exponential in the function. The per-head constant-`A` parameterization and the complex64 state (fp32 accumulation, never BF16 inside the SSD path) are what keep this contained in practice; see [docs/concepts/block-and-stability.md](docs/concepts/block-and-stability.md).

**Inter-chunk decay index alignment (live discrepancy, flagged 2026-08-04).** The theorem requires the state entering chunk $c$ to be $H_c = \sum_{c'<c} e^{\mathrm{CD}[c-1]-\mathrm{CD}[c']}\, S_{c'}$. The implementation builds `decay_chunk[b,h,z,c] = exp(CD[z]-CD[c])` and contracts it against the shifted list `[initial_states, S_0, S_1, ...]`, which yields a net factor $e^{\mathrm{CD}[z]-\mathrm{CD}[c+1]} = e^{\sum_{c''=c+2}^{z}\mathrm{ct}[c'']}$ for source chunk $c$ — equal to the theorem's $e^{\sum_{c''=c+1}^{z-1}\mathrm{ct}[c'']}$ **only when all chunk totals are equal**. That is exactly the condition the existing test exercises (`dt=0` and a per-head constant `A` make `A_log = ln(2)A` constant, hence all chunk totals identical), so `tests/test_ssd.py::test_chunkwise_matches_naive_complex` passes at ~1e-6. With input-dependent `dt` (the normal training regime — `dt` comes from the input projection), the chunkwise function deviates from `models/ssd_complex.py:ssd_naive_complex`; a 16-step/4-chunk random probe measured an absolute max difference of ~0.23 against the oracle while the `dt=0` version of the same probe measured ~1e-6. The intra-chunk term and `decay_states` are correct; only the chunk-level propagation is affected. This is a code bug, not a documentation choice — it has been reported to the coordinator (with a proposed fix and a regression-test recipe using time-varying `dt`) and the walkthrough above describes the code as it currently stands.

**Complex vs real output and padding.** `ssd_naive_complex` returns complex64 outputs; `ssd_complex_chunkwise` returns `Y.real` in float32. Any equivalence check must compare against `y_naive.real` (as the test does). And when $T$ is not a multiple of $C$, the padded positions must be sliced away after reassembly — forgetting `[:, :T]` returns a tensor of length $T_{\text{padded}} \ne T$.

## 8. Tests

- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` — the machine proof of the SSD decomposition: with `B,T,H,D,N = 2,16,2,4,4` and `chunk_size=4`, it asserts `torch.allclose(y_chunk, y_naive.real, atol=1e-4)`, plus dtype (`float32` vs `complex64`) and shape checks. As analyzed in the pitfalls, this test runs in the constant-`A_log` regime; it verifies the decomposition but does not exercise the inter-chunk propagation under non-uniform chunk totals.
- `tests/test_ssd.py::test_chunkwise_handles_uneven_T` — verifies padding correctness: `T=20` with `chunk_size=4` (not a multiple) must return shape `(B, 20, H, D)` with finite values.
- `tests/test_ssd.py::test_chunkwise_handles_T_equal_to_chunk` — the boundary `T == C`, i.e. a single chunk with no inter-chunk term at all.

The broader suite (37 tests collected (32 passed / 5 GPU-skipped on CPU)) also covers the Triton dispatch against this same chunkwise reference; the chunkwise-vs-naive equivalence above is the contract both dispatch paths must satisfy.
---

#Complex-Valued State Spaces

Mamba-3's sequence-mixing primitive packs decay *and* rotation into one complex parameter per head; this doc derives why, walks the code that builds and scans the complex state, and explains complex autograd, `torch.complex` mechanics, and the N-halving parity claim.

## 60-second summary

After reading this doc you will understand:

- A real SSM state can only *scale* per step (`e^{a·dt}`); oscillation requires rotation, which needs two real dimensions per mode. A complex state `h = u + iv` carries both in one scalar: `e^{(α+iβ)dt} = e^{αdt}(cos βdt + i sin βdt)`.
- One complex state = **two real degrees of freedom**, so a 64-dimensional complex state has the same coordinate count as a 128-dimensional real state. The "N=64 complex ≈ N=128 real" parity claim is plausible from that counting argument, but it is an empirical claim from the paper — marked `[INFERENCE]` below — not something this repo's tests verify.
- The repo's recurrence is `h_t = e^{softplus(dt)·A} h_{t-1} + B_t x_t` with per-head complex scalar `A` (init `−1.0`, i.e. pure decay), complex `B_t`/`C_t`, and a real readout `y_t = Re(C_t h_t)`. The state never leaves `complex64`; the logits are FP32 because the readout is a real linear map.
- PyTorch's complex autograd uses (conjugate) Wirtinger derivatives, so `.grad` on a `complex64` tensor is a complex tensor whose `.real`/`.imag` are the two coordinate gradients; the imaginary parts of `A`, `B`, `C` are genuinely trainable even though the loss is real.
- `torch.view_as_real` gives an interleaved stride-2 layout; `models/ssd_triton.py:_view_real_imag` materializes two contiguous float32 buffers for the Triton kernel — the `contiguous()` call is load-bearing.

## Why it exists

A diagonal real state space `h_t = e^{A·dt} h_{t-1}` with `A ∈ ℝ^N` models one thing per mode: a rate of decay (or growth). Its impulse response is a sum of decaying exponentials. Many sequence phenomena — alternating structure, periodic content, anything with a "phase" — are better described by *oscillating* modes, which require eigenvalues off the real axis. A real matrix can only have such eigenvalues in conjugate pairs, which forces a `2×2` real block per oscillating mode. In other words, in real arithmetic an oscillating mode costs **two** state dimensions.

A complex scalar `A = α + iβ` is the minimal object that carries both a decay rate `α` and an angular frequency `β`: after one discrete step of size `dt`, the state is multiplied by `e^{(α+iβ)dt} = e^{αdt}·e^{iβdt}`, i.e. scaled by `e^{αdt}` and rotated by `β·dt`. Mamba-2 used `N=128` real state dimensions per head; Mamba-3 replaces them with `N=64` **complex** state dimensions (`state_dim=64`, `complex64`) — the same number of real coordinates, with rotation for free. The architectural bet, stated in `AGENTS.md` and derived below, is that 64 complex states give perplexity parity with 128 real states while halving the state dimension (and the per-step state bandwidth in the scan).

## Intuition first

Think of a phasor on the complex plane. A complex number `a = e^{α+iβ}` has a radius `e^{α}` and an angle `β`. Multiplying a state by it does two independent things at once:

- `e^{α}` shrinks (α < 0) or grows (α > 0) the magnitude — **decay**;
- `e^{iβ} = cos β + i sin β` turns the state by angle `β` — **rotation**.

So *one* complex parameter is a scale *and* a rotation: two real controls (`α`, `β`) in a single multiplicand. A real scalar can only scale; to rotate you need two coupled real numbers (a `2×2` rotation matrix). That is the whole reason complex states exist here: the sequence model wants dynamics that turn, and the complex plane is the smallest place a turn lives.

The state itself is a little complex vector: each of its `N` complex entries `h_n = u_n + iv_n` is two real numbers that get *coupled* — the rotation mixes `u` and `v` on every step (that coupling is exactly what a real diagonal SSM cannot do). Readout is a real projection `y = Re(C h)`: it folds the two sub-states back into one real number per head-channel.

## Math

### The recurrence

Let `A = α + iβ ∈ ℂ` be a per-head scalar, `dt > 0` the per-token step, and `B_t, C_t ∈ ℂ^{H×N}` complex input/output maps. The repo's recurrence (see `models/ssd_complex.py:ssd_naive_complex`, the O(T) oracle) is

$$h_t = e^{(\alpha + i\beta)\,dt_t}\, h_{t-1} + B_t\, x_t, \qquad y_t = \operatorname{Re}\!\big(C_t\, h_t\big),$$

with `h_t ∈ ℂ^{N×D}` (here `N=64` states × `D=64` channels per head) and a real input `x_t` promoted to complex (imaginary part 0). The transition factor factors into the promised two effects:

$$e^{(\alpha + i\beta)dt} \;=\; e^{\alpha dt}\cdot e^{i\beta dt} \;=\; \underbrace{e^{\alpha dt}}_{\text{scale}}\;\big(\underbrace{\cos(\beta dt) + i\sin(\beta dt)}_{\text{rotation}}\big).$$

This is Euler's formula; it is not an approximation — it *is* the definition of the complex exponential.

### Unpacking: one complex state is two real states

Write `h_t = u_t + iv_t`, `B_t = b^R_t + i b^I_t`. Substituting and separating real/imaginary parts gives the equivalent **real** two-dimensional recurrence

$$
\begin{aligned}
u_t &= e^{\alpha dt}\big(\cos(\beta dt)\, u_{t-1} - \sin(\beta dt)\, v_{t-1}\big) + b^R_t\, x_t,\\
v_t &= e^{\alpha dt}\big(\sin(\beta dt)\, u_{t-1} + \cos(\beta dt)\, v_{t-1}\big) + b^I_t\, x_t,
\end{aligned}
$$

i.e. a `2×2` real transition matrix

$$\begin{pmatrix} u_t \\ v_t \end{pmatrix} = e^{\alpha dt}\underbrace{\begin{pmatrix} \cos(\beta dt) & -\sin(\beta dt) \\ \sin(\beta dt) & \cos(\beta dt) \end{pmatrix}}_{R(\beta dt)} \begin{pmatrix} u_{t-1} \\ v_{t-1} \end{pmatrix} + \begin{pmatrix} b^R_t \\ b^I_t \end{pmatrix} x_t .$$

Two observations fall out:

1. **Two real DOF per complex state.** Each complex entry is two real coordinates, and the recurrence is exactly a real linear recurrence over those coordinates. The complex model at `N=64` is therefore a linear SSM over **128 real coordinates** — the same coordinate count as Mamba-2's `N=128`.
2. **The transition is a scaled rotation.** A real *diagonal* SSM at `N=128` has transition `diag(e^{a_1 dt}, …, e^{a_{128} dt})` with `a_i ∈ ℝ`: it can scale each coordinate independently but can never rotate (a real diagonal matrix has real eigenvalues). The complex model's 128-coordinate transition is block-diagonal with `2×2` scaled-rotation blocks — a strictly richer dynamics class at the same coordinate count, because its eigenvalues `e^{α+ iβ}` may be off the real axis.

A real SSM *can* oscillate, but only by using `2×2` blocks (conjugate eigenvalue pairs), i.e. **two** real states per oscillating mode. So "oscillation capacity" measured in modes is `N_complex` complex states = `N_complex` oscillating modes = `2·N_complex` real states. That is the packing argument in one line: *one complex state carries two real sub-states, so N=64 complex achieves what N=128 real does* — same real DOF, plus the rotation the real diagonal model lacks.

### The parity claim — stated honestly

The stronger statement — that the 64-complex model reaches *perplexity parity* with the 128-real model — is an **empirical** claim from the Mamba-3 paper (Dao & Gu, 2025), not a theorem and not a fact this repository has measured: `[INFERENCE]`. What the counting argument above establishes is *plausibility* (equal real dimension, strictly more expressive transitions), and what the repo's tests establish is only *internal consistency* (chunkwise ≡ naive scan on the complex model, see Tests). No test here compares perplexity against a real-state model; treat "parity" as a paper claim the repo inherits, pending the 8B-token run (see `../concepts/block-and-stability.md`).

### Discretization

The repo discretizes the continuous-time rate `A` with a per-token step via `models/ssd_complex.py:_discretise`:

$$A\_bar = \exp\!\big(\operatorname{softplus}(dt)\cdot A\big), \qquad \operatorname{softplus}(dt) = \log(1 + e^{dt}).$$

Two design points, both deliberate:

- **`softplus` keeps the step positive and smooth.** `softplus(dt) > 0` for all real `dt`, is smooth (unlike `ReLU`), and behaves like `dt` for large `dt`. With `A = −1 + 0i` at init this makes every mode a contraction: `|e^{−softplus(dt)}| = e^{−softplus(dt)} ∈ (0,1)` — no blow-up, no sign flips, gradients flow through the smooth map.
- **Order matters.** The nonlinearity is applied to the *real* `dt` first, then multiplied by the *complex* `A` — `exp(softplus(dt)·A)`, never `exp(softplus(dt·A))`. The softplus operates in `ℝ` (complex `softplus` is not defined in PyTorch); the complex exponential then produces the rotation+scale factor. The same `A_log = softplus(dt)·A` quantity is the seed of every complex exponential inside `models/ssd_complex.py:ssd_complex_chunkwise` (the `L` matrix, `decay_states`, `decay_chunk`), which is why the whole chunkwise algorithm is just a batched, associative version of this one-step factor — detailed in `../concepts/ssd-theory.md`.

## Gradients through complex tensors

The loss is real: the SSD returns `Y.real` (float32), then MIMO, `out_proj`, and `lm_head` are real linear layers, and cross-entropy is real. Yet the parameters `A` (complex64), `B_t`, `C_t` (complex64) sit *upstream* of the `.real` truncation. How do real gradients reach them?

PyTorch's complex autograd uses **Wirtinger calculus**. For a real-valued loss `L(z)` with `z = u + iv`, the two Wirtinger derivatives are

$$\frac{\partial L}{\partial z} = \tfrac12\Big(\frac{\partial L}{\partial u} - i\frac{\partial L}{\partial v}\Big), \qquad \frac{\partial L}{\partial \bar z} = \tfrac12\Big(\frac{\partial L}{\partial u} + i\frac{\partial L}{\partial v}\Big),$$

and the *steepest-descent* direction in the `(u,v)` plane is the conjugate one: `−∂L/∂z̄`. PyTorch stores, for a complex input, exactly

$$\text{grad}_z = 2\,\frac{\partial L}{\partial \bar z} = \frac{\partial L}{\partial u} + i\,\frac{\partial L}{\partial v},$$

so that the update `z ← z − η·grad_z` is precisely real gradient descent on the two coordinates. Consequences for this repo:

- **`.grad` on a `complex64` tensor is `complex64`.** Its `.real` is `∂L/∂u` and its `.imag` is `∂L/∂v`. The two packed sub-states are trained as two real coordinates, which is exactly the packing argument in optimizer space.
- **The imaginary parts of `A`, `B`, `C` are genuinely trainable.** Although the loss only sees `Re(C h)`, the rotation makes the real output depend on `β` and on `Im B`/`Im C`: `∂Re(e^{iβdt}h)/∂β ≠ 0`. If the model needs oscillation, gradient descent can push `β` off its init value 0.
- **Non-holomorphic ops are handled.** `z.real`, `.abs`, `torch.view_as_real` are not holomorphic; autograd still propagates through them with the same conjugate-Wirtinger convention (e.g. the backward of `z.real` contributes a purely real `grad_z`). The chain through the SSD — holomorphic ops (`exp`, multiply, einsums) ending in `.real` — is therefore well-defined.
- **Optimizers accept complex parameters.** The `A` parameter (`torch.empty(H, dtype=torch.complex64)` in `models/mamba_block.py:Mamba3Block`) is optimized directly by AdamW; PyTorch optimizers treat a complex parameter as two real coordinates, consistent with the convention above.

### Why the state stays `complex64` while logits are FP32

The state is complex because the *dynamics* are complex — the recurrence is one complex multiply per step and the rotation cannot be expressed as a real scalar. Everything *after* the scan is real by design:

- the readout `y_t = Re(C_t h_t)` is a real linear functional of the 2N real coordinates (the `C` einsum in `ssd_complex_chunkwise` sums over `n` and takes `.real` at the end);
- a probability distribution over the 50,257-token vocabulary is real; complex logits have no meaning for cross-entropy;
- keeping `lm_head` real halves its memory (`50257 × 1024` complex64 would double the head's footprint) and keeps the logit path in FP32 as required by `AGENTS.md` ("Recurrent state stays in complex64; logits in FP32").

So `complex64` is confined to the SSD scan: `B_t`/`C_t` are complex only inside `ssd_complex_chunkwise`, the state tensors (`states`, starting from zeros of dtype `complex64`) never leave it, and `Y.real` hands a float32 tensor to the rest of the block. From there every subsequent op — MIMO, `out_proj`, residual, `norm_f`, `lm_head` in `models/transformer.py:Mamba3Transformer.forward` — is a real FP32 linear map, ending in real FP32 logits. The imaginary part of the scan output carries no probability mass — discarding it is the intended projection, not a lossy truncation.

## `torch` complex mechanics

Three operations carry the whole implementation:

- **`torch.complex(real, imag)`** — builds a `complex64` tensor from two equal-shaped `float32` tensors (real and imaginary parts). Used in `models/mamba_block.py:Mamba3Block._forward_impl` to assemble `B_t` and `C_t`, and in `models/ssd_triton.py` to reassemble `Y_diag`/`state` from the kernel's split float32 outputs. It requires `float32`/`float64` inputs — this is why the slices are `.float()`-cast before assembly (under BF16 autocast the `in_proj` output would be BF16).
- **`torch.view_as_real(z)`** — views a `complex64` tensor of shape `(…, N)` as `float32` of shape `(…, N, 2)`. In memory a complex64 tensor interleaves `[re₀, im₀, re₁, im₁, …]`, so the pair dimension has stride 1 while the element dimension has **stride 2** — the "stride-2 layout" named in `models/ssd_triton.py:_view_real_imag`. Any code that assumes a packed float32 buffer over this view reads every *other* float (i.e. real parts as imag and vice versa).
- **`torch.view_as_complex(real_pairs)`** — the inverse; it requires the pair dimension to have stride 1. Applied to a non-contiguous tensor it silently interleaves wrong elements — the "silent stride bug" `AGENTS.md` warns about ("`torch.view_as_complex` on raw real pairs can fail silently if the imaginary stride is wrong").

## Code walkthrough

### The block: `models/mamba_block.py:Mamba3Block._forward_impl`

The per-head complex scalar `A` is created in `__init__` and initialized to pure decay:

```python
self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
nn.init.constant_(self.A, -1.0)
```

`nn.init.constant_` fills with the scalar cast to `complex64`, i.e. `−1 + 0i`: `α = −1`, `β = 0`. At initialization every mode decays and nothing rotates; rotation must be learned. Note `A` is a **per-head scalar** (shape `(H,)` — 16 complex numbers in the whole model), shared across all `N=64` states of a head, unlike Mamba-2's per-state `A`. Selectivity per token/head comes from `dt`, not from `A`.

The forward path projects and slices. `in_proj` has width `H·(D + 4N + 1)` — one `D`-wide block for the token content, four `N`-wide blocks for the real/imaginary parts of `B` and `C`, and one slot per head for `dt`:

```python
proj = self.in_proj(h)
x_ssm = proj[..., :H * D].reshape(B, T, H, D).float()

B_real = proj[..., H * D:H * D + H * N].float()
B_imag = proj[..., H * D + H * N:H * D + 2 * H * N].float()
B_t = torch.complex(B_real, B_imag).reshape(B, T, H, N)

C_real = proj[..., H * D + 2 * H * N:H * D + 3 * H * N].float()
C_imag = proj[..., H * D + 3 * H * N:H * D + 4 * H * N].float()
C_t = torch.complex(C_real, C_imag).reshape(B, T, H, N)

dt = proj[..., -H:].float()
```

The slice boundaries, in `proj` columns: `x` occupies `[0, H·D)`, then four contiguous `H·N` blocks (`B_real`, `B_imag`, `C_real`, `C_imag`), then `dt` in the last `H` columns. This exact layout is the block's ABI — see `../concepts/block-and-stability.md` for the shape-by-shape dataflow. Three dtypes coexist: `x_ssm` stays `float32` (its imaginary part is 0 when promoted), `B_t`/`C_t` are `complex64`, `dt` is `float32`.

### The discretization: `models/ssd_complex.py:_discretise`

```python
def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.softplus(dt) * A)
```

`dt` is `(B, T, H)` real; `A` is `(H,)` complex64; the product broadcasts to `(B, T, H)` complex64 and the complex exponential yields the scale+rotation factor per token. The oracle `models/ssd_complex.py:ssd_naive_complex` applies this factor in a literal O(T) loop with a `complex64` state of shape `(B, H, N, D)`:

```python
A_bar = _discretise(dt, A)
s = torch.zeros(B_, H, N, D, dtype=torch.complex64, device=x.device)
for t in range(T):
    s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s             + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
    ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
```

Each step is one complex multiply of the whole state (`A_bar·s`, the scale+rotation) plus injection `B_t·x_t`; the readout sums `C_t·h_t` over the `N` states. This sequential scan is the ground truth the chunkwise algorithm must reproduce.

### The chunkwise pass: `models/ssd_complex.py:ssd_complex_chunkwise`

The production path replaces the O(T) loop with the chunked algorithm (derived in `../concepts/ssd-theory.md`); what matters here is that *every* recurrence factor is a complex exponential of `A_log = softplus(dt)·A`:

- intra-chunk coupling `L[l,s] = exp(A_cumsum[l] − A_cumsum[s])·1[l≥s]` and `decay_states = exp(A_cumsum[−1] − A_cumsum)`;
- inter-chunk `decay_chunk = exp(cd_seg)·tril` from the per-chunk cumsums;
- the output term `Y_off` weights propagated states by `exp(A_cumsum)`, and `Y = Y_diag + Y_off` is truncated with `Y = Y.real` before being sliced back to `T`.

All state tensors (`states`) are `complex64`; only the final `Y.real` (float32) escapes. When `ssd_dispatch='triton'`, the per-chunk `Y_diag` and `state` are produced by `models/ssd_triton.py:per_chunk_ssd_triton` and the inter-chunk propagation stays in PyTorch.

### The kernel-side split: `models/ssd_triton.py:_view_real_imag`

Triton has no complex type, so the host wrapper splits every complex64 input into two packed float32 buffers:

```python
pair = torch.view_as_real(z.contiguous())
return pair[..., 0].contiguous(), pair[..., 1].contiguous()
```

The `z.contiguous()` first guarantees the interleaved layout is materialized (a permuted complex tensor would have non-trivial strides), and the final `.contiguous()` on `pair[..., 0]`/`pair[..., 1]` copies the stride-2 slices into stride-1 buffers the kernel indexes linearly (the kernel's grid is one program per `(B, c, H)`, reading `Bc`/`Cc`/`Xc`/`A_log`/`decay_states` as flat pointers). The same split feeds the dtype guard (`complex64` required, else `TypeError`). The backward of `models/ssd_triton.py:_PerChunkSSDTriton` recomputes the per-chunk math with `models/ssd_triton.py:per_chunk_ssd_pytorch` seeded with the true `grad_outputs` — so the complex gradients described above are reproduced on the PyTorch path even when the forward ran on GPU. Full kernel mechanics: `../references/ssd-reference.md`.

## Pitfalls

1. **`N` must be even (AGENTS.md rule 4).** The parity argument ("`N=64` complex ≡ `N=128` real") is exact only when the state decomposes cleanly into real pairs; an odd `N` leaves the 2N-real-equivalent packing ambiguous. This is a repo rule, not an enforced assertion — `models/ssd_complex.py` has no even-`N` check, so a config with `state_dim=63` would silently run a model whose "packing" story is broken. Keep `state_dim=64`.
2. **Stride-2 view bugs.** `torch.view_as_real` output has stride 2 on the element dim; feeding it to anything assuming packed float32 (a Triton kernel, a reshape, `view_as_complex` on a non-contiguous pair) reads or interleaves the wrong floats *without error*. This is the "silent stride bug" of `AGENTS.md`. The `contiguous()` calls in `_view_real_imag` exist precisely to prevent it.
3. **`complex64` is 2× the element bandwidth of `float32`.** Every complex multiply is 4 real multiplies + 2 adds, and a complex64 tensor moves twice the bytes of a float32 tensor with the same shape. This is exactly why halving `N` matters: `N=64` complex state `(B,H,64,D)` occupies the same bytes as `N=128` real — the halving pays for the wider dtype. See `../concepts/block-and-stability.md` for the memory analysis.
4. **Discretization order.** `softplus` applies to real `dt`, then the product with complex `A` is exponentiated: `exp(softplus(dt)·A)`. `F.softplus` does not accept complex input, and `exp(softplus(dt·A))` would be a different (and wrong) recurrence. Don't "optimize" the order.
5. **`torch.complex` rejects BF16/float16.** Under BF16 autocast the `in_proj` output is BF16; the `.float()` casts in `Mamba3Block._forward_impl` exist so the complex assembly and the entire SSD run in FP32/complex64 — the "gating FP32" rule of `AGENTS.md`. Dropping the casts breaks under autocast.
6. **Only `.real` escapes the scan.** `Y.real` is a view into the complex result; the imaginary part is discarded by design (it carries no probability mass), but the truncation means the model cannot use `Im(C h)` directly — if a future variant wanted complex logits, the readout and the head would both change.

## Tests

`tests/test_ssd.py` machine-checks the claims of this doc:

- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` — the core equivalence: random complex `x` (dtype `complex64`), `A = torch.randn(H, dtype=torch.complex64) - 1.0` (negative real part → decay), random complex `B_t`/`C_t`, `dt = 0`, `chunk_size=4`; asserts `ssd_complex_chunkwise` output is `float32`, `ssd_naive_complex` output is `complex64`, shapes match, and `allclose(y_chunk, y_naive.real, atol=1e-4)`. Note what this proves: the chunkwise algorithm reproduces the O(T) complex scan *exactly on the real readout* — it verifies the math of this doc, **not** the parity claim (no real-state model is involved).
- `tests/test_ssd.py::test_chunkwise_handles_uneven_T` — `T=20`, random nonzero `dt` (so `softplus(dt)` varies and the rotation term is exercised); checks shape and finiteness.
- `tests/test_ssd.py::test_chunkwise_handles_T_equal_to_chunk` — `T = chunk_size = 4`; the single-chunk degenerate case.

One honest caveat: all three tests use `T` divisible by `chunk_size` (16/4, 20/4, 4/4), so the padding branch in `ssd_complex_chunkwise` (`(C − T%C) % C`) is not directly exercised by this file — the uneven-`T` path is covered only if a future test uses a non-multiple. The parity claim, again, is `[INFERENCE]` from the paper; this repo's tests establish internal consistency, not perplexity parity. The GPU path adds an end-to-end check (`tests/e2e_gpu_smoke.py`, CUDA + triton) and the reference docs `../references/ssd-reference.md` and `../references/ssd-reference.md` carry the full API contracts.

For the surrounding machinery: `../concepts/state-space-foundations.md` builds the real recurrence from first principles, `../concepts/ssd-theory.md` motivates why a scan can become a chunked matmul, and `../concepts/mimo.md` explains what happens to the real readout after it leaves the scan.
---

#The Chunkwise Complex SSD Algorithm

This doc derives the chunkwise complex state-space-duality (SSD) scan end to end — the sequence-mixing primitive at the heart of Mamba-3-Lite — and maps every einsum in `models/ssd_complex.py:ssd_complex_chunkwise` to its mathematics, einsum by einsum.

## The 60-second summary

After reading this doc you will understand: why the O(T) sequential scan is a research oracle rather than the production path; how splitting time into chunks of length $C$ turns the scan into "a matmul inside every chunk, a tiny sequential scan across chunks"; the exact derivation of the four einsums that implement it (`Y_diag`, the per-chunk states, the inter-chunk propagation, and `Y_off`), with a full tensor-shape table for each; a proof sketch that the chunkwise form computes exactly the same function as the naive scan; and a hand-computed 2×2 complex example in which both paths agree to the last digit.

## 1. Why: the sequential scan cannot use tensor cores

The reference implementation `models/ssd_complex.py:ssd_naive_complex` is the ground truth of this repo: a loop over $T$ timesteps that updates a `(B, H, N, D)` complex64 state one step at a time.

```python
for t in range(T):
    s = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) * s             + B_t[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-2)
    ys.append((C_t[:, t].unsqueeze(-1) * s).sum(dim=-2))
```

Every iteration is a handful of elementwise tensor ops on a state that is already in registers: multiply by a scalar per head, add one rank-1 (outer-product) update, contract against $C$. None of this is a matrix multiply, so none of it can run on the tensor cores of an NVIDIA GPU; the loop is also serial in $T$, so it underutilizes the machine the same way an RNN does. For a 2,048-token sequence, 28 layers, and 16 heads, that is 2,048 × 28 sequential micro-steps per batch — each one a memory-bound kernel launch (the scan's FLOPs are tiny; its latency is not). Nothing in this repo is wrong with the naive scan — it exists to be *correct* — but it is not the path that trains a 434M-parameter model. `[INFERENCE]` no throughput benchmark exists in this tree; the claim here is structural (GEMM vs elementwise loop), not a measured speedup number.

The chunkwise form in `models/ssd_complex.py:ssd_complex_chunkwise` is the production path: every layer of `models/mamba_block.py:Mamba3Block._ssd_with_dispatch` calls it (with `chunk_size=64`, `ssd_dispatch` from config), and inside it, all heavy work is batched matrix multiplication. This doc derives that function from the recurrence.

## 2. Intuition first

Think of the sequence as a stack of folders, each holding $C$ consecutive timesteps. Two rules make the algorithm:

1. **Inside a folder, everything is a matmul.** Within a chunk of $C$ positions, the output at position $l$ is a weighted sum over sources $s \le l$ of that chunk: each pair $(l, s)$ gets a decay factor, so the whole intra-chunk computation is "output-projection matrix × decay matrix × input-projection matrix" — three batched GEMMs.
2. **Across folders, you carry a state.** The only thing a chunk needs from the past is the complex state at its left edge. That state is a weighted sum of every earlier chunk's "contribution", with decay factors that telescope into products of per-chunk decays. So: compute one state contribution per chunk (a matmul), then run a tiny sequential scan over $\lceil T/C \rceil$ chunks.

The sequential scan is a special case with $C = 1$: no intra-chunk matmul (a 1×1 decay matrix), and the inter-chunk scan runs over $T$ steps. Chunking trades one long scalar loop for a short scalar loop plus big batched GEMMs — exactly the trade that makes tensor cores usable.

## 3. Notation and the recurrence

Let $T$ be the sequence length, $B$ the batch, $H$ heads, $D$ the head dimension, $N$ the (complex) state dimension, and $C$ the chunk size (64 in the repo config). Per timestep $t$:

$$h_t = \bar a_t \odot h_{t-1} + B_t \otimes x_t \in \mathbb{C}^{H \times D \times N}, \qquad y_t = \sum_{n=1}^{N} C_{t,n} \odot h_{t,n} \in \mathbb{C}^{H \times D},$$

where $x_t \in \mathbb{C}^{H \times D}$ is the token content, $B_t \in \mathbb{C}^{H \times N}$ and $C_t \in \mathbb{C}^{H \times N}$ are the input/output projections, $B_t \otimes x_t$ is the outer product into $H \times D \times N$, $\bar a_t = \exp(\mathrm{softplus}(dt_t) \cdot A) \in \mathbb{C}^H$ is the discretized per-head decay broadcast over the state (see `models/ssd_complex.py:_discretise` and the companion theory docs [01-ssm-foundations](../concepts/state-space-foundations.md), [03-complex-ssd](../concepts/ssd-theory.md)), and $A$ is the per-head complex scalar parameter. The naive scan keeps the state in $\mathbb{C}^{H \times N \times D}$; the chunkwise function uses the transposed layout $\mathbb{C}^{H \times D \times N}$ — for a zero initial state the two are equivalent (Section 7.5).

With zero initial state, unrolling the recurrence gives the closed form that everything below must reproduce:

$$y_t = \sum_{s \le t} C_t^{\top} \left(\prod_{u=s+1}^{t} \bar a_u\right) B_s x_s,$$

where $C_t^{\top}$ contracts the $N$ index (sum over $n$), and the empty product is 1.

## 4. Step-by-step derivation, mapped to the code

Every step below names the exact lines of `models/ssd_complex.py:ssd_complex_chunkwise` that implement it.

### 4.1 Step 1 — pad and reshape

Split time into $n_{\text{chunks}} = \lceil T / C \rceil$ chunks. If $T$ is not a multiple of $C$, pad on the right with `pad = (C - (T % C)) % C` zeros along the time axis, then reshape every tensor from `(B, T, …)` to `(B, n_chunks, C, …)` — a view, not a copy:

```python
pad = (C - (T % C)) % C
if pad > 0:
    x = F.pad(x, (0, 0, 0, 0, 0, pad))
    B_t = F.pad(B_t, (0, 0, 0, 0, 0, pad))
    C_t = F.pad(C_t, (0, 0, 0, 0, 0, pad))
    dt = F.pad(dt, (0, 0, 0, pad))
T_padded = T + pad
n_chunks = T_padded // C
...
def _chunk(t):
    return t.reshape(B_, n_chunks, C, *t.shape[2:])
```

The local `_chunk` helper is a pure reshape: because the padded time axis has size exactly $n_{\text{chunks}} \cdot C$, the row-major layout puts chunk $c$, position $l$ at index $c \cdot C + l$ — exactly the grouping we need. The two-step formula `(C - (T % C)) % C` is defensive: when $T \% C = 0$ the naive `C - T % C` would equal $C$, but the outer `% C` makes it 0, so the `F.pad` branch is skipped and `T_padded = T`.

After this step every sequence tensor carries indices $(c, l)$: chunk index $c \in [0, n_{\text{chunks}})$, within-chunk position $l \in [0, C)$, with global time $t = c \cdot C + l$.

### 4.2 Step 2 — the discretized decay and its cumsum

```python
A_log = F.softplus(dt) * A
...
Xc, Bc, Cc, Ac = _chunk(x).to(torch.complex64), _chunk(B_t), _chunk(C_t), _chunk(A_log)

A_cumsum = torch.cumsum(Ac, dim=2)
decay_states = torch.exp(A_cumsum[:, :, -1:, :] - A_cumsum)
```

`Ac` is the chunked per-position log-decay, shape `(B, n_chunks, C, H)`. Softplus is strictly positive, so for the repo's init ($\Re A = -1$) every $\bar a_t$ has magnitude $\exp(\mathrm{softplus}(dt_t) \cdot \Re A) \le \tfrac12$ — the state provably decays. (Why softplus rather than a raw exponent: the discretization theory is in [01-ssm-foundations](../concepts/state-space-foundations.md).)

`A_cumsum[c, l] = \sum_{u=0}^{l} A_log[c, u]` is the running log-decay *within* chunk $c$ (cumsum over `dim=2`, the chunk-position axis). The product of decays from source $s$ to target $l$ telescopes into one difference:

$$\prod_{u=s+1}^{l} \bar a_{c,u} \;=\; \exp\!\Big(A_{\text{cumsum}}[c,l] - A_{\text{cumsum}}[c,s]\Big).$$

`decay_states` is this formula specialized to $l = C-1$ (the *last* position of the chunk): `exp(A_cumsum[..., -1:, :] - A_cumsum)` keeps the `-1` dimension, so the result has shape `(B, n_chunks, C, H)` — the decay from each position $l$ to the end of its chunk. It is exactly what the per-chunk state needs in Step 5. Note the shape: `(B, n_chunks, C, H)`, i.e. `(b, c, l, h)`, the same axes as `Ac`; this is the tensor the `bclh` in the state einsum refers to.

### 4.3 Step 3 — the intra-chunk decay matrix $L$

The PyTorch branch builds the causal decay matrix of each chunk:

```python
Ac_perm = Ac.permute(0, 1, 3, 2).contiguous()
T_c = Ac_perm.size(-1)
Ac_cumsum = torch.cumsum(Ac_perm, dim=-1)
Ac_seg = Ac_cumsum.unsqueeze(-1) - Ac_cumsum.unsqueeze(-2)
mask = torch.tril(torch.ones(T_c, T_c, device=x.device, dtype=torch.bool))
L = torch.exp(Ac_seg) * mask
```

Reading this line by line: `Ac_perm` transposes `(b, c, l, h)` to `(b, c, h, l)` so the cumsum runs over the position axis; `Ac_cumsum.unsqueeze(-1) - Ac_cumsum.unsqueeze(-2)` forms the pairwise difference $A_{\text{cumsum}}[l] - A_{\text{cumsum}}[s]$ as a `(b, c, h, l, s)` tensor (this is exactly `bchls` — the `L` axis pair in the `Y_diag` einsum); the `bool` lower-triangular mask kills the $s > l$ entries. The result:

$$L[c, l, s] \;=\; \exp\!\Big(A_{\text{cumsum}}[c,l] - A_{\text{cumsum}}[c,s]\Big) \cdot \mathbf{1}[l \ge s].$$

**Real/imag of $\exp(\text{seg})$.** The segment $z = A_{\text{cumsum}}[c,l] - A_{\text{cumsum}}[c,s]$ is complex (each $A_{\text{log}}$ is a complex per-head scalar), so the decay factor is a rotation composed with a scale: writing $z = u + iv$,

$$\exp(z) \;=\; \exp(u)\big(\cos v + i \sin v\big),$$

with $u = \sum \mathrm{softplus}(dt)\Re A \le 0$ controlling decay magnitude and $v = \sum \mathrm{softplus}(dt)\Im A$ controlling phase rotation. This is the entire reason the state is complex: one parameter pair $(A, dt)$ encodes both how fast the memory fades and how it oscillates ([03-complex-ssd](../concepts/ssd-theory.md) derives the capacity argument). The code never splits real and imag — `torch.exp` on a complex tensor does it natively, and `L` stays complex64.

Each chunk's $L$ is a *structured* matrix: lower-triangular, with every entry determined by two numbers (the row and column cumsums). It is the chunk-level analogue of the causal mask in attention, and the duality that makes the whole thing a "masked linear attention" is developed in [02-state-space-duality](../concepts/ssd-theory.md).

### 4.4 Step 4 — $Y_{\text{diag}}$: the intra-chunk outputs

```python
Y_diag = torch.einsum("bclhn,bcshn,bchls,bcshp->bclhp", Cc, Bc, L, Xc)
```

**Shape table** (with $n := n_{\text{chunks}}$):

| letter | axis | `Cc` | `Bc` | `L` | `Xc` | output |
|---|---|---|---|---|---|---|
| `b` | batch | ✓ | ✓ | ✓ | ✓ | ✓ |
| `c` | chunk index | ✓ | ✓ | ✓ | ✓ | ✓ |
| `l` | target position in chunk | ✓ | – | ✓ (row) | – | ✓ |
| `s` | source position in chunk | – | ✓ | ✓ (col) | ✓ | – |
| `h` | head | ✓ | ✓ | ✓ | ✓ | ✓ |
| `n` | state index | ✓ | ✓ | – | – | – |
| `p` | head-dim index ($D$) | – | – | – | ✓ | ✓ |
| shape | | `(b,c,l,h,n)` | `(b,c,s,h,n)` | `(b,c,h,l,s)` | `(b,c,s,h,p)` | `(b,c,l,h,p)` |

**Plain-language reading:** for each chunk $c$, target position $l$, and head $h$, sum over sources $s$ (causal via $L$) and state indices $n$: take the output projection $C$ at $(c,l)$, the input projection $B$ at $(c,s)$, the decay from $s$ to $l$, and the token content at $(c,s)$, and contract. Algebraically it is the chunked version of the unrolled closed form with the carry-in state set to zero:

$$Y_{\text{diag}}[c,l,h,p] = \sum_{s \le l} \sum_{n} C[c,l,h,n]\, \exp\!\big(A_{\text{cumsum}}[c,l,h] - A_{\text{cumsum}}[c,s,h]\big)\, B[c,s,h,n]\, X[c,s,h,p].$$

The einsum contracts `s` and `n`, sums nothing else, and is a single batched GEMM chain over the batch axes `(b, c, h)`: the two `(l,s)`-indexed matrices ($L$ and the $B$·$X$ product) are multiplied with a triangular mask. This is the "everything inside a chunk is a matmul" half of the algorithm.

### 4.5 Step 5 — per-chunk states and the inter-chunk propagation

First, each chunk's *own* contribution to the state at the end of the chunk, assuming the chunk started from zero:

```python
states = torch.einsum("bclhn,bclh,bclhp->bchpn", Bc, decay_states, Xc)
```

**Shape table:**

| letter | axis | `Bc` | `decay_states` | `Xc` | output |
|---|---|---|---|---|---|
| `b` | batch | ✓ | ✓ | ✓ | ✓ |
| `c` | chunk index | ✓ | ✓ | ✓ | ✓ |
| `l` | position in chunk | ✓ | ✓ | ✓ | – |
| `h` | head | ✓ | ✓ | ✓ | ✓ |
| `n` | state index | ✓ | – | – | ✓ |
| `p` | head-dim index ($D$) | – | – | ✓ | ✓ |
| shape | | `(b,c,l,h,n)` | `(b,c,l,h)` | `(b,c,l,h,p)` | `(b,c,h,p,n)` |

**Plain-language reading:** sum over positions $l$: the input projection $B$ at $(c,l)$, the decay from $l$ to the *end* of chunk $c$ (`decay_states`, derived in Step 2), and the token content at $(c,l)$. The output drops the `l` axis and reorders to `(c, h, p, n)`:

$$S_c[h,p,n] = \sum_{l=0}^{C-1} \exp\!\big(A_{\text{cumsum}}[c,C-1,h] - A_{\text{cumsum}}[c,l,h]\big)\, B[c,l,h,n]\, X[c,l,h,p].$$

This is the state at the right edge of chunk $c$ given zero carry-in — the "state contribution" of chunk $c$. Now let $\Lambda_c[h] := A_{\text{cumsum}}[c, C-1, h]$ (the total log-decay across chunk $c$, the tensor called `chunk_decay` in the code) and $\Phi_c = \exp(\Lambda_c)$; the true state at the *end* of chunk $c$ follows the recurrence

$$H_c = \Phi_c \odot H_{c-1} + S_c,$$

with $H_{-1}$ the initial state. Unrolling over chunks, the state at the *start* of chunk $z$ — the carry-in state $G_z$ — is

$$G_z = \exp\!\Big(\sum_{u=0}^{z-1} \Lambda_u\Big) \odot H_{-1} \;+\; \sum_{c=0}^{z-1} \exp\!\Big(\sum_{u=c+1}^{z-1} \Lambda_u\Big) \odot S_c. \tag{★}$$

Both sums telescope through the cumulative per-chunk decay. Define $CD[k] := \sum_{u=0}^{k} \Lambda_u$ (with $CD[-1] := 0$); then $\sum_{u=0}^{z-1}\Lambda_u = CD[z-1]$ and $\sum_{u=c+1}^{z-1}\Lambda_u = CD[z-1] - CD[c]$. The code evaluates (★) in three moves — the cumulative decay, the strict-lower-triangular decay matrix, and the einsum:

```python
chunk_decay = A_cumsum[:, :, -1, :]
cd_perm = chunk_decay.permute(0, 2, 1).contiguous()
cd_cumsum = torch.cumsum(cd_perm, dim=-1)

---

## References

- [Mamba-3-Lite — SSD Foundations](state-space-foundations.md) — the real recurrence, ZOH discretization, and the S4 → Mamba-3 arc this doc builds on.
- [Mamba-3-Lite — MIMO Head Mixing](mimo.md) — what happens to the real readout after it leaves the scan.
- [Mamba-3-Lite — Block Anatomy and Numerical Stability](block-and-stability.md) — where the scan sits in the block and why every dtype choice exists.
- [Mamba-3-Lite — Config Reference](../references/config-reference.md) — `models/transformer.py:ModelConfig` and the annotated YAML (`chunk_size`, `state_dim`, `ssd_dispatch`).
- [Mamba-3-Lite — SSD Reference](../references/ssd-reference.md) — the API contracts of `models/ssd_complex.py:ssd_complex_chunkwise`, `models/ssd_complex.py:ssd_naive_complex`, and the Triton kernel.
- Mamba-2 / SSD — Dao & Gu, 2024 (arXiv:2405.21060); Mamba-3 — Dao & Gu, 2025 (arXiv:2603.15569).
