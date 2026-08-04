# State Space Duality: From Sequential Scan to Chunkwise Matrix Multiplication

This document proves the central structural result behind Mamba-2-style models — that a linear, diagonal state-space recurrence can be reorganized into block matrix multiplications whose intra-chunk term is an attention-like expression — and shows how that reorganization appears in `models/ssd_complex.py:ssd_complex_chunkwise`.

## 1. 60-second summary

After reading this document you will understand:

- **Why the sequential scan is slow**: it is a serial chain of $T$ tiny steps that underutilizes a GPU's Tensor Cores.
- **The SSD decomposition**: split the sequence into chunks of length $C$; outputs inside a chunk are computed by matrix multiplications against a structured causal "segment" matrix $L$, and only a cheap $T/C$-step recurrence remains for the state carried between chunks.
- **The attention link**: each chunk is a mini self-attention block with score matrix $C B^\top$, values $X$, and the decay $L[l,s]=\exp(A_{cs}[l]-A_{cs}[s])\cdot\mathbf{1}[l\ge s]$ in place of softmax scores — a *linear attention* with a data-dependent kernel.
- **The code mapping**: the chunked view is literally what `models/ssd_complex.py:ssd_complex_chunkwise` computes, and the equivalence is machine-checked by `tests/test_ssd.py::test_chunkwise_matches_naive_complex`.

## 2. Why it exists: the sequential scan is the bottleneck

A state space model (SSM) is the linear recurrence

$$h_t = \bar A_t\, h_{t-1} + B_t x_t, \qquad y_t = C_t h_t,$$

with $\bar A_t = \exp\big(\mathrm{softplus}(dt_t)\, A\big)$ (see `models/ssd_complex.py:_discretise` for the discretization; the foundations, including why $A$ is diagonal and why $dt$ is exponentiated through softplus, are developed in [docs/theory/01-ssm-foundations.md](docs/theory/01-ssm-foundations.md)). The reference implementation `models/ssd_complex.py:ssd_naive_complex` evaluates this literally:

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

which is exactly the unrolled recurrence. This is the **SSD decomposition**: the sequential scan is replaced by (a) per-chunk matrix products ($Y^{\mathrm{diag}}$, the per-chunk states $S_c$, the readout $Y^{\mathrm{off}}$) and (b) one short scan over $n = T/C$ chunks ($H_c$). A full einsum-by-einsum account of the implementation, including shapes and the padding contract, is in [docs/theory/04-chunkwise-algorithm.md](docs/theory/04-chunkwise-algorithm.md).

## 5. Complexity: what chunking buys

**Sequential scan (naive).** $T$ serial steps; per step and per $(b,h)$: $O(ND)$ flops (state scaling, outer product, readout) and $O(ND)$ complex numbers of state traffic; total $O(B\,H\,T\,N\,D)$ flops with a critical path of length $T$ and negligible arithmetic intensity. The scan is bandwidth-bound and latency-bound.

**Chunkwise.** Counting per $(b,h)$:

- Intra-chunk term $Y^{\mathrm{diag}}$: the score product is $C^2 N$ flops (a $(C\times N)\cdot(N\times C)$ multiply) and the masked application to $X$ is $C^2 D$ (a $(C\times C)\cdot(C\times D)$ multiply), per chunk. Over $n=T/C$ chunks: $O\big(T\,C\,(N{+}D)\big)$ — **linear in $T$**, with everything expressed as batched GEMMs.
- Per-chunk states $S_c$ and readout $Y^{\mathrm{off}}$: $O(TC\,ND) = O(TND)$ each, again einsum-shaped matmuls.
- Inter-chunk scan $H_c$: the decay matrix is $n\times n$, so $O(n^2 ND) = O\big((T/C)^2 ND\big)$ flops, with a serial dependency of length $n = T/C$ instead of $T$.

For the repo's dimensions ($T=2048$, $C=64$, $N=D=64$, $n=32$) the serial depth drops from 2048 to 32 steps, and the dominant work is dense, batched GEMMs of shape $(64\times 64)\cdot(64\times 64)$-style tiles — exactly what Tensor Cores execute. The flop count is *not* lower than the scan (chunking adds the $C^2$ and $n^2$ factors); the win is that the flops are now dense matmuls with high arithmetic intensity instead of a serial chain of tiny elementwise ops. The Triton path (`per_chunk_ssd_triton` in `models/ssd_triton.py`, behind `ssd_dispatch='triton'`) fuses the `L` construction, $Y^{\mathrm{diag}}$, and $S_c$ into one kernel, avoiding materializing $L$ in global memory; the inter-chunk scan stays in PyTorch.

**Why $C$ is a tunable knob.** Raising $C$: larger GEMM tiles (better Tensor-Core efficiency), fewer serial chunk steps, but the intra-chunk work grows as $C$ and, if $L$ is materialized, its storage grows as $C^2$ — $(B\cdot \tfrac{T}{C}\cdot H\cdot C^2)$ complex entries, i.e. $8\,B\,T\,H\,C$ bytes (for $B{=}16$, $T{=}2048$, $H{=}16$, $C{=}64$: $\approx 268$ MB in complex64). Lowering $C$: less memory, more serial steps, smaller tiles. $C=64$ matches $N=D=64$, giving square power-of-two tiles and a comfortable $T/C = 32$ chunk depth. All throughput figures here are `[INFERENCE]` — there is no `.benchmarks/` in the tree; the memory arithmetic above is derived from the shapes.

## 6. Code walkthrough: the chunked view in `models/ssd_complex.py:ssd_complex_chunkwise`

The function `models/ssd_complex.py:ssd_complex_chunkwise` is the theorem above, line by line. (The sibling doc [docs/theory/04-chunkwise-algorithm.md](docs/theory/04-chunkwise-algorithm.md) walks the full einsum chain; here we stay at the theorem level.)

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

`torch.exp(A_cumsum)` is $e^{A_{cs}[l]}$, the decay from the start of the chunk to position $l$; the einsum is $Y^{\mathrm{off}}_{c,l} = C_{c,l} e^{A_{cs}[c][l]} H_c$. The sum $Y^{\mathrm{diag}} + Y^{\mathrm{off}}$ is the full SSD decomposition. Only the real part is returned: the model's output projection is real-valued, and the imaginary part of the state is an internal representation (see [docs/theory/03-complex-ssd.md](docs/theory/03-complex-ssd.md) for why the state is complex at all). Finally the padding added in Step 1 is sliced away (`[:, :T]`).

## 7. Pitfalls

**Causality is only enforced by the mask.** Nothing in `torch.einsum` or `torch.cumsum` knows about causality; the only things making the computation causal are the `torch.tril` masks on $L$ and on `decay_chunk`. An upper-triangular or missing mask silently lets future positions leak into the present — and since the score matrix $C B^{\top}$ is not normalized, the leak is not bounded by any attention-style sum-to-one constraint. Note also that $L$'s diagonal is intentionally inclusive ($l\ge s$): the self-contribution $s=l$ has decay $e^0 = 1$ and *must* be included, matching the recurrence's own-input term. A strictly-lower-triangular mask (excluding $l=s$) would drop each position's own input.

**`cumsum` vs `segsum` conventions.** The decay from $s$ to $l$ is $e^{\Lambda(l)-\Lambda(s)}$ with an *inclusive* cumulative sum. If the cumsum is shifted (exclusive — summing positions strictly before the current one), every off-diagonal entry of $L$ picks up an extra factor $e^{A_{log}}$ and the chunkwise result silently disagrees with the naive scan. The code is consistent: `A_cumsum = cumsum(Ac, dim=2)` is inclusive, `chunk_decay = A_cumsum[:, :, -1, :]` is the inclusive chunk total, and the same inclusive convention feeds $L$, `decay_states`, and `decay_chunk`. Any change to one of the three must change all three.

**Numerical stability of `exp` of large-magnitude arguments.** $L$ is built as $\exp$ of differences of cumulative sums, not as a product of $T$ per-step decays; this is deliberate — products of many near-one factors accumulate error and underflow, while the log-space difference keeps the computation in a range where $\exp$ is well-conditioned. With the initial $A=-1$ and $dt\ge 0$, $A_{log}$ has a non-positive real part, so all differences $A_{cs}[l]-A_{cs}[j]$ for $l\ge j$ are $\le 0$ and $L\in(0,1]$ — no overflow, and underflow to 0 is the *intended* long-range forgetting. But `A` is learned and complex: if its real part drifts positive, `A_cumsum` grows, and `torch.exp(A_cumsum)` in the $Y^{\mathrm{off}}$ readout can overflow to `inf` (then `NaN` after the additions). This is the riskiest exponential in the function. The per-head constant-`A` parameterization and the complex64 state (fp32 accumulation, never BF16 inside the SSD path) are what keep this contained in practice; see [docs/theory/07-numerical-stability.md](docs/theory/07-numerical-stability.md).

**Inter-chunk decay index alignment (live discrepancy, flagged 2026-08-04).** The theorem requires the state entering chunk $c$ to be $H_c = \sum_{c'<c} e^{\mathrm{CD}[c-1]-\mathrm{CD}[c']}\, S_{c'}$. The implementation builds `decay_chunk[b,h,z,c] = exp(CD[z]-CD[c])` and contracts it against the shifted list `[initial_states, S_0, S_1, ...]`, which yields a net factor $e^{\mathrm{CD}[z]-\mathrm{CD}[c+1]} = e^{\sum_{c''=c+2}^{z}\mathrm{ct}[c'']}$ for source chunk $c$ — equal to the theorem's $e^{\sum_{c''=c+1}^{z-1}\mathrm{ct}[c'']}$ **only when all chunk totals are equal**. That is exactly the condition the existing test exercises (`dt=0` and a per-head constant `A` make `A_log = ln(2)A` constant, hence all chunk totals identical), so `tests/test_ssd.py::test_chunkwise_matches_naive_complex` passes at ~1e-6. With input-dependent `dt` (the normal training regime — `dt` comes from the input projection), the chunkwise function deviates from `models/ssd_complex.py:ssd_naive_complex`; a 16-step/4-chunk random probe measured an absolute max difference of ~0.23 against the oracle while the `dt=0` version of the same probe measured ~1e-6. The intra-chunk term and `decay_states` are correct; only the chunk-level propagation is affected. This is a code bug, not a documentation choice — it has been reported to the coordinator (with a proposed fix and a regression-test recipe using time-varying `dt`) and the walkthrough above describes the code as it currently stands.

**Complex vs real output and padding.** `ssd_naive_complex` returns complex64 outputs; `ssd_complex_chunkwise` returns `Y.real` in float32. Any equivalence check must compare against `y_naive.real` (as the test does). And when $T$ is not a multiple of $C$, the padded positions must be sliced away after reassembly — forgetting `[:, :T]` returns a tensor of length $T_{\text{padded}} \ne T$.

## 8. Tests

- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` — the machine proof of the SSD decomposition: with `B,T,H,D,N = 2,16,2,4,4` and `chunk_size=4`, it asserts `torch.allclose(y_chunk, y_naive.real, atol=1e-4)`, plus dtype (`float32` vs `complex64`) and shape checks. As analyzed in the pitfalls, this test runs in the constant-`A_log` regime; it verifies the decomposition but does not exercise the inter-chunk propagation under non-uniform chunk totals.
- `tests/test_ssd.py::test_chunkwise_handles_uneven_T` — verifies padding correctness: `T=20` with `chunk_size=4` (not a multiple) must return shape `(B, 20, H, D)` with finite values.
- `tests/test_ssd.py::test_chunkwise_handles_T_equal_to_chunk` — the boundary `T == C`, i.e. a single chunk with no inter-chunk term at all.

The broader suite (37 tests collected (32 passed / 5 GPU-skipped on CPU)) also covers the Triton dispatch against this same chunkwise reference; the chunkwise-vs-naive equivalence above is the contract both dispatch paths must satisfy.
