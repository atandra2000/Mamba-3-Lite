# The Chunkwise Complex SSD Algorithm

This doc derives the chunkwise complex state-space-duality (SSD) scan end to end — the sequence-mixing primitive at the heart of Mamba-3-Lite — and maps every einsum in `models/ssd_complex.py:ssd_complex_chunkwise` to its mathematics, einsum by einsum.

## 60-second summary

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

where $x_t \in \mathbb{C}^{H \times D}$ is the token content, $B_t \in \mathbb{C}^{H \times N}$ and $C_t \in \mathbb{C}^{H \times N}$ are the input/output projections, $B_t \otimes x_t$ is the outer product into $H \times D \times N$, $\bar a_t = \exp(\mathrm{softplus}(dt_t) \cdot A) \in \mathbb{C}^H$ is the discretized per-head decay broadcast over the state (see `models/ssd_complex.py:_discretise` and the companion theory docs [01-ssm-foundations](01-ssm-foundations.md), [03-complex-ssd](03-complex-ssd.md)), and $A$ is the per-head complex scalar parameter. The naive scan keeps the state in $\mathbb{C}^{H \times N \times D}$; the chunkwise function uses the transposed layout $\mathbb{C}^{H \times D \times N}$ — for a zero initial state the two are equivalent (Section 7.5).

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

`Ac` is the chunked per-position log-decay, shape `(B, n_chunks, C, H)`. Softplus is strictly positive, so for the repo's init ($\Re A = -1$) every $\bar a_t$ has magnitude $\exp(\mathrm{softplus}(dt_t) \cdot \Re A) \le \tfrac12$ — the state provably decays. (Why softplus rather than a raw exponent: the discretization theory is in [01-ssm-foundations](01-ssm-foundations.md).)

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

with $u = \sum \mathrm{softplus}(dt)\Re A \le 0$ controlling decay magnitude and $v = \sum \mathrm{softplus}(dt)\Im A$ controlling phase rotation. This is the entire reason the state is complex: one parameter pair $(A, dt)$ encodes both how fast the memory fades and how it oscillates ([03-complex-ssd](03-complex-ssd.md) derives the capacity argument). The code never splits real and imag — `torch.exp` on a complex tensor does it natively, and `L` stays complex64.

Each chunk's $L$ is a *structured* matrix: lower-triangular, with every entry determined by two numbers (the row and column cumsums). It is the chunk-level analogue of the causal mask in attention, and the duality that makes the whole thing a "masked linear attention" is developed in [02-state-space-duality](02-state-space-duality.md).

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
# cd_shift[z] = CD[z-1] with CD[-1] := 0: the total decay applied to the
# initial state as it travels through chunks 0..z-1.
cd_shift = torch.cat([torch.zeros_like(cd_cumsum[..., :1]), cd_cumsum[..., :-1]], dim=-1)
# M[z, c] = exp(CD[z-1] - CD[c]) · 1[z > c]: decay applied to chunk c's
# end-of-chunk state while it travels through chunks c+1..z-1.
cd_seg = cd_shift.unsqueeze(-1) - cd_cumsum.unsqueeze(-2)
decay_chunk = torch.exp(cd_seg) * torch.tril(
    torch.ones(n_chunks, n_chunks, device=x.device, dtype=torch.bool), diagonal=-1,
)

states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
if initial_states is None:
    initial_states = torch.zeros(B_, H, D, N, device=x.device, dtype=torch.complex64)
# The initial state predates chunk 0, so it decays through chunks 0..z-1.
init_decay = torch.exp(cd_shift).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
states = states + init_decay * initial_states.unsqueeze(1)
```

Decoding the construction: `cd_cumsum[b,h,z] = CD[z]` runs the cumsum over the chunk axis (after `cd_perm` transposes to `(B, H, n_chunks)`); `cd_shift` is the same cumsum shifted right by one slot with a zero prepended, so `cd_shift[z] = CD[z-1]` exactly as in (★). The pairwise difference `cd_seg[z, c] = cd_shift[z] - cd_cumsum[c] = CD[z-1] - CD[c]`, and the mask is `torch.tril(..., diagonal=-1)` — *strictly* lower triangular, so $\mathbf{1}[z > c]$ and the diagonal entries (which would be $\exp(CD[z-1]-CD[z]) = \exp(-\Lambda_z)$, an artifact of the shift) are excluded:

$$\text{decay\_chunk}[z, c] = \exp\!\big(CD[z-1] - CD[c]\big) \cdot \mathbf{1}[z > c].$$

**Shape table:**

| letter | axis | `decay_chunk` | `states` | output |
|---|---|---|---|---|
| `b` | batch | ✓ | ✓ | ✓ |
| `h` | head | ✓ | ✓ | ✓ |
| `z` | target chunk | ✓ | – | ✓ |
| `c` | source chunk | ✓ | ✓ | – |
| `p` | head-dim index | – | ✓ | ✓ |
| `n` | state index | – | ✓ | ✓ |
| shape | | `(b,h,z,c)` | `(b,c,h,p,n)` | `(b,z,h,p,n)` |

**Plain-language reading:** the carry-in state of chunk $z$ is the weighted sum of every earlier chunk's end-of-chunk contribution $S_c$ ($c < z$), each decayed by $\exp(CD[z-1] - CD[c])$ — the decay it suffers while travelling through chunks $c+1, \dots, z-1$. The einsum contracts `c` and outputs `(b, z, h, p, n)`, which is exactly the index order the next einsum reads as `bchpn`.

The initial state is added afterwards as a separate term, because it predates chunk 0 and therefore decays through chunks $0, \dots, z-1$ — a window one chunk longer than any $S_c$: `init_decay = exp(cd_shift)` reshaped to `(b, z, h, 1, 1)` times `initial_states` (shape `(B, H, D, N)`, broadcast over the `(p, n)` pair). With the default `initial_states = None` this term is exactly zero, but the contract accepts an explicit state, and the code handles it correctly. **Correctness note:** the decay window in (★) — and therefore in `decay_chunk` — must be $CD[z-1]-CD[c]$, not $CD[z]-CD[c]$. The latter would carry each contribution one chunk too far and only agree with the naive scan when all $\Lambda_c$ are equal (e.g. `dt = 0` everywhere). This exact off-by-one was a latent bug in the inter-chunk propagation and has been fixed in the current code; `tests/test_ssd.py::test_chunkwise_matches_naive_time_varying_dt` is the regression test that pins the general, time-varying-`dt` behavior (Section 8).

### 4.6 Step 6 — $Y_{\text{off}}$, the readout, and slicing back

```python
Y_off = torch.einsum("bclhn,bchpn,bclh->bclhp", Cc, states, torch.exp(A_cumsum))

Y = Y_diag + Y_off
Y = Y.real
return Y.reshape(B_, T_padded, H, D)[:, :T, :, :]
```

**Shape table:**

| letter | axis | `Cc` | `states` | `exp(A_cumsum)` | output |
|---|---|---|---|---|---|
| `b` | batch | ✓ | ✓ | ✓ | ✓ |
| `c` | chunk index | ✓ | ✓ | ✓ | ✓ |
| `l` | position in chunk | ✓ | – | ✓ | ✓ |
| `h` | head | ✓ | ✓ | ✓ | ✓ |
| `n` | state index | ✓ | ✓ | – | – |
| `p` | head-dim index | – | ✓ | – | ✓ |
| shape | | `(b,c,l,h,n)` | `(b,c,h,p,n)` | `(b,c,l,h)` | `(b,c,l,h,p)` |

**Plain-language reading:** for each chunk $c$, position $l$, and head $h$, sum over state indices $n$: the output projection $C$ at $(c,l)$, the carry-in state at the left edge of chunk $c$, and $\exp(A_{\text{cumsum}}[c,l])$ — the decay from the left edge of the chunk to position $l$ (which is $\prod_{u=0}^{l}\bar a_{c,u}$, the product the recurrence applies to any memory that enters the chunk). Algebraically,

$$Y_{\text{off}}[c,l,h,p] = \sum_n C[c,l,h,n]\, \exp\!\big(A_{\text{cumsum}}[c,l,h]\big)\, G_c[h,p,n].$$

`Y = Y_diag + Y_off` is then the full output: intra-chunk contributions plus everything inherited from earlier chunks. Two final steps deserve explanation:

- **`Y = Y.real`.** The scan is complex internally, but the layer's contract is a real residual branch: token embeddings are real, the block output must be real to be added to the residual stream, and `x_ssm` stays float32 ([06-block-anatomy](06-block-anatomy.md)). The imaginary part of the scan output is discarded by design — the complex state is an internal representation, and the model observes only its real projection. (The naive oracle returns complex64; the test asserts the chunkwise path returns float32.)
- **Slice back to `T`.** `reshape(B_, T_padded, H, D)` inverts the chunk reshape, and `[:, :T, :, :]` drops the padding inserted in Step 1. Without the slice, an uneven sequence would silently return $T_{\text{padded}}$ outputs.

## 5. Equivalence proof sketch

Why is the chunkwise function the *same function* as the naive scan? Both are instances of one algebraic fact. Define the transition of a length-$C$ block as the map that takes an incoming state $g$ and the chunk's tokens to (outgoing state, outputs). Because the recurrence is linear in the state, the block map is affine-linear in $g$, and composing two block maps is again a block map: this is the associative semiring of state transitions. The chunkwise algorithm factors the pointwise recurrence

$$h_{t} = \bar a_{t} \odot h_{t-1} + B_{t} \otimes x_{t}$$

into the *product* of chunk transitions $T_{n-1} \circ \cdots \circ T_{0}$, and associativity guarantees that grouping the factors differently cannot change the result:

- the naive scan evaluates the composition one $\bar a_t$-factor at a time;
- the chunkwise algorithm evaluates each chunk's internal product in closed form (the $L$ matrix, Steps 3–4), which is a valid *association* of the intra-chunk factors, and then composes the chunk products sequentially (Steps 5–6).

The only nontrivial step is that the intra-chunk closed form $Y_{\text{diag}}$ really is the chunk's internal product — and that is the telescoping identity $\prod_{u=s+1}^{l}\bar a_u = \exp(A_{\text{cumsum}}[l] - A_{\text{cumsum}}[s])$ from Step 2 — and that the inter-chunk factors in (★) telescope into $CD[z-1]-CD[c]$. Everything else is bookkeeping. The machine proofs are `tests/test_ssd.py::test_chunkwise_matches_naive_complex` (constant `dt`) and `tests/test_ssd.py::test_chunkwise_matches_naive_time_varying_dt` (random `dt`, the general case), both asserting `torch.allclose(y_chunk, y_naive.real, atol=1e-4)`.

Complexity. The naive scan needs $T$ sequential steps of $O(HND)$ elementwise work. The chunkwise form needs $O(T^2/C)$-scale *matmul* work inside chunks (a $C \times C$ triangular product per `(b, c, h)` tile) plus only $T/C$ sequential steps for the chunk-level scan — the sequential part shrinks by a factor $C$, and the parallel part is dense GEMM. With $C = 64$ the intra-chunk matmuls dominate, which is precisely what tensor cores and the Triton dispatch (`per_chunk_ssd_triton` host wrapper, [reference/03-ssd-triton](../reference/03-ssd-triton.md)) exploit.

## 6. Worked example: a 2×2 complex chunk

To see the einsums bite, take $B{=}1$, $H{=}1$, $D{=}1$, $N{=}1$, $T{=}4$, $C{=}2$ — two chunks of two positions. Use illustrative values chosen for hand-computability (not the repo's init): $A = -1 + i\frac{\pi}{\ln 2}$ and $dt = 0$, so $\bar a_t = \exp(\ln 2 \cdot A) = \exp(-\ln 2 + i\pi) = -\tfrac12$ at every position. Tokens $x = (1, 2, 3, 4)$; projections $b = (1,\ 1{+}i,\ 2,\ i)$ and $c = (1,\ -i,\ 1{+}i,\ 1)$.

**Naive path (ground truth).** $h_0 = b_0 x_0 = 1$, $y_0 = 1$. Then $h_1 = -\tfrac12 \cdot 1 + (1{+}i)\cdot 2 = \tfrac32 + 2i$, $y_1 = (-i)(\tfrac32 + 2i) = 2 - \tfrac32 i$. Next $h_2 = -\tfrac12(\tfrac32+2i) + 2\cdot 3 = \tfrac{21}{4} - i$, $y_2 = (1{+}i)(\tfrac{21}{4} - i) = \tfrac{25}{4} + \tfrac{17}{4}i$. Finally $h_3 = -\tfrac12(\tfrac{21}{4}-i) + i\cdot 4 = -\tfrac{21}{8} + \tfrac92 i$, $y_3 = -\tfrac{21}{8} + \tfrac92 i$. So

$$y = \Big(1,\ \ 2 - \tfrac32 i,\ \ \tfrac{25}{4} + \tfrac{17}{4}i,\ \ -\tfrac{21}{8} + \tfrac92 i\Big).$$

**Chunkwise path.** $A_{\text{log}} = -\ln 2 + i\pi$ per position, so $A_{\text{cumsum}}[c,0] = -\ln 2 + i\pi$, $A_{\text{cumsum}}[c,1] = -2\ln 2 + 2i\pi$.

- *Step 3, $L$:* $L[0,0]=L[1,1]=1$; $L[1,0] = \exp(A_{\text{cumsum}}[1] - A_{\text{cumsum}}[0]) = \exp(-\ln 2 + i\pi) = -\tfrac12$; $L[0,1] = 0$ (mask).
- *Step 4, $Y_{\text{diag}}$:* $Y_{\text{diag}}[0,0] = c_0\, L[0,0]\, b_0 x_0 = 1$ ✓. $Y_{\text{diag}}[0,1] = c_1\big(L[1,0] b_0 x_0 + L[1,1] b_1 x_1\big) = (-i)(-\tfrac12 + 2 + 2i) = 2 - \tfrac32 i$ ✓. Similarly $Y_{\text{diag}}[1,0] = 6 + 6i$, $Y_{\text{diag}}[1,1] = -3 + 4i$ (the zero-carry-in parts of $y_2, y_3$).
- *Step 5, per-chunk states:* $\text{decay\_states}[c,0] = \exp(A_{\text{cumsum}}[c,1] - A_{\text{cumsum}}[c,0]) = -\tfrac12$, $\text{decay\_states}[c,1] = 1$. So $S_0 = (-\tfrac12)\cdot 1 + 1\cdot (2+2i) = \tfrac32 + 2i$ (indeed $= h_1$), $S_1 = (-\tfrac12)\cdot 6 + 1\cdot 4i = -3 + 4i$.
- *Step 5, propagation:* $\Lambda_c = A_{\text{cumsum}}[c,1] = -2\ln 2 + 2i\pi$, so $\Phi_c = \exp(\Lambda_c) = \tfrac14(\cos 2\pi + i \sin 2\pi) = \tfrac14$. With zero initial state, (★) gives $G_0 = 0$ and $G_1 = \exp(CD[0])\cdot 0 + \exp(CD[0]-CD[0])\cdot S_0 = S_0 = \tfrac32 + 2i$ — the state entering chunk 1 (here $CD[0] = \Lambda_0$, and the strict mask keeps only the $c=0$ term).
- *Step 6, $Y_{\text{off}}$:* $\exp(A_{\text{cumsum}}[1,0]) = -\tfrac12$, $\exp(A_{\text{cumsum}}[1,1]) = \tfrac14$. So $Y_{\text{off}}[1,0] = c_2 \cdot (-\tfrac12)\cdot G_1 = (1{+}i)(-\tfrac12)(\tfrac32 + 2i) = \tfrac14 - \tfrac74 i$, and $Y_{\text{off}}[1,1] = c_3 \cdot \tfrac14 \cdot G_1 = \tfrac38 + \tfrac12 i$.
- *Sum:* $y_2 = (6+6i) + (\tfrac14 - \tfrac74 i) = \tfrac{25}{4} + \tfrac{17}{4}i$ ✓; $y_3 = (-3+4i) + (\tfrac38 + \tfrac12 i) = -\tfrac{21}{8} + \tfrac92 i$ ✓.

Both paths agree exactly. Note the rotation at work: the $i\pi$ phase makes $\exp(A_{\text{cumsum}}[1]-A_{\text{cumsum}}[0]) = -\tfrac12$ (a sign flip), and the full-chunk phase $2i\pi$ wraps around to $+1$, so $\Phi_c = \tfrac14 > 0$. These numbers are verified against the implementation: running them through `ssd_complex_chunkwise` and `ssd_naive_complex` reproduces $(1,\ 2,\ 6.25,\ -2.625)$ after `.real` to $10^{-7}$.

## 7. Pitfalls

1. **Uneven-$T$ padding must be sliced back.** The `[:, :T]` in the return is load-bearing: it removes the zeros appended in Step 1. Forgetting it yields `T_padded > T` outputs that silently corrupt the residual add. The `(C - T % C) % C` double-mod is deliberate — at $T \% C = 0$ it yields 0 and the `F.pad` branch is skipped entirely.
2. **`decay_states` shape vs `A_log` shape.** `decay_states` is `(B, n_chunks, C, H)` — the time axis of `A_log` has been reshaped to `(n_chunks, C)`. The `-1:` keepdim in `A_cumsum[:, :, -1:, :]` matters: dropping it would produce `(B, n_chunks, H)` and break the `bclh` broadcast in the state einsum. Contrast `chunk_decay = A_cumsum[:, :, -1, :]`, which deliberately *squeezes* to `(B, n_chunks, H)`.
3. **dtype promotion to complex64.** The chain is: `x` is promoted to complex64 inside `_chunk(x).to(torch.complex64)`; `B_t`, `C_t` are already complex64 (packed from real/imag pairs in `models/mamba_block.py:Mamba3Block._forward_impl`); `dt` is float32 but `A` is complex64, so `softplus(dt) * A` promotes to complex64 and `A_log` is complex. `Y.real` is float32 by construction — the function's return dtype is part of its contract (the test asserts it). If any tensor downstream of the einsums is real, the multiply implicitly promotes; silent promotions are what make dtype bugs cheap to introduce and hard to spot.
4. **Mask dtype and the two tril axes.** Both $L$ and `decay_chunk` are masked by *multiplication* with a `bool` tril tensor, not `masked_fill`. This is fine — `False` multiplies to an exact 0 — and avoids an extra float multiply. The subtle part is *which* axis pair each tril lives on and how strict it is: the intra-chunk `L` uses `torch.tril(ones(T_c, T_c))` (diagonal included, causality $l \ge s$), while the inter-chunk `decay_chunk` uses `torch.tril(ones(n_chunks, n_chunks), diagonal=-1)` — *strictly* below the diagonal, because the shift in `cd_shift` makes the diagonal entry $\exp(-\Lambda_z)$, which would be a spurious factor. Transposing or relaxing either mask silently breaks causality or reintroduces the off-by-one.
5. **The `initial_states` layout is `(B, H, D, N)`.** This matches the chunkwise state layout `(b, c, h, p, n)` (head-dim first), *not* the naive scan's internal `(B, H, N, D)`. With zeros — the default — the discrepancy is invisible; a nonzero `initial_states` compared against the naive oracle must be transposed first. Note also that the initial state is *not* passed through `decay_chunk`: it enters via its own `init_decay = exp(cd_shift)` term because it predates chunk 0 and decays one chunk longer than any $S_c$.
6. **Masking history: `dt = 0` hides propagation bugs.** Because all of the inter-chunk factors in (★) collapse to products of the same $\Lambda$ when the per-position decay is constant, an equivalence test with `dt = 0` cannot detect an off-by-one in the propagation window — the pre-fix code passed `test_chunkwise_matches_naive_complex` while being wrong for input-dependent `dt`. Any change to the propagation step must be validated with time-varying `dt` (as the regression test `tests/test_ssd.py::test_chunkwise_matches_naive_time_varying_dt` does), not just with the original test.

## 8. Tests

- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` — the constant-`dt` machine proof of Section 5: $B{=}2$, $T{=}16$, $H{=}2$, $D{=}4$, $N{=}4$, `chunk_size=4`, complex random $A, B_t, C_t, x$ and `dt = 0`; asserts `y_chunk` (float32) matches `y_naive.real` (complex64) to `atol=1e-4` and that the dtypes differ.
- `tests/test_ssd.py::test_chunkwise_matches_naive_time_varying_dt` — the general regression test: same shapes, but `dt = torch.randn(B, T, H)`, so the per-chunk decay totals differ and the inter-chunk propagation is exercised in full. This test fails on the pre-fix propagation (off-by-one decay window) and pins the corrected form of Step 5.
- `tests/test_ssd.py::test_chunkwise_handles_uneven_T` — $T{=}20$, `chunk_size=4`; asserts shape `(B, T, H, D)` and finiteness. Note that $20 \bmod 4 = 0$, so despite the name this test does *not* exercise the `F.pad` branch — the padding path is currently untested; the multi-chunk propagation here runs with random `dt` but is only checked for finiteness.
- `tests/test_ssd.py::test_chunkwise_handles_T_equal_to_chunk` — $T{=}4$, `chunk_size=4`: a single chunk, so `n_chunks = 1` and there is no inter-chunk step at all; asserts shape and finiteness.

For the API contract (signature, shapes, dispatch semantics) see [reference/02-ssd-complex](../reference/02-ssd-complex.md); the Triton dispatch that fuses `Y_diag` and the per-chunk state into one kernel is documented in [reference/03-ssd-triton](../reference/03-ssd-triton.md).

## Anchors cited

- `models/ssd_complex.py:ssd_complex_chunkwise`
- `models/ssd_complex.py:ssd_naive_complex`
- `models/ssd_complex.py:_discretise`
- `models/mamba_block.py:Mamba3Block._ssd_with_dispatch`
