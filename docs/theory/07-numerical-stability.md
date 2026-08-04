# Numerical Stability in Mamba-3-Lite: Why Every Dtype, Norm, and Precision Choice Exists

This doc explains the defense layers that keep a 28-block complex-valued recurrence trainable for 256,000 steps: where BF16, TF32, FP32, and complex64 each live in the pipeline, the math behind the NaN guard and checkpoint rollback, and why weight tying and the init scheme are stability features rather than conveniences.

## 60-second summary

After reading this doc you will understand: why the model runs *mixed* precision rather than any single dtype — BF16 activations inside the projection GEMMs, FP32 everywhere that feeds the recurrence, complex64 inside the SSD scan, and an FP32 loss; why BF16 beats FP16 here (range beats precision when the critical path is already FP32); why `torch.set_float32_matmul_precision("high")` (TF32) is safe; exactly which `.float()` casts in `models/mamba_block.py:Mamba3Block._forward_impl` enforce FP32 for the recurrence, and where logits actually end up (BF16 under autocast, with the *loss* promoted to FP32); the mechanism of the NaN guard in `training/pretrain.py:train_step` (check before backward, skip the step, wipe the accumulation window) and the five-consecutive rollback in `training/pretrain.py:Pretrainer.train`; why rollback beats gradient clipping for a recurrence whose decay factors are $\exp$ of trainable sums; and how weight tying (`models/transformer.py:Mamba3Transformer.__init__`) plus the `N(0, 0.02)` / eye / constant-`−1` init keep the residual stream and its gradients well conditioned.

## 1. Why it exists

An SSM layer is a recurrence: every output is a function of every earlier position through products of per-step transition factors, which multiplies numerical sensitivity beyond a feed-forward network's. A rounding error in one step's decay factor is compounded by every later step; one `inf` contaminates the state for the rest of the sequence; and the backward pass differentiates through $\exp$ chains whose intermediates can exceed any sane range. A NaN mid-training wastes hours of A100 time — and in a 256,000-step, 8-billion-token run, *when* a failure happens matters as much as *whether* it can.

This repo treats stability as a layered problem:

1. **Precision placement** — run each computation in the cheapest dtype that cannot corrupt it (BF16 GEMMs, FP32 gating, complex64 state, FP32 loss).
2. **Contractive dynamics by construction** — at init, every decay factor has magnitude ≤ 1, so the recurrence provably cannot amplify on its own.
3. **A NaN guard with rewind** — detect non-finite loss *before* backward, skip the poisoned update, and after five consecutive failures restore the last checkpoint.

This doc derives each layer's shape; [04-chunkwise-algorithm](04-chunkwise-algorithm.md) derives the chunkwise scan this stability math is about, [03-complex-ssd](03-complex-ssd.md) why the state is complex, and [02-state-space-duality](02-state-space-duality.md) the `L` matrix the decay analysis reuses.

## 2. Intuition: three defense layers

Think of a **precision budget** and a **failure budget**.

Precision: the only ops that tolerate low precision are the big projection GEMMs (`in_proj`, `out_proj`, `ffn_*`, `mimo`, `lm_head`). Their inputs are renormalized to $O(1)$ scale by RMSNorm, and their outputs are *deltas* added to a residual stream — a 0.4% rounding error in a delta is 0.4% of a small correction, not the state. Everything the recurrence depends on — the `dt` gate, the complex projections, the state — runs in FP32/complex64 ($2^{-24}$ rounding, not $2^{-8}$), and the residual stream is itself an FP32 accumulator: BF16 deltas are promoted on addition, so rounding never compounds across layers.

Failure: a divergence usually starts in the *weights* (a drift of $\Re A$ above zero, a `dt` spike), produces exponentially amplified forward activations, and only then surfaces as a non-finite loss. Clipping bounds the gradient norm but cannot repair the weights that caused the amplification; rewinding to the last checkpoint resets weights, optimizer moments, and scheduler to a known-good state. A five-step streak separates one transient NaN (skip it) from a genuine regime change (rewind).

## 3. The precision landscape

### 3.1 Where each dtype lives — verified flow

On CUDA the forward pass runs inside a BF16 autocast region produced by `training/pretrain.py:Pretrainer._amp_context`:

```python
def _amp_context(self):
    if torch.cuda.is_available():
        return autocast("cuda", dtype=self.amp_dtype)
    else:
        return autocast("cpu", enabled=False)
```

with `self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32` set in `Pretrainer.__init__` — BF16 on an A100 (`sm_80`); on CPU the region is disabled and everything runs FP32 (as the CPU test suite exercises).

Autocast casts *per op category*, not per graph. The verified dtype flow through a model structured exactly like `Mamba3Transformer` (probe run on torch 2.12 under `autocast("cpu", dtype=torch.bfloat16)`; CUDA BF16 autocast uses the same op categorization):

| Stage | Op | Result dtype |
|---|---|---|
| `self.embed(x)` | `nn.Embedding` | **float32** (gather; not in cast lists) |
| `self.norm1(x)` | `nn.RMSNorm` | **float32** (autocast promotes norms) |
| `self.in_proj(h)` | `nn.Linear` | **bfloat16** (matmul category) |
| `x_ssm = … .float()` | explicit cast | **float32** |
| `B_t, C_t = torch.complex(B_real, B_imag)` | cast + pack | **complex64** |
| `dt = proj[..., -H:].float()` | explicit cast | **float32** |
| SSD einsums (`ssd_complex_chunkwise`) | complex64 operands | **complex64** (autocast never casts complex) |
| `Y = Y.real` | view | **float32** |
| `self.mimo(y)` / `self.out_proj(y)` | `nn.Linear` | **bfloat16** |
| `x = residual + y` | elementwise add | **float32** (promotion) |
| `self.norm_f(x)` | `nn.RMSNorm` | **float32** |
| `logits = self.lm_head(h)` | `nn.Linear` | **bfloat16** |
| `F.cross_entropy(logits, …)` | `log_softmax` + `nll_loss` | **float32** (log_softmax promoted) |

Two facts are easy to get wrong:

- **The logits are BF16, not FP32.** The final `Linear` head is *not* outside autocast's scope — it is a matmul like any other, so its output is BF16. What is FP32 is the **loss**: autocast promotes `log_softmax` (hence `cross_entropy`), so the scalar the NaN guard inspects is always FP32. Since probabilities are never fed back into the recurrence, the BF16 rounding of logits is harmless.
- **The residual stream is an FP32 accumulator.** Each block's delta is BF16 but added to an FP32 residual with promotion per block — rounding never compounds across the 28 residual adds.

### 3.2 BF16 vs FP16: range vs precision

An IEEE binary format with $e$ exponent bits and $m$ explicit mantissa bits has unit roundoff $u = 2^{-(m+1)}$ and largest finite value $\approx 2^{2^{e-1}+1}$:

| Format | $e$ | $m$ | $u$ | max finite |
|---|---|---|---|---|
| FP16 | 5 | 10 | $2^{-11} \approx 4.9\times10^{-4}$ | $2^{16} \approx 6.6\times10^{4}$ |
| BF16 | 8 | 7 | $2^{-8} \approx 3.9\times10^{-3}$ | $2^{128} \approx 3.4\times10^{38}$ |
| FP32 | 8 | 23 | $2^{-24} \approx 6.0\times10^{-8}$ | $2^{128}$ |

FP16 has ~3 extra mantissa bits (8× finer relative precision) but its 5-bit exponent caps at 65,504. BF16 shares FP32's exponent field, so its range matches FP32's, paid for with 16 fewer mantissa bits.

Which is "safer" depends on what gets rounded. The precision-critical path never touches 16-bit arithmetic: the SSD runs in complex64 (two FP32 components) and `dt` is FP32. The 16-bit format only rounds projection GEMMs whose inputs are RMSNorm-normalized. What kills FP16 here is range, in three places:

1. **Loss scaling.** FP16 needs dynamic loss scaling (`GradScaler`) to keep underflowing gradients above $6\times10^{-8}$ — an extra feedback loop with its own silent failure mode. BF16 needs none, and the repo's discipline is explicitly "no GradScaler".
2. **Logits.** With 50,257 classes, logits reach tens in magnitude; FP16 values near 65,504 have ULP ≈ 32 — coarse quantization exactly where the softmax gradient is most sensitive.
3. **GEMM products.** Tensor cores accumulate in FP32 internally for both formats, but the products come from 16-bit-rounded inputs. BF16's $2^{-8}$ input rounding is acceptable *because the rounded quantity is a delta, not a state*.

Honest summary: BF16 trades 8× precision for ~10³⁰× range — correct here because the sensitive part of the graph never runs in 16 bits.

### 3.3 TF32: 19-bit mantissa truncation

In `Pretrainer.__init__`, on CUDA:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
```

TF32 is the Ampere-and-later tensor-core format: 8-bit exponent (FP32 range) but only 11 significant bits — 10 explicit plus the implicit one — versus FP32's 24. Its unit roundoff is $u_{\text{tf32}} = 2^{-11} \approx 4.9\times10^{-4}$; TF32 keeps the range and discards 13 mantissa bits, at roughly 8× the FP32 GEMM rate (vendor spec ratio; `[INFERENCE]` — no benchmark exists in this tree). `torch.set_float32_matmul_precision("high")` is torch's way of saying "FP32 matmuls may use TF32", equivalent to `allow_tf32` and additionally letting `torch.compile` pick TF32 kernels.

Why safe here? TF32 only truncates **real FP32 GEMMs**. The GEMMs that dominate run in BF16 under autocast (Section 3.1), and the SSD einsums operate on **complex64** operands: torch dispatches complex matmuls through cuBLAS's complex GEMM routines, which compute in FP32 and are not governed by the TF32 math-mode flag — the recurrence keeps full FP32 internal precision regardless of this setting. TF32 applies to the residual FP32 GEMMs (e.g. the CPU-fallback path where autocast is disabled). One caveat, an open verification item: the Triton dispatch `models/ssd_triton.py:per_chunk_ssd_triton` computes the per-chunk matmuls as *real* FP32 `tl.dot` calls (the complex multiply is expanded into four real GEMMs — `tl.dot(cc_re, bc_t_re) - tl.dot(cc_im, bc_t_im)` and friends), and Triton's default `input_precision` for FP32 dots on NVIDIA is TF32 unless overridden; the kernel does not set it, so the Triton path's internal GEMMs may run at TF32 precision. GPU-box verification item `[INFERENCE]` — this clone's CPU tests cannot exercise the kernel.

### 3.4 Where FP32 is enforced: the `.float()` casts

`models/mamba_block.py:Mamba3Block._forward_impl` enforces FP32 for everything that feeds the recurrence with four explicit casts:

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

(The plan outline's "gating sigmoid in FP32": this repo has no sigmoid — the gate is `softplus` — and these casts *are* the FP32 enforcement.) The casts are load-bearing:

1. **`torch.complex` requires FP32/FP64 inputs.** Under BF16 autocast `proj` is BF16; packing BF16 slices raises a type error.
2. **Autocast would otherwise round the recurrence's inputs.** `einsum` is in autocast's lower-precision set — empirically, FP32 einsum inputs under BF16 autocast produce BF16 output. With BF16 slices, the scan's products and exponentials would start from $2^{-8}$-rounded values, each decay factor carrying 0.4% relative error into a 64-deep cumsum. The casts guarantee FP32-precision inputs; the complex64 operands inside `models/ssd_complex.py:ssd_complex_chunkwise` are then never touched by autocast, so the whole scan runs in complex64 = FP32-pair arithmetic, and the state never leaves complex64 — only `Y.real` (FP32) escapes.

## 4. The decay bound: why the recurrence cannot amplify (at init)

The discretized transition is `models/ssd_complex.py:_discretise`:

```python
def _discretise(dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.softplus(dt) * A)
```

with `A` initialized to a constant real `−1.0` (verified: `constant_` on complex64 sets real = −1, imag = 0):

```python
self.A = nn.Parameter(torch.empty(self.n_heads, dtype=torch.complex64))
nn.init.constant_(self.A, -1.0)
```

Softplus is strictly positive — $\mathrm{softplus}(x)=\ln(1+e^x)\in(0,\infty)$ (empirically $4.5\times10^{-5}$ at $x=-10$, $0.693$ at $x=0$, $10.0$ at $x=10$). With $\Re A=-1$,

$$\ln|\bar a_t| = \mathrm{softplus}(dt_t)\cdot\Re A = -\mathrm{softplus}(dt_t) \le 0,$$

so $|\bar a_t| = e^{-\mathrm{softplus}(dt_t)} \in (0,1]$ — the recurrence is **non-expansive**: no step can amplify the state (strictly contractive except as $dt\to-\infty$). Consequences: the intra-chunk decay $L[l,s]=\exp(A_{\text{cs}}[l]-A_{\text{cs}}[s])\cdot\mathbf{1}[l\ge s]$ ([02-state-space-duality](02-state-space-duality.md)) and the inter-chunk propagation ([04-chunkwise-algorithm](04-chunkwise-algorithm.md)) — with $CD[k]:=\sum_{u=0}^{k}\Lambda_u$ ($\Lambda_u$ the chunk-$u$ total log-decay) and $CD[-1]:=0$, the fixed implementation carries chunk $c$'s end-of-chunk state into chunk $z$ with

$$\text{decay\_chunk}[z,c] = \exp\big(CD[z-1]-CD[c]\big)\cdot\mathbf{1}[z>c],$$

and the initial state with $\exp(CD[z-1])$ — both have non-positive real parts at init, so every decay factor is a contraction too. The entire forward pass is then a composition of contractions; blow-up must come from outside the decay path — from the $B_t\otimes x_t$ injection or from *weight drift*.

**The blow-up mechanism (why the guard exists).** $\Re A$ is trainable; nothing pins it below zero. If AdamW drives $\Re A>0$, the per-position amplification is $g = e^{\mathrm{softplus}(dt)\cdot\Re A} > 1$, and over $T=2048$ tokens the last position's memory scale is $g^{T}$: $\Re A=0.02$, $\mathrm{softplus}(dt)=1$ gives $g^{2048}=e^{40.96}\approx 6\times10^{17}$; $\Re A=0.05$ gives $e^{102}\approx 10^{44}$ — instant overflow of complex64 (max $\approx 3.4\times10^{38}$) to `inf`, propagating through every einsum to a NaN loss. The decay bound guarantees stability *conditionally* — at init and while weights stay contractive — and the NaN guard enforces the region constraint.

## 5. The NaN guard: detection, skip, rewind

### 5.1 Detection in `train_step` — before backward

`training/pretrain.py:train_step` (module-level, so tests call it directly) computes the loss inside the autocast region, then checks finiteness *before* `loss.backward()`:

```python
with amp_context:
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100,
    )
    ce_loss_val = float(loss.item())
    loss = loss / config.gradient_accumulation_steps

if config.nan_guard and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
    log(f"[nan-guard] NaN/Inf at micro_step={micro_step}, opt_steps={opt_steps}. Skipping backward.")
    optimizer.zero_grad(set_to_none=True)
    return None, opt_steps

loss.backward()
if is_opt_step:
    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    opt_steps += 1
```

Ordering is the design: the check runs **before** `backward()`, so a poisoned forward can never write NaN gradients into parameter buffers. The check reads the FP32 scalar loss (Section 3.1), so it is reliable and nearly free. It is also *stronger* than skipping one step: `zero_grad(set_to_none=True)` wipes the partial gradients accumulated from earlier micro-steps of the current window — a NaN anywhere in a window discards the window's partial credit, never mixing a clean loss's gradients with a poisoned one's. The returned `None` metric signals the skip to the caller.

### 5.2 The streak and rollback in `Pretrainer.train`

`training/pretrain.py:Pretrainer.train` interprets `None` with a consecutive counter:

```python
metrics = self.train_step(tokens, targets, global_step)
if metrics is None:
    nan_guard_streak += 1
    if nan_guard_streak >= self.config.nan_guard_max_consecutive:
        latest = self._find_latest_checkpoint()
        if latest is not None:
            self._log(f"[nan-guard] {nan_guard_streak} consecutive NaN/Inf — restoring checkpoint step {latest}.")
            global_step = self.load_checkpoint(latest)
        else:
            self._log("[nan-guard] No checkpoint to restore from. Aborting.")
            raise RuntimeError("NaN/Inf with no checkpoint to restore from")
        nan_guard_streak = 0
    continue
nan_guard_streak = 0
```

`nan_guard_max_consecutive = 5` (config default, `training/pretrain.py:TrainingConfig`). One non-finite loss just skips; five in a row declares a regime change and `load_checkpoint(latest)` rewinds the **whole training state** — weights, optimizer moments, scheduler position, `opt_steps` — to the most recent complete checkpoint via `utils/checkpoint.py:CheckpointManager`. `CheckpointManager.latest_step()` returns the highest step whose three files (`model_step_N.safetensors`, `optim_step_N.pt`, `meta_step_N.json`) all exist (`_checkpoint_complete`), so the rewind target is never a torn save. After a rewind, `global_step` resets to the checkpoint's step, the streak clears, and training continues; the triggering batch is consumed, not replayed.

Checkpoint cadence is `save_every = 4000` micro-steps (`save_interval` in `configs/pretrain_a100_400m.yaml`), and step 0 is never saved (`global_step % self.config.save_every == 0 and global_step > 0`), so the rollback discards up to 4,000 micro-steps and aborts (RuntimeError) if no checkpoint exists yet — the guard is only as good as its most recent save. The interval is a stability knob, not bookkeeping (see [guides/02-training-runbook](../guides/02-training-runbook.md) for operational semantics).

### 5.3 Why rollback beats clipping — honest rationale

Gradient clipping (`nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)`, `max_grad_norm = 1.0`, on optimizer steps) is present and useful — it bounds the *update* norm against large-but-finite gradients. But it cannot rescue a diverged run:

1. **NaN is not clip-able.** Once the forward produces `inf` (an amplified complex state overflowing $3.4\times10^{38}$), gradients are NaN, and $\text{clip}(\text{NaN})=\text{NaN}$. Clipping only exists between finite values.
2. **The damage is in the weights, not the step.** The Section 4 mechanism is a *parameter regime* ($\Re A>0$). The gradient there points along the amplifying direction, and AdamW's per-parameter normalization means even a fully clipped update keeps moving weights deeper into the bad regime. Clipping fixes step size, not trajectory.
3. **Loss can be finite while gradients are garbage.** The backward differentiates products of $\exp$ factors — `decay_chunk`, `L`, `exp(A_cumsum)` — and $\frac{d}{d\theta}\exp(f) = \exp(f)\,f'$ multiplies already-large exponentials by sums of softplus derivatives. In the amplifying regime these intermediates can be astronomical or NaN while the forward loss still rounds finite (one batch before overflow). A clipped step then silently corrupts weights from a "healthy" reading — exactly the silent failure the rollback targets.

Honest trade-off: rollback throws away up to 4,000 micro-steps and is a coarse instrument; the streak threshold keeps it coarse-on-purpose — one transient NaN (corrupted batch, data glitch) should not discard progress; five in a row signals the optimizer has left the contractive region.

## 6. Weight tying: one parameter, two roles

`models/transformer.py:Mamba3Transformer.__init__`:

```python
self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
...
self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

if cfg.weight_tying:
    self.lm_head.weight = self.embed.weight

self.apply(self._init_weights)
```

The assignment makes `lm_head.weight` *the same `Parameter` object* as `embed.weight` — identical `data_ptr`, not a copy. Consequences:

- **Parameter savings.** The matrix is $50257 \times 1024 = 51{,}463{,}168$ parameters (~51.5M). Untied, the model would have $433{,}662{,}400 + 51{,}463{,}168 = 485{,}125{,}568$; tying saves $\approx 10.6\%$ of parameters, plus $51{,}463{,}168 \times 2 \times 4\text{B} \approx 412\text{ MB}$ of AdamW moments (two FP32 per parameter).
- **Gradients accumulate automatically.** Both paths write through the same tensor, so `backward()` accumulates the embedding-path and head-path contributions into one shared `.grad` — no manual summing exists or is needed (verified: the tied parameter's `.grad` receives both). `Pretrainer.__init__` also deduplicates `model.parameters()` by `id()` when building the decay/no-decay groups, so AdamW holds one state entry.
- **A checkpoint nuance.** `CheckpointManager.save` walks the state dict tracking `data_ptr`:

```python
for k, v in state.items():
    ptr = v.data_ptr()
    if ptr in seen_ptrs:
        deduped[k] = v.contiguous().clone()
    else:
        seen_ptrs.add(ptr)
        deduped[k] = v.contiguous()
```

safetensors refuses to serialize shared storage (verified: `RuntimeError: Some tensors share memory…`), so the second occurrence is **cloned** — the file stores both `embed.weight` and `lm_head.weight` as independent full-size tensors, so the checkpoint is *not* smaller; the dedup exists to keep the file valid and self-contained. The tie survives a round-trip because `load_state_dict` copies values *in place* into the existing (still shared) parameter, so the two names alias the same storage again after `load_checkpoint`. See [reference/09-checkpoint](../reference/09-checkpoint.md) for the file format.

## 7. Init scheme: small normal, identity mixer, constant decay

`models/transformer.py:Mamba3Transformer._init_weights`, applied via `self.apply(...)` over the whole tree:

```python
def _init_weights(self, module):
    if getattr(module, "_identity_init", False):
        return
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
```

`init_std = 0.02` (`models/transformer.py:ModelConfig`). Three pieces:

**Why $N(0, 0.02)$ for a residual network.** A Linear with fan-in $d$ acting on unit-variance input gives output variance $\sigma^2 d$; with $\sigma=0.02$, $d=1024$:

$$\text{std}(\text{block delta}) = \sigma\sqrt{d} = 0.02 \times 32 = 0.64.$$

The variance-preserving scale is $1/\sqrt{d}\approx 0.031$; 0.02 is about two-thirds of it, so each block's contribution is *smaller* than the identity path it joins: unrolling the residual stream with roughly independent deltas gives $\mathrm{Var}(x_L) \approx 1 + L\cdot 0.64^2$, i.e. std ≈ 3.5 at $L=28$ — bounded. The stronger effect is backward: the residual Jacobian is $I + \sum\prod\partial\Delta/\partial x$, and small deltas keep it near-identity, so gradients neither vanish nor explode across 28 blocks. Pre-norm RMSNorm reinforces this by renormalizing each block input to unit RMS — which is also why BF16 rounding always operates on $O(1)$-scaled activations.

**Why eye for MIMO.** `models/mimo.py:MIMO` initializes the mixer to identity and flags it:

```python
self.mix = nn.Linear(n_heads * head_dim, n_heads * head_dim, bias=False)
nn.init.eye_(self.mix.weight)
# Mamba3Transformer._init_weights skips Linears flagged this way.
self.mix._identity_init = True
```

`_identity_init` is the escape hatch: the generic Linear branch would otherwise overwrite the eye with $N(0, 0.02)$ (that is the exact dead-code bug this flag prevents — see [05-mimo-mixing](05-mimo-mixing.md)). At init the block behaves like the classical SISO SSM (head $h$'s scan output feeds only head $h$'s projection), the mixing Jacobian is the identity — no scaling distortion of gradients — and cross-head communication grows as training moves `mix` off identity.

**Why a constant $−1$ complex $A$.** This anchors Section 4's contractive dynamics: a real −1 (verified imag = 0) puts every decay factor in $(0,1]$ at init, so the recurrence starts provably non-amplifying, while the imaginary part of $A$ — which encodes rotation, see [03-complex-ssd](03-complex-ssd.md) — is free to train away from zero. `A` is a `Parameter` of the block, not a submodule, so `_init_weights` never touches it; only AdamW moves it.

## 8. The loss path: FP32 cross-entropy, scaled accumulation

```python
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100,
)
ce_loss_val = float(loss.item())
loss = loss / config.gradient_accumulation_steps
```

- `logits.reshape(-1, 50257)` flattens `(B, T, V)`; `targets.reshape(-1)` flattens `(B, T)`; cross-entropy takes the mean over non-ignored positions.
- `ignore_index=-100` is the conventional mask sentinel but **inert in this pipeline**: `PretrainDataset` windows never produce a −100 target (pad id is 50,256; targets are always real token ids).
- `ce_loss_val` is recorded *before* the division (the logged loss is the per-micro-batch mean); the *backpropagated* loss is divided by `gradient_accumulation_steps`. With accumulation 2, two micro-batches of $\tfrac{1}{2}L_i$ sum to $\tfrac{1}{2}(L_1+L_2)$ — the mean over the 32 × 2048-token effective batch, whose gradients AdamW consumes after clipping.

## 9. Pitfalls

1. **NaN checks must run before `backward()`.** The guard's position in `train_step` is the point: after `backward()`, NaN gradients are already in the parameter buffers even if you skip `optimizer.step()`. Any edit must keep the check between the loss and the backward call.
2. **A complex NaN can hide in `.real`.** `torch.isnan` on complex64 checks *both* components (verified: `isnan(1 + NaN·i)` is `True`), so a direct complex check is safe. But the guard only sees the real scalar loss: a NaN in an imaginary component decoupled by the readout (e.g. `C_imag = 0` kills the imag contribution to `Y.real`) leaves the loss finite and the guard silent — it surfaces later as corrupt gradients and trips the guard next step. Debugging a state-side NaN: check `torch.isnan(state)`, never `torch.isnan(state.real)`.
3. **`torch.compile` changes numerics.** `Pretrainer.__init__` wraps with `torch.compile(..., mode=compile_mode)` (`max-autotune` default). Fusion and reassociation change rounding — last-bit differences, not semantic ones. The guard still works (it reads the scalar loss, outside the graph), but a *compiled* CUDA crash (illegal memory access) raises rather than producing NaN — it will not be caught. `--no-compile` is the debug escape.
4. **Rollback needs a recent checkpoint.** The rewind target is the latest *complete* checkpoint (`CheckpointManager.latest_step()` requires all three files), saved every 4,000 micro-steps, and step 0 is never saved. A divergence inside the first 4,000 steps aborts: `RuntimeError("NaN/Inf with no checkpoint to restore from")`. For early crash-tolerance, lower `save_interval`.
5. **The TF32 flag has narrow scope.** It affects real FP32 GEMMs; the complex64 einsums are dispatched through cuBLAS complex GEMMs and keep FP32 arithmetic. The one place TF32-style truncation could reach the recurrence is the Triton kernel's internal real `tl.dot`s, whose precision is Triton's default (not torch's flag) — verify `input_precision` on the GPU box `[INFERENCE]`.
6. **The decay bound is conditional.** $|\bar a_t|\le 1$ holds at init and while $\Re A \le 0$; `A` is unconstrained, so drifted $\Re A>0$ turns the recurrence into an amplifier — the failure the guard exists for. Also, $\exp(-\mathrm{softplus}(dt))$ for very negative `dt` rounds to exactly 1.0 in FP32: no amplification, but no decay either — memory then persists the full sequence length.
7. **Logits are BF16 under autocast.** Don't "fix" the head by casting logits to FP32 manually — the loss path is already FP32 and a `.float()` inside the region is a no-op promotion. If you ever call `softmax` on logits *outside* autocast for sampling, cast to FP32 first.

## 10. Tests

- `tests/test_train_step.py::test_train_step_on_tiny_model` — the end-to-end guard contract on a CPU-only 2-block tiny model: runs `training/pretrain.py:train_step` with `nan_guard=True`, asserts it returns a metric (the guard did **not** trip), `out["loss"]` is finite, and a step changed parameters. This would catch a guard placed after `backward()` (parameters still change, but the check is meaningless) or one tripping spuriously on healthy runs.
- `tests/test_ssd.py::test_chunkwise_matches_naive_complex` / `test_chunkwise_matches_naive_time_varying_dt` — pin the FP32/complex64 recurrence math this stability analysis rests on (chunkwise ≡ naive scan to `atol=1e-4`, float32-vs-complex64 dtypes asserted).
- `tests/test_ssd.py::test_chunkwise_handles_uneven_T` / `test_chunkwise_handles_T_equal_to_chunk` — shape and **finiteness** checks for the SSD edge cases: the cheapest regression net for the exp-chain paths.

## Anchors cited

- `training/pretrain.py:Pretrainer.__init__`
- `training/pretrain.py:Pretrainer._amp_context`
- `training/pretrain.py:train_step`
- `training/pretrain.py:Pretrainer.train`
- `training/pretrain.py:TrainingConfig`
- `models/transformer.py:Mamba3Transformer.__init__`
- `models/transformer.py:Mamba3Transformer._init_weights`
- `models/transformer.py:ModelConfig`
- `models/mamba_block.py:Mamba3Block._forward_impl`
- `models/ssd_complex.py:ssd_complex_chunkwise`
- `models/ssd_complex.py:_discretise`
- `models/mimo.py:MIMO`
- `models/ssd_triton.py:per_chunk_ssd_triton`
- `utils/checkpoint.py:CheckpointManager`
