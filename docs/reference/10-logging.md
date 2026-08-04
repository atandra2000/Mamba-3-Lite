# TrainingLogger — windowed metrics, throughput, and optional WandB

Reference for `utils/logging.py:TrainingLogger`: how the training loop turns raw per-step losses into console summaries and optional WandB records.

## 60-second summary

`TrainingLogger` is the step-driven logger for pre-training: every `log_every` steps it prints one line with the step number, average loss, perplexity (exp of the average loss), learning rate, and tokens-per-second, and optionally mirrors those values to WandB under `train/…` keys. The throughput number is `tps = log_every × seq_len × batch_size / elapsed`, where `elapsed` is the wall time between two logged windows. WandB is enabled purely by environment: setting `WANDB_PROJECT` triggers `wandb.init` at construction (`WANDB_RUN_NAME` sets the run name), and a missing `wandb` package is tolerated with a printed warning. `Pretrainer` builds one logger from `TrainingConfig` fields and calls `TrainingLogger.log` from inside `training/pretrain.py:Pretrainer.train` on steps divisible by `log_every`.

## Why it exists

Pre-training runs for up to `max_steps = 256,000` steps ([R7 `07-pretrain-cli.md`](07-pretrain-cli.md)); a line per step would drown a terminal, and per-step WandB points would bury a dashboard. Nothing else in the repo records training progress — checkpoints only capture state every `save_every = 4000` steps — so the logger is the only place a human (or an experiment tracker) sees loss, perplexity, and throughput over time. Its jobs: a greppable **console heartbeat** once per window, **perplexity** from a window average (caveat in [Pitfalls](#pitfalls)), and **throughput telemetry** — `tps` catches stalls and regressions in effective tokens/sec, the number that matters for scaling ([T8 `08-scaling-efficiency.md`](../theory/08-scaling-efficiency.md)).

## Intuition

Think of the logger as a **windowed accumulator with a clock**. Every call appends the loss to a Python list (`_loss_window`). When the incoming step is a multiple of `log_every`, the logger averages the window, divides the window's token count by the wall time since the previous flush (`elapsed`) to get `tps`, prints one line, optionally records one WandB entry keyed by the global step, then empties the window and restarts the clock. So each printed line summarizes exactly one window, and consecutive lines are independent — state is fully reset at every flush.

**Verified nuance about the "window":** the class docstring promises a "rolling-window summary every `log_every` steps", which implies the caller invokes `log` every step and the logger buffers. The actual integration does the opposite: `Pretrainer.train` only calls `TrainingLogger.log` on steps where `global_step % log_every == 0`, and `log` itself flushes on the same condition. So **every call flushes immediately and the window holds exactly one loss** in the shipped training loop; the averaging machinery only engages if a caller invokes `log` every step. The math below covers the general (windowed) case, with the integration-specific simplification called out.

## Math

**Throughput.** In the integration each training step consumes one DataLoader batch of `batch_size` sequences of `seq_len = max_seq_len` tokens (see [R8 `08-dataset.md`](08-dataset.md) for how windows are packed). Between two flushes there are exactly `log_every` such steps, so the tokens attributable to a window are

$$N_{\text{win}} = \text{log\_every} \times \text{seq\_len} \times \text{batch\_size}.$$

With $T_{\text{win}}$ the wall time between flushes (computed as `elapsed = max(time.time() - self._step_start, 1e-6)`), the logged throughput is

$$\text{tps} = \frac{N_{\text{win}}}{T_{\text{win}}} = \frac{\text{log\_every} \times \text{seq\_len} \times \text{batch\_size}}{\text{elapsed}}.$$

With the defaults (`log_every=50`, `seq_len=2048`, `batch_size=16`, all from `training/pretrain.py:TrainingConfig`), one window covers $50 \times 2048 \times 16 = 1{,}638{,}400$ tokens. The `1e-6` floor on `elapsed` only matters for degenerate zero-duration windows; it makes `tps` finite rather than a `ZeroDivisionError`.

**Perplexity.** The window loss is the arithmetic mean $\bar{l} = \frac{1}{W}\sum_{i=1}^{W} l_i$ over the $W$ losses in the window, and perplexity is

$$\text{ppl} = \exp\!\left(\bar{l}\right).$$

Because $\exp$ is convex, Jensen's inequality gives

$$\frac{1}{W}\sum_{i=1}^{W} \exp(l_i) \;\ge\; \exp\!\left(\frac{1}{W}\sum_{i=1}^{W} l_i\right),$$

so the logged ppl is **not** the mean of the per-step perplexities — it is at or below it, with equality only when all losses in the window are identical. Concretely: `ppl = torch.tensor(avg_loss).exp().item()` — exp is applied *after* averaging, never per step. In the current integration $W = 1$, so ppl degenerates to the single-step value $\exp(l)$.

## Code walkthrough

### Signature and semantics

`utils/logging.py:TrainingLogger` has two public methods.

- `TrainingLogger.__init__(self, log_every: int = 10, seq_len: int = 1024, batch_size: int = 1)` — records the three scalars, starts two clocks (`_start`, `_step_start`), initializes an empty `_loss_window: list[float]`, and conditionally initializes WandB.
- `TrainingLogger.log(self, step: int, loss: float, metrics: Optional[Dict[str, float]] = None, lr: float = 0.0) -> None` — appends `loss` to the window and flushes when `step` is a multiple of `log_every`.

### Construction and the WandB gate

```python
def __init__(self, log_every: int = 10, seq_len: int = 1024, batch_size: int = 1):
    self.log_every = log_every
    self.seq_len = seq_len
    self.batch_size = batch_size
    self._start = time.time()
    self._step_start = time.time()
    self._loss_window: list[float] = []
    self._wandb = None
    wandb_project = os.environ.get("WANDB_PROJECT")
    if wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project, name=os.environ.get("WANDB_RUN_NAME"), reinit=True)
            self._wandb = wandb
        except ImportError:
            print("[logging] wandb not installed -- skipping WandB integration")
```

Semantics worth stating precisely:

- WandB is enabled **entirely by environment**, never by a constructor flag: `WANDB_PROJECT` set → `wandb.init` runs at construction; unset → `self._wandb` stays `None` and `log` prints only.
- `WANDB_RUN_NAME` optionally names the run; `reinit=True` permits repeated `wandb.init` in one process (e.g., multiple logger constructions in tests).
- Only `ImportError` is caught — a `wandb` install that fails for another reason (bad API key, no network) raises out of `__init__` and aborts training.
- `_start` is recorded but never read — dead state, harmless.

### `log`: append, gate, flush

```python
def log(self, step: int, loss: float, metrics: Optional[Dict[str, float]] = None, lr: float = 0.0) -> None:
    self._loss_window.append(loss)
    if step % self.log_every != 0 or not self._loss_window:
        return
    avg_loss = sum(self._loss_window) / len(self._loss_window)
    elapsed = max(time.time() - self._step_start, 1e-6)
    tokens_per_sec = (self.log_every * self.seq_len * self.batch_size) / elapsed
    ppl = torch.tensor(avg_loss).exp().item()
    parts = [f"step={step:>7}", f"loss={avg_loss:.4f}", f"ppl={ppl:.2f}", f"lr={lr:.2e}", f"tps={tokens_per_sec:,.0f}"]
    if metrics:
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}")
    print(" | ".join(parts))
    if self._wandb is not None:
        log_dict = {"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr, "train/tokens_per_sec": tokens_per_sec}
        if metrics:
            log_dict.update({f"train/{k}": v for k, v in metrics.items()})
        self._wandb.log(log_dict, step=step)
    self._loss_window = []
    self._step_start = time.time()
```

Step by step:

1. **Append unconditionally**, then gate: `step % log_every != 0` or an empty window → early return. The window can only be empty if `loss` was never appended, which cannot happen — the guard is defensive.
2. **`avg_loss`** is a plain Python mean over `_loss_window`; `loss` arrives as a Python `float`, so no tensor arithmetic happens until ppl.
3. **`tokens_per_sec`** implements the formula above; `lr` is whatever the caller passed (default `0.0`).
4. **Print format**: fields joined by `" | "` — `step` right-aligned 7 wide, `loss` 4 decimals, `ppl` 2 decimals, `lr` scientific 2 decimals, `tps` thousands-separated with no decimals. Extra `metrics` entries append as `{k}={v:.4f}`.
5. **WandB**: exactly one `log_dict` per flush with the four `train/` keys (plus `train/{k}` per extra metric), recorded at `step=step` — the global training step, so WandB's x-axis is training progress even across resume runs.
6. **Reset**: window emptied and `_step_start` restarted, so the next `elapsed` measures the next window only.

### Integration in the training loop

`training/pretrain.py:Pretrainer.__init__` builds the logger once:

```python
self.logger = TrainingLogger(
    log_every=config.log_every, seq_len=config.max_seq_len, batch_size=config.batch_size,
)
```

i.e. `seq_len` is the training-window size `max_seq_len` (2048), not the model's context length per se (the module does not consume `max_seq_len` — see [R1 `01-model-config.md`](01-model-config.md)), and `batch_size` is the DataLoader batch size (16). The call site inside `training/pretrain.py:Pretrainer.train`:

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
if global_step % self.config.log_every == 0:
    lr = self.scheduler.get_last_lr()[0]
    self.logger.log(global_step, metrics["loss"], lr=lr, metrics={})
```

Precise facts about this call pattern:

- **The guard is only `global_step % self.config.log_every == 0`.** The "metrics non-None" condition is enforced *upstream*, not at the call site: `training/pretrain.py:train_step` returns `None` when `nan_guard` detects NaN/Inf (after zeroing grads and skipping backward), the loop `continue`s before reaching the logger, and a NaN step does not advance `global_step`. So `log` is never invoked with a NaN loss — but nothing in `TrainingLogger.log` itself would reject one.
- **`metrics["loss"]`** is the cross-entropy value from `train_step` (computed with `ignore_index=-100`), passed positionally as `loss`; **`lr`** is `scheduler.get_last_lr()[0]`, the current effective LR of the `SequentialLR` (linear warmup → cosine).
- **`metrics={}`** — an empty dict, so the extra-metrics branches in `log` (both the console `{k}={v:.4f}` and `train/{k}` WandB keys) are inert in this integration. `train_step` returns only `{"loss": …}` anyway; its dict is consumed via the `["loss"]` lookup, never passed whole.
- **Step 0 is logged**: `global_step` starts at 0 and `0 % log_every == 0`, so the first line appears immediately. Its `elapsed` covers logger construction through the first step, including dataset build, checkpoint scan, and any `load_checkpoint` ([R9 `09-checkpoint.md`](09-checkpoint.md)) — discard this line for steady-state tps.
- **After resume**, `global_step` is the restored step (`Pretrainer.load_checkpoint`), so the first flush occurs at the next multiple of `log_every`, with a one-loss window.

## Invariants

- A flush occurs **iff** the caller passes `step % log_every == 0` and the window is non-empty; every flush empties `_loss_window` and restarts `_step_start`.
- Exactly one console line and at most one WandB record per flush, keyed by the global `step`; WandB keys are `train/loss`, `train/ppl`, `train/lr`, `train/tokens_per_sec`, plus `train/{k}` for non-empty `metrics`.
- `tps` is only as accurate as its numerator assumption: `log_every × seq_len × batch_size` tokens must actually have been processed between flushes.

## Pitfalls

1. **The window is size 1 in the shipped integration.** Because the caller and the logger share the `% log_every == 0` gate, `avg_loss` is a single-step loss and ppl is `exp` of that step — there is no smoothing, despite the "rolling-window" docstring. To get true windowing, call `log` every step and let the internal gate do the buffering; the current `Pretrainer.train` does not.
2. **`tps` counts tokens including the batch factor, but only over logged windows.** It is correct *only because* the caller fires exactly once per window, so `elapsed` spans exactly `log_every` steps. Two ways it silently breaks: (a) NaN-guard skips consume wall time without advancing `global_step`, so a window's clock includes dead forward passes and `tps` reads low; (b) if the caller switched to calling `log` every step, `elapsed` would span one step while the numerator still counts `log_every × seq_len × batch_size` — inflating `tps` by a factor of `log_every`. Keep the two gates aligned.
3. **`ppl` is exp of the window-average loss, not the mean of per-step ppls** (Jensen; see [Math](#math)). Comparing the logged ppl against a mean of per-step perplexities will show a small, systematic gap.
4. **WandB initializes at construction, not at first log.** With `WANDB_PROJECT` set, `wandb.init` runs even if training crashes before the first log — you still get an empty experiment. Only `ImportError` is tolerated; auth/network failures propagate and abort training.
5. **First-line contamination.** The step-0 window's `elapsed` includes startup (dataset construction, checkpoint scan/load), so its `tps` is not representative; ignore it for steady-state readings.
6. **Cadence is config, not CLI.** `log_every` (default 50) comes from `TrainingConfig`/the yaml ([R12 `12-config-reference.md`](12-config-reference.md)); there is no `--log-every` flag ([R7 `07-pretrain-cli.md`](07-pretrain-cli.md)). A full 256k-step run emits ~5,120 console lines and WandB records.

## Tests

There are **no dedicated tests** for the logger: no `tests/test_utils.py` exists in the tree (the test files are `test_doc_refs.py`, `test_ssd.py`, `test_ssd_triton.py`, `test_mimo.py`, `test_train_step.py`, `test_transformer.py`, `test_grad_checkpoint.py`, and the GPU-gated `e2e_gpu_smoke.py`). The only automated check that touches `utils/logging.py` is `tests/test_doc_refs.py`, which verifies that doc citations like `` `utils/logging.py:TrainingLogger` `` resolve to real symbols — it checks anchors, not behavior. `tests/test_train_step.py` exercises `training/pretrain.py:train_step`, the function that produces the `{"loss": …}` dict the logger consumes, but never instantiates `TrainingLogger`. Behavioral coverage is indirect: `python training/pretrain.py --dry-run` (2 steps, per [R7 `07-pretrain-cli.md`](07-pretrain-cli.md)) logs step 0 — one line with `step=0` — and the [G2 runbook](../guides/02-training-runbook.md) covers reading the loss/ppl/tps columns and `WANDB_PROJECT` setup.
