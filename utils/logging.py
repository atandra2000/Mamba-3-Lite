"""Human-readable training metrics with optional WandB forwarding.

Set `WANDB_PROJECT` to enable the optional integration; local logging remains
available when WandB is absent or intentionally disabled.
"""
import os, time
import torch


class TrainingLogger:
    """Aggregate loss over logging intervals and report throughput and perplexity."""

    def __init__(self, log_every: int = 10, seq_len: int = 1024, batch_size: int = 1):
        self.log_every = log_every
        self.seq_len = seq_len
        self.batch_size = batch_size
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

    def log(self, step: int, loss: float, lr: float = 0.0) -> None:
        """Record one loss and emit/reset the interval summary when due."""
        self._loss_window.append(loss)
        if step % self.log_every != 0 or not self._loss_window:
            return
        avg_loss = sum(self._loss_window) / len(self._loss_window)
        elapsed = max(time.time() - self._step_start, 1e-6)
        tokens_per_sec = (self.log_every * self.seq_len * self.batch_size) / elapsed
        ppl = torch.tensor(avg_loss).exp().item()
        parts = [f"step={step:>7}", f"loss={avg_loss:.4f}", f"ppl={ppl:.2f}", f"lr={lr:.2e}", f"tps={tokens_per_sec:,.0f}"]
        print(" | ".join(parts))
        if self._wandb is not None:
            log_dict = {"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr, "train/tokens_per_sec": tokens_per_sec}
            self._wandb.log(log_dict, step=step)
        self._loss_window = []
        self._step_start = time.time()
