import argparse, math, os, sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import Mamba3Transformer, ModelConfig
from utils.checkpoint import CheckpointManager
from utils.logging import init_logging, get_logger

# ponytail: utils/distributed.py deleted — was a thin DEVICE-constant + device() wrapper.
# Inlined here as the only two call sites.
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



@dataclass
class TrainingConfig:
    model_config: dict = field(default_factory=dict)
    data_path: str = "data/pretrain_data.bin"
    checkpoint_dir: str = "checkpoints/pretrain"
    vocab_size: int = 50257
    max_seq_len: int = 2048
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_steps: int = 256000
    warmup_steps: int = 2000
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    grad_checkpoint: bool = True
    compile_model: bool = True
    save_every: int = 4000
    log_every: int = 50
    nan_guard: bool = True
    nan_guard_max_consecutive: int = 5


class PretrainDataset(Dataset):
    """Packed pre-training dataset backed by flat token tensors (single-file or sharded)."""
    def __init__(self, data_path: str, max_seq_len: int, vocab_size: int):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        if not os.path.exists(data_path):
            # For the purpose of dry-run and validation, we fallback to a dummy implementation if data doesn't exist
            print(f"[warn] Pre-training data not found: {data_path}. Using dummy data for testing.")
            self.layout = "dummy"
            self._n_samples = 1000
            return
            
        self._init_sharded(data_path) if os.path.isdir(data_path) else self._init_single(data_path)

    def _init_single(self, data_path: str) -> None:
        self.layout = "single"
        self.data = torch.load(data_path, weights_only=True)
        self._n_samples = (len(self.data) - 1) // self.max_seq_len

    def _get_window_single(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        chunk = self.data[start: start + self.max_seq_len + 1]
        return chunk[:-1], chunk[1:]

    def _init_sharded(self, data_dir: str) -> None:
        shard_paths = sorted(Path(data_dir).glob("shard_*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"No `shard_*.bin` files in {data_dir}")
        self.layout = "sharded"
        self.shards = [torch.load(p, weights_only=True, mmap=True) for p in shard_paths]
        self.shard_sizes = [s.numel() for s in self.shards]
        self.shard_offsets = [0] + [sum(self.shard_sizes[:i+1]) for i in range(len(self.shard_sizes)-1)]
        self._total_tokens = sum(self.shard_sizes)
        self._n_samples = (self._total_tokens - 1) // self.max_seq_len

    def _get_window_sharded(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        shard_idx, offset_in_shard = self._locate(start)
        if offset_in_shard + (self.max_seq_len + 1) <= self.shard_sizes[shard_idx]:
            chunk = self.shards[shard_idx][offset_in_shard: offset_in_shard + self.max_seq_len + 1]
            return chunk[:-1], chunk[1:]
        
        needed = self.max_seq_len + 1
        collected = []
        cursor = start
        while len(collected) < needed:
            s_idx, off = self._locate(cursor)
            take = min(needed - len(collected), self.shard_sizes[s_idx] - off)
            collected.extend(self.shards[s_idx][off: off + take].tolist())
            cursor += take
        chunk = torch.tensor(collected[:needed], dtype=torch.long)
        return chunk[:-1], chunk[1:]

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        return lo, global_idx - self.shard_offsets[lo]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layout == "dummy":
            return torch.randint(0, self.vocab_size, (self.max_seq_len,)), torch.randint(0, self.vocab_size, (self.max_seq_len,))
        return self._get_window_single(idx) if self.layout == "single" else self._get_window_sharded(idx)


def _build_minimal_pretrainer(model_config: dict) -> "Pretrainer":
    """Build a CPU-only Pretrainer with compile/grad-checkpoint disabled.
    Used by tests/test_train_step.py to avoid the heavy __init__ side effects.
    The data_path arg is intentionally absent — train_step does not touch the dataset.
    """
    p = Pretrainer.__new__(Pretrainer)
    p.config = TrainingConfig(
        model_config=model_config,
        max_seq_len=model_config.get("max_seq_len", 16),
        vocab_size=model_config.get("vocab_size", 64),
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        nan_guard=True,
    )
    p.device = torch.device("cpu")
    p.amp_dtype = torch.float32
    p._opt_steps = 0
    p._log = lambda msg: None  # swallow log output during tests; called as self._log(msg) so a 1-arg lambda is correct
    p._amp_context = lambda: torch.amp.autocast("cpu", enabled=False)
    p.model = Mamba3Transformer(ModelConfig(**model_config))
    p.raw_model = p.model
    p.optimizer = AdamW(p.model.parameters(), lr=1e-3, fused=False)
    p.scheduler = torch.optim.lr_scheduler.LambdaLR(p.optimizer, lr_lambda=lambda s: 1.0)
    return p


class Pretrainer:
    """BF16 pre-training loop for single GPU."""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = _DEVICE  # ponytail: inlined from utils/distributed.py.
        if not torch.cuda.is_available():
            print("[warn] CUDA not available — running on CPU (smoke-testing only).")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        init_logging(config.log_every, seq_len=config.max_seq_len)
        self.logger = get_logger()

        self._log("Initialising model...")
        # Inject training-only flags into the model config dict so each Mamba3Block
        # can read them at construction. (The YAML's grad_checkpoint lives on
        # TrainingConfig; the model needs it on its own config.)
        config.model_config.setdefault("grad_checkpoint", config.grad_checkpoint)
        raw_model = Mamba3Transformer(config.model_config).to(self.device)
        total, trainable = count_parameters(raw_model)
        self._log(f"Parameters: {total:,} total / {trainable:,} trainable")
        
        training_model = raw_model
        if config.compile_model and hasattr(torch, "compile"):
            compile_mode = os.environ.get("TORCH_COMPILE_MODE", "max-autotune")
            self._log(f"Compiling model with torch.compile (mode={compile_mode})...")
            training_model = torch.compile(training_model, mode=compile_mode, fullgraph=False)

        self.model = training_model
        self.raw_model = raw_model

        seen = set()
        all_params = []
        for p in self.model.parameters():
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                all_params.append(p)
        decay_params = [p for p in all_params if p.dim() >= 2]
        no_decay_params = [p for p in all_params if p.dim() < 2]
        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.lr, betas=(config.beta1, config.beta2), fused=torch.cuda.is_available())

        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=config.max_steps - config.warmup_steps, eta_min=config.lr * config.min_lr_ratio)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps])
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.ckpt_manager = CheckpointManager(config.checkpoint_dir)
        self._opt_steps = 0

    @staticmethod
    def _log(msg: str) -> None:
        print(msg)

    def _amp_context(self):
        if torch.cuda.is_available():
            return autocast("cuda", dtype=self.amp_dtype)
        else:
            return autocast("cpu", enabled=False)

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor, micro_step: int) -> Optional[Dict[str, float]]:
        is_opt_step = (micro_step + 1) % self.config.gradient_accumulation_steps == 0
        with self._amp_context():
            logits = self.model(tokens)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            _ce_loss_val = float(loss.item())
            loss = loss / self.config.gradient_accumulation_steps

        if self.config.nan_guard and (torch.isnan(loss).any().item() or torch.isinf(loss).any().item()):
            self._log(f"[nan-guard] NaN/Inf at micro_step={micro_step}, opt_steps={self._opt_steps}. Skipping backward.")
            self.optimizer.zero_grad(set_to_none=True)
            return None

        loss.backward()
        if is_opt_step:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._opt_steps += 1

        return {"loss": _ce_loss_val}

    def save_checkpoint(self, step: int, tag: str = "") -> None:
        state = self.raw_model.state_dict()
        extra_meta = {"scheduler": self.scheduler.state_dict(), "opt_steps": self._opt_steps,
                      "tag": tag or f"step_{step}", "config": asdict(self.config)}
        self.ckpt_manager.save(self.raw_model, self.optimizer, step, extra_meta=extra_meta, state_dict=state)
        self._log(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, step: int) -> int:
        meta = self.ckpt_manager.load(self.raw_model, step, device=str(self.device), optimizer=self.optimizer, strict=False)
        if "scheduler" in meta:
            self.scheduler.load_state_dict(meta["scheduler"])
        if "opt_steps" in meta:
            self._opt_steps = meta["opt_steps"]
        resumed_step = meta.get("step", step)
        self._log(f"Resumed from step {resumed_step}")
        return resumed_step

    def _find_latest_checkpoint(self) -> Optional[int]:
        return self.ckpt_manager.latest_step()

    def train(self) -> None:
        dataset = PretrainDataset(self.config.data_path, self.config.max_seq_len, self.config.vocab_size)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=0, drop_last=True)
        # Note: num_workers=0 to prevent multiprocessing issues during dry run.

        global_step = 0
        latest = self._find_latest_checkpoint()
        if latest is not None:
            try:
                global_step = self.load_checkpoint(latest)
            except Exception as exc:
                self._log(f"[warn] Could not load checkpoint: {exc}")

        self._log(f"Training from step {global_step} to {self.config.max_steps}")
        self.raw_model.train()
        nan_guard_streak = 0
        while global_step < self.config.max_steps:
            for tokens, targets in tqdm(loader):
                if global_step >= self.config.max_steps:
                    break
                tokens = tokens.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
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
                if global_step % self.config.save_every == 0 and global_step > 0:
                    self.save_checkpoint(global_step)
                global_step += 1
        self.save_checkpoint(global_step, tag="final")
        self._log("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba-3-Lite pre-training (single GPU)")
    parser.add_argument("--config", type=str, default="configs/pretrain_a100_400m.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint step number to resume from")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 steps to verify wiring")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    t = yaml_cfg.get("training", {})
    d = yaml_cfg.get("data", {})

    config = TrainingConfig(
        model_config=yaml_cfg.get("model", yaml_cfg),
        data_path=args.data_path or d.get("train_data_path", "data/pretrain_data.bin"),
        checkpoint_dir=args.checkpoint_dir or t.get("save_dir", "checkpoints/pretrain_a100"),
        max_seq_len=yaml_cfg.get("model", yaml_cfg).get("max_seq_len", 2048),
        vocab_size=yaml_cfg.get("model", yaml_cfg).get("vocab_size", 50257),
        batch_size=t.get("micro_batch_size", 16),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 2),
        max_steps=2 if args.dry_run else t.get("total_steps", 256000),
        warmup_steps=t.get("warmup_steps", 2000),
        lr=t.get("lr", 3.0e-4),
        min_lr_ratio=t.get("min_lr_ratio", 0.05),
        weight_decay=t.get("weight_decay", 0.1),
        max_grad_norm=t.get("grad_clip", 1.0),
        grad_checkpoint=t.get("grad_checkpoint", True) and not args.no_checkpoint,
        compile_model=t.get("compile", True) and not args.no_compile,
        save_every=t.get("save_interval", 4000),
        log_every=t.get("log_interval", 50),
        nan_guard=t.get("nan_guard", True),
        nan_guard_max_consecutive=t.get("nan_guard_max_consecutive", 5),
    )

    trainer = Pretrainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(int(args.resume))
    trainer.train()


if __name__ == "__main__":
    main()
