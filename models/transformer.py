"""Mamba-3 Transformer model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import Mamba3Block  # ponytail: RMSNorm import dropped — using native nn.RMSNorm.


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    head_dim: int = 64
    state_dim: int = 64
    chunk_size: int = 64
    ssd_dispatch: str = "pytorch"  # 'pytorch' | 'triton' (requires ENABLE_TRITON_KERNELS=1)
    ffn_dim: int = 2048
    max_seq_len: int = 2048
    dtype: str = "bf16"
    weight_tying: bool = True
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    grad_checkpoint: bool = False  # injected from TrainingConfig at construction time


class Mamba3Transformer(nn.Module):
    """Mamba-3 Lite Architecture."""

    def __init__(self, cfg: ModelConfig | dict):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = ModelConfig(**cfg)
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.layers = nn.ModuleList([
            Mamba3Block(cfg.__dict__, layer_idx=i)
            for i in range(cfg.n_layers)
        ])

        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)  # ponytail: native nn.RMSNorm.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, T, vocab_size)."""
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits
