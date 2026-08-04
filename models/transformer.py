"""Mamba-3 Transformer model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .mamba_block import Mamba3Block


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    head_dim: int = 64
    state_dim: int = 64
    chunk_size: int = 64
    ssd_dispatch: str = "pytorch"
    ffn_dim: int = 2048
    max_seq_len: int = 2048
    weight_tying: bool = True
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    grad_checkpoint: bool = False


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

        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if getattr(module, "_identity_init", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, T, vocab_size)."""
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits
