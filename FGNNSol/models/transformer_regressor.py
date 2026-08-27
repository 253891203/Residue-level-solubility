"""Transformer regressor over residue-level ESM2 representations."""
from __future__ import annotations

import math
import torch
from torch import nn


def sinusoidal_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=1280, d_model=256, num_layers=2, nhead=8,
                 dim_feedforward=1024, dropout=0.1, pooling="attention"):
        super().__init__()
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be 'attention' or 'mean'")
        self.config = dict(input_dim=input_dim, d_model=d_model, num_layers=num_layers,
                           nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                           pooling=pooling)
        self.pooling = pooling
        self.projection = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu",
            norm_first=True, batch_first=True)
        # norm_first=True is intentional. Disable the incompatible nested-tensor
        # fast path explicitly so PyTorch does not emit the same warning at
        # every checkpoint load. This does not alter model parameters/results.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.attention_score = nn.Linear(d_model, 1) if pooling == "attention" else None
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or padding_mask.shape != embeddings.shape[:2]:
            raise ValueError(f"Bad shapes: embeddings={tuple(embeddings.shape)}, mask={tuple(padding_mask.shape)}")
        x = self.projection(embeddings)
        x = x + sinusoidal_encoding(x.shape[1], x.shape[2], x.device, x.dtype).unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = ~padding_mask
        if not torch.all(valid.any(dim=1)):
            raise ValueError("At least one sample contains no valid residues")
        if self.pooling == "attention":
            scores = self.attention_score(x).squeeze(-1).masked_fill(padding_mask, float("-inf"))
            pooled = torch.sum(x * torch.softmax(scores, dim=1).unsqueeze(-1), dim=1)
        else:
            pooled = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1)
