from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1).masked_fill(~mask, torch.finfo(x.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class TransformerSolubilityClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        d_model: int = 256,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pooling = pooling
        self.attn_pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected embeddings with shape (batch, L, dim), got {tuple(x.shape)}")
        valid_mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=~valid_mask)
        if self.pooling == "mean":
            pooled = (h * valid_mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
        elif self.pooling == "attention":
            pooled = self.attn_pool(h, valid_mask)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.head(self.norm(pooled)).squeeze(-1)
