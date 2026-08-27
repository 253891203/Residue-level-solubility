from __future__ import annotations

import math

import torch
import torch.nn as nn


class ProteinSolubilityTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 2000,
        token_pool_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_max_len = max_len
        self.token_pool_size = max(1, int(token_pool_size))
        self.max_len = math.ceil(max_len / self.token_pool_size)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def make_padding_mask(self, lengths: torch.Tensor, pad_len: int) -> torch.Tensor:
        positions = torch.arange(pad_len, device=lengths.device).unsqueeze(0)
        residue_mask = positions >= lengths.unsqueeze(1)
        cls_mask = torch.zeros((lengths.shape[0], 1), dtype=torch.bool, device=lengths.device)
        return torch.cat([cls_mask, residue_mask], dim=1)

    def pool_tokens(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_pool_size == 1:
            return x, lengths

        batch_size, seq_len, dim = x.shape
        pooled_len = math.ceil(seq_len / self.token_pool_size)
        padded_len = pooled_len * self.token_pool_size
        valid = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        x = x * valid.unsqueeze(-1).to(dtype=x.dtype)

        if padded_len > seq_len:
            pad_tokens = x.new_zeros((batch_size, padded_len - seq_len, dim))
            pad_mask = torch.zeros((batch_size, padded_len - seq_len), dtype=torch.bool, device=x.device)
            x = torch.cat([x, pad_tokens], dim=1)
            valid = torch.cat([valid, pad_mask], dim=1)

        x = x.view(batch_size, pooled_len, self.token_pool_size, dim)
        valid = valid.view(batch_size, pooled_len, self.token_pool_size)
        counts = valid.sum(dim=2).clamp_min(1).unsqueeze(-1).to(dtype=x.dtype)
        x = x.sum(dim=2) / counts
        pooled_lengths = torch.div(lengths + self.token_pool_size - 1, self.token_pool_size, rounding_mode="floor")
        return x, pooled_lengths

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, L, input_dim), got {tuple(x.shape)}")
        if x.shape[1] > self.input_max_len:
            raise ValueError(f"Input length {x.shape[1]} exceeds model input_max_len {self.input_max_len}")
        x = self.input_proj(x)
        x, lengths = self.pool_tokens(x, lengths)
        pad_len = x.shape[1]
        if pad_len > self.max_len:
            raise ValueError(f"Pooled length {pad_len} exceeds model max_len {self.max_len}")
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        padding_mask = self.make_padding_mask(lengths, pad_len)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        cls_out = self.norm(x[:, 0, :])
        return self.classifier(cls_out).squeeze(-1)
