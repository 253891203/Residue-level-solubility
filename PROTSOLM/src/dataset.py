from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PDBSolEmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_path: str | Path,
        metadata_path: str | Path,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        normalize: bool = True,
        output_dtype: str = "float16",
    ) -> None:
        self.embedding_path = Path(embedding_path)
        self.metadata_path = Path(metadata_path)
        self.embeddings = np.load(self.embedding_path, mmap_mode="r")
        self.metadata = pd.read_csv(self.metadata_path)
        if len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                f"Metadata rows ({len(self.metadata)}) do not match embeddings "
                f"({self.embeddings.shape[0]}): {self.metadata_path}"
            )
        self.labels = self.metadata["label"].astype("float32").to_numpy()
        self.lengths = self.metadata["length"].astype("int64").to_numpy()
        self.names = self.metadata["name"].astype(str).tolist()
        self.normalize = normalize
        self.mean = self._prepare_stat(mean) if mean is not None else None
        self.std = self._prepare_stat(std) if std is not None else None
        if output_dtype not in {"float16", "float32"}:
            raise ValueError(f"output_dtype must be float16 or float32, got {output_dtype}")
        self.output_dtype = output_dtype

    def _prepare_stat(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape[-1] != self.embed_dim:
            raise ValueError(f"Normalization stat last dim must be {self.embed_dim}, got {arr.shape}")
        return arr.reshape(-1, self.embed_dim)[-1:].astype("float32")

    @property
    def pad_len(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.embeddings.shape[2])

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        emb = np.asarray(self.embeddings[idx], dtype=np.float32)
        if self.normalize and self.mean is not None and self.std is not None:
            emb = (emb - self.mean) / self.std
        if self.output_dtype == "float16":
            emb = emb.astype(np.float16, copy=False)
        return {
            "name": self.names[idx],
            "embedding": torch.from_numpy(emb),
            "length": torch.tensor(self.lengths[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def compute_train_norm_stats(
    embedding_path: str | Path,
    metadata_path: str | Path,
    out_path: str | Path,
    chunk_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    out_path = Path(out_path)
    if out_path.exists():
        data = np.load(out_path)
        return data["mean"].astype("float32"), data["std"].astype("float32")

    embeddings = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    embed_dim = embeddings.shape[2]
    total_sum = np.zeros(embed_dim, dtype=np.float64)
    total_sq = np.zeros(embed_dim, dtype=np.float64)
    total_count = 0

    for start in range(0, len(metadata), chunk_size):
        end = min(start + chunk_size, len(metadata))
        batch = np.asarray(embeddings[start:end], dtype=np.float32)
        lengths = metadata["length"].iloc[start:end].astype(int).to_numpy()
        for row, length in enumerate(lengths):
            valid = batch[row, :length, :]
            total_sum += valid.sum(axis=0)
            total_sq += np.square(valid, dtype=np.float64).sum(axis=0)
            total_count += int(length)

    mean = (total_sum / max(total_count, 1)).astype("float32").reshape(1, embed_dim)
    var = (total_sq / max(total_count, 1)) - np.square(mean.reshape(embed_dim), dtype=np.float64)
    std = np.sqrt(np.maximum(var, 1e-12)).astype("float32").reshape(1, embed_dim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mean=mean, std=std)
    return mean, std
