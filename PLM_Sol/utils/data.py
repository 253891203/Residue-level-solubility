from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def label_from_id(protein_id: str) -> int:
    try:
        return int(protein_id.rsplit("-", 1)[-1].strip())
    except (TypeError, ValueError):
        raise ValueError(f"Cannot extract a binary label from protein ID: {protein_id!r}")


def load_or_compute_normalization(train_h5, stats_path, embed_dim: int = 1280):
    stats_path = Path(stats_path)
    if stats_path.exists():
        saved = np.load(stats_path)
        return (
            torch.tensor(saved["mean"], dtype=torch.float32),
            torch.tensor(saved["std"], dtype=torch.float32),
        )

    total_sum = np.zeros(embed_dim, dtype=np.float64)
    total_square_sum = np.zeros(embed_dim, dtype=np.float64)
    total_count = 0
    with h5py.File(train_h5, "r") as handle:
        embeddings = handle["residue_embed"]
        for start in range(0, len(embeddings), 1000):
            sample_means = embeddings[start : start + 1000].mean(axis=1)
            total_sum += sample_means.sum(axis=0)
            total_square_sum += np.square(sample_means).sum(axis=0)
            total_count += len(sample_means)

    mean = total_sum / total_count
    std = np.sqrt(total_square_sum / total_count - np.square(mean)) + 1e-8
    mean = mean.reshape(1, embed_dim).astype(np.float32)
    std = std.reshape(1, embed_dim).astype(np.float32)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, mean=mean, std=std)
    return torch.from_numpy(mean), torch.from_numpy(std)


class ProteinHDF5Dataset(Dataset):
    def __init__(self, h5_path, mean, std):
        self.h5_path = str(h5_path)
        self.mean = mean
        self.std = std
        with h5py.File(self.h5_path, "r") as handle:
            self.sample_ids = [_decode(value) for value in handle["id"][:]]
            self.embedding_shape = tuple(handle["residue_embed"].shape[1:])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        with h5py.File(self.h5_path, "r") as handle:
            array = handle["residue_embed"][index]
        embedding = torch.tensor(array, dtype=torch.float32)
        embedding = (embedding - self.mean) / self.std
        protein_id = self.sample_ids[index]
        label = torch.tensor(label_from_id(protein_id), dtype=torch.float32)
        return protein_id, embedding, label
