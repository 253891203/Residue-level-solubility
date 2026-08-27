from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")
FOLD_KEYWORDS = ("fold", "split", "partition")


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _sequence_score(series: pd.Series) -> float:
    values = series.dropna().astype(str).str.replace(r"\s+", "", regex=True)
    if values.empty:
        return 0.0
    valid = values.map(lambda x: len(x) > 0 and set(x.upper()) <= AMINO_ACIDS)
    return float(valid.mean())


def identify_sequence_column(df: pd.DataFrame) -> str:
    preferred = ["fasta", "sequence", "seq", "aa_seq", "protein_sequence"]
    for col in preferred:
        if col in df.columns and _sequence_score(df[col]) >= 0.8:
            return col
    scores = {col: _sequence_score(df[col]) for col in df.columns}
    best = max(scores, key=scores.get)
    if scores[best] >= 0.8:
        return best
    raise ValueError(f"Cannot identify sequence column. Existing columns: {list(df.columns)}")


def identify_label_column(df: pd.DataFrame) -> str:
    preferred = [
        "solubility",
        "solublility|0=Insoluble|1=Soluble",
        "label",
        "target",
        "sol",
        "class",
    ]
    for col in preferred:
        if col in df.columns and _can_map_labels(df[col]):
            return col
    candidates = []
    for col in df.columns:
        name = col.lower()
        if any(key in name for key in ("sol", "label", "target", "class")) and _can_map_labels(df[col]):
            candidates.append(col)
    if candidates:
        return candidates[0]
    raise ValueError(f"Cannot identify label column. Existing columns: {list(df.columns)}")


def identify_id_column(df: pd.DataFrame) -> str | None:
    for col in ("sid", "id", "name", "protein_id", "accession"):
        if col in df.columns:
            return col
    return None


def identify_fold_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if any(key in col.lower() for key in FOLD_KEYWORDS):
            values = pd.to_numeric(df[col], errors="coerce")
            unique = sorted(values.dropna().unique().tolist())
            if len(unique) == 5:
                return col
    return None


def _can_map_labels(series: pd.Series) -> bool:
    try:
        mapped = map_labels(series)
    except ValueError:
        return False
    return set(mapped.dropna().unique()).issubset({0, 1}) and mapped.notna().all()


def map_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        labels = pd.to_numeric(series, errors="raise").astype(int)
    else:
        mapping = {
            "0": 0,
            "1": 1,
            "insoluble": 0,
            "insol": 0,
            "not soluble": 0,
            "negative": 0,
            "false": 0,
            "soluble": 1,
            "sol": 1,
            "positive": 1,
            "true": 1,
        }
        normalized = series.astype(str).str.strip().str.lower()
        if not normalized.isin(mapping).all():
            bad = sorted(normalized[~normalized.isin(mapping)].unique().tolist())
            raise ValueError(f"Unrecognized label values: {bad[:20]}")
        labels = normalized.map(mapping).astype(int)
    values = set(labels.unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Labels must be binary with 1=soluble and 0=insoluble, got {sorted(values)}")
    return labels


def load_netsolp_csv(path: Path, split_name: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    seq_col = identify_sequence_column(raw)
    label_col = identify_label_column(raw)
    id_col = identify_id_column(raw)
    fold_col = identify_fold_column(raw)

    df = pd.DataFrame()
    df["sample_id"] = raw[id_col].astype(str) if id_col else [f"{split_name}_{i}" for i in range(len(raw))]
    df["sequence"] = raw[seq_col].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    df["label"] = map_labels(raw[label_col]).astype(int)
    df["length"] = df["sequence"].str.len().astype(int)
    if fold_col is not None:
        df["fold"] = pd.to_numeric(raw[fold_col], errors="raise").astype(int)

    info = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(raw.columns),
        "sequence_column": seq_col,
        "label_column": label_col,
        "id_column": id_col,
        "fold_column": fold_col,
        "label_distribution": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "label_semantics": "1=soluble, 0=insoluble",
    }
    return df, info


def length_stats(df: pd.DataFrame) -> dict:
    lengths = df["length"].to_numpy()
    return {
        "n": int(len(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p95": float(np.quantile(lengths, 0.95)),
    }

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def embedding_path(embedding_dir: Path, split: str, row_index: int) -> Path:
    return embedding_dir / split / f"{row_index:06d}.pt"


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv: Path, embedding_dir: Path, indices: list[int] | None = None):
        self.meta = pd.read_csv(metadata_csv)
        self.embedding_dir = Path(embedding_dir)
        self.indices = list(range(len(self.meta))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        row_idx = self.indices[idx]
        row = self.meta.iloc[row_idx]
        emb_path = self.embedding_dir / str(row["split"]) / f"{int(row['row_index']):06d}.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding for row {row_idx}: {emb_path}")
        embedding = torch.load(emb_path, map_location="cpu")
        return {
            "embedding": embedding.float(),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "length": int(row["length"]),
            "sample_id": str(row["sample_id"]),
        }


def collate_embeddings(batch: list[dict]) -> dict:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.stack([item["label"] for item in batch])
    max_len = int(lengths.max().item())
    dim = int(batch[0]["embedding"].shape[-1])
    embeddings = torch.zeros((len(batch), max_len, dim), dtype=torch.float32)
    for i, item in enumerate(batch):
        emb = item["embedding"]
        embeddings[i, : emb.shape[0]] = emb
    return {
        "embeddings": embeddings,
        "lengths": lengths,
        "labels": labels,
        "sample_ids": [item["sample_id"] for item in batch],
    }
