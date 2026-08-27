import logging
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch


EMBED_DIM = 1280
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")
REPLACE_WITH_X = set("UBZOJ")


def setup_logging(log_file: str | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("pdbsol")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def round_up_to(length: int, multiple: int = 100) -> int:
    return int(math.ceil(length / multiple) * multiple)


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    candidates = {
        "name": ["name", "id", "protein_id", "protein", "pdb_id", "uniprot_id"],
        "aa_seq": ["aa_seq", "sequence", "seq", "protein_sequence", "aa_sequence", "fasta"],
        "label": ["label", "target", "solubility", "y", "class"],
    }
    resolved: Dict[str, str] = {}
    for key, names in candidates.items():
        for name in names:
            if name in lower_to_original:
                resolved[key] = lower_to_original[name]
                break
        if key not in resolved:
            raise ValueError(
                f"Could not find required column '{key}'. Columns are: {list(df.columns)}"
            )
    return resolved


def sanitize_sequence(seq: str) -> Tuple[str, Dict[str, int]]:
    seq = str(seq).strip().upper().replace(" ", "").replace("\n", "")
    counts: Dict[str, int] = {}
    cleaned = []
    for aa in seq:
        if aa in VALID_AA:
            cleaned.append(aa)
        elif aa in REPLACE_WITH_X:
            counts[aa] = counts.get(aa, 0) + 1
            cleaned.append("X")
        else:
            counts[aa] = counts.get(aa, 0) + 1
            cleaned.append("X")
    return "".join(cleaned), counts


def load_pdbsol_csv(path: str | Path, split: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{split} CSV not found: {path}")
    raw = pd.read_csv(path)
    logger.info("[%s] CSV: %s", split, path)
    logger.info("[%s] Columns: %s", split, list(raw.columns))
    cols = resolve_columns(raw)

    df = raw[[cols["name"], cols["aa_seq"], cols["label"]]].copy()
    df.columns = ["name", "aa_seq", "label"]
    df["name"] = df["name"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["aa_seq", "label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    invalid_totals: Dict[str, int] = {}
    cleaned = []
    for seq in df["aa_seq"].tolist():
        clean_seq, counts = sanitize_sequence(seq)
        cleaned.append(clean_seq)
        for aa, count in counts.items():
            invalid_totals[aa] = invalid_totals.get(aa, 0) + count
    df["aa_seq"] = cleaned
    df["length"] = df["aa_seq"].str.len().astype(int)
    df = df[df["length"] > 0].reset_index(drop=True)

    logger.info(
        "[%s] Samples=%d | length min=%d max=%d | labels=%s",
        split,
        len(df),
        int(df["length"].min()) if len(df) else 0,
        int(df["length"].max()) if len(df) else 0,
        df["label"].value_counts().sort_index().to_dict(),
    )
    if invalid_totals:
        logger.warning("[%s] Non-standard residues replaced with X: %s", split, invalid_totals)
    return df


def iter_split_csvs(paths: Dict[str, str]) -> Iterable[tuple[str, str]]:
    for split in ("train", "valid", "test"):
        yield split, paths[split]
