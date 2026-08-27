import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import EMBED_DIM, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Verify PDBSol embedding npy files before training.")
    parser.add_argument("--embedding_dir", default="outputs/embeddings")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--strict_zero_padding", action="store_true", default=True)
    parser.add_argument("--no_strict_zero_padding", action="store_false", dest="strict_zero_padding")
    return parser.parse_args()


def verify_split(split: str, embedding_dir: Path, chunk_size: int, strict_zero_padding: bool, logger) -> dict:
    emb_path = embedding_dir / f"PDBSol_{split}_embeddings.npy"
    meta_path = embedding_dir / f"PDBSol_{split}_metadata.csv"
    if not emb_path.exists():
        raise FileNotFoundError(f"[{split}] Missing embedding file: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"[{split}] Missing metadata file: {meta_path}")

    arr = np.load(emb_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    if arr.ndim != 3:
        raise ValueError(f"[{split}] Expected 3D embeddings, got shape={arr.shape}")
    n_samples, pad_len, embed_dim = arr.shape
    if embed_dim != EMBED_DIM:
        raise ValueError(f"[{split}] Expected embed_dim={EMBED_DIM}, got {embed_dim}")
    if len(meta) != n_samples:
        raise ValueError(f"[{split}] Metadata rows={len(meta)} but embeddings N={n_samples}")
    required_cols = {"index", "name", "aa_seq", "length", "label"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"[{split}] Metadata missing columns: {sorted(missing)}")

    lengths = meta["length"].astype(int).to_numpy()
    seq_lengths = meta["aa_seq"].astype(str).str.len().astype(int).to_numpy()
    if not np.array_equal(lengths, seq_lengths):
        bad = np.where(lengths != seq_lengths)[0][:10].tolist()
        raise ValueError(f"[{split}] length column does not match aa_seq length. First bad rows: {bad}")
    if lengths.min() <= 0:
        raise ValueError(f"[{split}] Non-positive sequence length found.")
    if lengths.max() > pad_len:
        raise ValueError(f"[{split}] max length={lengths.max()} exceeds pad_len={pad_len}")
    if not set(meta["label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"[{split}] Labels must be binary 0/1.")

    checked_padding = 0
    checked_finite = 0
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch = np.asarray(arr[start:end], dtype=np.float32)
        if not np.isfinite(batch).all():
            raise ValueError(f"[{split}] NaN/Inf found in embeddings rows {start}:{end}")
        checked_finite += end - start

        if strict_zero_padding:
            for row, length in enumerate(lengths[start:end]):
                if length < pad_len:
                    tail = batch[row, length:, :]
                    if not np.all(tail == 0):
                        raise ValueError(
                            f"[{split}] Non-zero padding found at metadata row {start + row}, "
                            f"length={length}, pad_len={pad_len}"
                        )
                checked_padding += 1

    logger.info(
        "[%s] OK | shape=%s | pad_len=%d | length max=%d | checked rows=%d",
        split,
        tuple(arr.shape),
        pad_len,
        int(lengths.max()),
        checked_finite,
    )
    return {
        "split": split,
        "embedding_path": str(emb_path),
        "metadata_path": str(meta_path),
        "shape": list(arr.shape),
        "n_samples": int(n_samples),
        "pad_len": int(pad_len),
        "embed_dim": int(embed_dim),
        "max_length": int(lengths.max()),
        "min_length": int(lengths.min()),
        "strict_zero_padding": bool(strict_zero_padding),
        "checked_padding_rows": int(checked_padding),
    }


def main():
    args = parse_args()
    embedding_dir = (PROJECT_ROOT / args.embedding_dir).resolve()
    logger = setup_logging(str(embedding_dir / "verify_embeddings.log"))
    reports = [
        verify_split(split, embedding_dir, args.chunk_size, args.strict_zero_padding, logger)
        for split in args.splits
    ]
    pad_lens = {item["pad_len"] for item in reports}
    embed_dims = {item["embed_dim"] for item in reports}
    if len(pad_lens) != 1:
        raise ValueError(f"Splits use different pad_len values: {sorted(pad_lens)}")
    if embed_dims != {EMBED_DIM}:
        raise ValueError(f"Unexpected embed dims: {sorted(embed_dims)}")
    out_path = embedding_dir / "embedding_verification.json"
    out_path.write_text(json.dumps({"splits": reports}, indent=2), encoding="utf-8")
    logger.info("All embedding checks passed. Report saved to %s", out_path)


if __name__ == "__main__":
    main()
