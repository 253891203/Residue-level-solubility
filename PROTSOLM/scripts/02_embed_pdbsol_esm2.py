import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import EMBED_DIM, human_bytes, load_pdbsol_csv, round_up_to, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ESM2 residue embeddings for PDBSol CSV files.")
    parser.add_argument("--train_csv", default="../data/pdbsol/train.csv")
    parser.add_argument("--valid_csv", default="../data/pdbsol/valid.csv")
    parser.add_argument("--test_csv", default="../data/pdbsol/test.csv")
    parser.add_argument("--out_dir", default="outputs/embeddings")
    parser.add_argument("--model_name", default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round_to", type=int, default=100)
    parser.add_argument("--repr_layer", type=int, default=33)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    return parser.parse_args()


def load_esm_model(model_name: str, device: torch.device):
    try:
        import esm
    except ImportError as exc:
        raise ImportError("Please install fair-esm first: pip install fair-esm") from exc
    if not hasattr(esm.pretrained, model_name):
        raise ValueError(f"Unknown ESM model name: {model_name}")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.to(device)
    model.eval()
    return model, alphabet.get_batch_converter()


def write_metadata(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "name", "aa_seq", "length", "label"])
        writer.writeheader()
        writer.writerows(rows)


def compact_memmap(path: Path, written: int, shape: tuple[int, int, int]) -> None:
    if written == shape[0]:
        return
    tmp = path.with_suffix(".compact.npy")
    src = np.load(path, mmap_mode="r")
    dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(written, shape[1], shape[2]))
    chunk = 64
    for start in range(0, written, chunk):
        end = min(start + chunk, written)
        dst[start:end] = src[start:end]
    dst.flush()
    del src, dst
    path.unlink()
    tmp.replace(path)


def embed_split(
    split: str,
    df: pd.DataFrame,
    model,
    batch_converter,
    pad_len: int,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    repr_layer: int,
    use_amp: bool,
    logger,
) -> None:
    out_path = out_dir / f"PDBSol_{split}_embeddings.npy"
    meta_path = out_dir / f"PDBSol_{split}_metadata.csv"
    failed_path = out_dir / f"PDBSol_{split}_failed_samples.csv"
    shape = (len(df), pad_len, EMBED_DIM)
    logger.info("[%s] Writing embeddings: %s | shape=%s | dtype=float16", split, out_path, shape)
    logger.info("[%s] Estimated size: %s", split, human_bytes(len(df) * pad_len * EMBED_DIM * 2))

    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=shape)
    metadata_rows: list[dict] = []
    failed_rows: list[dict] = []
    write_idx = 0

    records = list(df[["name", "aa_seq", "length", "label"]].itertuples(index=False, name=None))
    for start in tqdm(range(0, len(records), batch_size), desc=f"Embedding {split}"):
        batch = records[start : start + batch_size]
        batch_pairs = [(str(name), seq) for name, seq, _, _ in batch]
        try:
            _, _, tokens = batch_converter(batch_pairs)
            tokens = tokens.to(device)
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type, enabled=(use_amp and device.type == "cuda")
            ):
                outputs = model(tokens, repr_layers=[repr_layer], return_contacts=False)
                token_embeds = outputs["representations"][repr_layer]
        except Exception as exc:
            logger.warning("[%s] Batch failed at row %d: %s", split, start, exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch) > 1:
                for row, one in enumerate(batch):
                    name, seq, length, label = one
                    try:
                        _, _, tokens = batch_converter([(str(name), seq)])
                        tokens = tokens.to(device)
                        with torch.inference_mode(), torch.amp.autocast(
                            device_type=device.type, enabled=(use_amp and device.type == "cuda")
                        ):
                            outputs = model(tokens, repr_layers=[repr_layer], return_contacts=False)
                            one_embed = outputs["representations"][repr_layer][0, 1 : int(length) + 1, :].detach().cpu()
                        if tuple(one_embed.shape) != (int(length), EMBED_DIM):
                            raise ValueError(
                                f"Embedding shape {tuple(one_embed.shape)} != ({int(length)}, {EMBED_DIM})"
                            )
                        padded = np.zeros((pad_len, EMBED_DIM), dtype=np.float16)
                        padded[: int(length), :] = one_embed.numpy().astype(np.float16)
                        arr[write_idx] = padded
                        metadata_rows.append(
                            {
                                "index": write_idx,
                                "name": name,
                                "aa_seq": seq,
                                "length": int(length),
                                "label": int(label),
                            }
                        )
                        write_idx += 1
                        del tokens, outputs, one_embed
                    except Exception as single_exc:
                        failed_rows.append(
                            {
                                "index": start + row,
                                "name": name,
                                "length": length,
                                "label": label,
                                "error": str(single_exc),
                            }
                        )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
            name, seq, length, label = batch[0]
            failed_rows.append({"index": start, "name": name, "length": length, "label": label, "error": str(exc)})
            continue

        for row, (name, seq, length, label) in enumerate(batch):
            try:
                residue_embed = token_embeds[row, 1 : int(length) + 1, :].detach().cpu()
                if tuple(residue_embed.shape) != (int(length), EMBED_DIM):
                    raise ValueError(
                        f"Embedding shape {tuple(residue_embed.shape)} != ({int(length)}, {EMBED_DIM})"
                    )
                padded = np.zeros((pad_len, EMBED_DIM), dtype=np.float16)
                padded[: int(length), :] = residue_embed.numpy().astype(np.float16)
                arr[write_idx] = padded
                metadata_rows.append(
                    {
                        "index": write_idx,
                        "name": name,
                        "aa_seq": seq,
                        "length": int(length),
                        "label": int(label),
                    }
                )
                write_idx += 1
            except Exception as exc:
                failed_rows.append(
                    {"index": start + row, "name": name, "length": length, "label": label, "error": str(exc)}
                )
        del tokens, outputs, token_embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    arr.flush()
    del arr
    compact_memmap(out_path, write_idx, shape)
    write_metadata(meta_path, metadata_rows)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
        logger.warning("[%s] Failed samples: %d -> %s", split, len(failed_rows), failed_path)
    logger.info("[%s] Done. Successful samples=%d | metadata=%s", split, write_idx, meta_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir / "embed_pdbsol_esm2.log"))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    csv_paths = {
        "train": PROJECT_ROOT / args.train_csv,
        "valid": PROJECT_ROOT / args.valid_csv,
        "test": PROJECT_ROOT / args.test_csv,
    }
    dfs = {split: load_pdbsol_csv(path, split, logger) for split, path in csv_paths.items()}
    max_len = max(int(df["length"].max()) for df in dfs.values())
    pad_len = round_up_to(max_len, args.round_to)
    logger.info("Global max_len=%d | pad_len=%d", max_len, pad_len)
    (out_dir / "embedding_config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "repr_layer": args.repr_layer,
                "embed_dim": EMBED_DIM,
                "max_len": max_len,
                "pad_len": pad_len,
                "splits": {k: len(v) for k, v in dfs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model, batch_converter = load_esm_model(args.model_name, device)
    for split in ("train", "valid", "test"):
        embed_split(
            split=split,
            df=dfs[split],
            model=model,
            batch_converter=batch_converter,
            pad_len=pad_len,
            out_dir=out_dir,
            device=device,
            batch_size=args.batch_size,
            repr_layer=args.repr_layer,
            use_amp=args.use_amp,
            logger=logger,
        )


if __name__ == "__main__":
    main()
