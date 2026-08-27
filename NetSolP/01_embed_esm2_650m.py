from __future__ import annotations

import argparse
import csv
import contextlib
import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from utils.data_utils import embedding_path, length_stats, load_netsolp_csv, resolve_path, set_seed, write_json


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen ESM2-650M residue embeddings for NetSolP data.")
    parser.add_argument("--train_csv", default="../data/netsolp/PSI_Biology_solubility_trainset.csv")
    parser.add_argument("--test_csv", default="../data/netsolp/NESG_testset.csv")
    parser.add_argument("--out_dir", default="outputs/results")
    parser.add_argument("--embedding_dir", default="embeddings")
    parser.add_argument(
        "--model_name",
        default="esm2_t33_650M_UR50D",
        help="Local model name/path. Defaults to fair-esm local/cache name; no HuggingFace lookup is used.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "fair_esm", "transformers"],
        default="auto",
        help="Default auto loads local fair-esm cache/file first, then local HuggingFace-format directories.",
    )
    parser.add_argument("--local_files_only", action="store_true", default=True, help="Offline mode; enabled by default.")
    parser.add_argument("--allow_online", action="store_false", dest="local_files_only", help="Allow online lookup/download.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Dynamic batch token budget.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--truncate_long_sequences", action="store_true", default=False)
    parser.add_argument("--max_length", type=int, default=1022, help="Only used when --truncate_long_sequences is set.")
    return parser.parse_args()


def force_offline_if_needed(local_files_only: bool) -> None:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def normalize_fair_esm_name(model_name: str) -> str:
    if model_name == "facebook/esm2_t33_650M_UR50D":
        return "esm2_t33_650M_UR50D"
    return model_name.split("/")[-1]


def torch_checkpoint_candidates(model_name: str) -> list[Path]:
    fair_name = normalize_fair_esm_name(model_name)
    candidates = []
    for env_name in ("TORCH_HOME", "XDG_CACHE_HOME"):
        base = os.environ.get(env_name)
        if base:
            base_path = Path(base)
            candidates.extend(
                [
                    base_path / "hub" / "checkpoints" / f"{fair_name}.pt",
                    base_path / "torch" / "hub" / "checkpoints" / f"{fair_name}.pt",
                ]
            )
    home = Path.home()
    candidates.extend(
        [
            home / ".cache" / "torch" / "hub" / "checkpoints" / f"{fair_name}.pt",
            ROOT / "models" / f"{fair_name}.pt",
            ROOT / "pretrained_models" / f"{fair_name}.pt",
            ROOT / f"{fair_name}.pt",
        ]
    )
    return candidates


def local_model_candidates(model_name: str) -> list[Path]:
    name_path = Path(model_name).expanduser()
    fair_name = normalize_fair_esm_name(model_name)
    return [
        name_path,
        ROOT / model_name,
        ROOT / "models" / model_name,
        ROOT / "pretrained_models" / model_name,
        ROOT / "models" / fair_name,
        ROOT / "pretrained_models" / fair_name,
        ROOT / fair_name,
    ]


def find_local_transformers_dir(model_name: str) -> Path | None:
    for path in local_model_candidates(model_name):
        if path.is_dir() and (path / "config.json").exists():
            return path
    return None


def find_local_fair_esm_file(model_name: str) -> Path | None:
    for path in local_model_candidates(model_name):
        if path.is_file() and path.suffix == ".pt":
            return path
        if path.is_dir():
            for candidate in (path / f"{normalize_fair_esm_name(model_name)}.pt", path / "model.pt"):
                if candidate.exists():
                    return candidate
    for path in torch_checkpoint_candidates(model_name):
        if path.exists():
            return path
    return None


@contextlib.contextmanager
def torch_load_weights_only_false():
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def choose_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    if find_local_fair_esm_file(model_name) is not None:
        return "fair_esm"
    if find_local_transformers_dir(model_name) is not None:
        return "transformers"
    if normalize_fair_esm_name(model_name) == "esm2_t33_650M_UR50D":
        return "fair_esm"
    return "transformers"


def load_model_backend(model_name: str, backend: str, device: torch.device, local_files_only: bool):
    force_offline_if_needed(local_files_only)
    backend = choose_backend(model_name, backend)
    if backend == "fair_esm":
        try:
            import esm
        except ImportError as exc:
            raise ImportError("Please install fair-esm for the fair_esm backend: pip install fair-esm") from exc
        fair_name = normalize_fair_esm_name(model_name)
        local_pt = find_local_fair_esm_file(model_name)
        if local_files_only and local_pt is None:
            searched = [str(p) for p in local_model_candidates(model_name) + torch_checkpoint_candidates(model_name)]
            raise FileNotFoundError(
                "Offline mode is enabled, but no local fair-esm ESM2 checkpoint was found. "
                "Put esm2_t33_650M_UR50D.pt under ./models/, ./pretrained_models/, or torch hub cache. "
                f"Searched paths: {searched}"
            )
        if local_pt is not None:
            with torch_load_weights_only_false():
                model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(local_pt))
        elif not hasattr(esm.pretrained, fair_name):
            raise ValueError(f"Unknown fair-esm model name: {fair_name}")
        else:
            model, alphabet = getattr(esm.pretrained, fair_name)()
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return {
            "backend": "fair_esm",
            "model": model,
            "batch_converter": alphabet.get_batch_converter(),
            "model_source": str(local_pt) if local_pt is not None else fair_name,
        }

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Please install transformers for the transformers backend: pip install transformers") from exc
    local_dir = find_local_transformers_dir(model_name)
    if local_files_only and local_dir is None:
        searched = [str(p) for p in local_model_candidates(model_name)]
        raise FileNotFoundError(
            "Offline mode is enabled, but no local HuggingFace-format ESM2 directory with config.json was found. "
            f"Searched paths: {searched}"
        )
    source = str(local_dir if local_dir is not None else model_name)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(source, local_files_only=local_files_only).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"backend": "transformers", "model": model, "tokenizer": tokenizer, "model_source": source}


def make_batches(rows: list[tuple[int, str, str, int, int]], max_tokens: int) -> list[list[tuple[int, str, str, int, int]]]:
    batches = []
    current = []
    current_max = 0
    for row in sorted(rows, key=lambda x: x[3]):
        length = row[3]
        proposed_max = max(current_max, length + 2)
        if current and proposed_max * (len(current) + 1) > max_tokens:
            batches.append(current)
            current = []
            current_max = 0
        current.append(row)
        current_max = max(current_max, length + 2)
    if current:
        batches.append(current)
    return batches


def embed_split(
    split: str,
    df: pd.DataFrame,
    model_bundle: dict,
    embedding_dir: Path,
    device: torch.device,
    max_tokens: int,
    use_amp: bool,
    truncate_long_sequences: bool,
    max_length: int,
) -> pd.DataFrame:
    split_dir = embedding_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        (i, str(r.sample_id), str(r.sequence), int(r.length), int(r.label))
        for i, r in enumerate(df.itertuples(index=False))
    ]
    pending = [row for row in rows if not embedding_path(embedding_dir, split, row[0]).exists()]
    batches = make_batches(pending, max_tokens)

    for batch in tqdm(batches, desc=f"Embedding {split}"):
        row_indices, sample_ids, seqs, lengths, labels = zip(*batch)
        original_seqs = list(seqs)
        if truncate_long_sequences:
            seqs = tuple(seq[:max_length] for seq in seqs)
            lengths = tuple(min(length, max_length) for length in lengths)
        try:
            if model_bundle["backend"] == "fair_esm":
                _, _, tokens = model_bundle["batch_converter"](list(zip(sample_ids, seqs)))
                tokens = tokens.to(device)
                with torch.no_grad(), torch.amp.autocast(
                    device_type=device.type, enabled=use_amp and device.type == "cuda"
                ):
                    outputs = model_bundle["model"](tokens, repr_layers=[33], return_contacts=False)
                hidden = outputs["representations"][33].detach().cpu()
            else:
                tokens = model_bundle["tokenizer"](
                    list(seqs),
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    add_special_tokens=True,
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                with torch.no_grad(), torch.amp.autocast(
                    device_type=device.type, enabled=use_amp and device.type == "cuda"
                ):
                    outputs = model_bundle["model"](**tokens)
                hidden = outputs.last_hidden_state.detach().cpu()
            for pos, row_index in enumerate(row_indices):
                residue = hidden[pos, 1 : lengths[pos] + 1, :].contiguous().to(torch.float16)
                if residue.shape[0] != lengths[pos] or residue.shape[1] != 1280:
                    raise RuntimeError(
                        f"Unexpected embedding shape for {sample_ids[pos]}: {tuple(residue.shape)}; "
                        f"expected ({lengths[pos]}, 1280)"
                    )
                torch.save(residue, embedding_path(embedding_dir, split, row_index))
        except Exception as exc:
            details = [
                f"{sample_ids[i]}(row={row_indices[i]}, length={len(original_seqs[i])})"
                for i in range(len(batch))
            ]
            raise RuntimeError(
                f"ESM2 embedding failed for split={split}, sequences={details}. "
                f"No sequence was filtered or truncated automatically. Original error: {exc}"
            ) from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    meta = pd.DataFrame(
        {
            "split": split,
            "row_index": range(len(df)),
            "sample_id": df["sample_id"],
            "label": df["label"],
            "length": df["length"],
        }
    )
    meta.to_csv(embedding_dir / f"{split}_metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return meta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    force_offline_if_needed(args.local_files_only)
    out_dir = resolve_path(ROOT, args.out_dir)
    embedding_dir = resolve_path(ROOT, args.embedding_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_info = load_netsolp_csv(resolve_path(ROOT, args.train_csv), "train")
    test_df, test_info = load_netsolp_csv(resolve_path(ROOT, args.test_csv), "nesg")
    write_json(out_dir / "sequence_length_stats.json", {"train": length_stats(train_df), "NESG_test": length_stats(test_df)})

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_bundle = load_model_backend(args.model_name, args.backend, device, args.local_files_only)

    train_meta = embed_split(
        "train",
        train_df,
        model_bundle,
        embedding_dir,
        device,
        args.max_tokens,
        args.use_amp,
        args.truncate_long_sequences,
        args.max_length,
    )
    test_meta = embed_split(
        "nesg",
        test_df,
        model_bundle,
        embedding_dir,
        device,
        args.max_tokens,
        args.use_amp,
        args.truncate_long_sequences,
        args.max_length,
    )
    train_meta.to_csv(embedding_dir / "train_metadata.csv", index=False)
    test_meta.to_csv(embedding_dir / "nesg_metadata.csv", index=False)
    (embedding_dir / "embedding_config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "backend": model_bundle["backend"],
                "model_source": model_bundle.get("model_source"),
                "local_files_only": bool(args.local_files_only),
                "embedding_dim": 1280,
                "truncate_long_sequences": bool(args.truncate_long_sequences),
                "max_length_if_truncated": args.max_length,
                "train_info": train_info,
                "NESG_test_info": test_info,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("Embedding cache complete:", embedding_dir)


if __name__ == "__main__":
    main()
