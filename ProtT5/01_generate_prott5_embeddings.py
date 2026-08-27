import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "uesolds"
DEFAULT_MODEL_PATH = "Rostlab/prot_t5_xl_half_uniref50-enc"
DEFAULT_MODEL_TAG = "prott5_uniref50"
DEFAULT_FASTAS = {
    "train": DATA_DIR / "train.fasta",
    "validation": DATA_DIR / "validation.fasta",
    "test": DATA_DIR / "test.fasta",
}
FASTA_FALLBACKS = {"train": [], "validation": [], "test": []}


def parse_label(description: str) -> int:
    parts = description.split()
    for part in reversed(parts):
        if "-" in part:
            label = part.rsplit("-", 1)[-1]
            if label in {"0", "1"}:
                return int(label)
    raise ValueError(f"Cannot parse binary solubility label from FASTA header: {description}")


def clean_sequence(sequence: str) -> str:
    sequence = re.sub(r"\s+", "", sequence).upper()
    return re.sub(r"[UZOB]", "X", sequence)


def spaced_sequence(sequence: str) -> str:
    return " ".join(list(sequence))


def ensure_sentencepiece_available() -> None:
    try:
        import sentencepiece  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ProtT5 uses T5Tokenizer and requires sentencepiece. Install it with:\n"
            "python -m pip install sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc


def get_transformers():
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise ImportError(
            "ProtT5 embedding requires transformers. Install it with:\n"
            "python -m pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc
    return T5Tokenizer, T5EncoderModel


def get_seqio():
    try:
        from Bio import SeqIO
    except ImportError as exc:
        raise ImportError(
            "Reading PLMSol FASTA files requires biopython. Install it with:\n"
            "python -m pip install biopython -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc
    return SeqIO


def resolve_fasta_path(split_name: str, path: Path) -> Path:
    if path.exists():
        return path
    for candidate in FASTA_FALLBACKS.get(split_name, []):
        if candidate.exists():
            print(f"{split_name} FASTA not found at {path}; using fallback: {candidate}")
            return candidate
    candidates = [path] + FASTA_FALLBACKS.get(split_name, [])
    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Cannot find {split_name} FASTA. Checked:\n{candidate_text}\n"
        "Use --train_fasta/--validation_fasta/--test_fasta or set PLM_BASE_DIR."
    )


def read_fasta(path: Path, max_length: int, skip_long: bool) -> List[Tuple[str, str, int]]:
    SeqIO = get_seqio()
    records: List[Tuple[str, str, int]] = []
    skipped_long = 0
    for record in SeqIO.parse(str(path), "fasta"):
        sequence = clean_sequence(str(record.seq))
        if not sequence:
            continue
        if len(sequence) > max_length:
            if skip_long:
                skipped_long += 1
                continue
            sequence = sequence[:max_length]
        records.append((record.description, sequence, parse_label(record.description)))
    print(f"{path}: kept={len(records)}, skipped_long={skipped_long}")
    return records


@torch.inference_mode()
def embed_split(
    split_name: str,
    records: List[Tuple[str, str, int]],
    tokenizer,
    model,
    output_path: Path,
    model_name_or_path: str,
    model_tag: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    save_fp16: bool,
) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if save_fp16 else np.float32
    embedding_dim = int(model.config.d_model)

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["split"] = split_name
        h5f.attrs["max_length"] = max_length
        h5f.attrs["embedding_dim"] = embedding_dim
        h5f.attrs["model"] = "ProtT5"
        h5f.attrs["model_name_or_path"] = model_name_or_path
        h5f.attrs["model_tag"] = model_tag
        h5f.attrs["h5_key_format"] = f"{split_name}_00000000"

        for batch_start in tqdm(range(0, len(records), batch_size), desc=f"Embedding {split_name}"):
            batch = records[batch_start : batch_start + batch_size]
            ids = [item[0] for item in batch]
            sequences = [item[1] for item in batch]
            labels = [item[2] for item in batch]
            tokenized = tokenizer(
                [spaced_sequence(seq) for seq in sequences],
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            outputs = model(**tokenized).last_hidden_state.detach().cpu()

            for idx, (protein_id, sequence, label) in enumerate(zip(ids, sequences, labels)):
                sample_idx = batch_start + idx
                h5_key = f"{split_name}_{sample_idx:08d}"
                seq_len = min(len(sequence), max_length)
                residue_embedding = outputs[idx, :seq_len].float().numpy()
                if residue_embedding.shape[-1] != embedding_dim:
                    embedding_dim = int(residue_embedding.shape[-1])
                    h5f.attrs["embedding_dim"] = embedding_dim
                padded = np.zeros((max_length, residue_embedding.shape[-1]), dtype=dtype)
                padded[:seq_len] = residue_embedding.astype(dtype, copy=False)
                ds = h5f.create_dataset(
                    h5_key,
                    data=padded,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                ds.attrs["label"] = int(label)
                ds.attrs["length"] = int(seq_len)
                ds.attrs["sequence"] = sequence[:seq_len]
                ds.attrs["protein_id"] = protein_id
                ds.attrs["fasta_header"] = protein_id
                ds.attrs["source_index"] = int(sample_idx)
                ds.attrs["residue_embedding_shape"] = f"{seq_len}x{residue_embedding.shape[-1]}"

    return {
        "split": split_name,
        "records": len(records),
        "path": str(output_path),
        "model_name_or_path": model_name_or_path,
        "model_tag": model_tag,
        "embedding_dim": embedding_dim,
        "residue_embedding": f"Lx{embedding_dim}",
        "stored_shape": f"{max_length}x{embedding_dim}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ProtT5 residue embeddings for PLMSol FASTA splits.")
    parser.add_argument("--train_fasta", "--train-fasta", dest="train_fasta", default=str(DEFAULT_FASTAS["train"]))
    parser.add_argument("--validation_fasta", "--validation-fasta", dest="validation_fasta", default=str(DEFAULT_FASTAS["validation"]))
    parser.add_argument("--test_fasta", "--test-fasta", dest="test_fasta", default=str(DEFAULT_FASTAS["test"]))
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default=f"embeddings/plmsol_{DEFAULT_MODEL_TAG}")
    parser.add_argument("--model_name_or_path", "--model-name-or-path", dest="model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_tag", "--model-tag", dest="model_tag", default=DEFAULT_MODEL_TAG)
    parser.add_argument("--max_length", "--max-length", dest="max_length", type=int, default=500)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--local_files_only",
        "--local-files-only",
        dest="local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load tokenizer/model from local files only. Default: False.",
    )
    parser.add_argument("--skip_long", "--skip-long", dest="skip_long", action="store_true", help="Drop sequences longer than --max_length instead of truncating.")
    parser.add_argument("--save_fp16", "--save-fp16", dest="save_fp16", action="store_true", help="Store embeddings as float16 to reduce disk usage.")
    parser.add_argument("--half", action="store_true", help="Run ProtT5 in fp16 on CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_sentencepiece_available()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": resolve_fasta_path("train", Path(args.train_fasta)),
        "validation": resolve_fasta_path("validation", Path(args.validation_fasta)),
        "test": resolve_fasta_path("test", Path(args.test_fasta)),
    }
    print("PLMSol FASTA paths:")
    for split_name, fasta_path in split_paths.items():
        print(f"  {split_name}: {fasta_path}")

    print(f"Loading ProtT5 from local path: {args.model_name_or_path}")
    print(f"model_tag: {args.model_tag}")
    print(f"local_files_only: {args.local_files_only}")
    print("ProtT5 amino-acid sequence mode: U/Z/O/B -> X, then space-separated residues. No sequence-structure prefixes are used.")
    if os.environ.get("HF_ENDPOINT") and not args.local_files_only:
        print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    T5Tokenizer, T5EncoderModel = get_transformers()
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False, local_files_only=args.local_files_only)
    model = T5EncoderModel.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    model.eval().to(device)
    if args.half and device.type == "cuda":
        model.half()

    summary = []
    for split_name, fasta_path in split_paths.items():
        records = read_fasta(fasta_path, max_length=args.max_length, skip_long=args.skip_long)
        out_path = output_dir / f"{args.model_tag}_{split_name}_maxlen{args.max_length}.h5"
        summary.append(
            embed_split(
                split_name=split_name,
                records=records,
                tokenizer=tokenizer,
                model=model,
                output_path=out_path,
                model_name_or_path=args.model_name_or_path,
                model_tag=args.model_tag,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
                save_fp16=args.save_fp16,
            )
        )

    summary_path = output_dir / f"{args.model_tag}_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Embedding summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
