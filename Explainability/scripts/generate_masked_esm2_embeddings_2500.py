"""Generate test-only, residue-wise masked ESM2 embeddings for the 2500 pipeline.

Each target amino acid is replaced in the input string by ESM's literal
``<mask>`` token before tokenization. Only the test FASTA is processed. As in
the original full-length embedding script, each sequence is passed directly
through the 33-layer ESM2 model without window splitting or stitching.
Embeddings are stored at their valid sequence length; downstream code pads
them to 2500 exactly as required by the classifier.
"""

import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch


DEFAULT_FASTA = "../data/uesolds/test.fasta"
DEFAULT_OUTPUT_DIR = "masked_embeddings"
DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"
TARGET_LENGTH = 2500
EMBED_DIM = 1280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace every test-set residue with literal <mask> before ESM2 "
            "tokenization and save the resulting valid-length embeddings."
        )
    )
    parser.add_argument("--test-fasta", default=DEFAULT_FASTA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--esm-model", default=DEFAULT_ESM_MODEL)
    parser.add_argument(
        "--esm-model-location",
        default="",
        help="Optional local ESM checkpoint; otherwise --esm-model is loaded.",
    )
    parser.add_argument("--repr-layer", type=int, default=33)
    parser.add_argument("--device-ids", default="0,1")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "Safe batch size for a target-length sequence. Shorter sequences "
            "automatically use a larger batch unless --fixed-batch-size is set."
        ),
    )
    parser.add_argument(
        "--max-adaptive-batch-size",
        type=int,
        default=32,
        help="Maximum automatically selected batch size (default: 32).",
    )
    parser.add_argument(
        "--fixed-batch-size",
        action="store_true",
        help="Disable length-aware batching and always use --batch-size.",
    )
    parser.add_argument("--target-length", type=int, default=TARGET_LENGTH)
    parser.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Enable CUDA AMP inference (default: enabled).",
    )
    parser.add_argument(
        "--no-use-amp",
        dest="use_amp",
        action="store_false",
        help="Disable CUDA AMP inference.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--max-masks-per-seq", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--max-estimated-gb",
        type=float,
        default=500.0,
        help="Require --allow-large-output when raw valid-length data exceed this.",
    )
    parser.add_argument(
        "--allow-large-output",
        dest="allow_large_output",
        action="store_true",
        help="Allow the complete masked output even when it exceeds the size warning threshold (default: enabled).",
    )
    parser.add_argument(
        "--enforce-output-size-limit",
        dest="allow_large_output",
        action="store_false",
        help="Stop when the estimated output exceeds --max-estimated-gb.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted output instead of overwriting it.",
    )
    parser.set_defaults(use_amp=True, allow_large_output=True)
    return parser.parse_args()


def read_test_fasta(path: str) -> List[Tuple[str, str]]:
    try:
        from Bio import SeqIO
    except ImportError as exc:
        raise ImportError("Biopython is required: pip install biopython") from exc

    if not os.path.exists(path):
        raise FileNotFoundError(f"Test FASTA not found: {path}")
    records: List[Tuple[str, str]] = []
    for record in SeqIO.parse(path, "fasta"):
        protein_id = record.description.strip()
        sequence = str(record.seq).strip().upper()
        if sequence:
            records.append((protein_id, sequence))
    if not records:
        raise ValueError(f"No sequences found in test FASTA: {path}")
    return records


def parse_device_ids(text: str) -> List[int]:
    device_ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not device_ids:
        raise ValueError("--device-ids cannot be empty.")
    return device_ids


def load_esm(args: argparse.Namespace):
    try:
        import esm
    except ImportError as exc:
        raise ImportError("fair-esm is required: pip install fair-esm") from exc

    if args.esm_model_location:
        model, alphabet = esm.pretrained.load_model_and_alphabet(
            args.esm_model_location
        )
        model_source = args.esm_model_location
    else:
        if args.esm_model not in esm.pretrained.__dict__:
            raise ValueError(f"Unknown esm.pretrained model: {args.esm_model}")
        model, alphabet = esm.pretrained.__dict__[args.esm_model]()
        model_source = args.esm_model

    device_ids = parse_device_ids(args.device_ids)
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        invalid = [device_id for device_id in device_ids if device_id >= available]
        if invalid:
            raise ValueError(
                f"Requested CUDA ids {invalid}, but only {available} devices exist."
            )
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device = torch.device("cpu")
        device_ids = []

    model = model.to(device).eval()
    if device.type == "cuda" and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).eval()
    return model, alphabet, alphabet.get_batch_converter(), device, model_source


def literal_mask(sequence: str, position: int) -> str:
    if position < 0 or position >= len(sequence):
        raise IndexError(position)
    return sequence[:position] + "<mask>" + sequence[position + 1 :]


def embed_masked_strings(
    model,
    batch_converter,
    device: torch.device,
    repr_layer: int,
    items: Sequence[Tuple[str, str, int]],
    use_amp: bool,
) -> List[np.ndarray]:
    converter_items = [(name, masked_sequence) for name, masked_sequence, _ in items]
    _, _, tokens = batch_converter(converter_items)
    tokens = tokens.to(device)
    amp_context = (
        torch.amp.autocast("cuda", enabled=use_amp)
        if device.type == "cuda"
        else nullcontext()
    )
    # DataParallel may try to invoke an ESM2 replica without positional
    # ``tokens`` when the final partial batch is smaller than the GPU count
    # (for example, one remaining mask with GPUs 0 and 1). Run such a tail
    # batch on the primary replica; all full batches still use every GPU.
    forward_model = model
    if isinstance(model, torch.nn.DataParallel) and tokens.shape[0] < len(
        model.device_ids
    ):
        forward_model = model.module
    with torch.inference_mode(), amp_context:
        output = forward_model(tokens, repr_layers=[repr_layer])
        representations = output["representations"][repr_layer]

    arrays: List[np.ndarray] = []
    for index, (_, _, residue_count) in enumerate(items):
        expected_tokens = residue_count + 2
        if int((tokens[index] != 1).sum().item()) < expected_tokens:
            raise RuntimeError(
                "Tokenizer produced fewer tokens than expected; literal <mask> "
                "was not preserved as one ESM token."
            )
        array = representations[index, 1 : residue_count + 1].float().cpu().numpy()
        if array.shape != (residue_count, EMBED_DIM):
            raise RuntimeError(
                f"Unexpected ESM2 embedding shape {array.shape}; "
                f"expected {(residue_count, EMBED_DIM)}"
            )
        arrays.append(array)
    return arrays


def estimate_raw_gb(records: Iterable[Tuple[str, str]], bytes_per_value: int) -> float:
    values = sum(len(sequence) * len(sequence) * EMBED_DIM for _, sequence in records)
    return values * bytes_per_value / (1024**3)


def choose_batch_size(args: argparse.Namespace, sequence_length: int) -> int:
    """Keep roughly the same token load as batch=2 at length 2500."""
    if args.fixed_batch_size:
        return args.batch_size
    token_budget = args.batch_size * args.target_length
    length_aware = max(args.batch_size, token_budget // (sequence_length + 2))
    return min(args.max_adaptive_batch_size, length_aware)


def load_completed(index_path: Path) -> set:
    completed = set()
    if not index_path.exists():
        return completed
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            completed.add((int(row["seq_idx"]), int(row["mask_pos"])))
    return completed


def main() -> None:
    args = parse_args()
    if args.target_length != TARGET_LENGTH:
        raise ValueError(f"This pipeline requires --target-length {TARGET_LENGTH}.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_adaptive_batch_size < args.batch_size:
        raise ValueError(
            "--max-adaptive-batch-size must be greater than or equal to --batch-size."
        )

    records = read_test_fasta(args.test_fasta)
    if args.max_seqs is not None:
        records = records[: args.max_seqs]
    overlength = [
        (protein_id, len(sequence))
        for protein_id, sequence in records
        if len(sequence) > args.target_length
    ]
    if overlength:
        raise ValueError(
            f"{len(overlength)} test sequences exceed {args.target_length}; "
            "refusing to silently truncate them."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "masked_test_esm2_embeddings_2500.h5"
    index_path = output_dir / "masked_test_esm2_index_2500.csv"
    metadata_path = output_dir / "masked_test_esm2_metadata_2500.json"

    bytes_per_value = 2 if args.output_dtype == "float16" else 4
    estimated_gb = estimate_raw_gb(records, bytes_per_value)
    print(f"Test sequences: {len(records)}")
    print(f"Estimated uncompressed valid-length output: {estimated_gb:.2f} GiB")
    if estimated_gb > args.max_estimated_gb and not args.allow_large_output:
        raise RuntimeError(
            f"Estimated output exceeds {args.max_estimated_gb:.1f} GiB. "
            "Check free disk space, then rerun with --allow-large-output."
        )

    if not args.resume and (h5_path.exists() or index_path.exists()):
        raise FileExistsError(
            f"Output already exists. Use --resume or choose another --output-dir: "
            f"{h5_path}"
        )

    completed = load_completed(index_path) if args.resume else set()
    model, alphabet, batch_converter, device, model_source = load_esm(args)
    if getattr(alphabet, "mask_idx", None) is None:
        raise RuntimeError("Loaded ESM alphabet has no <mask> token.")

    dtype = np.float16 if args.output_dtype == "float16" else np.float32
    compression = None if args.compression == "none" else args.compression
    compression_opts = args.gzip_level if compression == "gzip" else None
    h5_mode = "a" if args.resume else "w"
    index_mode = "a" if args.resume else "w"
    write_header = not index_path.exists() or index_path.stat().st_size == 0
    total_written = 0
    started = time.time()

    with h5py.File(h5_path, h5_mode) as h5f, index_path.open(
        index_mode, newline="", encoding="utf-8"
    ) as index_handle:
        h5f.attrs["pipeline"] = "string_mask_before_esm2_2500"
        h5f.attrs["mask_token"] = "<mask>"
        h5f.attrs["target_length"] = TARGET_LENGTH
        h5f.attrs["embedding_dim"] = EMBED_DIM
        h5f.attrs["esm_model"] = model_source
        h5f.attrs["repr_layer"] = args.repr_layer
        h5f.attrs["sequence_processing"] = "direct_full_length_no_windowing"

        writer = csv.DictWriter(
            index_handle,
            fieldnames=[
                "seq_idx",
                "protein_id",
                "seq_len",
                "mask_pos",
                "mask_pos_1based",
                "orig_aa",
                "masked_key",
                "embedding_shape",
                "mask_strategy",
            ],
        )
        if write_header:
            writer.writeheader()

        for seq_idx, (protein_id, sequence) in enumerate(records):
            effective_batch_size = choose_batch_size(args, len(sequence))
            seq_group = h5f.require_group(f"seq_{seq_idx:06d}")
            seq_group.attrs["protein_id"] = protein_id
            seq_group.attrs["seq_len"] = len(sequence)
            seq_group.attrs["mask_strategy"] = "literal_<mask>_before_esm2"
            seq_group.attrs["inference_batch_size"] = effective_batch_size
            positions = list(range(len(sequence)))
            if args.max_masks_per_seq is not None:
                positions = positions[: args.max_masks_per_seq]

            pending: List[Tuple[str, str, int]] = []
            pending_positions: List[int] = []

            def save_pending() -> None:
                nonlocal total_written
                if not pending:
                    return
                arrays = embed_masked_strings(
                    model,
                    batch_converter,
                    device,
                    args.repr_layer,
                    pending,
                    args.use_amp,
                )
                for position, array in zip(pending_positions, arrays):
                    save_variant(position, array)
                pending.clear()
                pending_positions.clear()

            def save_variant(position: int, array: np.ndarray) -> None:
                nonlocal total_written
                orig_aa = sequence[position]
                masked_key = f"mask_{position:04d}_{orig_aa}_to_esm_mask"
                if masked_key in seq_group:
                    del seq_group[masked_key]
                dataset = seq_group.create_dataset(
                    masked_key,
                    data=array.astype(dtype, copy=False),
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=compression is not None,
                )
                dataset.attrs["mask_pos"] = position
                dataset.attrs["orig_aa"] = orig_aa
                writer.writerow(
                    {
                        "seq_idx": seq_idx,
                        "protein_id": protein_id,
                        "seq_len": len(sequence),
                        "mask_pos": position,
                        "mask_pos_1based": position + 1,
                        "orig_aa": orig_aa,
                        "masked_key": masked_key,
                        "embedding_shape": f"{array.shape[0]}x{array.shape[1]}",
                        "mask_strategy": "literal_<mask>_before_esm2",
                    }
                )
                total_written += 1
                if total_written % args.print_every == 0:
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        f"Written {total_written} variants "
                        f"({total_written / elapsed:.2f} variants/s)"
                    )

            for position in positions:
                if (seq_idx, position) in completed:
                    continue
                pending.append(
                    (
                        f"{protein_id}|mask={position}",
                        literal_mask(sequence, position),
                        len(sequence),
                    )
                )
                pending_positions.append(position)
                if len(pending) >= effective_batch_size:
                    save_pending()
            save_pending()
            index_handle.flush()
            h5f.flush()
            print(
                f"Completed test sequence {seq_idx + 1}/{len(records)}: "
                f"{protein_id[:60]} (L={len(sequence)}, "
                f"batch={effective_batch_size})"
            )

    metadata: Dict[str, object] = {
        "pipeline": "string_mask_before_esm2_2500",
        "test_only": True,
        "test_fasta": args.test_fasta,
        "test_sequence_count": len(records),
        "mask_token": "<mask>",
        "mask_strategy": "literal_<mask>_before_tokenization",
        "esm_model": model_source,
        "repr_layer": args.repr_layer,
        "target_length": TARGET_LENGTH,
        "embedding_dim": EMBED_DIM,
        "sequence_processing": "direct_full_length_no_windowing",
        "adaptive_batching": not args.fixed_batch_size,
        "base_batch_size": args.batch_size,
        "max_adaptive_batch_size": args.max_adaptive_batch_size,
        "output_dtype": args.output_dtype,
        "estimated_uncompressed_gib": estimated_gb,
        "h5_path": str(h5_path),
        "index_csv": str(index_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Masked ESM2 H5: {h5_path}")
    print(f"Masked index CSV: {index_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
