import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch


SPLITS = ("train", "validation", "test")


def read_fasta(path):
    records = []
    description = None
    sequence_lines = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">"):
                if description is not None:
                    sequence = "".join(sequence_lines)
                    if sequence:
                        records.append((description, sequence))
                description = line[1:].strip()
                sequence_lines = []
            elif line:
                sequence_lines.append(line)
    if description is not None:
        sequence = "".join(sequence_lines)
        if sequence:
            records.append((description, sequence))
    return records


def round_up(length, multiple):
    return ((length + multiple - 1) // multiple) * multiple


def parse_device_ids(value):
    return [int(item) for item in value.split(",") if item.strip()]


def legacy_shard_audit(records, batch_size=2, shard_size=1000):
    """Simulate the dictionary-key behavior of the original embedding code."""
    shard = {}
    shard_counts = []
    overwritten = 0
    for start in range(0, len(records), batch_size):
        for protein_id, sequence in records[start : start + batch_size]:
            if protein_id in shard:
                overwritten += 1
            shard[protein_id] = sequence
        if len(shard) >= shard_size:
            shard_counts.append(len(shard))
            shard = {}
    if shard:
        shard_counts.append(len(shard))
    return {
        "fasta_records": len(records),
        "shard_count": len(shard_counts),
        "effective_records": sum(shard_counts),
        "overwritten_within_shards": overwritten,
    }


def save_shard(shard, shard_index, split_dir):
    path = split_dir / f"shard_{shard_index:04d}.pt"
    torch.save(shard, path)
    return path


def merge_shards(shard_paths, output_path, max_len, embed_dim):
    total_count = 0
    storage_dtype = None
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        total_count += len(shard)
        if storage_dtype is None and shard:
            first = next(iter(shard.values()))["residue_embed"]
            storage_dtype = np.float16 if first.dtype == torch.float16 else np.float32
    if total_count == 0:
        raise RuntimeError(f"No embeddings were generated for {output_path.stem}")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        id_ds = handle.create_dataset("id", (total_count,), dtype=string_dtype)
        length_ds = handle.create_dataset("seq_len", (total_count,), dtype="int32")
        embedding_ds = handle.create_dataset(
            "residue_embed",
            (total_count, max_len, embed_dim),
            chunks=(1, max_len, embed_dim),
            compression="gzip",
            compression_opts=4,
            dtype=storage_dtype,
        )
        offset = 0
        for path in shard_paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            for protein_id, item in shard.items():
                embedding = item["residue_embed"]
                sequence_length = int(item["seq_len"])
                padded = torch.zeros((max_len, embed_dim), dtype=embedding.dtype)
                copy_length = min(len(embedding), max_len)
                padded[:copy_length] = embedding[:copy_length]
                id_ds[offset] = protein_id
                length_ds[offset] = sequence_length
                embedding_ds[offset] = padded.numpy().astype(storage_dtype, copy=False)
                offset += 1
        handle.attrs["max_len"] = max_len
        handle.attrs["embed_dim"] = embed_dim
        handle.attrs["legacy_dictionary_shards"] = True
    return total_count


def generate_split(args, split, model, alphabet, device, device_ids):
    records = read_fasta(args.data_dir / f"{split}.fasta")
    padded_length = round_up(max(len(sequence) for _, sequence in records), 100)
    split_dir = args.output_dir / "shards" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    batch_converter = alphabet.get_batch_converter()

    shard = {}
    shard_paths = []
    shard_index = 0
    failures = []
    overwritten = 0

    def embed_batch(batch):
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        active_model = model
        if isinstance(model, torch.nn.DataParallel) and len(batch) < len(device_ids):
            active_model = model.module
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=args.fp16):
            output = active_model(tokens=tokens, repr_layers=[args.repr_layer])
        return output["representations"][args.repr_layer]

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        try:
            token_embeddings = embed_batch(batch)
        except Exception as error:
            failures.extend((protein_id, len(sequence), str(error)) for protein_id, sequence in batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        for index, (protein_id, sequence) in enumerate(batch):
            sequence_length = len(sequence)
            if sequence_length > args.max_len:
                raise ValueError(
                    f"{protein_id}: sequence length {sequence_length} exceeds {args.max_len}"
                )
            residue_embedding = token_embeddings[index, 1 : sequence_length + 1].detach().cpu()
            residue_embedding = residue_embedding.half() if args.fp16 else residue_embedding.float()
            padded = torch.zeros(
                (padded_length, args.embed_dim), dtype=residue_embedding.dtype
            )
            padded[:sequence_length] = residue_embedding
            if protein_id in shard:
                overwritten += 1
            shard[protein_id] = {
                "residue_embed": padded,
                "seq_len": sequence_length,
                "padded_len": padded_length,
            }

        if len(shard) >= args.shard_size:
            shard_paths.append(save_shard(shard, shard_index, split_dir))
            shard = {}
            shard_index += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        processed = min(start + args.batch_size, len(records))
        if processed % args.progress_every < args.batch_size:
            print(f"{split}: processed {processed}/{len(records)} FASTA records", flush=True)

    if shard:
        shard_paths.append(save_shard(shard, shard_index, split_dir))

    manifest_path = args.output_dir / f"{split}_legacy_shards.txt"
    manifest_path.write_text(
        "".join(f"{path.relative_to(args.output_dir)}\n" for path in shard_paths),
        encoding="utf-8",
    )
    failure_path = args.output_dir / f"{split}_embedding_failures.csv"
    with failure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "sequence_length", "error"])
        writer.writerows(failures)

    output_h5 = args.output_dir / f"{split}_esm2_650m_maxlen2500.h5"
    effective_count = merge_shards(
        shard_paths, output_h5, args.max_len, args.embed_dim
    )
    print(
        f"{split}: FASTA={len(records)}, effective={effective_count}, "
        f"within-shard overwrites={overwritten}, failures={len(failures)}"
    )
    if failures and args.fail_on_embedding_error:
        raise RuntimeError(f"{len(failures)} records failed; see {failure_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate legacy-compatible full-residue ESM2-650M embeddings."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../data/uesolds"))
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--model-name", default="esm2_t33_650M_UR50D")
    parser.add_argument("--repr-layer", type=int, default=33)
    parser.add_argument("--max-len", type=int, default=2500)
    parser.add_argument("--embed-dim", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--device-ids", default="0,1")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--fail-on-embedding-error", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Reproduce legacy sample counts without loading ESM2 or generating embeddings.",
    )
    args = parser.parse_args()

    if args.audit_only:
        for split in args.splits:
            records = read_fasta(args.data_dir / f"{split}.fasta")
            report = legacy_shard_audit(records, args.batch_size, args.shard_size)
            print(f"{split}: {report}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical ESM2-650M embedding generation.")
    device_ids = parse_device_ids(args.device_ids)
    if not device_ids or max(device_ids) >= torch.cuda.device_count():
        raise ValueError(f"Unavailable CUDA device list: {device_ids}")

    import esm

    device = torch.device(f"cuda:{device_ids[0]}")
    model, alphabet = esm.pretrained.__dict__[args.model_name]()
    model = model.to(device)
    if args.fp16:
        model = model.half()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        generate_split(args, split, model, alphabet, device, device_ids)


if __name__ == "__main__":
    main()
