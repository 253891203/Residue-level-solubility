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


def parse_device_ids(text):
    return [int(item) for item in text.split(",") if item.strip()]


def generate_split(args, split, model, alphabet, device, device_ids):
    all_records = read_fasta(args.data_dir / f"{split}.fasta")
    records = [(identifier, sequence) for identifier, sequence in all_records if len(sequence) <= args.max_length]
    output_path = args.output_dir / f"esm2-500t-{split}_streamv3_maxlen500.h5"
    failure_path = args.output_dir / f"{split}_embedding_failures.csv"
    batch_converter = alphabet.get_batch_converter()
    dtype = np.float16 if args.fp16 else np.float32
    failures = []
    written = 0

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        id_ds = handle.create_dataset("id", (len(records),), maxshape=(None,), dtype=string_dtype)
        length_ds = handle.create_dataset("seq_len", (len(records),), maxshape=(None,), dtype="int32")
        embedding_ds = handle.create_dataset(
            "residue_embed",
            (len(records), args.max_length, args.embed_dim),
            maxshape=(None, args.max_length, args.embed_dim),
            chunks=(1, args.max_length, args.embed_dim),
            compression="lzf",
            shuffle=True,
            dtype=dtype,
        )

        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            active_model = model
            if isinstance(model, torch.nn.DataParallel) and len(batch) < len(device_ids):
                active_model = model.module
            try:
                with torch.inference_mode(), torch.amp.autocast("cuda", enabled=args.fp16):
                    output = active_model(tokens=tokens, repr_layers=[args.repr_layer])
                token_embeddings = output["representations"][args.repr_layer]
            except Exception as error:
                failures.extend((identifier, len(sequence), str(error)) for identifier, sequence in batch)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for index, (identifier, sequence) in enumerate(batch):
                length = len(sequence)
                residue_embedding = token_embeddings[index, 1 : length + 1].detach().cpu()
                padded = torch.zeros((args.max_length, args.embed_dim), dtype=residue_embedding.dtype)
                padded[:length] = residue_embedding
                id_ds[written] = identifier
                length_ds[written] = length
                embedding_ds[written] = padded.numpy().astype(dtype, copy=False)
                written += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            processed = min(start + args.batch_size, len(records))
            if processed % args.progress_every < args.batch_size:
                print(f"{split}: embedded {processed}/{len(records)}", flush=True)

        id_ds.resize((written,))
        length_ds.resize((written,))
        embedding_ds.resize((written, args.max_length, args.embed_dim))
        handle.attrs["model_name"] = args.model_name
        handle.attrs["repr_layer"] = args.repr_layer
        handle.attrs["max_length"] = args.max_length

    with failure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "sequence_length", "error"])
        writer.writerows(failures)
    print(
        f"{split}: source={len(all_records)}, retained_length_le_{args.max_length}={len(records)}, "
        f"written={written}, failures={len(failures)}"
    )
    if failures and not args.skip_failed:
        raise RuntimeError(f"{split}: embedding failures occurred; see {failure_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed-length ESM2-650M residue embeddings for the representation experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../data/uesolds"))
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--model-name", default="esm2_t33_650M_UR50D")
    parser.add_argument("--repr-layer", type=int, default=33)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--embed-dim", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device-ids", default="0,1")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--skip-failed", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical ESM2-650M embedding generation.")
    device_ids = parse_device_ids(args.device_ids)
    if not device_ids or max(device_ids) >= torch.cuda.device_count():
        raise ValueError(f"Unavailable CUDA device list: {device_ids}")

    import esm

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{device_ids[0]}")
    model, alphabet = esm.pretrained.__dict__[args.model_name]()
    model = model.to(device)
    if args.fp16:
        model = model.half()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    for split in args.splits:
        generate_split(args, split, model, alphabet, device, device_ids)


if __name__ == "__main__":
    main()
