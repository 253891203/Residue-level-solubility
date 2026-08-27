#!/usr/bin/env python3
"""Prepare unique protein queries for offline UniProt BLAST mapping.

The event CSV may contain many masked positions for the same protein. This
script writes each unique protein once, while retaining event counts and the
largest absolute sigmoid shift in the metadata CSV.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts).upper().rstrip("*")))
                header = line[1:].strip()
                sequence_parts = []
            else:
                sequence_parts.append(line)
    if header is not None:
        records.append((header, "".join(sequence_parts).upper().rstrip("*")))
    return records


def wrap_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[start : start + width] for start in range(0, len(sequence), width))


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_events(csv_path: Path, protein_col: str) -> tuple[OrderedDict[str, dict[str, object]], int]:
    proteins: OrderedDict[str, dict[str, object]] = OrderedDict()
    event_rows = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if protein_col not in fields:
            raise ValueError(f"Column {protein_col!r} not found in {csv_path}")
        for row_number, row in enumerate(reader, 1):
            event_rows += 1
            protein_id = (row.get(protein_col) or "").strip()
            if not protein_id:
                raise ValueError(f"Empty protein ID at CSV data row {row_number}")
            item = proteins.setdefault(
                protein_id,
                {
                    "protein_id": protein_id,
                    "label": (row.get("label") or "").strip(),
                    "transition": (row.get("transition") or "").strip(),
                    "event_count": 0,
                    "max_abs_delta": None,
                    "representative_mask_pos_1based": "",
                    "representative_orig_aa": "",
                    "first_event_row": row_number,
                },
            )
            item["event_count"] = int(item["event_count"]) + 1
            abs_delta = safe_float(row.get("abs_delta") or "")
            current_max = item["max_abs_delta"]
            if abs_delta is not None and (current_max is None or abs_delta > current_max):
                item["max_abs_delta"] = abs_delta
                item["representative_mask_pos_1based"] = (
                    row.get("mask_pos_1based") or row.get("mask_pos") or ""
                )
                item["representative_orig_aa"] = row.get("orig_aa") or ""
    return proteins, event_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique structure-query proteins from a masked-transition event CSV."
    )
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--source-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="structure_queries")
    parser.add_argument("--protein-col", default="protein_id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.events_csv, args.source_fasta):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    proteins, event_rows = load_events(args.events_csv, args.protein_col)
    fasta_records = read_fasta(args.source_fasta)
    records_by_header: dict[str, tuple[int, str]] = {}
    duplicate_headers: list[str] = []
    for fasta_index, (header, sequence) in enumerate(fasta_records):
        if header in records_by_header and records_by_header[header][1] != sequence:
            duplicate_headers.append(header)
        records_by_header.setdefault(header, (fasta_index, sequence))
    if duplicate_headers:
        raise ValueError(
            "Source FASTA contains duplicate headers with conflicting sequences: "
            + ", ".join(duplicate_headers[:10])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = args.output_dir / f"{args.output_prefix}.fasta"
    metadata_path = args.output_dir / f"{args.output_prefix}_metadata.csv"
    missing_path = args.output_dir / f"{args.output_prefix}_missing.csv"

    matched_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    with fasta_path.open("w", encoding="utf-8", newline="\n") as fasta_out:
        for protein_id, item in proteins.items():
            match = records_by_header.get(protein_id)
            if match is None:
                missing_rows.append(
                    {
                        "protein_id": protein_id,
                        "event_count": item["event_count"],
                        "first_event_row": item["first_event_row"],
                    }
                )
                continue
            fasta_index, sequence = match
            fasta_out.write(f">{protein_id}\n{wrap_sequence(sequence)}\n")
            matched_rows.append(
                {
                    **item,
                    "fasta_index": fasta_index,
                    "sequence_length": len(sequence),
                }
            )

    metadata_fields = [
        "protein_id",
        "label",
        "transition",
        "event_count",
        "max_abs_delta",
        "representative_mask_pos_1based",
        "representative_orig_aa",
        "first_event_row",
        "fasta_index",
        "sequence_length",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields)
        writer.writeheader()
        writer.writerows(matched_rows)
    with missing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "event_count", "first_event_row"])
        writer.writeheader()
        writer.writerows(missing_rows)

    print(f"Event rows: {event_rows}")
    print(f"Unique event proteins: {len(proteins)}")
    print(f"Source FASTA records: {len(fasta_records)}")
    print(f"Matched unique proteins: {len(matched_rows)}")
    print(f"Missing unique proteins: {len(missing_rows)}")
    print(f"Query FASTA: {fasta_path}")
    print(f"Metadata CSV: {metadata_path}")
    print(f"Missing CSV: {missing_path}")
    return 1 if missing_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
