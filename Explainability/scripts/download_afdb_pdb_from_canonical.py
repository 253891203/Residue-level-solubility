#!/usr/bin/env python3
"""
Download AlphaFold DB PDB files for canonical UniProt accessions.

Input:
  uniprot_mapping_results_canonical_from_existing/
    plmsol_uniprot_canonical_accession_per_protein.csv

The script keeps only rows with afdb_exists=True and a non-empty accession.
For each accession it first queries the AlphaFold DB prediction API and uses
the returned pdbUrl/latestVersion when available. If the API request fails, it
falls back to the common v4 URL pattern:
  https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb

The output is resumable:
  - existing non-empty PDB files are skipped
  - successful downloads are recorded in downloaded_afdb_pdb.csv
  - failures are recorded in failed_afdb_pdb.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
AFDB_FALLBACK_PDB = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb"
USER_AGENT = "plmsol-afdb-pdb-downloader/1.0"


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def request_bytes(url: str, timeout: float, retries: int, retry_sleep: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                sleep_for = retry_sleep * attempt
                print(f"  request failed ({exc}); retrying in {sleep_for:.1f}s")
                time.sleep(sleep_for)
    raise RuntimeError(f"Request failed after {retries} tries: {url}: {last_error}")


def request_json(url: str, timeout: float, retries: int, retry_sleep: float) -> Any:
    return json.loads(request_bytes(url, timeout=timeout, retries=retries, retry_sleep=retry_sleep).decode("utf-8"))


def get_afdb_prediction(accession: str, timeout: float, retries: int, retry_sleep: float) -> dict[str, Any]:
    payload = request_json(
        AFDB_API.format(accession=accession),
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    if isinstance(payload, list) and payload:
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


def resolve_pdb_url(accession: str, timeout: float, retries: int, retry_sleep: float) -> tuple[str, str, str]:
    try:
        prediction = get_afdb_prediction(accession, timeout=timeout, retries=retries, retry_sleep=retry_sleep)
        pdb_url = prediction.get("pdbUrl", "")
        latest_version = prediction.get("latestVersion", "")
        entry_id = prediction.get("entryId", f"AF-{accession}-F1")
        if pdb_url:
            return pdb_url, str(latest_version), str(entry_id)
    except Exception as exc:
        print(f"  API lookup failed for {accession}; using fallback URL ({exc})")

    return AFDB_FALLBACK_PDB.format(accession=accession), "v4_fallback", f"AF-{accession}-F1"


def unique_afdb_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    selected: list[dict[str, str]] = []
    for row in rows:
        accession = row.get("accession", "").strip() or row.get("selected_uniprot_accession", "").strip()
        if not accession or accession in seen:
            continue
        availability = row.get("afdb_exists", row.get("selected_afdb_exists"))
        if availability is not None and not as_bool(availability):
            continue
        copied = dict(row)
        copied["accession"] = accession
        selected.append(copied)
        seen.add(accession)
    return selected


def download_one(
    row: dict[str, str],
    pdb_dir: Path,
    timeout: float,
    retries: int,
    retry_sleep: float,
    force: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    accession = row["accession"]
    out_path = pdb_dir / f"AF-{accession}-F1.pdb"

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return (
            {
                "protein_id": row.get("protein_id", ""),
                "accession": accession,
                "pdb_path": str(out_path),
                "bytes": out_path.stat().st_size,
                "status": "already_exists",
                "pdb_url": "",
                "latest_version": "",
                "entry_id": "",
            },
            None,
        )

    pdb_url, latest_version, entry_id = resolve_pdb_url(
        accession,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )

    try:
        content = request_bytes(pdb_url, timeout=timeout, retries=retries, retry_sleep=retry_sleep)
        if not content.startswith(b"HEADER") and b"\nATOM" not in content[:5000]:
            raise RuntimeError("Downloaded content does not look like a PDB file")
        out_path.write_bytes(content)
        return (
            {
                "protein_id": row.get("protein_id", ""),
                "accession": accession,
                "pdb_path": str(out_path),
                "bytes": len(content),
                "status": "downloaded",
                "pdb_url": pdb_url,
                "latest_version": latest_version,
                "entry_id": entry_id,
            },
            None,
        )
    except Exception as exc:
        return (
            None,
            {
                "protein_id": row.get("protein_id", ""),
                "accession": accession,
                "pdb_url": pdb_url,
                "latest_version": latest_version,
                "entry_id": entry_id,
                "error": str(exc),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AFDB PDB files listed in a canonical mapping or release manifest.")
    parser.add_argument(
        "--canonical-csv",
        type=Path,
        default=Path("data/afdb_download_manifest.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("afdb_structures"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0, help="Download only the first N accessions; 0 means all.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel download workers.")
    parser.add_argument("--force", action="store_true", help="Re-download even if a PDB already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.canonical_csv)
    targets = unique_afdb_rows(rows)
    if args.limit > 0:
        targets = targets[: args.limit]

    pdb_dir = args.out_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    print(f"AFDB targets: {len(targets)}")

    def run_target(index_and_row: tuple[int, dict[str, str]]) -> tuple[int, str, dict[str, Any] | None, dict[str, Any] | None]:
        index, row = index_and_row
        accession = row["accession"]
        success, failed = download_one(
            row,
            pdb_dir=pdb_dir,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            force=args.force,
        )
        return index, accession, success, failed

    indexed_targets = list(enumerate(targets, 1))
    def record_result(index: int, accession: str, success: dict[str, Any] | None, failed: dict[str, Any] | None) -> None:
        print(f"[{index}/{len(targets)}] {accession}")
        if success:
            print(f"  {success['status']}: {success['pdb_path']} ({success['bytes']} bytes)")
            success_rows.append(success)
        if failed:
            print(f"  failed: {failed['error']}")
            failed_rows.append(failed)

        write_csv(
            args.out_dir / "downloaded_afdb_pdb.csv",
            success_rows,
            ["protein_id", "accession", "pdb_path", "bytes", "status", "pdb_url", "latest_version", "entry_id"],
        )
        write_csv(
            args.out_dir / "failed_afdb_pdb.csv",
            failed_rows,
            ["protein_id", "accession", "pdb_url", "latest_version", "entry_id", "error"],
        )

    if args.workers <= 1:
        for item in indexed_targets:
            record_result(*run_target(item))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_target, item) for item in indexed_targets]
            for future in as_completed(futures):
                record_result(*future.result())

    print(f"Downloaded/already present: {len(success_rows)}")
    print(f"Failed: {len(failed_rows)}")
    print(f"PDB directory: {pdb_dir}")


if __name__ == "__main__":
    main()
