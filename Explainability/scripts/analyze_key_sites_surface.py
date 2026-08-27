"""
Map PLMSol key sites onto downloaded AFDB structures and estimate whether the
mapped residues are on the protein surface.

Surface is estimated with a small Shrake-Rupley solvent accessible surface area
(SASA) implementation. A residue is called surface-exposed when its relative
SASA is >= --surface-rsa-threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
PLM_ROOT = PROJECT_DIR.parent
STRUCTURE_PREP_DIR = PROJECT_DIR / "data"
AFDB_BLAST80_DIR = PROJECT_DIR / "afdb_structures"
SUCCESS_STATUSES = {"downloaded", "skipped_existing"}


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}

ATOM_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
}

# Tien et al. 2013 maximum residue ASA values, extended with U/O fallback.
MAX_ASA = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
    "U": 167.0,
    "O": 236.0,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:]
                records[current] = []
            elif current is not None:
                records[current].append(line)
    return {name: "".join(parts) for name, parts in records.items()}


def accession_from_pdb_name(path: Path) -> str:
    name = path.stem
    if name.startswith("AF-") and name.endswith("-F1"):
        return name[3:-3]
    return name


def build_pdb_index(pdb_dir: Path) -> dict[str, Path]:
    pdb_by_accession: dict[str, Path] = {}
    for path in sorted(pdb_dir.glob("*.pdb")):
        pdb_by_accession[accession_from_pdb_name(path)] = path
    return pdb_by_accession


def read_afdb_mapping_summary(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_protein: dict[str, dict[str, str]] = {}
    for row in rows:
        protein_id = row.get("protein_id", "")
        accession = row.get("accession", "") or row.get("selected_uniprot_accession", "")
        if not protein_id or not accession:
            continue
        by_protein[protein_id] = row
    return by_protein


def normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = normalize_id(row.get(name, ""))
        if value:
            return value
    return ""


def abs_delta_above_threshold(row: dict[str, str], threshold: float) -> bool:
    try:
        return float(row.get("abs_delta", "")) > threshold
    except ValueError:
        return False


def resolve_site_protein_id(
    site: dict[str, str],
    protein_ids: set[str],
    seq_idx_to_protein: dict[str, str],
    fasta_idx_to_protein: dict[str, str],
) -> tuple[str, str]:
    protein_id = normalize_id(site.get("protein_id", ""))
    if protein_id in protein_ids:
        return protein_id, "protein_id"

    seq_idx = first_present(site, ("seq_idx", "sequence_idx", "dataset_idx", "row_idx", "index"))
    if seq_idx in seq_idx_to_protein:
        return seq_idx_to_protein[seq_idx], "seq_idx"

    fasta_idx = first_present(site, ("fasta_idx", "fasta_index", "query_idx", "query_index"))
    if fasta_idx in fasta_idx_to_protein:
        return fasta_idx_to_protein[fasta_idx], "fasta_idx"

    return "", "unmatched"


def atom_radius(element: str, atom_name: str) -> float:
    key = (element or "").strip().upper()
    if not key:
        key = "".join(ch for ch in atom_name if ch.isalpha())[:1].upper()
    return ATOM_RADII.get(key, 1.70)


def parse_pdb_atoms(path: Path) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    residues: list[dict[str, Any]] = []
    atoms: list[dict[str, Any]] = []
    residue_index: dict[tuple[str, int, str], int] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            resname = line[17:20].strip()
            aa = AA3_TO_1.get(resname)
            if aa is None:
                continue
            chain = line[21].strip() or "A"
            resseq = int(line[22:26])
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key not in residue_index:
                residue_index[key] = len(residues)
                residues.append(
                    {
                        "chain": chain,
                        "resseq": resseq,
                        "icode": icode,
                        "resname": resname,
                        "aa": aa,
                    }
                )
            atom = {
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "radius": atom_radius(line[76:78], line[12:16]),
                "residue_index": residue_index[key],
            }
            atoms.append(atom)
    sequence = "".join(residue["aa"] for residue in residues)
    return sequence, residues, atoms


def fibonacci_sphere(n_points: int) -> list[tuple[float, float, float]]:
    points = []
    offset = 2.0 / n_points
    increment = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(n_points):
        y = ((index * offset) - 1.0) + (offset / 2.0)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        phi = index * increment
        points.append((math.cos(phi) * radius, y, math.sin(phi) * radius))
    return points


def calculate_residue_sasa(
    atoms: list[dict[str, Any]],
    n_residues: int,
    probe_radius: float,
    n_sphere_points: int,
) -> list[float]:
    if not atoms:
        return []

    sphere = fibonacci_sphere(n_sphere_points)
    max_radius = max(atom["radius"] + probe_radius for atom in atoms)
    cell_size = max_radius * 2.0
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for atom_index, atom in enumerate(atoms):
        cell = (
            math.floor(atom["x"] / cell_size),
            math.floor(atom["y"] / cell_size),
            math.floor(atom["z"] / cell_size),
        )
        grid[cell].append(atom_index)

    residue_sasa = [0.0 for _ in range(n_residues)]
    for atom_index, atom in enumerate(atoms):
        expanded_radius = atom["radius"] + probe_radius
        accessible = 0
        cell = (
            math.floor(atom["x"] / cell_size),
            math.floor(atom["y"] / cell_size),
            math.floor(atom["z"] / cell_size),
        )
        neighbor_indices: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_indices.extend(grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []))

        for sx, sy, sz in sphere:
            px = atom["x"] + sx * expanded_radius
            py = atom["y"] + sy * expanded_radius
            pz = atom["z"] + sz * expanded_radius
            buried = False
            for other_index in neighbor_indices:
                if other_index == atom_index:
                    continue
                other = atoms[other_index]
                other_expanded = other["radius"] + probe_radius
                distance2 = (px - other["x"]) ** 2 + (py - other["y"]) ** 2 + (pz - other["z"]) ** 2
                if distance2 < other_expanded * other_expanded:
                    buried = True
                    break
            if not buried:
                accessible += 1

        atom_area = 4.0 * math.pi * expanded_radius * expanded_radius * (accessible / n_sphere_points)
        residue_sasa[atom["residue_index"]] += atom_area
    return residue_sasa


def smith_waterman_map(query: str, target: str) -> tuple[dict[int, int], float, int]:
    match_score = 2
    mismatch_score = -1
    gap_score = -2
    rows = len(query) + 1
    cols = len(target) + 1
    scores = [[0] * cols for _ in range(rows)]
    traces = [[""] * cols for _ in range(rows)]
    best_score = 0
    best_cell = (0, 0)

    for i in range(1, rows):
        for j in range(1, cols):
            diag = scores[i - 1][j - 1] + (match_score if query[i - 1] == target[j - 1] else mismatch_score)
            up = scores[i - 1][j] + gap_score
            left = scores[i][j - 1] + gap_score
            value = max(0, diag, up, left)
            scores[i][j] = value
            if value == 0:
                traces[i][j] = ""
            elif value == diag:
                traces[i][j] = "D"
            elif value == up:
                traces[i][j] = "U"
            else:
                traces[i][j] = "L"
            if value > best_score:
                best_score = value
                best_cell = (i, j)

    mapping: dict[int, int] = {}
    matches = 0
    aligned_pairs = 0
    i, j = best_cell
    while i > 0 and j > 0 and scores[i][j] > 0:
        trace = traces[i][j]
        if trace == "D":
            aligned_pairs += 1
            if query[i - 1] == target[j - 1]:
                matches += 1
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif trace == "U":
            i -= 1
        elif trace == "L":
            j -= 1
        else:
            break

    identity = matches / aligned_pairs if aligned_pairs else 0.0
    return mapping, identity, aligned_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AFDB structure availability and key-site surface exposure.")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=STRUCTURE_PREP_DIR / "key_site_proteins_metadata.csv",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=STRUCTURE_PREP_DIR / "key_site_proteins.fasta",
    )
    parser.add_argument(
        "--key-sites-csv",
        type=Path,
        default=PROJECT_DIR / "outputs/masking/all_correct_to_wrong_2500.csv",
    )
    parser.add_argument(
        "--afdb-summary-csv",
        type=Path,
        default=STRUCTURE_PREP_DIR / "afdb_download_manifest.csv",
        help="AFDB download summary generated from BLAST 80/80 accessions.",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=AFDB_BLAST80_DIR,
        help="Directory containing AFDB PDB files named as {accession}.pdb.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_DIR / "outputs/structure/key_site_surface_analysis.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_DIR / "outputs/structure/key_site_surface_summary.json",
    )
    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument("--sphere-points", type=int, default=32)
    parser.add_argument("--surface-rsa-threshold", type=float, default=0.25)
    parser.add_argument(
        "--abs-delta-threshold",
        type=float,
        default=0.05,
        help="Also report a second summary using only key sites with abs_delta above this threshold.",
    )
    args = parser.parse_args()

    metadata = read_csv(args.metadata_csv)
    fasta = read_fasta(args.fasta)
    key_sites = read_csv(args.key_sites_csv)
    afdb_by_protein = read_afdb_mapping_summary(args.afdb_summary_csv)
    pdb_by_accession = build_pdb_index(args.pdb_dir)

    protein_ids = {normalize_id(row["protein_id"]) for row in metadata}
    seq_idx_to_protein = {
        normalize_id(row["seq_idx"]): row["protein_id"]
        for row in metadata
        if normalize_id(row.get("seq_idx", ""))
    }
    fasta_idx_to_protein = {
        normalize_id(row["fasta_idx"]): row["protein_id"]
        for row in metadata
        if normalize_id(row.get("fasta_idx", ""))
    }
    grouped_sites: dict[str, list[dict[str, str]]] = defaultdict(list)
    site_match_counts = defaultdict(int)
    site_match_counts_above_abs_delta_threshold = defaultdict(int)
    unmatched_site_rows = 0
    unmatched_site_rows_above_abs_delta_threshold = 0
    key_site_rows_above_abs_delta_threshold = 0
    for row in key_sites:
        is_above_abs_delta_threshold = abs_delta_above_threshold(row, args.abs_delta_threshold)
        if is_above_abs_delta_threshold:
            key_site_rows_above_abs_delta_threshold += 1

        protein_id, matched_by = resolve_site_protein_id(
            row,
            protein_ids=protein_ids,
            seq_idx_to_protein=seq_idx_to_protein,
            fasta_idx_to_protein=fasta_idx_to_protein,
        )
        site_match_counts[matched_by] += 1
        if is_above_abs_delta_threshold:
            site_match_counts_above_abs_delta_threshold[matched_by] += 1
        if not protein_id:
            unmatched_site_rows += 1
            if is_above_abs_delta_threshold:
                unmatched_site_rows_above_abs_delta_threshold += 1
            continue
        row = dict(row)
        row["_matched_by"] = matched_by
        grouped_sites[protein_id].append(row)

    structure_cache: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    proteins_with_structure = set()
    proteins_with_mapped_surface_sites = set()

    for protein_row in metadata:
        protein_id = protein_row["protein_id"]
        afdb_row = afdb_by_protein.get(protein_id, {})
        accession = afdb_row.get("accession", "") or afdb_row.get("selected_uniprot_accession", "")
        afdb_status = afdb_row.get("status", "")
        pdb_path = pdb_by_accession.get(accession)
        if afdb_status and afdb_status not in SUCCESS_STATUSES:
            pdb_path = None
        query_sequence = fasta.get(protein_id, "")
        sites = grouped_sites.get(protein_id, [])
        if pdb_path:
            proteins_with_structure.add(protein_id)

        structure = None
        if pdb_path:
            if accession not in structure_cache:
                pdb_sequence, residues, atoms = parse_pdb_atoms(pdb_path)
                residue_sasa = calculate_residue_sasa(
                    atoms,
                    n_residues=len(residues),
                    probe_radius=args.probe_radius,
                    n_sphere_points=args.sphere_points,
                )
                position_map, identity, aligned_pairs = smith_waterman_map(query_sequence, pdb_sequence)
                structure_cache[accession] = {
                    "sequence": pdb_sequence,
                    "residues": residues,
                    "residue_sasa": residue_sasa,
                    "position_map": position_map,
                    "alignment_identity": identity,
                    "aligned_pairs": aligned_pairs,
                }
            structure = structure_cache[accession]

        for site in sites:
            mask_pos = int(site["mask_pos"])
            query_aa = query_sequence[mask_pos] if 0 <= mask_pos < len(query_sequence) else ""
            mapped_residue_index = None
            residue = None
            sasa = ""
            rsa = ""
            surface_call = "no_structure"
            mapping_status = "no_structure"
            pdb_aa = ""

            if structure:
                mapped_residue_index = structure["position_map"].get(mask_pos)
                if mapped_residue_index is None:
                    surface_call = "unmapped"
                    mapping_status = "site_not_aligned_to_afdb_structure"
                else:
                    residue = structure["residues"][mapped_residue_index]
                    pdb_aa = residue["aa"]
                    sasa_value = structure["residue_sasa"][mapped_residue_index]
                    max_asa = MAX_ASA.get(pdb_aa, 200.0)
                    rsa_value = sasa_value / max_asa
                    sasa = round(sasa_value, 3)
                    rsa = round(rsa_value, 4)
                    surface_call = "surface" if rsa_value >= args.surface_rsa_threshold else "buried"
                    mapping_status = "mapped" if query_aa == pdb_aa else "mapped_aa_mismatch"
                    if surface_call == "surface":
                        proteins_with_mapped_surface_sites.add(protein_id)

            output_rows.append(
                {
                    "seq_idx": site.get("seq_idx", "") or site.get("index_row", ""),
                    "key_site_matched_by": site.get("_matched_by", ""),
                    "protein_id": protein_id,
                    "label": protein_row.get("label", ""),
                    "accession": accession,
                    "selected_identity_percent": afdb_row.get("selected_identity_percent", ""),
                    "selected_query_coverage_percent": afdb_row.get("selected_query_coverage_percent", ""),
                    "selected_evalue": afdb_row.get("selected_evalue", ""),
                    "selected_bit_score": afdb_row.get("selected_bit_score", ""),
                    "quality_tier": afdb_row.get("quality_tier", ""),
                    "afdb_download_status": afdb_status,
                    "reviewed": afdb_row.get("isReviewed", ""),
                    "organism": afdb_row.get("organismScientificName", ""),
                    "afdb_pdb_exists": bool(pdb_path),
                    "pdb_path": str(pdb_path) if pdb_path else "",
                    "mask_pos_0_based": mask_pos,
                    "residue_position_1_based_in_query": mask_pos + 1,
                    "orig_aa": site.get("orig_aa", ""),
                    "query_aa": query_aa,
                    "pdb_chain": residue["chain"] if residue else "",
                    "pdb_resseq": residue["resseq"] if residue else "",
                    "pdb_icode": residue["icode"] if residue else "",
                    "pdb_aa": pdb_aa,
                    "residue_sasa": sasa,
                    "residue_rsa": rsa,
                    "surface_call": surface_call,
                    "mapping_status": mapping_status,
                    "alignment_identity": round(structure["alignment_identity"], 4) if structure else "",
                    "aligned_pairs": structure["aligned_pairs"] if structure else "",
                    "baseline_sigmoid": site.get("baseline_sigmoid", "")
                    or site.get("baseline_probability", ""),
                    "masked_sigmoid": site.get("masked_sigmoid", "")
                    or site.get("masked_probability", ""),
                    "delta": site.get("delta", ""),
                    "abs_delta": site.get("abs_delta", ""),
                }
            )

    fieldnames = [
        "seq_idx",
        "key_site_matched_by",
        "protein_id",
        "label",
        "accession",
        "selected_identity_percent",
        "selected_query_coverage_percent",
        "selected_evalue",
        "selected_bit_score",
        "quality_tier",
        "afdb_download_status",
        "reviewed",
        "organism",
        "afdb_pdb_exists",
        "pdb_path",
        "mask_pos_0_based",
        "residue_position_1_based_in_query",
        "orig_aa",
        "query_aa",
        "pdb_chain",
        "pdb_resseq",
        "pdb_icode",
        "pdb_aa",
        "residue_sasa",
        "residue_rsa",
        "surface_call",
        "mapping_status",
        "alignment_identity",
        "aligned_pairs",
        "baseline_sigmoid",
        "masked_sigmoid",
        "delta",
        "abs_delta",
    ]
    write_csv(args.out_csv, output_rows, fieldnames)

    def summarize_rows(
        rows: list[dict[str, Any]],
        key_site_rows_total: int,
        unmatched_rows: int,
        match_counts: dict[str, int],
    ) -> dict[str, Any]:
        by_call = defaultdict(int)
        by_mapping = defaultdict(int)
        by_afdb_status = defaultdict(int)
        exact_mapped_surface_calls = defaultdict(int)
        structure_resolved_surface_calls = defaultdict(int)
        for row in rows:
            by_call[row["surface_call"]] += 1
            by_mapping[row["mapping_status"]] += 1
            by_afdb_status[row["afdb_download_status"] or "not_in_afdb_summary"] += 1
            if row["mapping_status"] == "mapped":
                exact_mapped_surface_calls[row["surface_call"]] += 1
            if row["surface_call"] in ("surface", "buried"):
                structure_resolved_surface_calls[row["surface_call"]] += 1

        unique_proteins_with_sites = {row["protein_id"] for row in rows}
        proteins_with_downloaded_structure = {
            row["protein_id"] for row in rows if row["afdb_pdb_exists"]
        }
        proteins_with_surface_sites = {
            row["protein_id"] for row in rows if row["surface_call"] == "surface"
        }
        proteins_all_surface = set()
        proteins_any_buried = set()
        output_by_protein: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            output_by_protein[row["protein_id"]].append(row)
        for protein_id, protein_rows in output_by_protein.items():
            mapped = [row for row in protein_rows if row["surface_call"] in ("surface", "buried")]
            if mapped and all(row["surface_call"] == "surface" for row in mapped):
                proteins_all_surface.add(protein_id)
            if any(row["surface_call"] == "buried" for row in mapped):
                proteins_any_buried.add(protein_id)

        surface_count = by_call.get("surface", 0)
        buried_count = by_call.get("buried", 0)
        exact_mapped_count = by_mapping.get("mapped", 0)
        aa_mismatch_count = by_mapping.get("mapped_aa_mismatch", 0)
        structure_resolved_count = surface_count + buried_count
        return {
            "key_site_rows_total": key_site_rows_total,
            "key_site_rows_for_metadata": len(rows),
            "key_site_rows_unmatched_to_metadata": unmatched_rows,
            "key_site_match_counts": dict(sorted(match_counts.items())),
            "unique_proteins_with_key_sites": len(unique_proteins_with_sites),
            "proteins_with_downloaded_afdb_structure": len(proteins_with_downloaded_structure),
            "key_site_surface_call_counts": dict(sorted(by_call.items())),
            "key_site_mapping_status_counts": dict(sorted(by_mapping.items())),
            "key_site_surface_call_counts_for_exact_aa_mapped": dict(sorted(exact_mapped_surface_calls.items())),
            "key_site_surface_call_counts_for_structure_resolved": dict(sorted(structure_resolved_surface_calls.items())),
            "structure_resolved_key_sites_surface_plus_buried": structure_resolved_count,
            "exact_aa_mapped_key_sites": exact_mapped_count,
            "aa_mismatch_but_structure_resolved_key_sites": aa_mismatch_count,
            "structure_resolved_equals_mapped_plus_mismatch": structure_resolved_count
            == exact_mapped_count + aa_mismatch_count,
            "key_site_afdb_download_status_counts": dict(sorted(by_afdb_status.items())),
            "proteins_with_at_least_one_surface_key_site": len(proteins_with_surface_sites),
            "proteins_where_all_mapped_key_sites_are_surface": len(proteins_all_surface),
            "proteins_with_at_least_one_buried_key_site": len(proteins_any_buried),
        }

    output_rows_above_abs_delta_threshold = [
        row for row in output_rows if abs_delta_above_threshold(row, args.abs_delta_threshold)
    ]
    all_sites_summary = summarize_rows(
        output_rows,
        key_site_rows_total=len(key_sites),
        unmatched_rows=unmatched_site_rows,
        match_counts=site_match_counts,
    )
    abs_delta_filtered_summary = summarize_rows(
        output_rows_above_abs_delta_threshold,
        key_site_rows_total=key_site_rows_above_abs_delta_threshold,
        unmatched_rows=unmatched_site_rows_above_abs_delta_threshold,
        match_counts=site_match_counts_above_abs_delta_threshold,
    )
    summary = {
        "metadata_csv": str(args.metadata_csv),
        "fasta": str(args.fasta),
        "key_sites_csv": str(args.key_sites_csv),
        "afdb_summary_csv": str(args.afdb_summary_csv),
        "pdb_dir": str(args.pdb_dir),
        "output_csv": str(args.out_csv),
        "metadata_proteins": len(metadata),
        "proteins_in_afdb_summary": len(afdb_by_protein),
        "pdb_files_indexed": len(pdb_by_accession),
        "proteins_with_downloaded_afdb_structure_in_metadata": len(proteins_with_structure),
        "surface_rsa_threshold": args.surface_rsa_threshold,
        "abs_delta_threshold": args.abs_delta_threshold,
        "probe_radius_angstrom": args.probe_radius,
        "sphere_points_per_atom": args.sphere_points,
        "statistics_all_sites": all_sites_summary,
        f"statistics_abs_delta_gt_{args.abs_delta_threshold:g}": abs_delta_filtered_summary,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
