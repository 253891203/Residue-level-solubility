"""
Analyze local structural environments for high- and low-impact key sites.

This script is intentionally independent from training, inference, and the
existing structure-mapping pipeline. It reuses previously generated key-site
surface mapping results when available, then computes local CA-neighborhood
hydrophobicity and charge features from PDB files.
"""

from __future__ import annotations

import argparse
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
except ImportError:  # pragma: no cover - depends on local environment
    mannwhitneyu = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

COMMON_SEARCH_DIRS = (
    Path("."),
    Path("results"),
    Path("structure_query_preparation"),
    Path("structure_analysis"),
    Path("afdb"),
    Path("databases/afdb"),
    Path("model_results"),
    Path("3Dstructure"),
    Path("3Dstructure/structure_query_preparation"),
    Path("3Dstructure/structure_analysis"),
    Path("3Dstructure/afdb_blast80_structures"),
)

PROTEIN_ID_FIELDS = ("protein_id", "sequence_id", "accession", "seq_id")
POSITION_FIELDS = (
    "mask_pos",
    "position",
    "residue_index",
    "mask_pos_0_based",
    "residue_position_1_based_in_query",
)
RSA_FIELDS = ("residue_rsa", "rsa", "relative_sasa", "relative_asa")
SASA_FIELDS = ("residue_sasa", "sasa", "asa")
RESIDUE_FIELDS = ("pdb_aa", "query_aa", "orig_aa", "residue_type", "aa")

HYDROPHOBIC_RESIDUES = set("AVILMFWY")
POSITIVE_RESIDUES = set("KRH")
NEGATIVE_RESIDUES = set("DE")

KYTE_DOOLITTLE = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}

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

RSA_BINS = [
    ("RSA < 5\\%", 0.0, 0.05),
    ("5\\% <= RSA < 25\\%", 0.05, 0.25),
    ("25\\% <= RSA < 50\\%", 0.25, 0.50),
    ("RSA >= 50\\%", 0.50, math.inf),
]


def group_order(df: pd.DataFrame) -> list[str]:
    preferred = ["Positive high", "Negative high", "High impact", "Low impact"]
    present = set(df["impact_group"].dropna().astype(str))
    ordered = [group for group in preferred if group in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("local_structure_environment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "analysis.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def resolve_path(path_text: str | Path, base_dirs: tuple[Path, ...]) -> Path:
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate.resolve()
    matches = sorted(PROJECT_ROOT.rglob(path.name))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"Input file not found: {path_text}")


def first_existing_field(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def log_detected_input_fields(df: pd.DataFrame, logger: logging.Logger) -> dict[str, str | None]:
    fields = {
        "protein_id": first_existing_field(list(df.columns), PROTEIN_ID_FIELDS),
        "position": first_existing_field(list(df.columns), POSITION_FIELDS),
        "abs_delta": first_existing_field(list(df.columns), ("abs_delta",)),
        "rsa": first_existing_field(list(df.columns), RSA_FIELDS),
        "sasa": first_existing_field(list(df.columns), SASA_FIELDS),
        "surface_call": first_existing_field(list(df.columns), ("surface_call", "surface", "buried")),
    }
    logger.info("Detected input fields: %s", fields)

    required_missing = [
        logical_name
        for logical_name in ("protein_id", "position", "abs_delta")
        if fields[logical_name] is None
    ]
    if required_missing:
        raise ValueError(
            "Missing required input fields: "
            + ", ".join(required_missing)
            + ". Candidate field names are: protein_id/sequence_id/accession, "
            "mask_pos/position/residue_index, abs_delta."
        )
    if fields["rsa"] is None and fields["sasa"] is None and fields["surface_call"] is None:
        logger.warning(
            "Input file has no RSA/SASA/surface fields. A structure mapping result file is required."
        )
    return fields


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def find_structure_mapping_file(input_path: Path, explicit_path: Path | None, logger: logging.Logger) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Structure mapping file not found: {explicit_path}")
        return explicit_path.resolve()

    candidate_names = (
        "key_site_surface_analysis_blast80.csv",
        "key_site_surface_analysis.csv",
        "site_surface_analysis.csv",
    )
    roots = [PROJECT_ROOT / search_dir for search_dir in COMMON_SEARCH_DIRS]
    roots.extend([input_path.parent, SCRIPT_DIR])

    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        for name in candidate_names:
            candidates.extend(root.rglob(name))

    scored_candidates: list[tuple[int, Path]] = []
    required = {"abs_delta", "pdb_path", "pdb_chain", "pdb_resseq", "mapping_status"}
    for candidate in sorted(set(candidates)):
        try:
            columns = set(pd.read_csv(candidate, nrows=0, encoding="utf-8-sig").columns)
        except Exception as exc:
            logger.warning("Could not inspect candidate mapping file %s: %s", candidate, exc)
            continue
        score = len(required.intersection(columns))
        if "residue_rsa" in columns or "residue_sasa" in columns:
            score += 2
        if "blast80" in candidate.name.lower():
            score += 1
        if score >= 4:
            scored_candidates.append((score, candidate.resolve()))

    if not scored_candidates:
        raise FileNotFoundError(
            "No usable structure mapping result file found. Required fields include "
            "pdb_path, pdb_chain, pdb_resseq, mapping_status, and RSA/SASA fields."
        )

    scored_candidates.sort(key=lambda item: (-item[0], str(item[1])))
    selected = scored_candidates[0][1]
    logger.info("Selected structure mapping result file: %s", selected)
    return selected


def normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def add_impact_group(
    df: pd.DataFrame,
    threshold: float,
    grouping: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    df = df.copy()
    df["abs_delta_numeric"] = pd.to_numeric(df["abs_delta"], errors="coerce")
    if grouping == "threshold":
        df["impact_group"] = np.where(df["abs_delta_numeric"] > threshold, "High impact", "Low impact")
        logger.info("Grouping mode: threshold, high impact is abs_delta > %.3f", threshold)
        return df

    if grouping == "top_bottom_quartile":
        low_cutoff = df["abs_delta_numeric"].quantile(0.25)
        high_cutoff = df["abs_delta_numeric"].quantile(0.75)
        df["impact_group"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df.loc[df["abs_delta_numeric"] >= high_cutoff, "impact_group"] = "High impact"
        df.loc[df["abs_delta_numeric"] <= low_cutoff, "impact_group"] = "Low impact"
        logger.info(
            "Grouping mode: top_bottom_quartile, low cutoff <= %.6f, high cutoff >= %.6f",
            low_cutoff,
            high_cutoff,
        )
        logger.info("Rows in middle 50%% excluded from grouped analyses: %d", int(df["impact_group"].isna().sum()))
        return df

    if grouping == "signed_top_bottom":
        if "delta" not in df.columns:
            raise ValueError(
                "signed_top_bottom grouping requires a delta column. In the masked sigmoid workflow, "
                "delta should be masked_sigmoid - baseline_sigmoid."
            )
        df["delta_numeric"] = pd.to_numeric(df["delta"], errors="coerce")
        if {"masked_sigmoid", "baseline_sigmoid"}.issubset(df.columns):
            masked = pd.to_numeric(df["masked_sigmoid"], errors="coerce")
            baseline = pd.to_numeric(df["baseline_sigmoid"], errors="coerce")
            max_error = (df["delta_numeric"] - (masked - baseline)).abs().max()
            logger.info(
                "Validated delta direction against masked_sigmoid - baseline_sigmoid; max abs error = %.6g",
                max_error,
            )
        low_cutoff = df["abs_delta_numeric"].quantile(0.25)
        high_cutoff = df["abs_delta_numeric"].quantile(0.75)
        df["impact_group"] = pd.Series(pd.NA, index=df.index, dtype="string")
        high_mask = df["abs_delta_numeric"] >= high_cutoff
        df.loc[high_mask & (df["delta_numeric"] > 0), "impact_group"] = "Positive high"
        df.loc[high_mask & (df["delta_numeric"] < 0), "impact_group"] = "Negative high"
        df.loc[df["abs_delta_numeric"] <= low_cutoff, "impact_group"] = "Low impact"
        logger.info(
            "Grouping mode: signed_top_bottom, low cutoff <= %.6f, high cutoff >= %.6f",
            low_cutoff,
            high_cutoff,
        )
        logger.info(
            "Delta interpretation: delta = masked_sigmoid - baseline_sigmoid; "
            "positive means masking increases predicted soluble probability."
        )
        logger.info(
            "Because label 1 is soluble, positive high sites are interpreted as residues whose presence "
            "lowers predicted solubility; negative high sites are interpreted as residues whose presence "
            "raises predicted solubility."
        )
        logger.info("Rows excluded from grouped analyses: %d", int(df["impact_group"].isna().sum()))
        return df

    raise ValueError(f"Unsupported grouping mode: {grouping}")
    return df


def prepare_rsa(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = df.copy()
    rsa_field = first_existing_field(list(df.columns), RSA_FIELDS)
    if rsa_field:
        logger.info("Using RSA field: %s", rsa_field)
        df["rsa_numeric"] = pd.to_numeric(df[rsa_field], errors="coerce")
        return df

    sasa_field = first_existing_field(list(df.columns), SASA_FIELDS)
    residue_field = first_existing_field(list(df.columns), RESIDUE_FIELDS)
    if sasa_field and residue_field:
        logger.info("RSA field missing; computing RSA from %s / max_ASA(%s)", sasa_field, residue_field)
        sasa = pd.to_numeric(df[sasa_field], errors="coerce")
        max_asa = df[residue_field].astype(str).str.upper().map(MAX_ASA)
        df["rsa_numeric"] = sasa / max_asa
        return df

    raise ValueError(
        "Missing RSA information. Need one of residue_rsa/rsa/relative_sasa, "
        "or residue_sasa plus a residue type field such as pdb_aa/query_aa/orig_aa."
    )


def filter_reliable_structure_rows(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, int]]:
    required = ("pdb_path", "pdb_chain", "pdb_resseq", "pdb_aa")
    missing = [field for field in required if field not in df.columns]
    if missing:
        raise ValueError(
            "Missing fields required for local neighborhood analysis: "
            + ", ".join(missing)
            + ". Need a structure mapping result with PDB path and mapped residue identifiers."
        )

    skip_counts: dict[str, int] = {}
    mask = pd.Series(True, index=df.index)
    if "mapping_status" in df.columns:
        reliable_mapping = df["mapping_status"].isin(["mapped", "mapped_aa_mismatch"])
        skip_counts["unmapped_or_no_structure"] = int((~reliable_mapping).sum())
        mask &= reliable_mapping
        logger.info("Using mapping_status to exclude no_structure/unmapped rows.")
    if "surface_call" in df.columns:
        reliable_surface = df["surface_call"].isin(["surface", "buried"])
        skip_counts["non_surface_resolved"] = int((~reliable_surface).sum())
        mask &= reliable_surface
    if "afdb_pdb_exists" in df.columns:
        has_structure = df["afdb_pdb_exists"].map(normalize_bool)
        skip_counts["pdb_missing_flag"] = int((~has_structure).sum())
        mask &= has_structure

    has_pdb_path = df["pdb_path"].fillna("").astype(str).str.len() > 0
    skip_counts["missing_pdb_path"] = int((~has_pdb_path).sum())
    mask &= has_pdb_path

    reliable = df[mask].copy()
    logger.info("Reliable structure-mapped rows: %d", len(reliable))
    return reliable, skip_counts


def parse_pdb_ca_residues(pdb_path: Path) -> list[dict[str, Any]]:
    residues: list[dict[str, Any]] = []
    with pdb_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16].strip()
            atom_name = line[12:16].strip()
            if atom_name != "CA" or altloc not in ("", "A"):
                continue
            resname = line[17:20].strip()
            aa = AA3_TO_1.get(resname)
            if aa is None:
                continue
            residues.append(
                {
                    "chain": line[21].strip() or "A",
                    "resseq": int(line[22:26]),
                    "icode": line[26].strip(),
                    "aa": aa,
                    "coord": np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=float,
                    ),
                }
            )
    return residues


@lru_cache(maxsize=None)
def cached_pdb_ca_residues(pdb_path_text: str) -> tuple[dict[str, Any], ...]:
    return tuple(parse_pdb_ca_residues(Path(pdb_path_text)))


def clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def residue_key(chain: Any, resseq: Any, icode: Any) -> tuple[str, int, str] | None:
    try:
        resseq_int = int(float(str(resseq).strip()))
    except ValueError:
        return None
    chain_text = clean_optional_text(chain) or "A"
    return (chain_text, resseq_int, clean_optional_text(icode))


def compute_local_environment(
    rows: pd.DataFrame,
    radius: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, int]]:
    features: list[dict[str, Any]] = []
    skip_counts = {
        "pdb_file_not_found": 0,
        "pdb_parse_failed": 0,
        "center_residue_missing_ca": 0,
        "no_neighbors": 0,
    }
    logger.info("Neighborhood definition: CA distance <= %.2f A, excluding center residue.", radius)

    for row_index, row in rows.iterrows():
        pdb_path = Path(str(row["pdb_path"]))
        if not pdb_path.exists():
            alternate = PROJECT_ROOT / pdb_path
            pdb_path = alternate if alternate.exists() else pdb_path
        if not pdb_path.exists():
            skip_counts["pdb_file_not_found"] += 1
            continue

        try:
            residues = list(cached_pdb_ca_residues(str(pdb_path.resolve())))
        except Exception as exc:
            logger.warning("Could not parse PDB %s: %s", pdb_path, exc)
            skip_counts["pdb_parse_failed"] += 1
            continue

        center_key = residue_key(row["pdb_chain"], row["pdb_resseq"], row.get("pdb_icode", ""))
        if center_key is None:
            skip_counts["center_residue_missing_ca"] += 1
            continue

        center = None
        for residue in residues:
            current_key = (residue["chain"], residue["resseq"], residue["icode"])
            if current_key == center_key:
                center = residue
                break
        if center is None:
            skip_counts["center_residue_missing_ca"] += 1
            continue

        neighbors = []
        for residue in residues:
            current_key = (residue["chain"], residue["resseq"], residue["icode"])
            if current_key == center_key:
                continue
            distance = float(np.linalg.norm(residue["coord"] - center["coord"]))
            if distance <= radius:
                neighbors.append(residue)

        if not neighbors:
            skip_counts["no_neighbors"] += 1
            continue

        neighbor_aas = [residue["aa"] for residue in neighbors]
        neighbor_count = len(neighbor_aas)
        hydrophobic_count = sum(aa in HYDROPHOBIC_RESIDUES for aa in neighbor_aas)
        positive_count = sum(aa in POSITIVE_RESIDUES for aa in neighbor_aas)
        negative_count = sum(aa in NEGATIVE_RESIDUES for aa in neighbor_aas)
        charged_count = positive_count + negative_count
        hydrophobicity_values = [KYTE_DOOLITTLE[aa] for aa in neighbor_aas if aa in KYTE_DOOLITTLE]

        features.append(
            {
                "source_row_index": row_index,
                "protein_id": row.get("protein_id", ""),
                "abs_delta": row["abs_delta_numeric"],
                "impact_group": row["impact_group"],
                "pdb_path": str(pdb_path),
                "pdb_chain": row["pdb_chain"],
                "pdb_resseq": row["pdb_resseq"],
                "pdb_icode": row.get("pdb_icode", ""),
                "neighbor_count": neighbor_count,
                "hydrophobic_count": hydrophobic_count,
                "hydrophobic_ratio": hydrophobic_count / neighbor_count,
                "mean_hydrophobicity": float(np.mean(hydrophobicity_values)) if hydrophobicity_values else np.nan,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "charged_count": charged_count,
                "net_charge": positive_count - negative_count,
                "charged_ratio": charged_count / neighbor_count,
                "positive_ratio": positive_count / neighbor_count,
                "negative_ratio": negative_count / neighbor_count,
            }
        )

    return pd.DataFrame(features), skip_counts


def mann_whitney_pvalue(df: pd.DataFrame, metric: str, group_a: str, group_b: str) -> float:
    if mannwhitneyu is None:
        return np.nan
    values_a = df.loc[df["impact_group"] == group_a, metric].dropna()
    values_b = df.loc[df["impact_group"] == group_b, metric].dropna()
    if values_a.empty or values_b.empty:
        return np.nan
    return float(mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)


def summarize_metrics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    groups = group_order(df)
    has_low = "Low impact" in groups
    has_positive_negative = {"Positive high", "Negative high"}.issubset(groups)
    for metric in metrics:
        p_high_vs_low = mann_whitney_pvalue(df, metric, "High impact", "Low impact") if "High impact" in groups and has_low else np.nan
        p_positive_vs_low = mann_whitney_pvalue(df, metric, "Positive high", "Low impact") if "Positive high" in groups and has_low else np.nan
        p_negative_vs_low = mann_whitney_pvalue(df, metric, "Negative high", "Low impact") if "Negative high" in groups and has_low else np.nan
        p_positive_vs_negative = mann_whitney_pvalue(df, metric, "Positive high", "Negative high") if has_positive_negative else np.nan
        for group_name in groups:
            values = pd.to_numeric(df.loc[df["impact_group"] == group_name, metric], errors="coerce").dropna()
            rows.append(
                {
                    "metric": metric,
                    "group": group_name,
                    "mean": values.mean() if len(values) else np.nan,
                    "median": values.median() if len(values) else np.nan,
                    "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "min": values.min() if len(values) else np.nan,
                    "max": values.max() if len(values) else np.nan,
                    "n": len(values),
                    "mann_whitney_u_p_value_high_vs_low": p_high_vs_low,
                    "mann_whitney_u_p_value_positive_vs_low": p_positive_vs_low,
                    "mann_whitney_u_p_value_negative_vs_low": p_negative_vs_low,
                    "mann_whitney_u_p_value_positive_vs_negative": p_positive_vs_negative,
                }
            )
    return pd.DataFrame(rows)


def make_rsa_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df[df["impact_group"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    groups = group_order(df)
    p_high_vs_low = mann_whitney_pvalue(df, "rsa_numeric", "High impact", "Low impact") if {"High impact", "Low impact"}.issubset(groups) else np.nan
    p_positive_vs_low = mann_whitney_pvalue(df, "rsa_numeric", "Positive high", "Low impact") if {"Positive high", "Low impact"}.issubset(groups) else np.nan
    p_negative_vs_low = mann_whitney_pvalue(df, "rsa_numeric", "Negative high", "Low impact") if {"Negative high", "Low impact"}.issubset(groups) else np.nan
    p_positive_vs_negative = mann_whitney_pvalue(df, "rsa_numeric", "Positive high", "Negative high") if {"Positive high", "Negative high"}.issubset(groups) else np.nan
    for group_name in groups:
        group = df[df["impact_group"] == group_name].copy()
        total = int(group["rsa_numeric"].notna().sum())
        for label, lower, upper in RSA_BINS:
            values = group["rsa_numeric"]
            if math.isinf(upper):
                count = int((values >= lower).sum())
            else:
                count = int(((values >= lower) & (values < upper)).sum())
            rows.append(
                {
                    "group": group_name,
                    "rsa_bin": label,
                    "count": count,
                    "percent": (count / total * 100.0) if total else np.nan,
                    "total_with_rsa": total,
                    "mann_whitney_u_p_value_high_vs_low": p_high_vs_low,
                    "mann_whitney_u_p_value_positive_vs_low": p_positive_vs_low,
                    "mann_whitney_u_p_value_negative_vs_low": p_negative_vs_low,
                    "mann_whitney_u_p_value_positive_vs_negative": p_positive_vs_negative,
                }
            )
    return pd.DataFrame(rows)


def plot_rsa_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    labels = [item[0].replace("\\%", "%") for item in RSA_BINS]
    groups = [group for group in ["Positive high", "Negative high", "High impact", "Low impact"] if group in set(distribution["group"])]
    x = np.arange(len(labels))
    width = min(0.8 / max(len(groups), 1), 0.38)
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518"]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    for index, group_name in enumerate(groups):
        values = (
            distribution[distribution["group"] == group_name]
            .set_index("rsa_bin")
            .loc[[item[0] for item in RSA_BINS]]
        )
        offset = (index - (len(groups) - 1) / 2) * width
        ax.bar(x + offset, values["percent"], width, label=group_name, color=colors[index % len(colors)])
    ax.set_title("RSA Distribution: High vs Low Impact Sites")
    ax.set_xlabel("RSA bin")
    ax.set_ylabel("Percentage of sites")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, metrics: list[str], titles: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 5.2), dpi=180)
    if len(metrics) == 1:
        axes = [axes]
    groups = group_order(df)
    for ax, metric, title in zip(axes, metrics, titles):
        data = [
            pd.to_numeric(df.loc[df["impact_group"] == group_name, metric], errors="coerce").dropna()
            for group_name in groups
        ]
        tick_labels = [group.replace(" impact", "").replace(" high", "+ high").replace("Negative", "Neg").replace("Positive", "Pos") for group in groups]
        ax.boxplot(data, tick_labels=tick_labels, showmeans=True)
        ax.set_title(title)
        ax.set_xlabel("Impact group")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    rsa_df: pd.DataFrame,
    neighborhood_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    rsa_csv = output_dir / "rsa_distribution_high_vs_low.csv"
    rsa_png = output_dir / "rsa_distribution_high_vs_low.png"
    hydrophobic_csv = output_dir / "local_hydrophobicity_high_vs_low.csv"
    hydrophobic_png = output_dir / "local_hydrophobicity_high_vs_low.png"
    charge_csv = output_dir / "local_charge_high_vs_low.csv"
    charge_png = output_dir / "local_charge_high_vs_low.png"
    per_site_csv = output_dir / "local_environment_per_site.csv"

    rsa_df = rsa_df[rsa_df["impact_group"].notna()].copy()
    neighborhood_df = neighborhood_df[neighborhood_df["impact_group"].notna()].copy()

    rsa_distribution = make_rsa_distribution(rsa_df)
    rsa_distribution.to_csv(rsa_csv, index=False, encoding="utf-8-sig")
    plot_rsa_distribution(rsa_distribution, rsa_png)

    hydrophobic_metrics = [
        "neighbor_count",
        "hydrophobic_count",
        "hydrophobic_ratio",
        "mean_hydrophobicity",
    ]
    charge_metrics = [
        "positive_count",
        "negative_count",
        "charged_count",
        "net_charge",
        "charged_ratio",
        "positive_ratio",
        "negative_ratio",
    ]
    summarize_metrics(neighborhood_df, hydrophobic_metrics).to_csv(
        hydrophobic_csv, index=False, encoding="utf-8-sig"
    )
    summarize_metrics(neighborhood_df, charge_metrics).to_csv(
        charge_csv, index=False, encoding="utf-8-sig"
    )
    neighborhood_df.to_csv(per_site_csv, index=False, encoding="utf-8-sig")

    plot_boxplots(
        neighborhood_df,
        ["hydrophobic_ratio", "mean_hydrophobicity"],
        ["Local Hydrophobic Ratio", "Mean Local Hydrophobicity"],
        hydrophobic_png,
    )
    plot_boxplots(
        neighborhood_df,
        ["charged_ratio", "net_charge"],
        ["Local Charged Ratio", "Local Net Charge"],
        charge_png,
    )

    for path in (rsa_csv, rsa_png, hydrophobic_csv, hydrophobic_png, charge_csv, charge_png, per_site_csv):
        logger.info("Saved output: %s", path.resolve())


def print_terminal_summary(rsa_df: pd.DataFrame, neighborhood_df: pd.DataFrame) -> None:
    rsa_df = rsa_df[rsa_df["impact_group"].notna()].copy()
    neighborhood_df = neighborhood_df[neighborhood_df["impact_group"].notna()].copy()
    groups = group_order(rsa_df)

    print("\nSummary")
    for group_name in groups:
        group_rsa = rsa_df[rsa_df["impact_group"] == group_name]["rsa_numeric"].dropna()
        group_neighbors = neighborhood_df[neighborhood_df["impact_group"] == group_name]
        print(
            f"{group_name}: n_rsa={len(group_rsa)}, mean_rsa={group_rsa.mean():.4f}, "
            f"mean_hydrophobic_ratio={group_neighbors['hydrophobic_ratio'].mean():.4f}, "
            f"mean_charged_ratio={group_neighbors['charged_ratio'].mean():.4f}, "
            f"mean_net_charge={group_neighbors['net_charge'].mean():.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local structure environments for high- and low-impact key sites."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/masking/all_correct_to_wrong_2500.csv"),
        help="Key-site CSV containing abs_delta and site identifiers.",
    )
    parser.add_argument(
        "--structure-results",
        type=Path,
        default=None,
        help="Optional existing structure mapping result CSV with RSA and PDB mapping fields.",
    )
    parser.add_argument("--radius", type=float, default=8.0, help="CA-neighborhood radius in Angstrom.")
    parser.add_argument("--impact-threshold", type=float, default=0.05)
    parser.add_argument(
        "--grouping",
        choices=("threshold", "top_bottom_quartile", "signed_top_bottom"),
        default="threshold",
        help=(
            "threshold: high is abs_delta > impact-threshold and low is <= threshold; "
            "top_bottom_quartile: compare top 25%% vs bottom 25%% abs_delta and exclude the middle 50%%; "
            "signed_top_bottom: split top 25%% by delta sign and compare against bottom 25%%."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/structure/local_environment_analysis"),
    )
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir, (Path.cwd(), PROJECT_ROOT)) if args.output_dir.exists() else args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    logger = setup_logging(output_dir)

    if mannwhitneyu is None:
        logger.warning("scipy is not available; Mann-Whitney U tests will be skipped.")
    else:
        logger.info("scipy is available; Mann-Whitney U tests will be included.")

    input_path = resolve_path(args.input, (Path.cwd(), SCRIPT_DIR, PROJECT_ROOT))
    logger.info("Input key-site file: %s", input_path)
    input_df = read_csv(input_path)
    input_fields = log_detected_input_fields(input_df, logger)
    logger.info("Total key-site rows in input: %d", len(input_df))

    structure_results_path = find_structure_mapping_file(input_path, args.structure_results, logger)
    structure_df = read_csv(structure_results_path)
    logger.info("Structure result rows: %d", len(structure_df))
    logger.info("Structure result columns: %s", list(structure_df.columns))

    if "abs_delta" not in structure_df.columns:
        raise ValueError("Structure result file is missing abs_delta. Please add or merge this field.")
    structure_df = add_impact_group(structure_df, args.impact_threshold, args.grouping, logger)
    structure_df = prepare_rsa(structure_df, logger)

    grouped_counts = structure_df["impact_group"].value_counts(dropna=True).to_dict()
    logger.info("Impact group counts before structure filtering: %s", grouped_counts)
    logger.info("Ungrouped/excluded sites: %d", int(structure_df["impact_group"].isna().sum()))

    reliable_df, structure_skip_counts = filter_reliable_structure_rows(structure_df, logger)
    neighborhood_df, neighborhood_skip_counts = compute_local_environment(reliable_df, args.radius, logger)
    logger.info("Rows used for local structure neighborhood analysis: %d", len(neighborhood_df))
    logger.info("Skipped before neighborhood analysis: %s", structure_skip_counts)
    logger.info("Skipped during neighborhood analysis: %s", neighborhood_skip_counts)
    logger.info("Input field choices retained for provenance: %s", input_fields)

    save_outputs(reliable_df, neighborhood_df, output_dir, logger)
    print_terminal_summary(reliable_df, neighborhood_df)


if __name__ == "__main__":
    main()
