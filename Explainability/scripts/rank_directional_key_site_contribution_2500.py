"""Rank amino acids by frequency-corrected directional C2W contribution.

This is a new post-processing pipeline for the Transformer-2500 mask results.
It does not repeat masked-embedding generation or masked-site inference.

For amino acid ``a`` it calculates, separately for baseline-correct soluble and
baseline-correct insoluble proteins,

    key_rate = C2W_count / all_tested_occurrences_in_baseline_correct_proteins
    directional_importance = key_rate * mean_abs_delta_given_C2W

which is exactly

    directional_importance = sum_abs_delta_C2W / all_tested_occurrences.

The final score is

    net_directional_score = importance_soluble - importance_insoluble.

Only one inexpensive baseline prediction per test protein is needed to decide
which index rows belong in the denominators. Baseline predictions are cached;
the roughly one-million masked variants are never inferred again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    import batch_annotate_and_rank_amino_acid_2500 as original_pipeline
except ImportError as exc:
    raise ImportError(
        "Place this script in the same directory as "
        "batch_annotate_and_rank_amino_acid_2500.py"
    ) from exc


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
TARGET_LENGTH = 2500
DEFAULT_BASE_DIR = "../PLM_Sol/embeddings"
DEFAULT_RESULTS_DIR = "outputs/masking"
DEFAULT_MODEL_PATH = "../PLM_Sol/outputs/checkpoints/transformer_2500maxlen_paper.pt"
DEFAULT_MASKED_DIR = "masked_embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frequency-corrected directional key-site ranking from "
            "existing Transformer-2500 C2W results. Masked inference is not run."
        )
    )
    parser.add_argument(
        "--train-h5",
        default=os.path.join(
            DEFAULT_BASE_DIR, "train_esm2_650m_maxlen2500.h5"
        ),
        help="Used only if the saved normalization statistics are unavailable.",
    )
    parser.add_argument(
        "--test-h5",
        default=os.path.join(
            DEFAULT_BASE_DIR, "test_esm2_650m_maxlen2500.h5"
        ),
        help="Unmasked test embeddings used for one baseline prediction/protein.",
    )
    parser.add_argument(
        "--masked-index-csv",
        default=os.path.join(
            DEFAULT_MASKED_DIR,
            "masked_test_esm2_index_2500.csv",
        ),
        help="Complete one-row-per-tested-position mask index.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--c2w-csv",
        default=os.path.join(DEFAULT_RESULTS_DIR, "all_correct_to_wrong_2500.csv"),
        help="Existing correct-to-wrong transition CSV.",
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--baseline-cache",
        default=None,
        help=(
            "Optional baseline-prediction cache path. The default is "
            "<results-dir>/baseline_predictions_directional_2500.csv."
        ),
    )
    parser.add_argument(
        "--recompute-baseline",
        action="store_true",
        help="Ignore a compatible baseline cache and recompute baseline predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200000,
        help="Print mask-index scan progress every N rows; use 0 to disable.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def require_columns(
    fieldnames: Optional[Sequence[str]], required: Iterable[str], path: Path
) -> None:
    available = set(fieldnames or [])
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")


def atomic_write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    if not rows:
        temporary_path.write_text("", encoding="utf-8")
        os.replace(temporary_path, path)
        return
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def baseline_signature(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "test_h5": os.path.abspath(args.test_h5),
        "model_path": os.path.abspath(args.model_path),
        "threshold": float(args.threshold),
        "target_length": TARGET_LENGTH,
    }


def save_baseline_cache(
    baseline_map: Mapping[str, Mapping[str, object]],
    cache_path: Path,
    signature: Mapping[str, object],
) -> None:
    rows: List[Dict[str, object]] = []
    for protein_id, values in baseline_map.items():
        rows.append(
            {
                "protein_id": protein_id,
                "label": int(values["label"]),
                "baseline_probability": float(values["probability"]),
                "baseline_prediction": int(values["prediction"]),
                "baseline_correct": int(bool(values["correct"])),
            }
        )
    atomic_write_csv(rows, cache_path)
    atomic_write_json(
        {"signature": dict(signature), "protein_count": len(rows)},
        cache_path.with_suffix(cache_path.suffix + ".metadata.json"),
    )


def load_baseline_cache(
    cache_path: Path, expected_signature: Mapping[str, object]
) -> Dict[str, Dict[str, object]]:
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".metadata.json")
    if not metadata_path.exists():
        raise ValueError(f"Baseline cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != dict(expected_signature):
        raise ValueError(
            "Baseline cache was produced with different test/model/threshold settings: "
            f"{metadata_path}"
        )

    baseline_map: Dict[str, Dict[str, object]] = {}
    with cache_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            reader.fieldnames,
            [
                "protein_id",
                "label",
                "baseline_probability",
                "baseline_prediction",
                "baseline_correct",
            ],
            cache_path,
        )
        for row_number, row in enumerate(reader, start=1):
            protein_id = row["protein_id"]
            if protein_id in baseline_map:
                raise ValueError(
                    f"Duplicate protein_id in baseline cache row {row_number}: "
                    f"{protein_id!r}"
                )
            label = int(row["label"])
            prediction = int(row["baseline_prediction"])
            probability = float(row["baseline_probability"])
            correct = bool(int(row["baseline_correct"]))
            if label not in (0, 1) or prediction not in (0, 1):
                raise ValueError(f"Invalid binary value for {protein_id!r}")
            if correct != (prediction == label):
                raise ValueError(f"Inconsistent baseline correctness for {protein_id!r}")
            baseline_map[protein_id] = {
                "label": label,
                "probability": probability,
                "prediction": prediction,
                "correct": correct,
            }

    expected_count = int(metadata.get("protein_count", -1))
    if expected_count != len(baseline_map):
        raise ValueError(
            f"Baseline cache count mismatch: metadata={expected_count}, "
            f"CSV={len(baseline_map)}"
        )
    return baseline_map


def get_or_build_baseline_map(
    args: argparse.Namespace, cache_path: Path
) -> Tuple[Dict[str, Dict[str, object]], str, Optional[str]]:
    signature = baseline_signature(args)
    if cache_path.exists() and not args.recompute_baseline:
        baseline_map = load_baseline_cache(cache_path, signature)
        print(f"Loaded cached baseline predictions: {len(baseline_map)} proteins")
        return baseline_map, "cache", None

    for required in (args.train_h5, args.test_h5, args.model_path):
        if not os.path.exists(required):
            raise FileNotFoundError(required)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; using CPU instead of {requested_device}.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    model, _ = original_pipeline.load_verified_model(args.model_path, device)
    test_h5 = original_pipeline.EmbeddingH5(args.test_h5)
    mean, std, stats_path = (
        original_pipeline.load_or_compute_training_normalization(args.train_h5)
    )
    baseline_map = original_pipeline.build_baseline_map(
        test_h5=test_h5,
        model=model,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.baseline_batch_size,
        threshold=args.threshold,
    )
    save_baseline_cache(baseline_map, cache_path, signature)
    print(f"Saved baseline prediction cache: {cache_path}")
    return baseline_map, "computed", stats_path


def load_c2w(
    path: Path,
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, Tuple[str, int, str]], int, int]:
    stats: Dict[str, Dict[str, float]] = {
        aa: {
            "count_soluble": 0,
            "count_insoluble": 0,
            "sum_soluble": 0.0,
            "sum_insoluble": 0.0,
        }
        for aa in AA20
    }
    expected_index_rows: Dict[int, Tuple[str, int, str]] = {}
    total_rows = 0
    canonical_rows = 0

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            reader.fieldnames,
            [
                "index_row",
                "protein_id",
                "label",
                "orig_aa",
                "abs_delta",
                "transition",
                "baseline_prediction",
                "masked_prediction",
            ],
            path,
        )
        for row in reader:
            total_rows += 1
            index_row = int(row["index_row"])
            protein_id = row["protein_id"]
            label = int(row["label"])
            amino_acid = row["orig_aa"].strip().upper()
            abs_delta = abs(float(row["abs_delta"]))
            baseline_prediction = int(row["baseline_prediction"])
            masked_prediction = int(row["masked_prediction"])

            if row["transition"].strip() != "correct_to_wrong":
                raise ValueError(f"Non-C2W transition at index row {index_row}")
            if label not in (0, 1):
                raise ValueError(f"Invalid label {label} at index row {index_row}")
            if baseline_prediction != label or masked_prediction == label:
                raise ValueError(
                    f"Row {index_row} does not satisfy correct-to-wrong semantics"
                )
            if index_row in expected_index_rows:
                raise ValueError(f"Duplicate C2W index_row: {index_row}")
            if not math.isfinite(abs_delta):
                raise ValueError(f"Non-finite abs_delta at index row {index_row}")

            expected_index_rows[index_row] = (amino_acid, label, protein_id)
            if amino_acid not in stats:
                continue
            canonical_rows += 1
            direction = "soluble" if label == 1 else "insoluble"
            stats[amino_acid][f"count_{direction}"] += 1
            stats[amino_acid][f"sum_{direction}"] += abs_delta

    return stats, expected_index_rows, total_rows, canonical_rows


def scan_correct_exposures(
    index_path: Path,
    baseline_map: Mapping[str, Mapping[str, object]],
    expected_c2w_rows: Mapping[int, Tuple[str, int, str]],
    progress_every: int,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    exposure: Dict[str, Dict[str, int]] = {
        aa: {"correct_soluble": 0, "correct_insoluble": 0} for aa in AA20
    }
    matched_c2w_rows = set()
    counters = {
        "index_rows": 0,
        "baseline_correct_rows": 0,
        "baseline_wrong_rows": 0,
        "noncanonical_correct_rows": 0,
        "missing_baseline_rows": 0,
    }

    with index_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            reader.fieldnames, ["protein_id", "orig_aa"], index_path
        )
        for index_row, row in enumerate(reader, start=1):
            counters["index_rows"] = index_row
            protein_id = row["protein_id"]
            amino_acid = row["orig_aa"].strip().upper()
            baseline = baseline_map.get(protein_id)
            if baseline is None:
                counters["missing_baseline_rows"] += 1
                continue

            label = int(baseline["label"])
            if bool(baseline["correct"]):
                counters["baseline_correct_rows"] += 1
                if amino_acid in exposure:
                    direction = "soluble" if label == 1 else "insoluble"
                    exposure[amino_acid][f"correct_{direction}"] += 1
                else:
                    counters["noncanonical_correct_rows"] += 1
            else:
                counters["baseline_wrong_rows"] += 1

            expected = expected_c2w_rows.get(index_row)
            if expected is not None:
                expected_aa, expected_label, expected_protein = expected
                if (amino_acid, label, protein_id) != (
                    expected_aa,
                    expected_label,
                    expected_protein,
                ):
                    raise ValueError(
                        "C2W/index mismatch at row "
                        f"{index_row}: C2W={expected}, "
                        f"index={(amino_acid, label, protein_id)}"
                    )
                if not bool(baseline["correct"]):
                    raise ValueError(
                        f"C2W row {index_row} belongs to a baseline-wrong protein"
                    )
                matched_c2w_rows.add(index_row)

            if progress_every > 0 and index_row % progress_every == 0:
                print(f"Scanned mask index rows: {index_row}")

    missing = sorted(set(expected_c2w_rows) - matched_c2w_rows)
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(
            f"{len(missing)} C2W rows were not matched to the index; "
            f"first missing index rows: {preview}"
        )
    if counters["missing_baseline_rows"]:
        raise ValueError(
            "Mask index contains rows without baseline predictions: "
            f"{counters['missing_baseline_rows']}"
        )
    return exposure, counters


def divide(numerator: float, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def build_ranking_rows(
    c2w_stats: Mapping[str, Mapping[str, float]],
    exposure: Mapping[str, Mapping[str, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for amino_acid in AA20:
        c2w = c2w_stats[amino_acid]
        counts = exposure[amino_acid]
        n_soluble = int(c2w["count_soluble"])
        n_insoluble = int(c2w["count_insoluble"])
        total_soluble = int(counts["correct_soluble"])
        total_insoluble = int(counts["correct_insoluble"])
        sum_soluble = float(c2w["sum_soluble"])
        sum_insoluble = float(c2w["sum_insoluble"])

        key_rate_soluble = divide(n_soluble, total_soluble)
        key_rate_insoluble = divide(n_insoluble, total_insoluble)
        mean_soluble = divide(sum_soluble, n_soluble)
        mean_insoluble = divide(sum_insoluble, n_insoluble)
        importance_soluble = divide(sum_soluble, total_soluble)
        importance_insoluble = divide(sum_insoluble, total_insoluble)

        values = (
            key_rate_soluble,
            key_rate_insoluble,
            mean_soluble,
            mean_insoluble,
            importance_soluble,
            importance_insoluble,
        )
        complete = all(value is not None for value in values)
        if complete:
            # These identities are central to the requested ranking definition.
            assert math.isclose(
                float(importance_soluble),
                float(key_rate_soluble) * float(mean_soluble),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            assert math.isclose(
                float(importance_insoluble),
                float(key_rate_insoluble) * float(mean_insoluble),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            net_score: object = float(importance_soluble) - float(
                importance_insoluble
            )
        else:
            net_score = ""

        rows.append(
            {
                "rank": "",
                "amino_acid": amino_acid,
                "count_soluble_to_insoluble": n_soluble,
                "count_insoluble_to_soluble": n_insoluble,
                "total_occurrence_in_correct_soluble": total_soluble,
                "total_occurrence_in_correct_insoluble": total_insoluble,
                "key_rate_soluble": (
                    "" if key_rate_soluble is None else key_rate_soluble
                ),
                "key_rate_insoluble": (
                    "" if key_rate_insoluble is None else key_rate_insoluble
                ),
                "mean_abs_delta_soluble": (
                    "" if mean_soluble is None else mean_soluble
                ),
                "mean_abs_delta_insoluble": (
                    "" if mean_insoluble is None else mean_insoluble
                ),
                "directional_importance_soluble": (
                    "" if importance_soluble is None else importance_soluble
                ),
                "directional_importance_insoluble": (
                    "" if importance_insoluble is None else importance_insoluble
                ),
                "net_directional_score": net_score,
                "sum_abs_delta_soluble": sum_soluble,
                "sum_abs_delta_insoluble": sum_insoluble,
                "raw_net_sum_abs_delta": sum_soluble - sum_insoluble,
                "ranking_status": "ok" if complete else "missing_data",
            }
        )

    comparable = [row for row in rows if row["ranking_status"] == "ok"]
    comparable.sort(
        key=lambda row: (
            -float(row["net_directional_score"]), str(row["amino_acid"])
        )
    )
    for rank, row in enumerate(comparable, start=1):
        row["rank"] = rank
    missing = sorted(
        (row for row in rows if row["ranking_status"] != "ok"),
        key=lambda row: str(row["amino_acid"]),
    )
    return comparable + missing


def save_plot(rows: Sequence[Mapping[str, object]], path: Path) -> bool:
    comparable = [row for row in rows if row["ranking_status"] == "ok"]
    if not comparable:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV/JSON outputs were still created.")
        return False

    amino_acids = [str(row["amino_acid"]) for row in comparable]
    scores = [float(row["net_directional_score"]) for row in comparable]
    colors = ["#54A24B" if score >= 0 else "#E45756" for score in scores]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(amino_acids, scores, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Amino acid")
    axis.set_ylabel("Frequency-corrected net directional score")
    axis.set_title("Transformer-2500 directional key-site contribution ranking")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def main() -> None:
    args = parse_args()
    if args.baseline_batch_size <= 0:
        raise ValueError("--baseline-batch-size must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.masked_index_csv)
    c2w_path = Path(args.c2w_csv)
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not c2w_path.exists():
        raise FileNotFoundError(c2w_path)

    cache_path = (
        Path(args.baseline_cache)
        if args.baseline_cache
        else results_dir / "baseline_predictions_directional_2500.csv"
    )
    baseline_map, baseline_source, normalization_stats = get_or_build_baseline_map(
        args, cache_path
    )
    c2w_stats, expected_rows, c2w_total, canonical_c2w_total = load_c2w(
        c2w_path
    )
    exposure, counters = scan_correct_exposures(
        index_path=index_path,
        baseline_map=baseline_map,
        expected_c2w_rows=expected_rows,
        progress_every=args.progress_every,
    )
    ranking_rows = build_ranking_rows(c2w_stats, exposure)

    ranking_path = results_dir / "directional_key_site_ranking_2500.csv"
    plot_path = results_dir / "directional_key_site_ranking_2500.png"
    summary_path = results_dir / "directional_key_site_ranking_summary_2500.json"
    atomic_write_csv(ranking_rows, ranking_path)
    plot_created = False if args.no_plot else save_plot(ranking_rows, plot_path)

    ranked = [row for row in ranking_rows if row["ranking_status"] == "ok"]
    ranking_text = " > ".join(str(row["amino_acid"]) for row in ranked)
    summary = {
        "pipeline": "transformer_2500_directional_key_site_postprocessing",
        "runs_masked_inference": False,
        "baseline_predictions": baseline_source,
        "baseline_cache": str(cache_path),
        "normalization_stats": normalization_stats,
        "model_path": args.model_path,
        "threshold": args.threshold,
        "masked_index_csv": str(index_path),
        "correct_to_wrong_csv": str(c2w_path),
        "denominator_definition": (
            "all tested occurrences of the amino acid within baseline-correct "
            "proteins, calculated separately for y=1 and y=0"
        ),
        "directional_importance_soluble_formula": (
            "sum(abs_delta | C2W, amino_acid, y=1) / "
            "count(all tested occurrences | amino_acid, y=1, baseline_correct)"
        ),
        "directional_importance_insoluble_formula": (
            "sum(abs_delta | C2W, amino_acid, y=0) / "
            "count(all tested occurrences | amino_acid, y=0, baseline_correct)"
        ),
        "net_directional_score_formula": (
            "directional_importance_soluble - directional_importance_insoluble"
        ),
        "counters": counters,
        "c2w_rows": c2w_total,
        "canonical_c2w_rows": canonical_c2w_total,
        "ranking": ranking_text,
        "ranking_csv": str(ranking_path),
        "ranking_plot": str(plot_path) if plot_created else None,
    }
    atomic_write_json(summary, summary_path)

    print("\n===== Directional key-site contribution ranking complete =====")
    print(f"Baseline source: {baseline_source}")
    print(f"Index rows: {counters['index_rows']}")
    print(f"Baseline-correct denominator rows: {counters['baseline_correct_rows']}")
    print(f"C2W rows validated: {c2w_total}")
    print(f"Ranking: {ranking_text}")
    print(f"Ranking CSV: {ranking_path}")
    print(f"Plot: {plot_path if plot_created else 'not created'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
