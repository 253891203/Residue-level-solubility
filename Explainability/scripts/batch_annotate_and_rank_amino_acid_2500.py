"""Analyze 2500-length Transformer mask predictions and rank amino acids.

The script first keeps only site masks that change a correct baseline
prediction into an incorrect prediction. It then ranks the 20 canonical amino
acids with the class-balanced score

    mean(|delta probability| for y=1) - mean(|delta probability| for y=0).

It is intentionally separate from the legacy max-length-500 analysis.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
TARGET_LENGTH = 2500
EMBED_DIM = 1280
DEFAULT_BASE_DIR = "../PLM_Sol/embeddings"
DEFAULT_RESULTS_DIR = "outputs/masking"
DEFAULT_MODEL_PATH = "../PLM_Sol/outputs/checkpoints/transformer_2500maxlen_paper.pt"
DEFAULT_MASKED_DIR = "masked_embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the full-length Transformer-2500 checkpoint to extract "
            "correct-to-wrong masked sites and build a class-balanced AA ranking."
        )
    )
    parser.add_argument(
        "--train-h5",
        default=os.path.join(
            DEFAULT_BASE_DIR, "train_esm2_650m_maxlen2500.h5"
        ),
        help="Training H5 used only for the original normalization statistics.",
    )
    parser.add_argument(
        "--test-h5",
        default=os.path.join(
            DEFAULT_BASE_DIR, "test_esm2_650m_maxlen2500.h5"
        ),
    )
    parser.add_argument(
        "--masked-h5",
        default=os.path.join(
            DEFAULT_MASKED_DIR,
            "masked_test_esm2_embeddings_2500.h5",
        ),
    )
    parser.add_argument(
        "--masked-index-csv",
        default=os.path.join(
            DEFAULT_MASKED_DIR,
            "masked_test_esm2_index_2500.csv",
        ),
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-batch-size", type=int, default=4)
    parser.add_argument("--masked-batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Atomically save transition CSVs, ranking, and resume state every N index rows.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore an existing progress state and restart the analysis from row 1.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


class ProteinSolubilityTransformer2500(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = TARGET_LENGTH,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(embeddings)
        cls = self.cls_token.expand(hidden.size(0), -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embed[:, : hidden.size(1)]
        hidden = self.encoder(hidden)
        hidden = self.norm(hidden)
        return self.classifier(hidden[:, 0]).squeeze(-1)


def load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_verified_model(
    model_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, object]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer-2500 checkpoint not found: {model_path}")
    checkpoint = load_checkpoint(model_path, device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no model_state_dict.")
    config = dict(checkpoint.get("config", {}))
    max_len = int(config.get("max_len", -1))
    embed_dim = int(config.get("embed_dim", -1))
    if max_len != TARGET_LENGTH or embed_dim != EMBED_DIM:
        raise ValueError(
            "Refusing a non-2500 checkpoint: "
            f"config.max_len={max_len}, config.embed_dim={embed_dim}"
        )

    state = checkpoint["model_state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    pos_embed = state.get("pos_embed")
    if pos_embed is None or tuple(pos_embed.shape[:2]) != (1, TARGET_LENGTH + 1):
        raise ValueError(
            f"Checkpoint position embedding is not full length: "
            f"{None if pos_embed is None else tuple(pos_embed.shape)}"
        )

    model = ProteinSolubilityTransformer2500(
        embed_dim=embed_dim,
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 8)),
        num_layers=int(config.get("num_layers", 4)),
        dim_feedforward=int(config.get("dim_feedforward", 512)),
        dropout=float(config.get("dropout", 0.1)),
        max_len=max_len,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def extract_label(protein_id: str) -> int:
    try:
        label = int(protein_id.rsplit("-", 1)[-1].strip())
    except Exception as exc:
        raise ValueError(
            f"Cannot extract binary label from protein ID ending: {protein_id}"
        ) from exc
    if label not in (0, 1):
        raise ValueError(f"Non-binary label in protein ID: {protein_id}")
    return label


class EmbeddingH5:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        with h5py.File(path, "r") as handle:
            if "id" in handle and "residue_embed" in handle:
                self.layout = "table"
                self.ids = [decode_text(value) for value in handle["id"][:]]
                self.shape = tuple(handle["residue_embed"].shape[1:])
            else:
                self.layout = "legacy"
                self.ids = list(handle.keys())
                if not self.ids:
                    raise ValueError(f"Empty H5: {path}")
                first = handle[self.ids[0]]
                if isinstance(first, h5py.Group) and "residue_embed" in first:
                    self.shape = tuple(first["residue_embed"].shape)
                else:
                    self.shape = tuple(first.shape)
        if self.shape != (TARGET_LENGTH, EMBED_DIM):
            raise ValueError(
                f"Expected full-length shape {(TARGET_LENGTH, EMBED_DIM)}, "
                f"got {self.shape} from {path}"
            )

    def read_many(self, indices: Sequence[int]) -> np.ndarray:
        with h5py.File(self.path, "r") as handle:
            arrays = []
            for index in indices:
                if self.layout == "table":
                    array = handle["residue_embed"][index]
                else:
                    item = handle[self.ids[index]]
                    array = (
                        item["residue_embed"][:]
                        if isinstance(item, h5py.Group)
                        else item[:]
                    )
                arrays.append(np.asarray(array, dtype=np.float32))
        return np.stack(arrays)


def normalization_stats_path(train_h5: str) -> str:
    basename = os.path.basename(train_h5).replace(".h5", ".npz")
    return os.path.join(os.path.dirname(train_h5), f"norm_stats_{basename}")


def compute_training_normalization(
    train_h5: EmbeddingH5, batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    total_sum = np.zeros(EMBED_DIM, dtype=np.float64)
    total_sq_sum = np.zeros(EMBED_DIM, dtype=np.float64)
    count = 0
    for start in range(0, len(train_h5.ids), batch_size):
        indices = list(range(start, min(start + batch_size, len(train_h5.ids))))
        batch = train_h5.read_many(indices)
        per_sample_mean = batch.mean(axis=1, dtype=np.float64)
        total_sum += per_sample_mean.sum(axis=0)
        total_sq_sum += np.square(per_sample_mean).sum(axis=0)
        count += len(indices)
        if start % (batch_size * 10) == 0:
            print(f"Normalization scan: {min(start + batch_size, len(train_h5.ids))}/{len(train_h5.ids)}")
    mean = total_sum / count
    variance = total_sq_sum / count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
    return mean.reshape(1, EMBED_DIM).astype(np.float32), std.reshape(
        1, EMBED_DIM
    ).astype(np.float32)


def load_or_compute_training_normalization(
    train_h5_path: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    stats_path = normalization_stats_path(train_h5_path)
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        mean = np.asarray(data["mean"], dtype=np.float32).reshape(1, EMBED_DIM)
        std = np.asarray(data["std"], dtype=np.float32).reshape(1, EMBED_DIM)
        return mean, std, stats_path
    train_h5 = EmbeddingH5(train_h5_path)
    mean, std = compute_training_normalization(train_h5)
    np.savez(stats_path, mean=mean, std=std)
    return mean, std, stats_path


def predict_probabilities(
    model: nn.Module,
    arrays: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized = (arrays.astype(np.float32) - mean) / std
    tensor = torch.from_numpy(normalized).to(device)
    with torch.inference_mode():
        return torch.sigmoid(model(tensor)).cpu().numpy()


def build_baseline_map(
    test_h5: EmbeddingH5,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for start in range(0, len(test_h5.ids), batch_size):
        indices = list(range(start, min(start + batch_size, len(test_h5.ids))))
        arrays = test_h5.read_many(indices)
        probabilities = predict_probabilities(model, arrays, mean, std, device)
        for index, probability in zip(indices, probabilities):
            protein_id = test_h5.ids[index]
            label = extract_label(protein_id)
            prediction = int(float(probability) >= threshold)
            result[protein_id] = {
                "label": label,
                "probability": float(probability),
                "prediction": prediction,
                "correct": prediction == label,
            }
        print(f"Baseline predictions: {min(start + batch_size, len(test_h5.ids))}/{len(test_h5.ids)}")
    return result


def pad_masked_embedding(array: np.ndarray, sequence_length: int) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] != EMBED_DIM:
        raise ValueError(f"Invalid masked embedding shape: {array.shape}")
    if sequence_length > TARGET_LENGTH or array.shape[0] > TARGET_LENGTH:
        raise ValueError(
            f"Masked embedding exceeds {TARGET_LENGTH}: "
            f"seq_len={sequence_length}, shape={array.shape}"
        )
    if array.shape[0] < sequence_length:
        raise ValueError(
            f"Masked embedding is shorter than seq_len: {array.shape[0]} < {sequence_length}"
        )
    padded = np.zeros((TARGET_LENGTH, EMBED_DIM), dtype=np.float32)
    padded[:sequence_length] = array[:sequence_length].astype(np.float32)
    return padded


def build_class_balanced_ranking(
    correct_to_wrong_rows: Iterable[Dict[str, object]],
) -> List[Dict[str, object]]:
    stats = defaultdict(
        lambda: {
            "count_y1": 0,
            "sum_abs_delta_y1": 0.0,
            "count_y0": 0,
            "sum_abs_delta_y0": 0.0,
        }
    )
    for row in correct_to_wrong_rows:
        amino_acid = str(row["orig_aa"]).strip().upper()
        label = int(row["label"])
        if amino_acid not in AA20 or label not in (0, 1):
            continue
        stats[amino_acid][f"count_y{label}"] += 1
        stats[amino_acid][f"sum_abs_delta_y{label}"] += abs(
            float(row["abs_delta"])
        )

    rows: List[Dict[str, object]] = []
    for amino_acid in AA20:
        values = stats[amino_acid]
        count_y1 = int(values["count_y1"])
        count_y0 = int(values["count_y0"])
        mean_y1: Optional[float] = (
            values["sum_abs_delta_y1"] / count_y1 if count_y1 else None
        )
        mean_y0: Optional[float] = (
            values["sum_abs_delta_y0"] / count_y0 if count_y0 else None
        )
        score: Optional[float] = (
            mean_y1 - mean_y0
            if mean_y1 is not None and mean_y0 is not None
            else None
        )
        rows.append(
            {
                "rank": "",
                "amino_acid": amino_acid,
                "total_correct_to_wrong_count": count_y1 + count_y0,
                "count_y1_soluble": count_y1,
                "count_y0_insoluble": count_y0,
                "sum_abs_delta_y1_soluble": values["sum_abs_delta_y1"],
                "sum_abs_delta_y0_insoluble": values["sum_abs_delta_y0"],
                "mean_abs_delta_y1_soluble": "" if mean_y1 is None else mean_y1,
                "mean_abs_delta_y0_insoluble": "" if mean_y0 is None else mean_y0,
                "class_balanced_mean_difference": "" if score is None else score,
                "ranking_status": "ok" if score is not None else "missing_one_class",
            }
        )

    ranked = [row for row in rows if row["ranking_status"] == "ok"]
    ranked.sort(
        key=lambda row: (
            -float(row["class_balanced_mean_difference"]),
            str(row["amino_acid"]),
        )
    )
    missing = sorted(
        (row for row in rows if row["ranking_status"] != "ok"),
        key=lambda row: str(row["amino_acid"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked + missing


def save_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    if not rows:
        temporary_path.write_text("", encoding="utf-8")
        os.replace(temporary_path, path)
        return
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def load_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_json_atomic(payload: Dict[str, object], path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def save_ranking_plot(rows: Sequence[Dict[str, object]], path: Path) -> bool:
    comparable = [row for row in rows if row["ranking_status"] == "ok"]
    if not comparable:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    amino_acids = [str(row["amino_acid"]) for row in comparable]
    scores = [
        float(row["class_balanced_mean_difference"]) for row in comparable
    ]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in scores]
    plt.figure(figsize=(12, 5))
    plt.bar(amino_acids, scores, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Amino acid")
    plt.ylabel("Mean |ΔP| (y=1) - Mean |ΔP| (y=0)")
    plt.title("Transformer-2500 class-balanced amino-acid ranking")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def print_complete_ranking(rows: Sequence[Dict[str, object]]) -> None:
    comparable = [row for row in rows if row["ranking_status"] == "ok"]
    missing = [row for row in rows if row["ranking_status"] != "ok"]
    print("\n完整20种标准氨基酸排名（类别平衡平均差降序）:")
    print(" > ".join(str(row["amino_acid"]) for row in comparable))
    for row in comparable:
        print(
            f"{int(row['rank']):2d}. {row['amino_acid']}: "
            f"score={float(row['class_balanced_mean_difference']):.8f}, "
            f"mean_y1={float(row['mean_abs_delta_y1_soluble']):.8f}, "
            f"mean_y0={float(row['mean_abs_delta_y0_insoluble']):.8f}, "
            f"n_y1={row['count_y1_soluble']}, "
            f"n_y0={row['count_y0_insoluble']}"
        )
    if missing:
        print(
            "无法排名（缺少y=0或y=1记录）: "
            + ", ".join(str(row["amino_acid"]) for row in missing)
        )


def main() -> None:
    args = parse_args()
    if args.baseline_batch_size <= 0 or args.masked_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")
    for required in (
        args.train_h5,
        args.test_h5,
        args.masked_h5,
        args.masked_index_csv,
        args.model_path,
    ):
        if not os.path.exists(required):
            raise FileNotFoundError(required)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    model, model_config = load_verified_model(args.model_path, device)
    test_h5 = EmbeddingH5(args.test_h5)
    mean, std, stats_path = load_or_compute_training_normalization(args.train_h5)
    baseline_map = build_baseline_map(
        test_h5=test_h5,
        model=model,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.baseline_batch_size,
        threshold=args.threshold,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    c2w_path = results_dir / "all_correct_to_wrong_2500.csv"
    w2c_path = results_dir / "all_wrong_to_correct_2500.csv"
    ranking_path = results_dir / "amino_acid_ranking_c2w_balanced_2500.csv"
    plot_path = results_dir / "amino_acid_ranking_c2w_balanced_2500.png"
    metadata_path = results_dir / "run_metadata_2500.json"
    progress_path = results_dir / "analysis_progress_2500.json"

    counters = {
        "index_rows": 0,
        "baseline_correct_rows": 0,
        "baseline_wrong_rows": 0,
        "missing_baseline_rows": 0,
        "correct_to_wrong": 0,
        "correct_to_correct": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    c2w_rows: List[Dict[str, object]] = []
    w2c_rows: List[Dict[str, object]] = []
    resume_row = 0
    run_signature = {
        "masked_h5": os.path.abspath(args.masked_h5),
        "masked_index_csv": os.path.abspath(args.masked_index_csv),
        "model_path": os.path.abspath(args.model_path),
        "threshold": float(args.threshold),
        "target_length": TARGET_LENGTH,
    }
    if progress_path.exists() and not args.restart:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("run_signature") != run_signature:
            raise ValueError(
                "Existing resume state belongs to different inputs/model/options. "
                f"Use another --results-dir or pass --restart: {progress_path}"
            )
        resume_row = int(progress.get("processed_index_rows", 0))
        saved_counters = progress.get("counters", {})
        for key in counters:
            counters[key] = int(saved_counters.get(key, counters[key]))
        c2w_rows = load_csv(c2w_path)
        w2c_rows = load_csv(w2c_path)
        if len(c2w_rows) != counters["correct_to_wrong"]:
            raise ValueError(
                "Resume C2W CSV/count mismatch: "
                f"{len(c2w_rows)} != {counters['correct_to_wrong']}"
            )
        if len(w2c_rows) != counters["wrong_to_correct"]:
            raise ValueError(
                "Resume W2C CSV/count mismatch: "
                f"{len(w2c_rows)} != {counters['wrong_to_correct']}"
            )
        print(
            f"Resuming after index row {resume_row}: "
            f"C2W={len(c2w_rows)}, W2C={len(w2c_rows)}"
        )
    elif args.restart:
        print("Restart requested: existing transition/progress outputs will be replaced.")

    pending_arrays: List[np.ndarray] = []
    pending_info: List[Dict[str, object]] = []

    def flush_pending() -> None:
        if not pending_arrays:
            return
        arrays = np.stack(pending_arrays)
        probabilities = predict_probabilities(model, arrays, mean, std, device)
        for info, masked_probability in zip(pending_info, probabilities):
            base = info["baseline"]
            label = int(base["label"])
            masked_probability = float(masked_probability)
            masked_prediction = int(masked_probability >= args.threshold)
            baseline_correct = bool(base["correct"])
            masked_correct = masked_prediction == label
            if baseline_correct and masked_correct:
                counters["correct_to_correct"] += 1
                continue
            if not baseline_correct and not masked_correct:
                counters["wrong_to_wrong"] += 1
                continue
            baseline_probability = float(base["probability"])
            delta = masked_probability - baseline_probability
            transition = (
                "correct_to_wrong" if baseline_correct else "wrong_to_correct"
            )
            record = {
                "index_row": info["index_row"],
                "protein_id": info["protein_id"],
                "label": label,
                "mask_pos": info["mask_pos"],
                "mask_pos_1based": int(info["mask_pos"]) + 1,
                "orig_aa": info["orig_aa"],
                "baseline_probability": baseline_probability,
                "masked_probability": masked_probability,
                "delta": delta,
                "abs_delta": abs(delta),
                "baseline_prediction": int(base["prediction"]),
                "masked_prediction": masked_prediction,
                "transition": transition,
                "model_path": args.model_path,
                "max_len": TARGET_LENGTH,
            }
            if baseline_correct:
                c2w_rows.append(record)
                counters["correct_to_wrong"] += 1
            else:
                w2c_rows.append(record)
                counters["wrong_to_correct"] += 1
        pending_arrays.clear()
        pending_info.clear()

    def save_progress(processed_index_rows: int, status: str = "running") -> None:
        c2w_rows.sort(key=lambda row: int(row["index_row"]))
        w2c_rows.sort(key=lambda row: int(row["index_row"]))
        save_csv(c2w_rows, c2w_path)
        save_csv(w2c_rows, w2c_path)
        save_csv(build_class_balanced_ranking(c2w_rows), ranking_path)
        save_json_atomic(
            {
                "status": status,
                "processed_index_rows": processed_index_rows,
                "run_signature": run_signature,
                "counters": counters,
                "correct_to_wrong_csv": str(c2w_path),
                "wrong_to_correct_csv": str(w2c_path),
                "ranking_csv_c2w_only": str(ranking_path),
                "saved_unix_time": time.time(),
            },
            progress_path,
        )
        print(
            f"Checkpoint saved at row {processed_index_rows}: "
            f"C2W={len(c2w_rows)}, W2C={len(w2c_rows)}"
        )

    with h5py.File(args.masked_h5, "r") as masked_h5, open(
        args.masked_index_csv, "r", newline="", encoding="utf-8"
    ) as index_handle:
        if int(masked_h5.attrs.get("target_length", -1)) != TARGET_LENGTH:
            raise ValueError(
                "Masked H5 is not tagged as the 2500 string-mask pipeline."
            )
        if str(masked_h5.attrs.get("mask_token", "")) != "<mask>":
            raise ValueError("Masked H5 was not generated with literal <mask>.")

        for index_row, row in enumerate(csv.DictReader(index_handle), start=1):
            if index_row <= resume_row:
                continue
            counters["index_rows"] = index_row
            protein_id = row["protein_id"]
            baseline = baseline_map.get(protein_id)
            if baseline is None:
                counters["missing_baseline_rows"] += 1
            else:
                if bool(baseline["correct"]):
                    counters["baseline_correct_rows"] += 1
                else:
                    counters["baseline_wrong_rows"] += 1

                seq_idx = int(row["seq_idx"])
                mask_pos = int(row["mask_pos"])
                sequence_length = int(row["seq_len"])
                dataset_path = f"seq_{seq_idx:06d}/{row['masked_key']}"
                if dataset_path not in masked_h5:
                    raise KeyError(f"Masked dataset missing: {dataset_path}")
                valid_array = masked_h5[dataset_path][:]
                pending_arrays.append(
                    pad_masked_embedding(valid_array, sequence_length)
                )
                pending_info.append(
                    {
                        "index_row": index_row,
                        "protein_id": protein_id,
                        "mask_pos": mask_pos,
                        "orig_aa": row["orig_aa"],
                        "baseline": baseline,
                    }
                )
                if len(pending_arrays) >= args.masked_batch_size:
                    flush_pending()
            if index_row % args.save_every == 0:
                flush_pending()
                save_progress(index_row)
                print(
                    f"Masked rows: {index_row}, "
                    f"correct_to_wrong: {counters['correct_to_wrong']}, "
                    f"wrong_to_correct: {counters['wrong_to_correct']}"
                )
    flush_pending()
    save_progress(counters["index_rows"], status="complete")

    c2w_rows.sort(key=lambda row: float(row["abs_delta"]), reverse=True)
    w2c_rows.sort(key=lambda row: float(row["abs_delta"]), reverse=True)
    save_csv(c2w_rows, c2w_path)
    save_csv(w2c_rows, w2c_path)
    ranking_rows = build_class_balanced_ranking(c2w_rows)
    save_csv(ranking_rows, ranking_path)
    plot_created = False if args.no_plot else save_ranking_plot(ranking_rows, plot_path)

    class_counts = {
        "correct_to_wrong_y1": sum(
            1 for row in c2w_rows if int(row["label"]) == 1
        ),
        "correct_to_wrong_y0": sum(
            1 for row in c2w_rows if int(row["label"]) == 0
        ),
        "wrong_to_correct_y1": sum(
            1 for row in w2c_rows if int(row["label"]) == 1
        ),
        "wrong_to_correct_y0": sum(
            1 for row in w2c_rows if int(row["label"]) == 0
        ),
    }
    metadata = {
        "pipeline": "transformer_2500_string_mask_correct_to_wrong_ranking",
        "model_type": "ProteinSolubilityTransformer",
        "model_path": args.model_path,
        "model_config": model_config,
        "verified_max_len": TARGET_LENGTH,
        "train_h5": args.train_h5,
        "test_h5": args.test_h5,
        "masked_h5": args.masked_h5,
        "masked_index_csv": args.masked_index_csv,
        "normalization_stats": stats_path,
        "mask_strategy": "literal_<mask>_before_esm2",
        "recorded_transitions": [
            "baseline_correct_and_masked_incorrect",
            "baseline_incorrect_and_masked_correct",
        ],
        "ranking_formula": (
            "mean(abs_delta | amino_acid, label=1, correct_to_wrong) - "
            "mean(abs_delta | amino_acid, label=0, correct_to_wrong)"
        ),
        "ranking_uses": "correct_to_wrong_only",
        "counters": counters,
        "class_counts": class_counts,
        "correct_to_wrong_csv": str(c2w_path),
        "wrong_to_correct_csv": str(w2c_path),
        "ranking_csv": str(ranking_path),
        "ranking_plot": str(plot_path) if plot_created else None,
        "progress_state": str(progress_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Transformer-2500 masked analysis complete =====")
    print(f"Verified checkpoint: {args.model_path}")
    print(f"Correct-to-wrong CSV: {c2w_path}")
    print(f"Wrong-to-correct CSV: {w2c_path}")
    print(f"Class-balanced ranking CSV: {ranking_path}")
    print(f"C2W y=1/y=0: {class_counts['correct_to_wrong_y1']}/{class_counts['correct_to_wrong_y0']}")
    print(f"W2C y=1/y=0: {class_counts['wrong_to_correct_y1']}/{class_counts['wrong_to_correct_y0']}")
    print_complete_ranking(ranking_rows)


if __name__ == "__main__":
    main()
