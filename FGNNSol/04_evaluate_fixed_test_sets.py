from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import TransformerRegressor
from utils.data_utils import EmbeddingDataset, TokenBatchSampler, collate_embeddings
from utils.metrics import METRIC_NAMES, calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the five FGNNSol paper checkpoints.")
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--embedding_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens_per_batch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def predict(checkpoint_path, dataset, args, device, seed):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TransformerRegressor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    sampler = TokenBatchSampler(
        [row["sequence_length"] for row in dataset.rows],
        args.batch_size,
        args.max_tokens_per_batch,
        False,
        seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_embeddings,
        num_workers=args.num_workers,
    )

    rows = []
    with torch.inference_mode():
        for embeddings, padding_mask, labels, metadata, _ in loader:
            predictions = model(
                embeddings.to(device),
                padding_mask.to(device),
            ).cpu().numpy()
            for label, prediction, row in zip(labels.numpy(), predictions, metadata):
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "original_id": row["original_id"],
                        "sequence_length": row["sequence_length"],
                        "true_solubility": label,
                        "predicted_solubility": float(prediction),
                        "residual": float(prediction - label),
                        "true_class_at_0.5": int(label >= 0.5),
                        "predicted_class_at_0.5": int(prediction >= 0.5),
                        "seed": seed,
                        "split": "internal_test",
                    }
                )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = args.project_root.resolve()
    embedding_dir = (args.embedding_dir or root / "cache" / "esm2_650m_embeddings").resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EmbeddingDataset(
        embedding_dir / "embedding_manifest.csv",
        "internal_test",
        embedding_dir,
    )

    rows_by_seed = []
    for seed in args.seeds:
        checkpoint_path = checkpoint_dir / f"seed_{seed}_best_model.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        seed_result_dir = result_dir / f"seed_{seed}"
        seed_result_dir.mkdir(parents=True, exist_ok=True)
        predictions = predict(checkpoint_path, dataset, args, device, seed)
        predictions.to_csv(seed_result_dir / "internal_test_predictions.csv", index=False)
        metrics = calculate_metrics(
            predictions["true_solubility"],
            predictions["predicted_solubility"],
        )
        (seed_result_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        rows_by_seed.append({"seed": seed, **metrics})

    metrics_by_seed = pd.DataFrame(rows_by_seed)
    metrics_by_seed.to_csv(result_dir / "internal_test_metrics_by_seed.csv", index=False)
    summary = {
        metric: {
            "mean": float(metrics_by_seed[metric].mean()),
            "std": float(np.std(metrics_by_seed[metric], ddof=0)),
        }
        for metric in METRIC_NAMES
    }
    summary["std_ddof"] = 0
    (result_dir / "internal_test_mean_std.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# FGNNSol five-seed internal-test results",
        "",
        "Classification threshold: 0.5. AUC uses continuous predictions.",
        "Standard deviation uses `ddof=0`.",
        "",
        "| Metric | Mean ± std |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {metric} | {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f} |"
        for metric in METRIC_NAMES
    )
    (result_dir / "experiment_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    for metric in METRIC_NAMES:
        print(
            f"{metric}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}"
        )


if __name__ == "__main__":
    main()
