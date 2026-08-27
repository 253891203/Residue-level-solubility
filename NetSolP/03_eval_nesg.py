from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer_classifier import TransformerSolubilityClassifier
from utils.data_utils import EmbeddingDataset, collate_embeddings, resolve_path
from utils.metrics import binary_metrics


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fold checkpoints on independent NESG test set.")
    parser.add_argument("--embedding_dir", default="embeddings")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    return parser.parse_args()


def predict(model, loader, device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    probs_all = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            x = batch["embeddings"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].numpy()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(x, lengths)
            labels_all.extend(labels.tolist())
            probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return np.asarray(labels_all).astype(int), np.asarray(probs_all).astype(float)


def add_metric_row(rows: list[dict], fold: str | int, threshold_type: str, threshold: float, labels, probs) -> None:
    metrics = binary_metrics(labels, probs, threshold=threshold)
    rows.append(
        {
            "fold": fold,
            "threshold_type": threshold_type,
            "threshold": threshold,
            "ACC": metrics["ACC"],
            "PRE": metrics["PRE"],
            "MCC": metrics["MCC"],
            "AUC": metrics["AUC"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
        }
    )


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(ROOT, args.out_dir)
    result_dir = out_dir / "results"
    embedding_dir = resolve_path(ROOT, args.embedding_dir)
    metadata_csv = embedding_dir / "nesg_metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing NESG embedding metadata: {metadata_csv}. Run 01_embed_esm2_650m.py first.")
    thresholds_path = result_dir / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing thresholds from CV: {thresholds_path}. Run 02_train_transformer_5fold.py first.")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    mean_youden = float(thresholds["mean_youden_threshold"])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        EmbeddingDataset(metadata_csv, embedding_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_embeddings,
    )
    rows = []
    fold_probs = []
    labels_ref = None
    checkpoint_paths = sorted((out_dir / "checkpoints").glob("fold_*_best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {out_dir / 'checkpoints'}")

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        fold = ckpt["fold"]
        model = TransformerSolubilityClassifier(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        labels, probs = predict(model, loader, device, args.use_amp)
        labels_ref = labels if labels_ref is None else labels_ref
        fold_probs.append(probs)
        add_metric_row(rows, fold, "fixed_0.5", 0.5, labels, probs)
        add_metric_row(rows, fold, "mean_validation_youden", mean_youden, labels, probs)

    ensemble_probs = np.mean(np.vstack(fold_probs), axis=0)
    add_metric_row(rows, "ensemble", "fixed_0.5", 0.5, labels_ref, ensemble_probs)
    add_metric_row(rows, "ensemble", "mean_validation_youden", mean_youden, labels_ref, ensemble_probs)
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_dir / "nesg_test_metrics.csv", index=False)
    print("NESG independent test complete:", result_dir / "nesg_test_metrics.csv")


if __name__ == "__main__":
    main()
