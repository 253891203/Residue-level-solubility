from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer_classifier import TransformerSolubilityClassifier
from utils.data_utils import (
    EmbeddingDataset,
    collate_embeddings,
    identify_fold_column,
    load_netsolp_csv,
    resolve_path,
    set_seed,
    write_json,
)
from utils.metrics import binary_metrics, youden_threshold


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train downstream Transformer with NetSolP 5-fold CV.")
    parser.add_argument("--train_csv", default="../data/netsolp/PSI_Biology_solubility_trainset.csv")
    parser.add_argument("--embedding_dir", default="embeddings")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=1280)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--monitor", default="val_loss", choices=["val_loss"])
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    return parser.parse_args()


def build_folds(train_df: pd.DataFrame, train_csv: Path, seed: int) -> tuple[np.ndarray, str]:
    raw = pd.read_csv(train_csv)
    fold_col = identify_fold_column(raw)
    if fold_col is not None:
        folds = pd.to_numeric(raw[fold_col], errors="raise").astype(int).to_numpy()
        return folds, f"Using original fold column '{fold_col}' from PSI_Biology_solubility_trainset.csv."
    print("未发现原始 fold 列，因此使用分层 5 折划分")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.zeros(len(train_df), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(train_df)), train_df["label"].to_numpy())):
        folds[val_idx] = fold
    return folds, "未发现原始 fold 列，因此使用分层 5 折划分"


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=True):
    train = optimizer is not None
    model.train(train)
    losses = []
    labels_all = []
    probs_all = []
    for batch in tqdm(loader, leave=False):
        x = batch["embeddings"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(x, lengths)
            loss = criterion(logits, labels)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu().item()) * len(labels))
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    avg_loss = float(np.sum(losses) / max(1, len(labels_all)))
    return avg_loss, np.asarray(labels_all), np.asarray(probs_all)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = resolve_path(ROOT, args.out_dir)
    embedding_dir = resolve_path(ROOT, args.embedding_dir)
    checkpoint_dir = out_dir / "checkpoints"
    result_dir = out_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_csv = resolve_path(ROOT, args.train_csv)
    train_df, _ = load_netsolp_csv(train_csv, "train")
    folds, fold_message = build_folds(train_df, train_csv, args.seed)
    metadata_csv = embedding_dir / "train_metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing embedding metadata: {metadata_csv}. Run 01_embed_esm2_650m.py first.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_config = {
        "input_dim": args.input_dim,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "pooling": args.pooling,
    }
    criterion = torch.nn.BCEWithLogitsLoss()
    fold_rows = []
    train_log_rows = []
    thresholds = {"fold_thresholds": {}, "monitor": "val_loss", "mode": "min", "fold_message": fold_message}

    for fold in sorted(np.unique(folds)):
        fold = int(fold)
        train_idx = np.where(folds != fold)[0].tolist()
        val_idx = np.where(folds == fold)[0].tolist()
        train_loader = DataLoader(
            EmbeddingDataset(metadata_csv, embedding_dir, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_embeddings,
        )
        val_loader = DataLoader(
            EmbeddingDataset(metadata_csv, embedding_dir, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_embeddings,
        )
        model = TransformerSolubilityClassifier(**model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")
        best_val_loss = float("inf")
        best_epoch = -1
        best_labels = None
        best_probs = None
        bad_epochs = 0
        ckpt_path = checkpoint_dir / f"fold_{fold}_best.pt"

        for epoch in range(1, args.max_epochs + 1):
            train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, args.use_amp)
            val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, device, None, None, args.use_amp)
            train_log_rows.append({"fold": fold, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"fold={fold} epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_labels = val_labels
                best_probs = val_probs
                bad_epochs = 0
                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "model_state_dict": model.state_dict(),
                        "model_config": model_config,
                        "monitor": "val_loss",
                        "mode": "min",
                    },
                    ckpt_path,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping fold={fold}: val_loss did not improve for {args.patience} epochs.")
                    break

        threshold = youden_threshold(best_labels, best_probs)
        metrics = binary_metrics(best_labels, best_probs, threshold=0.5)
        fold_rows.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_ACC": metrics["ACC"],
                "val_PRE": metrics["PRE"],
                "val_MCC": metrics["MCC"],
                "val_AUC": metrics["AUC"],
                "val_Recall": metrics["Recall"],
                "val_F1": metrics["F1"],
                "youden_threshold": threshold,
            }
        )
        thresholds["fold_thresholds"][str(fold)] = threshold

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(result_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(train_log_rows).to_csv(result_dir / "training_log.csv", index=False)
    thresholds["mean_youden_threshold"] = float(np.mean(list(thresholds["fold_thresholds"].values())))
    write_json(result_dir / "thresholds.json", thresholds)

    metric_cols = ["val_ACC", "val_PRE", "val_MCC", "val_AUC", "val_Recall", "val_F1", "best_val_loss"]
    summary = {
        col: {"mean": float(fold_metrics[col].mean()), "std": float(fold_metrics[col].std(ddof=1))}
        for col in metric_cols
    }
    write_json(result_dir / "mean_std_metrics.json", summary)
    write_json(result_dir / "model_config.json", model_config)
    print("Training complete. Results:", result_dir)


if __name__ == "__main__":
    main()
