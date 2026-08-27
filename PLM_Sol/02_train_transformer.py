import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models import ProteinSolubilityTransformer
from utils import (
    ProteinHDF5Dataset,
    classification_metrics,
    evaluate,
    load_or_compute_normalization,
)


def build_model(config):
    return ProteinSolubilityTransformer(
        embed_dim=config["embed_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    )


def main():
    parser = argparse.ArgumentParser(description="Train the full-matrix Transformer classifier.")
    parser.add_argument("--embedding-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "max_len": 2500,
        "embed_dim": 1280,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
    }
    train_h5 = args.embedding_dir / "train_esm2_650m_maxlen2500.h5"
    validation_h5 = args.embedding_dir / "validation_esm2_650m_maxlen2500.h5"
    test_h5 = args.embedding_dir / "test_esm2_650m_maxlen2500.h5"
    for path in (train_h5, validation_h5, test_h5):
        if not path.exists():
            raise FileNotFoundError(path)

    checkpoint_dir = args.output_dir / "checkpoints"
    result_dir = args.output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    history_path = result_dir / "training_history.csv"
    metrics_path = result_dir / "metrics.json"

    mean, std = load_or_compute_normalization(
        train_h5, args.embedding_dir / "normalization_stats.npz"
    )
    train_dataset = ProteinHDF5Dataset(train_h5, mean, std)
    validation_dataset = ProteinHDF5Dataset(validation_h5, mean, std)
    test_dataset = ProteinHDF5Dataset(test_h5, mean, std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_validation_acc = float("-inf")
    patience_counter = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        labels, probabilities = [], []
        for step, (_, embeddings, batch_labels) in enumerate(train_loader, start=1):
            embeddings = embeddings.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(embeddings)
                loss = criterion(logits, batch_labels) / args.grad_accum_steps
            scaler.scale(loss).backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * args.grad_accum_steps * len(batch_labels)
            labels.extend(batch_labels.detach().cpu().numpy())
            probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy())

        train_metrics = classification_metrics(labels, probabilities)
        train_metrics["loss"] = total_loss / len(train_loader.dataset)
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["acc"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": validation_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_acc": validation_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "val_f1": validation_metrics["f1"],
            "val_mcc": validation_metrics["mcc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(json.dumps(row))

        if validation_metrics["acc"] > best_validation_acc:
            best_validation_acc = validation_metrics["acc"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "best_val_acc": best_validation_acc,
                    "epoch": epoch,
                    "dataset_counts": {
                        "train": len(train_dataset),
                        "validation": len(validation_dataset),
                        "test": len(test_dataset),
                    },
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    output = {
        "best_val_acc": best_validation_acc,
        "test_loss": test_metrics.pop("loss"),
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "train_count": len(train_dataset),
        "validation_count": len(validation_dataset),
        "test_count": len(test_dataset),
        "max_len": config["max_len"],
        "embed_dim": config["embed_dim"],
    }
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
