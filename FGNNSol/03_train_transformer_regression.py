from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import TransformerRegressor
from utils.data_utils import EmbeddingDataset, TokenBatchSampler, collate_embeddings
from utils.seed_utils import seed_everything
from utils.training_utils import git_commit, regression_metrics, runtime_versions, str2bool


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train the FGNNSol comparison model.")
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--embedding_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model_name_or_path", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens_per_batch", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--early_stopping", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--scheduler", choices=["step", "plateau"], default="step")
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("--step_gamma", type=float, default=0.75)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    return parser.parse_args()


def make_loader(dataset, batch_size, max_tokens, workers, shuffle, seed, device):
    sampler = TokenBatchSampler(
        [row["sequence_length"] for row in dataset.rows],
        batch_size,
        max_tokens,
        shuffle,
        seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_embeddings,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    return loader, sampler


def evaluate(model, loader, device, mixed_precision):
    model.eval()
    labels, predictions = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for embeddings, padding_mask, targets, _, _ in loader:
            embeddings = embeddings.to(device)
            padding_mask = padding_mask.to(device)
            targets = targets.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                output = model(embeddings, padding_mask)
            total_loss += nn.functional.mse_loss(output, targets, reduction="sum").item()
            labels.extend(targets.cpu().tolist())
            predictions.extend(output.float().cpu().tolist())
    return total_loss / len(labels), regression_metrics(labels, predictions)


def main():
    args = parse_args()
    if args.batch_size > 32:
        raise ValueError("batch_size must be <= 32")

    root = args.project_root.resolve()
    embedding_dir = (args.embedding_dir or root / "cache" / "esm2_650m_embeddings").resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = embedding_dir / "embedding_manifest.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EmbeddingDataset(manifest_path, "train", embedding_dir)
    validation_dataset = EmbeddingDataset(manifest_path, "validation", embedding_dir)

    for seed in args.seeds:
        seed_everything(seed)
        seed_result_dir = result_dir / f"seed_{seed}"
        seed_result_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / f"seed_{seed}_best_model.pt"

        model_config = {
            key: getattr(args, key)
            for key in ("d_model", "num_layers", "nhead", "dim_feedforward", "dropout", "pooling")
        }
        model = TransformerRegressor(**model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.step_size,
                gamma=args.step_gamma,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.plateau_factor,
                patience=args.plateau_patience,
                min_lr=args.min_lr,
            )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=args.mixed_precision and device.type == "cuda",
        )

        train_loader, train_sampler = make_loader(
            train_dataset,
            args.batch_size,
            args.max_tokens_per_batch,
            args.num_workers,
            True,
            seed,
            device,
        )
        validation_loader, _ = make_loader(
            validation_dataset,
            args.batch_size,
            args.max_tokens_per_batch,
            args.num_workers,
            False,
            seed,
            device,
        )

        best_validation_mse = float("inf")
        stale_epochs = 0
        history = []

        for epoch in range(1, args.max_epochs + 1):
            start = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            sample_count = 0
            train_sampler.set_epoch(epoch)

            for step, (embeddings, padding_mask, targets, _, lengths) in enumerate(train_loader, 1):
                embeddings = embeddings.to(device)
                padding_mask = padding_mask.to(device)
                targets = targets.to(device)
                try:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=args.mixed_precision and device.type == "cuda",
                    ):
                        loss = nn.functional.mse_loss(
                            model(embeddings, padding_mask),
                            targets,
                        ) / args.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    if step % args.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    loss_sum += loss.item() * args.gradient_accumulation_steps * len(targets)
                    sample_count += len(targets)
                except torch.cuda.OutOfMemoryError as exc:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(
                        f"CUDA OOM for batch sequence lengths={lengths}; "
                        "automatically retrying each sample with batch_size=1"
                    )
                    try:
                        for index in range(len(targets)):
                            length = lengths[index]
                            with torch.autocast(
                                device_type=device.type,
                                dtype=torch.float16,
                                enabled=args.mixed_precision and device.type == "cuda",
                            ):
                                single_loss = nn.functional.mse_loss(
                                    model(
                                        embeddings[index : index + 1, :length],
                                        padding_mask[index : index + 1, :length],
                                    ),
                                    targets[index : index + 1],
                                )
                            scaler.scale(single_loss).backward()
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            loss_sum += single_loss.item()
                            sample_count += 1
                    except torch.cuda.OutOfMemoryError as single_exc:
                        raise RuntimeError(
                            "CUDA OOM persists at batch_size=1; "
                            f"all original batch lengths={lengths}"
                        ) from single_exc

            if step % args.gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            validation_mse, validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                args.mixed_precision,
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            if args.scheduler == "plateau":
                scheduler.step(validation_mse)
            else:
                scheduler.step()

            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_mse": loss_sum / sample_count,
                "val_mse": validation_mse,
                "val_R2": validation_metrics["R2"],
                "val_Pearson": validation_metrics["Pearson"],
                "val_RMSE": validation_metrics["RMSE"],
                "elapsed_time": time.time() - start,
                "gpu_memory_mb": (
                    torch.cuda.max_memory_allocated() / 2**20
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            history.append(row)
            print({"seed": seed, **row}, flush=True)

            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                stale_epochs = 0
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "best_val_loss": best_validation_mse,
                    "model_config": model.config,
                    "training_config": vars(args),
                    "dataset_paths": {"manifest": str(manifest_path)},
                    "git_commit": git_commit(root),
                    "versions": runtime_versions(),
                }
                torch.save(state, best_model_path)
            else:
                stale_epochs += 1

            pd.DataFrame(history).to_csv(
                seed_result_dir / "training_log.csv",
                index=False,
            )
            if args.early_stopping and args.patience > 0 and stale_epochs >= args.patience:
                break

        print(
            f"Completed seed={seed}: best_validation_mse={best_validation_mse:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
