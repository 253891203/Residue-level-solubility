import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import PDBSolEmbeddingDataset, compute_train_norm_stats
from src.metrics import binary_metrics
from src.model import ProteinSolubilityTransformer
from src.utils import set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer on PDBSol ESM2 embeddings.")
    parser.add_argument("--embedding_dir", default="outputs/embeddings")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output run directory. If omitted, creates outputs/transformer_run_YYYYmmdd_HHMMSS_seedN.",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token_pool_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--train_metrics", action="store_true")
    parser.add_argument("--input_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    return parser.parse_args()


def make_run_dir(project_root: Path, out_dir_arg: str | None, run_name: str | None, seed: int) -> Path:
    if out_dir_arg is None:
        name = run_name or f"transformer_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
        run_dir = project_root / "outputs" / name
    else:
        run_dir = Path(out_dir_arg)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
        if run_name:
            run_dir = run_dir / run_name

    original = run_dir
    suffix = 1
    while run_dir.exists():
        run_dir = original.parent / f"{original.name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def paths_for(embedding_dir: Path, split: str) -> tuple[Path, Path]:
    return (
        embedding_dir / f"PDBSol_{split}_embeddings.npy",
        embedding_dir / f"PDBSol_{split}_metadata.csv",
    )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    grad_accum_steps,
    use_amp,
    train: bool,
    collect_outputs: bool = True,
):
    model.train(train)
    total_loss = torch.zeros((), device=device)
    labels_all: list[float] = []
    probs_all: list[float] = []
    names_all: list[str] = []
    iterator = tqdm(loader, desc="Train" if train else "Eval")
    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(iterator):
        embeddings = batch["embedding"].to(device, non_blocking=True)
        if embeddings.ndim == 4 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)
        if embeddings.ndim != 3:
            raise ValueError(f"Expected embeddings with shape (batch, L, 1280), got {tuple(embeddings.shape)}")
        lengths = batch["length"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits = model(embeddings, lengths)
                loss = criterion(logits, labels)
                scaled_loss = loss / grad_accum_steps
            if train:
                scaler.scale(scaled_loss).backward()
                is_update = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader)
                if is_update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss = total_loss + loss.detach() * embeddings.shape[0]
        if collect_outputs:
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            probs_all.extend(probs.tolist())
            names_all.extend(batch["name"])

    avg_loss = float((total_loss / max(len(loader.dataset), 1)).detach().cpu())
    metrics = binary_metrics(labels_all, probs_all) if collect_outputs else {}
    return avg_loss, metrics, names_all, labels_all, probs_all


def save_predictions(path: Path, names, labels, probs) -> None:
    preds = (np.asarray(probs) >= 0.5).astype(int)
    df = pd.DataFrame({"name": names, "label": labels, "prob": probs, "pred": preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def dataloader_kwargs(num_workers: int, device: torch.device) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    embedding_dir = (PROJECT_ROOT / args.embedding_dir).resolve()
    out_dir = make_run_dir(PROJECT_ROOT, args.out_dir, args.run_name, args.seed).resolve()
    ckpt_dir = out_dir / "checkpoints"
    result_dir = out_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(result_dir / "train_transformer.log"))
    logger.info("Run output directory: %s", out_dir)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_emb, train_meta = paths_for(embedding_dir, "train")
    valid_emb, valid_meta = paths_for(embedding_dir, "valid")
    test_emb, test_meta = paths_for(embedding_dir, "test")
    for path in [train_emb, train_meta, valid_emb, valid_meta, test_emb, test_meta]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    normalize = not args.no_normalize
    mean = std = None
    if normalize:
        mean, std = compute_train_norm_stats(train_emb, train_meta, embedding_dir / "PDBSol_train_norm_stats.npz")
        logger.info("Loaded/computed train normalization stats: mean=%s std=%s", mean.shape, std.shape)

    input_dtype = args.input_dtype if device.type == "cuda" else "float32"
    train_ds = PDBSolEmbeddingDataset(train_emb, train_meta, mean, std, normalize, output_dtype=input_dtype)
    valid_ds = PDBSolEmbeddingDataset(valid_emb, valid_meta, mean, std, normalize, output_dtype=input_dtype)
    test_ds = PDBSolEmbeddingDataset(test_emb, test_meta, mean, std, normalize, output_dtype=input_dtype)
    logger.info(
        "Datasets: train=%d valid=%d test=%d | embedding shape=%s | token_pool_size=%d | input_dtype=%s",
        len(train_ds),
        len(valid_ds),
        len(test_ds),
        tuple(train_ds.embeddings.shape[1:]),
        args.token_pool_size,
        input_dtype,
    )
    sample_shape = tuple(train_ds[0]["embedding"].shape)
    logger.info("First normalized training sample shape: %s", sample_shape)
    logger.info(
        "Train loader: batch_size=%d | shuffle=%s | num_workers=%d",
        args.batch_size,
        args.shuffle_train,
        args.num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        drop_last=False,
        **dataloader_kwargs(args.num_workers, device),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs(args.num_workers, device),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs(args.num_workers, device),
    )

    model = ProteinSolubilityTransformer(
        input_dim=train_ds.embed_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=train_ds.pad_len,
        token_pool_size=args.token_pool_size,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    monitor_metric = "ACC"
    best_score = -1.0
    best_path = ckpt_dir / "PDBSol_transformer_best.pt"
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, *_ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args.grad_accum_steps,
            args.amp,
            train=True,
            collect_outputs=args.train_metrics,
        )
        valid_loss, valid_metrics, *_ = run_epoch(
            model,
            valid_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args.grad_accum_steps,
            args.amp,
            train=False,
            collect_outputs=True,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(result_dir / "PDBSol_transformer_history.csv", index=False)
        logger.info(
            "Epoch %03d | train_loss=%.5f | valid_loss=%.5f | valid_ACC=%.4f | valid_AUC=%.4f | valid_MCC=%.4f",
            epoch,
            train_loss,
            valid_loss,
            valid_metrics["ACC"],
            valid_metrics["AUC"],
            valid_metrics["MCC"],
        )

        score = valid_metrics[monitor_metric]
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "pad_len": train_ds.pad_len,
                    "embed_dim": train_ds.embed_dim,
                    "best_valid_metrics": valid_metrics,
                    "monitor_metric": monitor_metric,
                    "best_valid_score": best_score,
                    "normalize": normalize,
                },
                best_path,
            )
            logger.info("Saved best checkpoint by valid_%s=%.4f: %s", monitor_metric, best_score, best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping after %d epochs without improvement.", args.patience)
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, names, labels, probs = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer,
        device,
        scaler,
        args.grad_accum_steps,
        args.amp,
        train=False,
        collect_outputs=True,
    )
    test_metrics["loss"] = test_loss
    metrics_path = result_dir / "PDBSol_transformer_metrics.json"
    pred_path = result_dir / "PDBSol_transformer_predictions.csv"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    save_predictions(pred_path, names, labels, probs)
    logger.info("Test metrics: %s", test_metrics)
    logger.info("Saved metrics: %s", metrics_path)
    logger.info("Saved predictions: %s", pred_path)
    print("\nFinal test metrics")
    for key in ["ACC", "Precision", "Recall", "AUC", "MCC"]:
        print(f"{key}: {test_metrics[key]:.6f}")


if __name__ == "__main__":
    main()
