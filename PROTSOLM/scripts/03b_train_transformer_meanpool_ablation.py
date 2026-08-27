import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import compute_train_norm_stats
from src.metrics import binary_metrics
from src.model import ProteinSolubilityTransformer
from src.utils import set_seed, setup_logging


class PDBSolMeanPooledEmbeddingDataset(Dataset):
    """Mean-pool valid residue embeddings to one 1280-d vector per protein."""

    def __init__(
        self,
        embedding_path,
        metadata_path,
        mean=None,
        std=None,
        normalize=True,
        output_dtype="float16",
    ):
        self.embedding_path = Path(embedding_path)
        self.metadata_path = Path(metadata_path)
        self.embeddings = np.load(self.embedding_path, mmap_mode="r")
        self.metadata = pd.read_csv(self.metadata_path)
        if len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                f"Metadata rows ({len(self.metadata)}) do not match embeddings "
                f"({self.embeddings.shape[0]}): {self.metadata_path}"
            )
        self.labels = self.metadata["label"].astype("float32").to_numpy()
        self.lengths = self.metadata["length"].astype("int64").to_numpy()
        self.names = self.metadata["name"].astype(str).tolist()
        self.normalize = normalize
        self.mean = self._prepare_stat(mean) if mean is not None else None
        self.std = self._prepare_stat(std) if std is not None else None
        if output_dtype not in {"float16", "float32"}:
            raise ValueError(f"output_dtype must be float16 or float32, got {output_dtype}")
        self.output_dtype = output_dtype

    def _prepare_stat(self, value):
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape[-1] != self.embed_dim:
            raise ValueError(f"Normalization stat last dim must be {self.embed_dim}, got {arr.shape}")
        return arr.reshape(-1, self.embed_dim)[-1:].astype("float32")

    @property
    def pad_len(self):
        return int(self.embeddings.shape[1])

    @property
    def embed_dim(self):
        return int(self.embeddings.shape[2])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        length = int(self.lengths[idx])
        length = max(1, min(length, self.pad_len))
        valid = np.asarray(self.embeddings[idx, :length, :], dtype=np.float32)
        if self.normalize and self.mean is not None and self.std is not None:
            valid = (valid - self.mean) / self.std
        pooled = valid.mean(axis=0)
        if self.output_dtype == "float16":
            pooled = pooled.astype(np.float16, copy=False)
        return {
            "name": self.names[idx],
            "embedding": torch.from_numpy(pooled),
            "length": torch.tensor(1, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation: train Transformer on length-mean pooled PDBSol embeddings."
    )
    parser.add_argument("--embedding_dir", default="outputs/embeddings")
    parser.add_argument("--out_dir", default="outputs_meanpool_ablation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
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
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--train_metrics", action="store_true")
    parser.add_argument("--input_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    return parser.parse_args()


def paths_for(embedding_dir: Path, split: str):
    return (
        embedding_dir / f"PDBSol_{split}_embeddings.npy",
        embedding_dir / f"PDBSol_{split}_metadata.csv",
    )


def dataloader_kwargs(num_workers: int, device: torch.device):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


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
    labels_all = []
    probs_all = []
    names_all = []
    iterator = tqdm(loader, desc="Train" if train else "Eval")
    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(iterator):
        embeddings = batch["embedding"].to(device, non_blocking=True)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected mean-pooled embeddings with shape (batch, 1280), got {tuple(embeddings.shape)}")
        embeddings = embeddings.unsqueeze(1)
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


def save_predictions(path: Path, names, labels, probs):
    preds = (np.asarray(probs) >= 0.5).astype(int)
    df = pd.DataFrame({"name": names, "label": labels, "prob": probs, "pred": preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    ckpt_dir = out_dir / "checkpoints"
    result_dir = out_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(result_dir / "train_transformer_meanpool_ablation.log"))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Ablation input: mean over valid residue dimension, 2000x1280 -> 1280 -> 1x1280")

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
    train_ds = PDBSolMeanPooledEmbeddingDataset(train_emb, train_meta, mean, std, normalize, output_dtype=input_dtype)
    valid_ds = PDBSolMeanPooledEmbeddingDataset(valid_emb, valid_meta, mean, std, normalize, output_dtype=input_dtype)
    test_ds = PDBSolMeanPooledEmbeddingDataset(test_emb, test_meta, mean, std, normalize, output_dtype=input_dtype)
    logger.info(
        "Datasets: train=%d valid=%d test=%d | source embedding shape=%s | ablation shape=(1, %d) | input_dtype=%s",
        len(train_ds),
        len(valid_ds),
        len(test_ds),
        tuple(train_ds.embeddings.shape[1:]),
        train_ds.embed_dim,
        input_dtype,
    )
    logger.info("First mean-pooled training sample shape: %s", tuple(train_ds[0]["embedding"].shape))
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
        max_len=1,
        token_pool_size=1,
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
    best_path = ckpt_dir / "PDBSol_transformer_meanpool_ablation_best.pt"
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
        pd.DataFrame(history).to_csv(result_dir / "PDBSol_transformer_meanpool_ablation_history.csv", index=False)
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
                    "ablation": "length_mean_pool",
                    "input_shape": "1x1280",
                    "source_pad_len": train_ds.pad_len,
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
    metrics_path = result_dir / "PDBSol_transformer_meanpool_ablation_metrics.json"
    pred_path = result_dir / "PDBSol_transformer_meanpool_ablation_predictions.csv"
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
