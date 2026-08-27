import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import PDBSolEmbeddingDataset
from src.metrics import binary_metrics
from src.model import ProteinSolubilityTransformer
from src.utils import setup_logging


class PDBSolMeanPooledEmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_path,
        metadata_path,
        mean=None,
        std=None,
        normalize=True,
        output_dtype="float32",
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
    parser = argparse.ArgumentParser(description="Evaluate a trained PDBSol Transformer checkpoint.")
    parser.add_argument(
        "--target",
        choices=["both", "pdbsol", "external", "custom"],
        default="both",
        help="Evaluation target. Default runs PDBSol test and ExternalTest together.",
    )
    parser.add_argument(
        "--model_set",
        choices=["all", "full", "meanpool"],
        default="all",
        help="Which trained model(s) to evaluate. Default runs full Transformer and mean-pool ablation.",
    )
    parser.add_argument("--embedding_dir", default="outputs/external_embeddings")
    parser.add_argument(
        "--embedding_prefix",
        default="ExternalTest",
        help="Embedding filename prefix for --target custom.",
    )
    parser.add_argument("--checkpoint", default="outputs/checkpoints/PDBSol_transformer_best.pt")
    parser.add_argument(
        "--meanpool_checkpoint",
        default="outputs_meanpool_ablation/checkpoints/PDBSol_transformer_meanpool_ablation_best.pt",
    )
    parser.add_argument("--split", default="external")
    parser.add_argument("--out_dir", default="outputs/external_results")
    parser.add_argument(
        "--stats_dir",
        default="outputs/embeddings",
        help="Directory containing PDBSol_train_norm_stats.npz.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def evaluate_target(args, target, model_spec, device):
    checkpoint_path = (PROJECT_ROOT / model_spec["checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    result_dir = (PROJECT_ROOT / target["out_dir"]).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(result_dir / f"eval_{model_spec['name']}_{target['split']}.log"))

    embedding_dir = (PROJECT_ROOT / target["embedding_dir"]).resolve()
    embedding_prefix = target["embedding_prefix"]
    emb_path = embedding_dir / f"{embedding_prefix}_embeddings.npy"
    meta_path = embedding_dir / f"{embedding_prefix}_metadata.csv"
    mean = std = None
    stats_dir = (PROJECT_ROOT / target["stats_dir"]).resolve() if target["stats_dir"] else embedding_dir
    stats_path = stats_dir / "PDBSol_train_norm_stats.npz"
    if checkpoint.get("normalize", True) and not stats_path.exists():
        raise FileNotFoundError(
            f"Checkpoint expects normalized embeddings, but stats file was not found: {stats_path}"
        )
    if checkpoint.get("normalize", True):
        stats = np.load(stats_path)
        mean, std = stats["mean"], stats["std"]
    input_dtype = "float32"
    dataset_cls = PDBSolMeanPooledEmbeddingDataset if model_spec["meanpool"] else PDBSolEmbeddingDataset
    dataset = dataset_cls(
        emb_path,
        meta_path,
        mean,
        std,
        checkpoint.get("normalize", True),
        output_dtype=input_dtype,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    saved_args = checkpoint.get("args", {})
    max_len = 1 if model_spec["meanpool"] else checkpoint.get("pad_len", dataset.pad_len)
    model = ProteinSolubilityTransformer(
        input_dim=checkpoint.get("embed_dim", dataset.embed_dim),
        d_model=saved_args.get("d_model", 256),
        nhead=saved_args.get("nhead", 8),
        num_layers=saved_args.get("num_layers", 4),
        dim_feedforward=saved_args.get("dim_feedforward", 512),
        dropout=saved_args.get("dropout", 0.1),
        max_len=max_len,
        token_pool_size=saved_args.get("token_pool_size", 1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    names, labels, probs = [], [], []
    for batch in loader:
        embeddings = batch["embedding"].to(device)
        if model_spec["meanpool"]:
            embeddings = embeddings.unsqueeze(1)
        lengths = batch["length"].to(device)
        y = batch["label"].to(device)
        logits = model(embeddings, lengths)
        loss = criterion(logits, y)
        total_loss += float(loss.detach().cpu()) * embeddings.shape[0]
        names.extend(batch["name"])
        labels.extend(y.cpu().numpy().tolist())
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    metrics = binary_metrics(labels, probs)
    metrics["loss"] = total_loss / max(len(dataset), 1)
    output_name = f"{embedding_prefix}_{model_spec['output_suffix']}"
    metrics_path = result_dir / f"{output_name}_transformer_metrics.json"
    pred_path = result_dir / f"{output_name}_transformer_predictions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preds = (np.asarray(probs) >= 0.5).astype(int)
    pd.DataFrame({"name": names, "label": labels, "prob": probs, "pred": preds}).to_csv(pred_path, index=False)
    logger.info("Metrics: %s", metrics)
    logger.info("Saved: %s | %s", metrics_path, pred_path)
    print(f"\n[{model_spec['display_name']} | {target['name']}]")
    for key in ["ACC", "F1", "Precision", "Recall", "AUC", "MCC"]:
        print(f"{key}: {metrics[key]:.6f}")


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    default_targets = {
        "pdbsol": {
            "name": "PDBSol test",
            "embedding_dir": "outputs/embeddings",
            "embedding_prefix": "PDBSol_test",
            "split": "test",
            "out_dir": "outputs/results",
            "stats_dir": "outputs/embeddings",
        },
        "external": {
            "name": "ExternalTest",
            "embedding_dir": "outputs/external_embeddings",
            "embedding_prefix": "ExternalTest",
            "split": "external",
            "out_dir": "outputs/external_results",
            "stats_dir": "outputs/embeddings",
        },
        "custom": {
            "name": args.embedding_prefix,
            "embedding_dir": args.embedding_dir,
            "embedding_prefix": args.embedding_prefix or f"PDBSol_{args.split}",
            "split": args.split,
            "out_dir": args.out_dir,
            "stats_dir": args.stats_dir,
        },
    }

    if args.target == "both":
        targets = [default_targets["pdbsol"], default_targets["external"]]
    else:
        targets = [default_targets[args.target]]

    model_specs = {
        "full": {
            "name": "full",
            "display_name": "Full Transformer",
            "checkpoint": args.checkpoint,
            "meanpool": False,
            "output_suffix": "full",
        },
        "meanpool": {
            "name": "meanpool",
            "display_name": "Mean-pool ablation",
            "checkpoint": args.meanpool_checkpoint,
            "meanpool": True,
            "output_suffix": "meanpool_ablation",
        },
    }
    if args.model_set == "all":
        selected_models = [model_specs["full"], model_specs["meanpool"]]
    else:
        selected_models = [model_specs[args.model_set]]

    for model_spec in selected_models:
        for target in targets:
            evaluate_target(args, target, model_spec, device)


if __name__ == "__main__":
    main()
