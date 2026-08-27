import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import ProteinSolubilityTransformer
from utils import ProteinHDF5Dataset, evaluate, load_or_compute_normalization


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PLM_Sol Transformer checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/transformer_2500maxlen_paper.pt"),
    )
    parser.add_argument("--embedding-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/results/evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    train_h5 = args.embedding_dir / "train_esm2_650m_maxlen2500.h5"
    test_h5 = args.embedding_dir / "test_esm2_650m_maxlen2500.h5"
    mean, std = load_or_compute_normalization(
        train_h5, args.embedding_dir / "normalization_stats.npz"
    )
    test_dataset = ProteinHDF5Dataset(test_h5, mean, std)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ProteinSolubilityTransformer(
        embed_dim=config["embed_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss().to(device)
    metrics, protein_ids, labels, probabilities = evaluate(
        model, loader, criterion, device, return_outputs=True
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "true_label", "probability", "predicted_label"])
        for protein_id, label, probability in zip(protein_ids, labels, probabilities):
            writer.writerow([protein_id, int(label), float(probability), int(probability >= 0.5)])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
