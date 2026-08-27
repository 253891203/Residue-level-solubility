import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def is_stream_h5(handle):
    return all(name in handle for name in ("residue_embed", "id", "seq_len"))


def read_h5_keys(path):
    with h5py.File(path, "r") as handle:
        if is_stream_h5(handle):
            return [str(index) for index in range(handle["residue_embed"].shape[0])]
        return list(handle.keys())


def h5_shape(path):
    with h5py.File(path, "r") as handle:
        if is_stream_h5(handle):
            return tuple(handle["residue_embed"].shape[1:3])
        return tuple(handle[next(iter(handle.keys()))].shape)


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sample_id(handle, key):
    if is_stream_h5(handle):
        return decode(handle["id"][int(key)])
    dataset = handle[key]
    for attribute in ("protein_id", "fasta_header", "id"):
        if attribute in dataset.attrs:
            return decode(dataset.attrs[attribute])
    return key


def label(handle, key):
    if not is_stream_h5(handle) and "label" in handle[key].attrs:
        return int(handle[key].attrs["label"])
    return int(sample_id(handle, key).rsplit("-", 1)[-1])


def sequence_length(handle, key):
    if is_stream_h5(handle):
        return int(handle["seq_len"][int(key)])
    dataset = handle[key]
    if "length" in dataset.attrs:
        return int(dataset.attrs["length"])
    array = dataset[:]
    nonzero = np.where(np.abs(array).sum(axis=1) > 0)[0]
    return int(nonzero[-1] + 1) if len(nonzero) else len(array)


def embedding(handle, key, length):
    if is_stream_h5(handle):
        array = handle["residue_embed"][int(key), :length]
    else:
        array = handle[key][:length]
    return array.astype(np.float32)


def compute_stats(path, sample_limit, seed):
    keys = read_h5_keys(path)
    if sample_limit > 0 and len(keys) > sample_limit:
        keys = random.Random(seed).sample(keys, sample_limit)
    total = 0
    total_sum = None
    total_square_sum = None
    with h5py.File(path, "r") as handle:
        for key in keys:
            length = sequence_length(handle, key)
            array = embedding(handle, key, length).astype(np.float64)
            current_sum = array.sum(axis=0)
            current_square_sum = np.square(array).sum(axis=0)
            total_sum = current_sum if total_sum is None else total_sum + current_sum
            total_square_sum = (
                current_square_sum
                if total_square_sum is None
                else total_square_sum + current_square_sum
            )
            total += length
    mean = total_sum / total
    variance = total_square_sum / total - np.square(mean)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)


class NDataset(Dataset):
    def __init__(self, path, mean, std):
        self.path = str(path)
        self.keys = read_h5_keys(self.path)
        self.max_length, self.embed_dim = h5_shape(self.path)
        self.mean = mean
        self.std = std
        self._handle = None

    def __len__(self):
        return len(self.keys)

    def _open(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index):
        key = self.keys[index]
        handle = self._open()
        length = min(sequence_length(handle, key), self.max_length)
        array = embedding(handle, key, length)
        array = ((array - self.mean) / self.std).astype(np.float32)
        return (
            torch.from_numpy(array),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(float(label(handle, key)), dtype=torch.float32),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state


def collate_batch(batch):
    matrices, lengths, labels = zip(*batch)
    length_tensor = torch.stack(lengths)
    padded = torch.zeros(
        (len(matrices), int(length_tensor.max()), matrices[0].shape[1]),
        dtype=torch.float32,
    )
    for index, matrix in enumerate(matrices):
        padded[index, : len(matrix)] = matrix
    return padded, length_tensor, torch.stack(labels)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


class ResidueReducer(nn.Module):
    def __init__(self, embed_dim, reduce_dim):
        super().__init__()
        self.network = nn.Linear(embed_dim, reduce_dim)

    def forward(self, inputs):
        return self.network(inputs)


class ProtT5MLPN(nn.Module):
    def __init__(self, embed_dim, max_length, reduce_dim, hidden_dims=(512, 256), dropout=0.2):
        super().__init__()
        self.max_length = max_length
        self.reducer = ResidueReducer(embed_dim, reduce_dim)
        self.classifier = MLPClassifier(
            max_length * reduce_dim, hidden_dims, dropout
        )

    def forward(self, inputs, lengths):
        reduced = self.reducer(inputs)
        positions = torch.arange(reduced.shape[1], device=inputs.device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        reduced = reduced.masked_fill(~valid.unsqueeze(-1), 0.0)
        if reduced.shape[1] < self.max_length:
            reduced = torch.nn.functional.pad(
                reduced, (0, 0, 0, self.max_length - reduced.shape[1])
            )
        return self.classifier(reduced.flatten(1))


@dataclass
class Metrics:
    loss: float
    accuracy: float
    f1: float
    mcc: float


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled):
    model.eval()
    labels: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    for inputs, lengths, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(inputs, lengths)
            loss = criterion(logits, targets)
        total_loss += float(loss.item()) * len(targets)
        total_samples += len(targets)
        labels.append(targets.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    labels_array = np.concatenate(labels).astype(np.int64)
    probability_array = np.concatenate(probabilities)
    predictions = (probability_array >= 0.5).astype(np.int64)
    return Metrics(
        loss=total_loss / total_samples,
        accuracy=float(accuracy_score(labels_array, predictions)),
        f1=float(f1_score(labels_array, predictions, zero_division=0)),
        mcc=float(matthews_corrcoef(labels_array, predictions)),
    )


def make_loader(dataset, batch_size, shuffle, workers, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    options = {"prefetch_factor": 1} if workers > 0 else {}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(dataset) % batch_size == 1,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate_batch,
        **options,
    )


def train_dimension(args, dimension, datasets, device):
    set_seed(args.seed)
    train_dataset, validation_dataset, test_dataset = datasets
    loaders = [
        make_loader(train_dataset, args.batch_size, True, args.num_workers, device, args.seed),
        make_loader(validation_dataset, args.batch_size, False, args.num_workers, device, args.seed),
        make_loader(test_dataset, args.batch_size, False, args.num_workers, device, args.seed),
    ]
    model = ProtT5MLPN(
        train_dataset.embed_dim,
        train_dataset.max_length,
        dimension,
        tuple(args.hidden_dims),
        args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 3), min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    amp_enabled = device.type == "cuda" and not args.no_amp
    checkpoint_path = args.output_dir / "checkpoints" / f"N{dimension}_best.pt"
    history_path = args.output_dir / "history" / f"N{dimension}_history.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    best_validation = None
    stalled = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for inputs, lengths, targets in loaders[0]:
            inputs = inputs.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs, lengths)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(targets)
            total_samples += len(targets)

        validation = evaluate(model, loaders[1], criterion, device, amp_enabled)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_samples,
            "val_loss": validation.loss,
            "val_accuracy": validation.accuracy,
            "val_f1": validation.f1,
            "val_mcc": validation.mcc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if validation.accuracy > best_accuracy:
            best_accuracy = validation.accuracy
            best_validation = validation
            stalled = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stalled += 1
        scheduler.step(validation.accuracy)
        if epoch >= args.min_epochs and stalled >= args.patience:
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test = evaluate(model, loaders[2], criterion, device, amp_enabled)
    validation = best_validation
    return {
        "model": "MLP",
        "representation": f"N-{dimension}",
        "reduce_dim": dimension,
        "input_dim": train_dataset.max_length * dimension,
        "monitor_metric": "val_accuracy",
        **{f"val_{key}": value for key, value in asdict(validation).items()},
        **{f"test_{key}": value for key, value in asdict(test).items()},
        "checkpoint": str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the fixed ProtT5 MLP-N baseline for the later dimensions."
    )
    parser.add_argument("--embedding-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generated/n"))
    parser.add_argument("--dims", default="16,32")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stat-sample-proteins", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    paths = {
        split: args.embedding_dir / f"prott5_uniref50_{split}_maxlen500.h5"
        for split in ("train", "validation", "test")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing embedding files:\n" + "\n".join(missing))
    mean, std = compute_stats(paths["train"], args.stat_sample_proteins, args.seed)
    datasets = (
        NDataset(paths["train"], mean, std),
        NDataset(paths["validation"], mean, std),
        NDataset(paths["test"], mean, std),
    )
    device = torch.device(args.device)
    rows = []
    for dimension in [int(value) for value in args.dims.split(",") if value.strip()]:
        result = train_dimension(args, dimension, datasets, device)
        rows.append(result)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_dir / "n_representation_results.csv", index=False)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
