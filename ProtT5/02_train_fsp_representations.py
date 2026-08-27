import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_TAG = "prott5_uniref50"
DEFAULT_EMBEDDING_DIR = SCRIPT_DIR / "embeddings"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "generated" / "fsp"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_stream_h5(h5f: h5py.File) -> bool:
    return "residue_embed" in h5f and "id" in h5f and "seq_len" in h5f


def read_h5_keys(path: str) -> List[str]:
    with h5py.File(path, "r") as h5f:
        if is_stream_h5(h5f):
            return [str(i) for i in range(h5f["residue_embed"].shape[0])]
        return list(h5f.keys())


def h5_shape(path: str) -> Tuple[int, int]:
    with h5py.File(path, "r") as h5f:
        if is_stream_h5(h5f):
            return tuple(h5f["residue_embed"].shape[1:3])
        key = next(iter(h5f.keys()))
        return tuple(h5f[key].shape)


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get_label_from_id(sample_id: str) -> int:
    label = sample_id.rsplit("-", 1)[-1]
    return int(label)


def get_sample_id(h5f: h5py.File, key: str) -> str:
    if is_stream_h5(h5f):
        return decode_h5_string(h5f["id"][int(key)])
    ds = h5f[key]
    for attr_name in ("protein_id", "fasta_header", "id"):
        if attr_name in ds.attrs:
            return decode_h5_string(ds.attrs[attr_name])
    return key


def get_label(h5f: h5py.File, key: str) -> int:
    if is_stream_h5(h5f):
        return get_label_from_id(get_sample_id(h5f, key))
    ds = h5f[key]
    if "label" in ds.attrs:
        return int(ds.attrs["label"])
    return get_label_from_id(get_sample_id(h5f, key))


def get_length(h5f: h5py.File, key: str) -> int:
    if is_stream_h5(h5f):
        return int(h5f["seq_len"][int(key)])
    ds = h5f[key]
    if "length" in ds.attrs:
        return int(ds.attrs["length"])
    arr = ds[:]
    nonzero = np.where(np.abs(arr).sum(axis=1) > 0)[0]
    return int(nonzero[-1] + 1) if len(nonzero) else arr.shape[0]


def get_embedding(h5f: h5py.File, key: str, max_length: Optional[int] = None) -> np.ndarray:
    if is_stream_h5(h5f):
        arr = h5f["residue_embed"][int(key)]
    else:
        arr = h5f[key][:]
    if max_length is not None:
        arr = arr[:max_length]
    return arr.astype(np.float32)


def compute_embedding_stats(h5_path: str, sample_limit: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    keys = read_h5_keys(h5_path)
    if sample_limit > 0 and len(keys) > sample_limit:
        keys = random.sample(keys, sample_limit)
    total = 0
    sum_vec = None
    sq_sum_vec = None
    with h5py.File(h5_path, "r") as h5f:
        for key in tqdm(keys, desc="Embedding stats"):
            length = get_length(h5f, key)
            x = get_embedding(h5f, key, max_length=length)
            if sum_vec is None:
                sum_vec = x.sum(axis=0)
                sq_sum_vec = (x * x).sum(axis=0)
            else:
                sum_vec += x.sum(axis=0)
                sq_sum_vec += (x * x).sum(axis=0)
            total += length
    mean = sum_vec / max(total, 1)
    var = sq_sum_vec / max(total, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def fit_pca(
    h5_path: str,
    n_components: int,
    output_path: Path,
    max_residue_samples: int,
    batch_size: int,
    seed: int,
) -> IncrementalPCA:
    if output_path.exists():
        return joblib.load(output_path)

    rng = np.random.default_rng(seed)
    keys = read_h5_keys(h5_path)
    residue_batches: List[np.ndarray] = []
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    seen = 0

    with h5py.File(h5_path, "r") as h5f:
        pbar = tqdm(keys, desc=f"Fitting PCA-{n_components}")
        for key in pbar:
            length = get_length(h5f, key)
            x = get_embedding(h5f, key, max_length=length)
            if max_residue_samples > 0 and seen + length > max_residue_samples:
                remaining = max_residue_samples - seen
                if remaining <= 0:
                    break
                idx = rng.choice(length, size=min(remaining, length), replace=False)
                x = x[idx]
            residue_batches.append(x)
            seen += len(x)
            total_rows = sum(batch.shape[0] for batch in residue_batches)
            if total_rows >= max(batch_size, n_components):
                block = np.concatenate(residue_batches, axis=0)
                pca.partial_fit(block)
                residue_batches = []
        if residue_batches:
            block = np.concatenate(residue_batches, axis=0)
            if block.shape[0] >= n_components:
                pca.partial_fit(block)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, output_path)
    return pca


class RepresentationDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        representation: str,
        pca: Optional[IncrementalPCA] = None,
        normalize_embedding: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_length: Optional[int] = None,
    ):
        self.h5_path = h5_path
        self.representation = representation.upper()
        self.pca = pca
        self.keys = read_h5_keys(h5_path)
        self.max_length, self.embed_dim = h5_shape(h5_path)
        if max_length is not None:
            self.max_length = min(self.max_length, max_length)
        self.normalize_embedding = normalize_embedding
        self._h5 = None

    def __len__(self) -> int:
        return len(self.keys)

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int):
        key = self.keys[index]
        h5f = self._open()
        label = float(get_label(h5f, key))
        length = min(get_length(h5f, key), self.max_length)
        x = get_embedding(h5f, key, max_length=self.max_length)
        if self.normalize_embedding is not None:
            mean, std = self.normalize_embedding
            x = (x - mean) / std
        if self.representation == "F":
            vec = x[:length].mean(axis=0)
        elif self.representation == "S":
            vec = np.zeros((self.max_length,), dtype=np.float32)
            vec[:length] = x[:length].mean(axis=1)
        elif self.representation == "P":
            if self.pca is None:
                raise ValueError("P representation requires a fitted PCA object.")
            reduced = self.pca.transform(x[:length])
            vec = np.zeros((self.max_length, reduced.shape[1]), dtype=np.float32)
            vec[:length] = reduced.astype(np.float32)
            vec = vec.reshape(-1)
        elif self.representation == "N":
            vec = x
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
        return key, torch.from_numpy(vec), torch.tensor(label, dtype=torch.float32)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (512, 256), dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TrainableNClassifier(nn.Module):
    """Legacy N model used for the original 1/2/4/6/8-dimensional sweep."""

    def __init__(self, embed_dim: int, max_length: int, reduce_dim: int):
        super().__init__()
        self.reducer = nn.Linear(embed_dim, reduce_dim)
        self.classifier = MLPClassifier(max_length * reduce_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.reducer(x).flatten(1))


@dataclass
class Metrics:
    loss: float
    accuracy: float
    f1: float
    mcc: float


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Metrics:
    model.eval()
    losses: List[float] = []
    labels_all: List[float] = []
    probs_all: List[float] = []
    with torch.no_grad():
        for _, x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            labels_all.extend(y.cpu().numpy().tolist())
            probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    pred = [1 if p >= 0.5 else 0 for p in probs_all]
    return Metrics(
        loss=float(np.mean(losses)),
        accuracy=float(accuracy_score(labels_all, pred)),
        f1=float(f1_score(labels_all, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(labels_all, pred)),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_path: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[Metrics, Metrics]:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    monitor_metric = "accuracy"
    best_score = -1.0
    best_val = None
    stalled = 0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for _, x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

        val = evaluate(model, val_loader, criterion, device)
        score = getattr(val, monitor_metric)
        if score > best_score:
            best_score = score
            best_val = val
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break

    if output_path.exists():
        model.load_state_dict(torch.load(output_path, map_location=device))
    test = evaluate(model, test_loader, criterion, device)
    return best_val or evaluate(model, val_loader, criterion, device), test


def infer_input_dim(representation: str, max_length: int, embed_dim: int, reduce_dim: Optional[int]) -> int:
    representation = representation.upper()
    if representation == "F":
        return embed_dim
    if representation == "S":
        return max_length
    if representation in {"P", "N"}:
        if reduce_dim is None:
            raise ValueError(f"{representation} requires reduce_dim.")
        return max_length * reduce_dim
    raise ValueError(f"Input dim for {representation} is model-defined.")


def parse_csv_list(text: str, cast=str) -> List:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the original ProtT5 F/S/P/N representation sweep with an MLP.")
    parser.add_argument("--embedding-dir", default=str(DEFAULT_EMBEDDING_DIR))
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    parser.add_argument("--train-h5", default=None)
    parser.add_argument("--val-h5", default=None)
    parser.add_argument("--test-h5", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", default="mlp", help="Only mlp is supported in this cleaned ProtT5 script.")
    parser.add_argument("--representations", default="F,S,P")
    parser.add_argument("--dims", default="1,2,4,6,8,16,32")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pca-max-residue-samples", type=int, default=300000)
    parser.add_argument("--pca-batch-size", type=int, default=8192)
    parser.add_argument("--stat-sample-proteins", type=int, default=5000)
    return parser.parse_args()


def find_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def default_h5_candidates(embedding_dir: Path, model_tag: str, split: str) -> List[Path]:
    return [embedding_dir / f"{model_tag}_{split}_maxlen500.h5"]


def resolve_default_h5_paths(args: argparse.Namespace) -> None:
    embedding_dir = Path(args.embedding_dir)
    if args.train_h5 is None:
        found = find_existing_path(default_h5_candidates(embedding_dir, args.model_tag, "train"))
        args.train_h5 = str(found or default_h5_candidates(embedding_dir, args.model_tag, "train")[0])
    if args.val_h5 is None:
        found = find_existing_path(default_h5_candidates(embedding_dir, args.model_tag, "validation"))
        args.val_h5 = str(found or default_h5_candidates(embedding_dir, args.model_tag, "validation")[0])
    if args.test_h5 is None:
        found = find_existing_path(default_h5_candidates(embedding_dir, args.model_tag, "test"))
        args.test_h5 = str(found or default_h5_candidates(embedding_dir, args.model_tag, "test")[0])

    missing = [path for path in [args.train_h5, args.val_h5, args.test_h5] if not Path(path).exists()]
    if missing:
        checked = "\n".join(f"  - {path}" for path in [args.train_h5, args.val_h5, args.test_h5])
        raise FileNotFoundError(
            "Cannot find ProtT5 embedding H5 files. Checked:\n"
            f"{checked}\n"
            "Pass --embedding-dir if the old H5 files are in another folder, or pass --train-h5/--val-h5/--test-h5 explicitly."
        )


def main() -> None:
    args = parse_args()
    resolve_default_h5_paths(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    pca_dir = output_dir / "pca"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    models = [m.lower() for m in parse_csv_list(args.models)]
    unsupported_models = sorted(set(models) - {"mlp"})
    if unsupported_models:
        raise ValueError(f"Only mlp is supported in this ProtT5 script, got: {unsupported_models}")
    representations = [r.upper() for r in parse_csv_list(args.representations)]
    unsupported_representations = sorted(set(representations) - {"F", "S", "P", "N"})
    if unsupported_representations:
        raise ValueError(f"Unsupported representations: {unsupported_representations}")
    dims = parse_csv_list(args.dims, int)
    print("ProtT5 representation sweep paths:")
    print(f"  train_h5: {args.train_h5}")
    print(f"  val_h5:   {args.val_h5}")
    print(f"  test_h5:  {args.test_h5}")
    print(f"  output:   {output_dir}")
    max_length, embed_dim = h5_shape(args.train_h5)

    embed_mean, embed_std = compute_embedding_stats(args.train_h5, sample_limit=args.stat_sample_proteins)
    norm = (embed_mean, embed_std)

    rows: List[Dict[str, object]] = []
    for representation in representations:
        dim_list: List[Optional[int]] = dims if representation in {"P", "N"} else [None]
        for reduce_dim in dim_list:
            pca = None
            if representation == "P":
                pca = fit_pca(
                    h5_path=args.train_h5,
                    n_components=int(reduce_dim),
                    output_path=pca_dir / f"pca_dim{reduce_dim}.joblib",
                    max_residue_samples=args.pca_max_residue_samples,
                    batch_size=args.pca_batch_size,
                    seed=args.seed,
                )
            for model_name in models:
                tag = f"{model_name}_{representation}" + (f"{reduce_dim}" if reduce_dim is not None else "")
                print(f"\nRunning {tag}")
                train_ds = RepresentationDataset(args.train_h5, representation, pca=pca, normalize_embedding=norm)
                val_ds = RepresentationDataset(args.val_h5, representation, pca=pca, normalize_embedding=norm)
                test_ds = RepresentationDataset(args.test_h5, representation, pca=pca, normalize_embedding=norm)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
                test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

                input_dim = infer_input_dim(representation, max_length, embed_dim, reduce_dim)
                if representation == "N":
                    model = TrainableNClassifier(embed_dim, max_length, int(reduce_dim))
                else:
                    model = MLPClassifier(input_dim)

                val_metrics, test_metrics = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    output_path=checkpoint_dir / f"{tag}_best.pt",
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                )
                row = {
                    "model": model_name.upper(),
                    "representation": representation if reduce_dim is None else f"{representation}-{reduce_dim}",
                    "reduce_dim": reduce_dim if reduce_dim is not None else "",
                    "input_dim": input_dim,
                    "monitor_metric": "val_accuracy",
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_f1": val_metrics.f1,
                    "val_mcc": val_metrics.mcc,
                    "test_loss": test_metrics.loss,
                    "test_accuracy": test_metrics.accuracy,
                    "test_f1": test_metrics.f1,
                    "test_mcc": test_metrics.mcc,
                    "checkpoint": str(checkpoint_dir / f"{tag}_best.pt"),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_dir / "representation_sweep_results.csv", index=False)
                print(json.dumps(row, indent=2))

if __name__ == "__main__":
    main()
