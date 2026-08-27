from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class EmbeddingDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, embedding_dir: Path | None = None, limit: int = 0):
        frame = pd.read_csv(manifest_path)
        frame = frame[(frame["split"] == split) & (frame["status"] == "ok")].copy()
        if limit > 0: frame = frame.head(limit)
        if frame.empty: raise ValueError(f"No successful embeddings for split={split} in {manifest_path}")
        self.rows = frame.to_dict("records"); self.embedding_dir = embedding_dir

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]; path = Path(row["embedding_path"])
        if not path.is_absolute() and self.embedding_dir: path = self.embedding_dir / path
        obj = torch.load(path, map_location="cpu", weights_only=False)
        emb = obj["embedding"] if isinstance(obj, dict) else obj
        if emb.ndim != 2 or emb.shape[1] != 1280: raise ValueError(f"Invalid embedding {path}: {tuple(emb.shape)}")
        return emb.float(), float(row["solubility"]), row


def collate_embeddings(items):
    lengths = [x[0].shape[0] for x in items]; dim = items[0][0].shape[1]
    batch = torch.zeros(len(items), max(lengths), dim); mask = torch.ones(len(items), max(lengths), dtype=torch.bool)
    for i, (emb, _, _) in enumerate(items): batch[i, :len(emb)] = emb; mask[i, :len(emb)] = False
    return batch, mask, torch.tensor([x[1] for x in items], dtype=torch.float32), [x[2] for x in items], lengths


class TokenBatchSampler(Sampler):
    def __init__(self, lengths, max_batch_size=32, max_tokens=0, shuffle=True, seed=0):
        self.lengths=list(map(int,lengths)); self.max_batch_size=max_batch_size; self.max_tokens=max_tokens; self.shuffle=shuffle; self.seed=seed; self.epoch=0
    def set_epoch(self, epoch): self.epoch=epoch
    def __iter__(self):
        import random
        order=sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        buckets=[order[i:i+64] for i in range(0,len(order),64)]
        rng=random.Random(self.seed+self.epoch)
        if self.shuffle:
            for b in buckets: rng.shuffle(b)
            rng.shuffle(buckets)
        batch=[]; maxlen=0
        for idx in [x for b in buckets for x in b]:
            proposed=max(maxlen,self.lengths[idx]); exceeds=batch and (len(batch)>=self.max_batch_size or (self.max_tokens>0 and proposed*(len(batch)+1)>self.max_tokens))
            if exceeds: yield batch; batch=[]; maxlen=0
            batch.append(idx); maxlen=max(maxlen,self.lengths[idx])
        if batch: yield batch
    def __len__(self): return (len(self.lengths)+self.max_batch_size-1)//self.max_batch_size
