import numpy as np
import torch

from .metrics import classification_metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_outputs=False):
    model.eval()
    total_loss = 0.0
    protein_ids, labels, probabilities = [], [], []
    for batch_ids, embeddings, batch_labels in loader:
        embeddings = embeddings.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        logits = model(embeddings)
        loss = criterion(logits, batch_labels)
        total_loss += loss.item() * len(batch_labels)
        protein_ids.extend(batch_ids)
        labels.extend(batch_labels.cpu().numpy())
        probabilities.extend(torch.sigmoid(logits).cpu().numpy())

    metrics = classification_metrics(labels, probabilities)
    metrics["loss"] = total_loss / len(loader.dataset)
    if return_outputs:
        return metrics, protein_ids, np.asarray(labels), np.asarray(probabilities)
    return metrics
