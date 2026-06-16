from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits.detach()).cpu().numpy()
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    y = labels.detach().cpu().numpy()
    m = mask.detach().cpu().numpy().astype(bool)
    if m.sum() == 0:
        return {"AUC": 0.5, "ACC": 0.0, "AP": 0.0, "RMSE": 1.0}
    probs = probs[m]
    y = y[m]
    auc = roc_auc_score(y, probs) if len(set(y.tolist())) > 1 else 0.5
    acc = accuracy_score(y, probs >= 0.5)
    ap = average_precision_score(y, probs) if len(set(y.tolist())) > 1 else 0.0
    rmse = float(np.sqrt(((y - probs) ** 2).mean()))
    return {"AUC": float(auc), "ACC": float(acc), "AP": float(ap), "RMSE": rmse}
