from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .metrics import metrics_from_logits


def make_param_groups(model: torch.nn.Module, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    wd = float(cfg.get("weight_decay", 0.0))
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias") or "norm" in name.lower() or "emb" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, aux: dict[str, torch.Tensor], cfg: dict[str, Any]) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss(reduction="none")
    task_loss = (bce(logits, labels.float()) * mask).sum() / mask.sum().clamp_min(1.0)
    return (
        task_loss
        + float(cfg.get("router_entropy_alpha", 0.0)) * aux["router_entropy"]
        + float(cfg.get("router_quad_beta", 0.0)) * aux["router_quad"]
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    cfg: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        if hasattr(torch, "amp"):
            autocast_context = torch.amp.autocast(device_type="cuda", enabled=use_amp)
        else:
            autocast_context = torch.cuda.amp.autocast(enabled=use_amp)
        with autocast_context:
            logits, aux = model(batch, return_aux=True)
            loss = compute_loss(logits, batch["label_seq"], batch["mask"], aux, cfg)
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, cfg: dict[str, Any]):
    model.eval()
    logits_all, labels_all, masks_all = [], [], []
    target_len = int(cfg.get("seq_len", 100))
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch, return_aux=False)
        labels = batch["label_seq"]
        masks = batch["mask"]
        cur_len = logits.size(1)
        if cur_len < target_len:
            pad_len = target_len - cur_len
            logits = torch.nn.functional.pad(logits, (pad_len, 0), value=0.0)
            labels = torch.nn.functional.pad(labels, (pad_len, 0), value=0.0)
            masks = torch.nn.functional.pad(masks, (pad_len, 0), value=0.0)
        elif cur_len > target_len:
            logits = logits[:, -target_len:]
            labels = labels[:, -target_len:]
            masks = masks[:, -target_len:]
        logits_all.append(logits)
        labels_all.append(labels)
        masks_all.append(masks)
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
    masks = torch.cat(masks_all, 0)
    return metrics_from_logits(logits, labels, masks), logits, labels, masks


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
