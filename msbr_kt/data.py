from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):

    def __init__(self, raw_pkl: str | Path, seq_len: int = 100):
        self.raw_pkl = Path(raw_pkl)
        with open(self.raw_pkl, "rb") as f:
            self.data: dict[int, list[dict[str, Any]]] = pickle.load(f)
        self.users = sorted(self.data.keys())
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.users)

    @staticmethod
    def _prev(x: np.ndarray) -> np.ndarray:
        z = np.zeros_like(x)
        if x.shape[0] > 1:
            z[1:] = x[:-1]
        return z

    def __getitem__(self, idx: int) -> dict[str, Any]:
        uid = int(self.users[idx])
        seq = self.data[uid]
        item = np.array([e["problem_id"] for e in seq], dtype=np.int64)
        skill = np.array([e["skill_id"] for e in seq], dtype=np.int64)
        y = np.array([e["correct"] for e in seq], dtype=np.float32)
        attempt = np.array([e.get("attempt_count", 0.0) for e in seq], dtype=np.float32)
        pos_ratio = np.array([e.get("position_ratio", 0.0) for e in seq], dtype=np.float32)
        past_correct_rate = np.array([e.get("past_correct_rate", 0.5) for e in seq], dtype=np.float32)
        delta_ts = np.array([e.get("delta_ts", 0.0) for e in seq], dtype=np.float32)
        response_time = np.array([e.get("response_time", 0.0) for e in seq], dtype=np.float32)

        return {
            "user_id": uid,
            "len": len(seq),
            "item_seq": torch.from_numpy(item),
            "skill_seq": torch.from_numpy(skill),
            "label_seq": torch.from_numpy(y),
            "dt_seq": torch.from_numpy(delta_ts),
            "prev_y_seq": torch.from_numpy(self._prev(y)),
            "prev_rt_seq": torch.from_numpy(self._prev(response_time)),
            "prev_ac_seq": torch.from_numpy(self._prev(attempt)),
            "prev_posr_seq": torch.from_numpy(self._prev(pos_ratio)),
            "prev_pcr_seq": torch.from_numpy(self._prev(past_correct_rate)),
            "prev_dt_seq": torch.from_numpy(self._prev(delta_ts)),
        }


def collate_batch(samples: list[dict[str, Any]], seq_len: int = 100) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    t_max = min(max(s["len"] for s in samples), int(seq_len))

    def pad_left(x: torch.Tensor, fill: float | int = 0) -> torch.Tensor:
        t = x.size(0)
        if t >= t_max:
            return x[-t_max:]
        pad = torch.full((t_max - t,), fill, dtype=x.dtype)
        return torch.cat([pad, x], dim=0)

    keys = [
        "item_seq",
        "skill_seq",
        "label_seq",
        "dt_seq",
        "prev_y_seq",
        "prev_rt_seq",
        "prev_ac_seq",
        "prev_posr_seq",
        "prev_pcr_seq",
        "prev_dt_seq",
    ]
    out = {key: torch.stack([pad_left(s[key]) for s in samples], dim=0) for key in keys}

    pad_mask = torch.zeros((batch_size, t_max), dtype=torch.bool)
    mask = torch.zeros((batch_size, t_max), dtype=torch.float32)
    for i, sample in enumerate(samples):
        t = min(sample["len"], t_max)
        pad_mask[i, : t_max - t] = True
        mask[i, t_max - t :] = 1.0
    out["pad_mask"] = pad_mask
    out["mask"] = mask
    out["user_id"] = torch.tensor([s["user_id"] for s in samples], dtype=torch.long)
    out["orig_len"] = torch.tensor([s["len"] for s in samples], dtype=torch.long)
    return out


def build_loader(
    raw_pkl: str | Path,
    batch_size: int,
    seq_len: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SequenceDataset(raw_pkl, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda xs: collate_batch(xs, seq_len=seq_len),
    )


def build_split_loaders(data_root: str | Path, cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(data_root)
    batch_size = int(cfg["batch_size"])
    seq_len = int(cfg["seq_len"])
    num_workers = int(cfg.get("num_workers", 0))
    return (
        build_loader(root / "train_raw.pkl", batch_size, seq_len, True, num_workers),
        build_loader(root / "valid_raw.pkl", batch_size, seq_len, False, num_workers),
        build_loader(root / "test_raw.pkl", batch_size, seq_len, False, num_workers),
    )
