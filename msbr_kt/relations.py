from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


BIAS_NAMES = ["B_time", "B_item", "B_skill", "B_dir", "B_right", "B_wrong"]
REQUIRED_RELATION_KEYS = [
    "item_dir_idx",
    "item_dir_val",
    "item_right_idx",
    "item_right_val",
    "item_wrong_idx",
    "item_wrong_val",
]


class RelationResources:
    """Container for precomputed relation tensors used by MSBR-KT.

    The project intentionally does not include raw-data preprocessing code. The
    processed data directory is expected to provide ``relations.pt`` with the
    keys listed in ``REQUIRED_RELATION_KEYS``.
    """

    def __init__(self, rel_dict: dict[str, Any], device: torch.device | str | None = None):
        missing = [k for k in REQUIRED_RELATION_KEYS if k not in rel_dict]
        if missing:
            raise KeyError(f"relations.pt is missing required keys: {missing}")
        if device is not None:
            rel_dict = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in rel_dict.items()
            }
        self.data = rel_dict
        self.__dict__.update(rel_dict)

    @classmethod
    def load(cls, data_root: str | Path, device: torch.device | str | None = None) -> "RelationResources":
        path = Path(data_root) / "relations.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Please place the precomputed relation tensors "
                "in the processed data directory."
            )
        rel_dict = torch.load(path, map_location=device or "cpu")
        return cls(rel_dict, device=device)


def build_time_bias(dt_seq: torch.Tensor, valid_mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Build the temporal structured bias matrix.

    Args:
        dt_seq: Tensor of shape ``[B, T]``. Each value is the interval since the
            previous interaction.
        valid_mask: Optional tensor of shape ``[B, T]`` where True/1 indicates a
            valid interaction.
        eps: Numerical stability constant.

    Returns:
        Tensor of shape ``[B, T, T]``. Diagonal and invalid pairs are zeroed.
    """
    dt_seq = torch.nan_to_num(dt_seq.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tau = torch.cumsum(dt_seq, dim=1)
    delta = torch.abs(tau.unsqueeze(2) - tau.unsqueeze(1))

    if valid_mask is not None:
        valid = valid_mask.bool()
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        denom = pair_valid.float().sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
        mean = (delta * pair_valid.float()).sum(dim=(-1, -2), keepdim=True) / denom
        var = (((delta - mean) * pair_valid.float()) ** 2).sum(dim=(-1, -2), keepdim=True) / denom
        sigma = torch.sqrt(var).clamp_min(eps)
    else:
        sigma = delta.std(dim=(-1, -2), keepdim=True).clamp_min(eps)

    bias = -torch.log1p(delta / sigma)
    eye = torch.eye(dt_seq.size(1), device=dt_seq.device, dtype=torch.bool).unsqueeze(0)
    bias = bias.masked_fill(eye, 0.0)
    if valid_mask is not None:
        bias = bias.masked_fill(~pair_valid, 0.0)
    return bias
