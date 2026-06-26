
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


default_config: dict[str, Any] = {
    "dataset": "custom",
    "seq_len": 100,
    "skill_num": 0,
    "item_num": 0,
    "student_num": 0,

    "d_model": 128,
    "n_heads": 8,
    "n_layers": 2,
    "dropout": 0.1,

    "lr": 2e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epoch": 200,
    "patience": 20,
    "grad_clip": 1.0,
    "seed": 2025,
    "num_workers": 0,
    "use_amp": True,
    "scheduler": "cosine",

    "router_entropy_alpha": 0.003,
    "router_quad_beta": 0.001,
}


def make_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(default_config)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
