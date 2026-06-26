from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import make_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(obj), f, indent=2, ensure_ascii=False)


def load_global_maps(data_root: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(data_root) / "global_maps.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. The processed data directory must contain global_maps.pt."
        )
    return torch.load(path, map_location="cpu")


def initialize_run(
    data_root: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    overrides: Mapping[str, Any] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], torch.device]:
    config_overrides: dict[str, Any] = {}
    if config_path is not None:
        config_overrides.update(load_json(config_path))
    if overrides:
        config_overrides.update({k: v for k, v in overrides.items() if v is not None})

    cfg = make_config(config_overrides)
    data_root = Path(data_root)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    global_maps = load_global_maps(data_root)
    cfg["student_num"] = int(global_maps.get("U", cfg.get("student_num", 0)))
    cfg["item_num"] = int(global_maps.get("I", cfg.get("item_num", 0)))
    cfg["skill_num"] = int(global_maps.get("S", cfg.get("skill_num", 0)))
    cfg["data_root"] = str(data_root)

    set_seed(int(cfg.get("seed", 2025)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_json(cfg, run_dir / "config.json")
    return cfg, device
