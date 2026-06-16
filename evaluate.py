from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from msbr_kt.data import build_loader
from msbr_kt.model import MSBRKT
from msbr_kt.relations import BIAS_NAMES, RelationResources
from msbr_kt.training import evaluate
from msbr_kt.utils import initialize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MSBR-KT and optionally dump step-level predictions.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to <run_dir>/best.ckpt.")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--dump_pred_csv", default=None)
    return parser.parse_args()


@torch.no_grad()
def dump_predictions(model, loader, device, out_csv: str | Path) -> None:
    model.eval()
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user_id",
        "t_padded",
        "pos_in_seq",
        "problem_id",
        "skill_id",
        "label",
        "prob",
    ] + [f"pi_{name}" for name in BIAS_NAMES]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits, aux = model(batch, return_aux=True)
            probs = torch.sigmoid(logits).detach().cpu()
            labels = batch["label_seq"].detach().cpu()
            items = batch["item_seq"].detach().cpu()
            skills = batch["skill_seq"].detach().cpu()
            masks = batch["mask"].detach().cpu().bool()
            users = batch["user_id"].detach().cpu()
            pi = aux["pi"].detach().cpu()

            for b in range(probs.size(0)):
                valid_idx = torch.where(masks[b])[0].tolist()
                pos_map = {int(t): int(p) for p, t in enumerate(valid_idx)}
                for t in valid_idx:
                    row = {
                        "user_id": int(users[b]),
                        "t_padded": int(t),
                        "pos_in_seq": pos_map[int(t)],
                        "problem_id": int(items[b, t]),
                        "skill_id": int(skills[b, t]),
                        "label": float(labels[b, t]),
                        "prob": float(probs[b, t]),
                    }
                    for k, name in enumerate(BIAS_NAMES):
                        row[f"pi_{name}"] = float(pi[b, t, k])
                    writer.writerow(row)
    print(f"[DUMP] predictions -> {out_csv}")


def main() -> None:
    args = parse_args()
    cfg, device = initialize_run(args.data_root, args.run_dir, overrides=None, config_path=Path(args.run_dir) / "config.json")
    relations = RelationResources.load(args.data_root)
    loader = build_loader(
        Path(args.data_root) / f"{args.split}_raw.pkl",
        int(cfg["batch_size"]),
        int(cfg["seq_len"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    model = MSBRKT(cfg, relations).to(device)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(args.run_dir) / "best.ckpt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    metrics, _, _, _ = evaluate(model, loader, device, cfg)
    print(
        f"[{args.split.upper()}] AUC={metrics['AUC']:.4f} | ACC={metrics['ACC']:.4f} | "
        f"AP={metrics['AP']:.4f} | RMSE={metrics['RMSE']:.4f}"
    )
    if args.dump_pred_csv:
        dump_predictions(model, loader, device, args.dump_pred_csv)


if __name__ == "__main__":
    main()
