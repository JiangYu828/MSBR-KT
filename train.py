from __future__ import annotations

import argparse
from pathlib import Path

import torch

from msbr_kt.data import build_split_loaders
from msbr_kt.model import MSBRKT
from msbr_kt.relations import RelationResources
from msbr_kt.training import evaluate, make_param_groups, save_checkpoint, train_one_epoch
from msbr_kt.utils import initialize_run, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MSBR-KT on processed sequence data.")
    parser.add_argument("--data_root", required=True, help="Directory containing train_raw.pkl, valid_raw.pkl, test_raw.pkl, global_maps.pt, and relations.pt.")
    parser.add_argument("--run_dir", default="runs/msbrkt", help="Directory for checkpoints and metrics.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Router entropy regularization coefficient.")
    parser.add_argument("--beta", type=float, default=None, help="Router quadratic regularization coefficient.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "max_epoch": args.max_epoch,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "scheduler": args.scheduler,
        "router_entropy_alpha": args.alpha,
        "router_quad_beta": args.beta,
    }
    if args.no_amp:
        overrides["use_amp"] = False

    cfg, device = initialize_run(args.data_root, args.run_dir, overrides=overrides, config_path=args.config)
    train_loader, valid_loader, test_loader = build_split_loaders(args.data_root, cfg)
    relations = RelationResources.load(args.data_root)
    model = MSBRKT(cfg, relations).to(device)

    optimizer = torch.optim.AdamW(make_param_groups(model, cfg), lr=float(cfg["lr"]))
    scheduler = None
    if cfg.get("scheduler", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["max_epoch"])))
    amp_enabled = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    run_dir = Path(args.run_dir)
    best_auc = -1.0
    best_epoch = -1
    best_metrics = None
    bad_epochs = 0

    for epoch in range(1, int(cfg["max_epoch"]) + 1):
        train_loss = train_one_epoch(model, train_loader, cfg, device, optimizer, scaler)
        valid_metrics, _, _, _ = evaluate(model, valid_loader, device, cfg)
        if scheduler is not None:
            scheduler.step()

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | "
            f"VAL AUC={valid_metrics['AUC']:.4f} ACC={valid_metrics['ACC']:.4f} "
            f"AP={valid_metrics['AP']:.4f} RMSE={valid_metrics['RMSE']:.4f}"
        )

        if valid_metrics["AUC"] > best_auc:
            best_auc = valid_metrics["AUC"]
            best_epoch = epoch
            best_metrics = valid_metrics
            bad_epochs = 0
            save_checkpoint(model, run_dir / "best.ckpt")
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["patience"]):
                print(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(run_dir / "best.ckpt", map_location=device))
    test_metrics, _, _, _ = evaluate(model, test_loader, device, cfg)
    print(
        f"[TEST] AUC={test_metrics['AUC']:.4f} | ACC={test_metrics['ACC']:.4f} | "
        f"AP={test_metrics['AP']:.4f} | RMSE={test_metrics['RMSE']:.4f}"
    )
    save_json({"best_epoch": best_epoch, "valid": best_metrics, "test": test_metrics}, run_dir / "metrics.json")


if __name__ == "__main__":
    main()
