#!/usr/bin/env python3
"""Train a two-hidden-layer state MLP (BC) on LeRobot ``pusht_keypoints``.

Uses :func:`data.dataloader_keypoints.build_keypoints_dataloaders` with
``horizon=1`` (single-step actions).  States and actions are min–max normalized
to ``[-1, 1]`` on the train split, matching :mod:`scripts.act_train_keypoints`.

Example:
    python scripts/bc_mlp_keypoints_train.py --output_dir models/bc_mlp_keypoints
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataloader_keypoints import build_keypoints_dataloaders, min_max_normalize
from models.bc_mlp_keypoints import BCKeypointsMLP, DEFAULT_STATE_DIM, DEFAULT_ACTION_DIM

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht_keypoints"
    output_dir: str = "models/bc_mlp_keypoints"
    seed: int = 42
    val_ratio: float = 0.1
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    num_workers: int = 4
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None


def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    wandb_run = None
    if config.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. pip install wandb or omit --wandb.")
        if not config.wandb_project or not config.wandb_entity:
            raise ValueError("--wandb requires --wandb_project and --wandb_entity")
        run_name = f"{Path(config.output_dir).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=asdict(config),
            tags=["bc_mlp", "keypoints", "train"],
        )

    print(f"Loading {config.dataset_id} (horizon=1 for single-step BC)...")
    # Default DataLoader has persistent_workers=False, so val workers respawn every epoch;
    # use a single val worker (when train uses workers) to keep that cost small.
    val_nw = 1 if config.num_workers > 0 else 0
    bundle = build_keypoints_dataloaders(
        dataset_id=config.dataset_id,
        horizon=1,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
        val_num_workers=val_nw,
        pin_memory=True,
    )

    state_min_np = bundle.state_min
    state_max_np = bundle.state_max
    action_min_np = bundle.action_min
    action_max_np = bundle.action_max
    if bundle.state_dim != DEFAULT_STATE_DIM or bundle.action_dim != DEFAULT_ACTION_DIM:
        raise ValueError(
            f"Expected state_dim={DEFAULT_STATE_DIM} and action_dim={DEFAULT_ACTION_DIM}, "
            f"got {bundle.state_dim} and {bundle.action_dim}."
        )

    state_min = torch.tensor(state_min_np, dtype=torch.float32, device=device)
    state_max = torch.tensor(state_max_np, dtype=torch.float32, device=device)
    action_min = torch.tensor(action_min_np, dtype=torch.float32, device=device)
    action_max = torch.tensor(action_max_np, dtype=torch.float32, device=device)

    model = BCKeypointsMLP(
        state_dim=bundle.state_dim,
        action_dim=bundle.action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val = float("inf")
    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            t_loss = 0.0
            for batch in tqdm(bundle.train_loader, desc=f"train {epoch}"):
                s = batch["state"].to(device, non_blocking=True)
                a = batch["action_chunk"][:, 0, :].to(device, non_blocking=True)

                s_n = min_max_normalize(s, state_min, state_max)
                a_n = min_max_normalize(a, action_min, action_max)
                pred = model(s_n)
                loss = loss_fn(pred, a_n)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                t_loss += loss.item()

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(bundle.val_loader, desc=f"val {epoch}"):
                    s = batch["state"].to(device, non_blocking=True)
                    a = batch["action_chunk"][:, 0, :].to(device, non_blocking=True)
                    s_n = min_max_normalize(s, state_min, state_max)
                    a_n = min_max_normalize(a, action_min, action_max)
                    pred = model(s_n)
                    v_loss += loss_fn(pred, a_n).item()

            avg_t = t_loss / max(len(bundle.train_loader), 1)
            avg_v = v_loss / max(len(bundle.val_loader), 1)
            print(f"Epoch {epoch:03d} | Train: {avg_t:.6f} | Val: {avg_v:.6f}")
            if wandb_run is not None:
                wandb.log(
                    {"epoch": epoch, "train/loss": float(avg_t), "val/loss": float(avg_v)},
                    step=epoch,
                )

            ckpt = {
                "model_state_dict": model.state_dict(),
                "state_min": state_min_np.astype(np.float32),
                "state_max": state_max_np.astype(np.float32),
                "action_min": action_min_np.astype(np.float32),
                "action_max": action_max_np.astype(np.float32),
                "config": {
                    **asdict(config),
                    "norm_mode": "min_max",
                    "state_dim": bundle.state_dim,
                    "action_dim": bundle.action_dim,
                },
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(config.output_dir, "latest.pt"))
            if avg_v < best_val:
                best_val = avg_v
                torch.save(ckpt, os.path.join(config.output_dir, "best.pt"))
                print(f"  --> new best val loss: {best_val:.6f}")
                if wandb_run is not None:
                    wandb.log({"best/val_loss": float(best_val)}, step=epoch)

        with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = float(best_val)
    finally:
        if wandb_run is not None:
            wandb.finish()


def main() -> None:
    p = argparse.ArgumentParser(description="Train BC MLP on lerobot/pusht_keypoints")
    p.add_argument("--dataset_id", type=str, default="lerobot/pusht_keypoints")
    p.add_argument("--output_dir", type=str, default="models/bc_mlp_keypoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    args = p.parse_args()
    train(
        TrainConfig(
            dataset_id=args.dataset_id,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
    )


if __name__ == "__main__":
    main()
