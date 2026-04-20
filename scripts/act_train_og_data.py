#!/usr/bin/env python3
"""Train ACT policy on the local Push-T zarr replay buffer.

Example:
    python scripts/act_train_og_data.py --output_dir models/act_og_data \
        --wandb --wandb_project introML-proj-graphs \
        --wandb_entity yizhoul2-carnegie-mellon-university
"""
import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataloader_pusht import build_pusht_dataloaders
from models.act import ACTPolicy

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainConfig:
    zarr_path: str = "data/pusht/pusht_cchi_v7_replay.zarr"
    output_dir: str = "models/act_pusht"
    seed: int = 42
    val_ratio: float = 0.1
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    horizon: int = 20
    state_dim: int = 5
    action_dim: int = 2
    hidden_dim: int = 256
    latent_dim: int = 32
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 7
    kl_beta: float = 10.0
    image_shift_px: int = 4
    num_workers: int = 4
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None


def random_shift_batch(images: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return images

    b, c, h, w = images.shape
    pad = int(max_shift)
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    shift_y = torch.randint(0, 2 * pad + 1, (b,), device=images.device)
    shift_x = torch.randint(0, 2 * pad + 1, (b,), device=images.device)

    out = torch.empty_like(images)
    for i in range(b):
        y0 = int(shift_y[i].item())
        x0 = int(shift_x[i].item())
        out[i] = padded[i, :, y0 : y0 + h, x0 : x0 + w]
    return out


def preprocess_image_batch(
    images: torch.Tensor, image_normalize, random_shift_px: int = 0
) -> torch.Tensor:
    # Images come from PushTZarrDataset already as (B,3,96,96) float32 in [0,1].
    images = images.to(dtype=torch.float32)
    images = random_shift_batch(images, max_shift=random_shift_px)
    images = image_normalize(images)
    return images


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    valid = (~is_pad).unsqueeze(-1)
    abs_error = torch.abs(pred - target)
    valid_error = abs_error * valid.float()
    denom = valid.float().sum().clamp_min(1.0) * pred.size(-1)
    return valid_error.sum() / denom


def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    image_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    print(f"Loading Push-T zarr from {config.zarr_path}...")
    bundle = build_pusht_dataloaders(
        zarr_path=config.zarr_path,
        horizon=config.horizon,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    print(
        f"Push-T loaded: {bundle.num_train} train steps, {bundle.num_val} val steps, "
        f"state_dim={bundle.state_dim}, action_dim={bundle.action_dim}"
    )

    if bundle.state_dim != config.state_dim:
        print(
            f"[warn] config.state_dim={config.state_dim} differs from data state_dim "
            f"={bundle.state_dim}; using data state_dim."
        )
    if bundle.action_dim != config.action_dim:
        print(
            f"[warn] config.action_dim={config.action_dim} differs from data action_dim "
            f"={bundle.action_dim}; using data action_dim."
        )

    state_dim = bundle.state_dim
    action_dim = bundle.action_dim

    model = ACTPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=config.horizon,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
    ).to(device)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training ACT with {num_parameters} trainable parameters")

    wandb_run = None
    if config.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install dependencies from requirements.txt or disable --wandb.")
        if not config.wandb_project or not config.wandb_entity:
            raise ValueError("--wandb requires both --wandb_project and --wandb_entity")
        run_name = f"{Path(config.output_dir).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=asdict(config),
            tags=["act", "train", "pusht-zarr"],
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    state_mean_t = torch.tensor(bundle.state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(bundle.state_std, dtype=torch.float32, device=device)
    action_mean_t = torch.tensor(bundle.action_mean, dtype=torch.float32, device=device)
    action_std_t = torch.tensor(bundle.action_std, dtype=torch.float32, device=device)

    best_val = float("inf")

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_recon_sum = 0.0
            train_kl_sum = 0.0
            train_batches = 0

            for batch in tqdm(bundle.train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False):
                images = preprocess_image_batch(
                    batch["image"].to(device),
                    image_normalize,
                    random_shift_px=config.image_shift_px,
                )
                states_b = batch["state"].to(device)
                action_chunk = batch["action_chunk"].to(device)
                action_is_pad_b = batch["action_is_pad"].to(device)

                states_norm = (states_b - state_mean_t) / state_std_t
                target_actions = (action_chunk - action_mean_t.view(1, 1, -1)) / action_std_t.view(1, 1, -1)

                pred_actions, mu, logvar = model(images, states_norm, target_actions)
                recon_loss = masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
                kl_loss = ACTPolicy.kl_divergence(mu, logvar)
                loss = recon_loss + config.kl_beta * kl_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += float(loss.item())
                train_recon_sum += float(recon_loss.item())
                train_kl_sum += float(kl_loss.item())
                train_batches += 1

            train_loss = train_loss_sum / max(1, train_batches)
            train_recon = train_recon_sum / max(1, train_batches)
            train_kl = train_kl_sum / max(1, train_batches)

            model.eval()
            val_loss_sum = 0.0
            val_recon_sum = 0.0
            val_kl_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in tqdm(bundle.val_loader, desc=f"Val {epoch}/{config.epochs}", leave=False):
                    images = preprocess_image_batch(
                        batch["image"].to(device),
                        image_normalize,
                        random_shift_px=0,
                    )
                    states_b = batch["state"].to(device)
                    action_chunk = batch["action_chunk"].to(device)
                    action_is_pad_b = batch["action_is_pad"].to(device)

                    states_norm = (states_b - state_mean_t) / state_std_t
                    target_actions = (action_chunk - action_mean_t.view(1, 1, -1)) / action_std_t.view(1, 1, -1)

                    pred_actions, mu, logvar = model(images, states_norm, target_actions)
                    recon_loss = masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
                    kl_loss = ACTPolicy.kl_divergence(mu, logvar)
                    loss = recon_loss + config.kl_beta * kl_loss

                    val_loss_sum += float(loss.item())
                    val_recon_sum += float(recon_loss.item())
                    val_kl_sum += float(kl_loss.item())
                    val_batches += 1

            val_loss = val_loss_sum / max(1, val_batches)
            val_recon = val_recon_sum / max(1, val_batches)
            val_kl = val_kl_sum / max(1, val_batches)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "state_mean": bundle.state_mean.astype(np.float32),
                "state_std": bundle.state_std.astype(np.float32),
                "action_mean": bundle.action_mean.astype(np.float32),
                "action_std": bundle.action_std.astype(np.float32),
                "config": asdict(config),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_loss": val_loss,
                "val_recon": val_recon,
                "val_kl": val_kl,
                "state_dim": state_dim,
                "action_dim": action_dim,
            }

            latest_path = os.path.join(config.output_dir, "latest.pt")
            torch.save(checkpoint, latest_path)
            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(config.output_dir, "best.pt")
                torch.save(checkpoint, best_path)
                if wandb_run is not None:
                    wandb.log({"best/val_loss": float(best_val)}, step=epoch)

            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"train={train_loss:.6f} (recon={train_recon:.6f}, kl={train_kl:.6f}) | "
                f"val={val_loss:.6f} (recon={val_recon:.6f}, kl={val_kl:.6f})"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": float(train_loss),
                        "val/loss": float(val_loss),
                        "train/recon_loss": float(train_recon),
                        "train/kl_loss": float(train_kl),
                        "val/recon_loss": float(val_recon),
                        "val/kl_loss": float(val_kl),
                        "train/kl_beta": float(config.kl_beta),
                    },
                    step=epoch,
                )

        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = float(best_val)
    finally:
        if wandb_run is not None:
            wandb.finish()

    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    np.savez(
        os.path.join(config.output_dir, "normalization_stats.npz"),
        state_mean=bundle.state_mean.astype(np.float32),
        state_std=bundle.state_std.astype(np.float32),
        action_mean=bundle.action_mean.astype(np.float32),
        action_std=bundle.action_std.astype(np.float32),
    )

    print(f"Saved ACT artifacts to: {config.output_dir}")
    print(f"Best validation loss: {best_val:.6f}")


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train ACT on local Push-T zarr buffer")
    parser.add_argument("--zarr_path", type=str, default=defaults.zarr_path)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--val_ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--horizon", type=int, default=defaults.horizon)
    parser.add_argument("--state_dim", type=int, default=defaults.state_dim)
    parser.add_argument("--action_dim", type=int, default=defaults.action_dim)
    parser.add_argument("--hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--latent_dim", type=int, default=defaults.latent_dim)
    parser.add_argument("--nhead", type=int, default=defaults.nhead)
    parser.add_argument("--num_encoder_layers", type=int, default=defaults.num_encoder_layers)
    parser.add_argument("--num_decoder_layers", type=int, default=defaults.num_decoder_layers)
    parser.add_argument("--kl_beta", type=float, default=defaults.kl_beta)
    parser.add_argument("--image_shift_px", type=int, default=defaults.image_shift_px)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--wandb", action="store_true", default=defaults.wandb)
    parser.add_argument("--wandb_project", type=str, default=defaults.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=defaults.wandb_entity)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
