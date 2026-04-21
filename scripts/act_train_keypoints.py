#!/usr/bin/env python3
"""Train ACT (LeRobot-config variant) on the LeRobot ``pusht_keypoints`` dataset.

Uses :class:`models.act_lerobot.ACTLeRobotPolicy`, configured via
:class:`models.act_lerobot.ACTLeRobotConfig`.  The dataset still provides an
18-D concatenated state ``[agent_pos (2) | keypoints (16)]``; at the model
boundary it is split into the LeRobot two-key observation dict
``{"observation.state": (B, 2), "observation.environment_state": (B, 16)}``
to match ``input_shapes`` in the LeRobot ACT config.

Example:
    python scripts/act_train_keypoints.py --output_dir models/act_keypoints \
        --wandb --wandb_project introML-proj-graphs \
        --wandb_entity yizhoul2-carnegie-mellon-university
"""
import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataloader_keypoints import build_keypoints_dataloaders, min_max_normalize
from models.act_lerobot import ACTLeRobotConfig, ACTLeRobotPolicy

try:
    import wandb
except ImportError:
    wandb = None


# Default LeRobot ACT config for the pusht_keypoints task.
DEFAULT_LEROBOT_CFG: Dict = {
    "chunk_size": 16,
    "dim_feedforward": 3200,
    "dim_model": 512,
    "dropout": 0.1,
    "feedforward_activation": "gelu",
    "input_normalization_modes": {
        "observation.environment_state": "min_max",
        "observation.state": "min_max",
    },
    "input_shapes": {
        "observation.environment_state": [16],
        "observation.state": [2],
    },
    "kl_weight": 10.0,
    "latent_dim": 32,
    "n_action_steps": 16,
    "n_decoder_layers": 4,
    "n_encoder_layers": 4,
    "n_heads": 8,
    "n_obs_steps": 1,
    "n_vae_encoder_layers": 4,
    "output_normalization_modes": {"action": "min_max"},
    "output_shapes": {"action": [2]},
    "pre_norm": False,
    "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
    "replace_final_stride_with_dilation": False,
    "temporal_ensemble_momentum": None,
    "use_vae": True,
    "vision_backbone": "resnet18",
}

AGENT_POS_DIM = 2
ENV_STATE_DIM = 16
STATE_DIM = AGENT_POS_DIM + ENV_STATE_DIM
ACTION_DIM = 2


@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht_keypoints"
    output_dir: str = "models/act_keypoints"
    seed: int = 42
    val_ratio: float = 0.1
    epochs: int = 250
    batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    num_workers: int = 4
    # The LeRobot ACT config dict (any subset of ACTLeRobotConfig fields).
    # CLI flags below allow overriding common scalar fields.
    lerobot_cfg: Dict = field(default_factory=lambda: dict(DEFAULT_LEROBOT_CFG))
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None


def split_state_to_obs(state: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split the 18-D concatenated state into the LeRobot two-key obs dict."""
    return {
        "observation.state": state[..., :AGENT_POS_DIM],
        "observation.environment_state": state[..., AGENT_POS_DIM:],
    }


def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    lerobot_cfg = ACTLeRobotConfig.from_dict(config.lerobot_cfg)
    horizon = int(lerobot_cfg.chunk_size)
    kl_weight = float(lerobot_cfg.kl_weight)

    # Sanity-check the input/output shapes match the keypoints data layout.
    expected_input_shapes = {
        "observation.state": [AGENT_POS_DIM],
        "observation.environment_state": [ENV_STATE_DIM],
    }
    if dict(lerobot_cfg.input_shapes) != expected_input_shapes:
        raise ValueError(
            "lerobot_cfg.input_shapes must match the keypoints layout "
            f"{expected_input_shapes}; got {dict(lerobot_cfg.input_shapes)}"
        )
    if list(lerobot_cfg.output_shapes.get("action", [])) != [ACTION_DIM]:
        raise ValueError(
            f"lerobot_cfg.output_shapes['action'] must be [{ACTION_DIM}]"
        )

    print(f"Loading {config.dataset_id}...")
    bundle = build_keypoints_dataloaders(
        dataset_id=config.dataset_id,
        horizon=horizon,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    print(
        f"Keypoints loaded: {bundle.num_train} train steps, {bundle.num_val} val steps, "
        f"state_dim={bundle.state_dim}, action_dim={bundle.action_dim}"
    )
    if bundle.state_dim != STATE_DIM:
        raise ValueError(f"Expected concatenated state_dim={STATE_DIM}, got {bundle.state_dim}")
    if bundle.action_dim != ACTION_DIM:
        raise ValueError(f"Expected action_dim={ACTION_DIM}, got {bundle.action_dim}")

    model = ACTLeRobotPolicy(lerobot_cfg).to(device)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"training ACTLeRobotPolicy with {num_parameters:,} trainable parameters | "
        f"chunk_size={horizon} dim_model={lerobot_cfg.dim_model} "
        f"n_enc/n_dec/n_vae={lerobot_cfg.n_encoder_layers}/{lerobot_cfg.n_decoder_layers}/"
        f"{lerobot_cfg.n_vae_encoder_layers} use_vae={lerobot_cfg.use_vae}"
    )

    wandb_run = None
    if config.wandb:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install dependencies from requirements.txt or disable --wandb."
            )
        if not config.wandb_project or not config.wandb_entity:
            raise ValueError("--wandb requires both --wandb_project and --wandb_entity")
        run_name = f"{Path(config.output_dir).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config={**asdict(config), "lerobot_cfg": asdict(lerobot_cfg)},
            tags=["act", "train", "pusht-keypoints", "lerobot-cfg"],
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    state_min_t = torch.tensor(bundle.state_min, dtype=torch.float32, device=device)
    state_max_t = torch.tensor(bundle.state_max, dtype=torch.float32, device=device)
    action_min_t = torch.tensor(bundle.action_min, dtype=torch.float32, device=device)
    action_max_t = torch.tensor(bundle.action_max, dtype=torch.float32, device=device)

    best_val = float("inf")

    def _step(batch, train_mode: bool):
        states_b = batch["state"].to(device)
        action_chunk = batch["action_chunk"].to(device)
        action_is_pad_b = batch["action_is_pad"].to(device)

        states_norm = min_max_normalize(states_b, state_min_t, state_max_t)
        target_actions = min_max_normalize(
            action_chunk,
            action_min_t.view(1, 1, -1),
            action_max_t.view(1, 1, -1),
        )

        obs = split_state_to_obs(states_norm)
        pred_actions, mu, logvar = model(obs, target_actions if train_mode else target_actions)
        recon_loss = ACTLeRobotPolicy.masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
        if mu is not None and logvar is not None:
            kl_loss = ACTLeRobotPolicy.kl_divergence(mu, logvar)
        else:
            kl_loss = torch.zeros((), device=device)
        loss = recon_loss + kl_weight * kl_loss
        return loss, recon_loss, kl_loss

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            train_loss_sum = train_recon_sum = train_kl_sum = 0.0
            train_batches = 0

            for batch in tqdm(bundle.train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False):
                loss, recon_loss, kl_loss = _step(batch, train_mode=True)

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
            val_loss_sum = val_recon_sum = val_kl_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in tqdm(bundle.val_loader, desc=f"Val {epoch}/{config.epochs}", leave=False):
                    loss, recon_loss, kl_loss = _step(batch, train_mode=False)
                    val_loss_sum += float(loss.item())
                    val_recon_sum += float(recon_loss.item())
                    val_kl_sum += float(kl_loss.item())
                    val_batches += 1

            val_loss = val_loss_sum / max(1, val_batches)
            val_recon = val_recon_sum / max(1, val_batches)
            val_kl = val_kl_sum / max(1, val_batches)

            checkpoint_config = asdict(config)
            checkpoint_config["lerobot_cfg"] = asdict(lerobot_cfg)
            checkpoint_config["norm_mode"] = "min_max"
            checkpoint_config["model_arch"] = "act_lerobot"

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "state_min": bundle.state_min.astype(np.float32),
                "state_max": bundle.state_max.astype(np.float32),
                "action_min": bundle.action_min.astype(np.float32),
                "action_max": bundle.action_max.astype(np.float32),
                "config": checkpoint_config,
                "lerobot_cfg": asdict(lerobot_cfg),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_loss": val_loss,
                "val_recon": val_recon,
                "val_kl": val_kl,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
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
                        "train/kl_weight": float(kl_weight),
                    },
                    step=epoch,
                )

        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = float(best_val)
    finally:
        if wandb_run is not None:
            wandb.finish()

    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump({**asdict(config), "lerobot_cfg": asdict(lerobot_cfg)}, file, indent=2)

    np.savez(
        os.path.join(config.output_dir, "normalization_stats.npz"),
        state_min=bundle.state_min.astype(np.float32),
        state_max=bundle.state_max.astype(np.float32),
        action_min=bundle.action_min.astype(np.float32),
        action_max=bundle.action_max.astype(np.float32),
    )

    print(f"Saved ACT (LeRobot) artifacts to: {config.output_dir}")
    print(f"Best validation loss: {best_val:.6f}")


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train ACT (LeRobot-config variant) on lerobot/pusht_keypoints (state-only)"
    )
    parser.add_argument("--dataset_id", type=str, default=defaults.dataset_id)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--val_ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)

    # Optional path to a JSON file with a (partial) LeRobot ACT config dict.
    parser.add_argument(
        "--lerobot_cfg_json",
        type=str,
        default=None,
        help="Optional JSON file with LeRobot ACT config overrides (deep-merged into defaults).",
    )

    # Common scalar overrides (subset of ACTLeRobotConfig fields).
    cfg = DEFAULT_LEROBOT_CFG
    parser.add_argument("--chunk_size", type=int, default=cfg["chunk_size"])
    parser.add_argument("--dim_model", type=int, default=cfg["dim_model"])
    parser.add_argument("--dim_feedforward", type=int, default=cfg["dim_feedforward"])
    parser.add_argument("--dropout", type=float, default=cfg["dropout"])
    parser.add_argument("--feedforward_activation", type=str, default=cfg["feedforward_activation"])
    parser.add_argument("--latent_dim", type=int, default=cfg["latent_dim"])
    parser.add_argument("--n_heads", type=int, default=cfg["n_heads"])
    parser.add_argument("--n_encoder_layers", type=int, default=cfg["n_encoder_layers"])
    parser.add_argument("--n_decoder_layers", type=int, default=cfg["n_decoder_layers"])
    parser.add_argument("--n_vae_encoder_layers", type=int, default=cfg["n_vae_encoder_layers"])
    parser.add_argument("--n_obs_steps", type=int, default=cfg["n_obs_steps"])
    parser.add_argument("--n_action_steps", type=int, default=cfg["n_action_steps"])
    parser.add_argument("--kl_weight", type=float, default=cfg["kl_weight"])
    parser.add_argument("--pre_norm", action=argparse.BooleanOptionalAction, default=cfg["pre_norm"])
    parser.add_argument("--use_vae", action=argparse.BooleanOptionalAction, default=cfg["use_vae"])
    parser.add_argument("--vision_backbone", type=str, default=cfg["vision_backbone"])
    parser.add_argument("--pretrained_backbone_weights", type=str, default=cfg["pretrained_backbone_weights"])
    parser.add_argument(
        "--replace_final_stride_with_dilation",
        action=argparse.BooleanOptionalAction,
        default=cfg["replace_final_stride_with_dilation"],
    )

    parser.add_argument("--wandb", action="store_true", default=defaults.wandb)
    parser.add_argument("--wandb_project", type=str, default=defaults.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=defaults.wandb_entity)
    args = parser.parse_args()

    lerobot_cfg = dict(DEFAULT_LEROBOT_CFG)
    if args.lerobot_cfg_json is not None:
        with open(args.lerobot_cfg_json, "r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        lerobot_cfg.update(user_cfg)
    # Apply scalar overrides on top.
    lerobot_cfg.update(
        {
            "chunk_size": args.chunk_size,
            "dim_model": args.dim_model,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "feedforward_activation": args.feedforward_activation,
            "latent_dim": args.latent_dim,
            "n_heads": args.n_heads,
            "n_encoder_layers": args.n_encoder_layers,
            "n_decoder_layers": args.n_decoder_layers,
            "n_vae_encoder_layers": args.n_vae_encoder_layers,
            "n_obs_steps": args.n_obs_steps,
            "n_action_steps": args.n_action_steps,
            "kl_weight": args.kl_weight,
            "pre_norm": bool(args.pre_norm),
            "use_vae": bool(args.use_vae),
            "vision_backbone": args.vision_backbone,
            "pretrained_backbone_weights": args.pretrained_backbone_weights,
            "replace_final_stride_with_dilation": bool(args.replace_final_stride_with_dilation),
        }
    )

    return TrainConfig(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        lerobot_cfg=lerobot_cfg,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    train(parse_args())
