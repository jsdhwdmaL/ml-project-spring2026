#!/usr/bin/env python3
"""Train ACT policy on lerobot/pusht using chunked action supervision."""

import argparse
import json
import os
import sys
from datetime import datetime
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.build_chunk import build_action_chunks_by_episode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from models.act import ACTPolicy

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    split: str = "train"
    output_dir: str = "models/act"
    seed: int = 42
    val_ratio: float = 0.1
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    horizon: int = 20
    hidden_dim: int = 256
    latent_dim: int = 32
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 7
    kl_beta: float = 0.5
    kl_warmup_epochs: int = 100
    kl_free_nats: float = 0.1
    image_shift_px: int = 4
    ensemble_decay: float = 0.05
    amp: bool = True
    compile_model: bool = True
    use_vision: bool = True
    """If False, train on pymunk-style ground-truth state only (no ResNet on pixels)."""
    gt_state_zarr: str | None = None
    """Optional path to a replay .zarr with data/state (N, 5) aligned with lerobot frame order."""
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None


class ACTStepDataset(Dataset):
    def __init__(
        self,
        base_dataset: LeRobotDataset,
        step_indices: np.ndarray,
        action_chunks: np.ndarray,
        action_is_pad: np.ndarray,
    ):
        self.base_dataset = base_dataset
        self.step_indices = step_indices.astype(np.int64)
        self.action_chunks = action_chunks
        self.action_is_pad = action_is_pad

    def __len__(self) -> int:
        return int(self.step_indices.shape[0])

    def __getitem__(self, idx: int):
        global_idx = int(self.step_indices[idx])
        sample = self.base_dataset[global_idx]
        return {
            "image": sample["observation.image"].float(),
            "state": sample["observation.state"].float(),
            "action_chunk": torch.from_numpy(self.action_chunks[global_idx]).float(),
            "action_is_pad": torch.from_numpy(self.action_is_pad[global_idx]).bool(),
        }


class ACTStepDatasetStateOnly(Dataset):
    """Same as ACTStepDataset but uses a precomputed state array (no image loading)."""

    def __init__(
        self,
        step_indices: np.ndarray,
        states: np.ndarray,
        action_chunks: np.ndarray,
        action_is_pad: np.ndarray,
    ):
        self.step_indices = step_indices.astype(np.int64)
        self.states = states.astype(np.float32)
        self.action_chunks = action_chunks
        self.action_is_pad = action_is_pad

    def __len__(self) -> int:
        return int(self.step_indices.shape[0])

    def __getitem__(self, idx: int):
        global_idx = int(self.step_indices[idx])
        return {
            "state": torch.from_numpy(self.states[global_idx]).float(),
            "action_chunk": torch.from_numpy(self.action_chunks[global_idx]).float(),
            "action_is_pad": torch.from_numpy(self.action_is_pad[global_idx]).bool(),
        }


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


def preprocess_image_batch(images: torch.Tensor, image_normalize, random_shift_px: int = 0) -> torch.Tensor:
    images = images.to(dtype=torch.float32)
    if images.ndim != 4:
        raise ValueError(f"Expected image batch with shape (B,C,H,W) or (B,H,W,C), got {tuple(images.shape)}")

    if images.shape[1] != 3 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)

    if images.numel() > 0 and torch.max(images) > 1.0:
        images = images / 255.0
    if images.shape[-2:] != (96, 96):
        images = F.interpolate(images, size=(96, 96), mode="bilinear", align_corners=False)
    images = random_shift_batch(images, max_shift=random_shift_px)
    images = image_normalize(images)
    return images


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    valid = (~is_pad).unsqueeze(-1)
    abs_error = torch.abs(pred - target)
    valid_error = abs_error * valid.float()
    denom = valid.float().sum().clamp_min(1.0) * pred.size(-1)
    return valid_error.sum() / denom


def split_episode_indices(episode_index: np.ndarray, val_ratio: float, seed: int):
    unique_eps = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)

    n_val_eps = max(1, int(len(unique_eps) * val_ratio)) if val_ratio > 0 else 0
    val_eps = set(unique_eps[:n_val_eps].tolist())

    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]

    if train_idx.size == 0:
        raise ValueError("Empty ACT train split")
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def get_model_state_dict_for_saving(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a checkpoint-friendly state_dict regardless of torch.compile wrapping."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print(f"Loading dataset {config.dataset_id}...")
    dataset = LeRobotDataset(config.dataset_id)
    if hasattr(dataset, "hf_dataset") and hasattr(dataset.hf_dataset, "keys") and config.split in dataset.hf_dataset.keys():
        dataset.hf_dataset = dataset.hf_dataset[config.split]
    elif hasattr(dataset, "keys") and config.split in dataset.keys():
        dataset = dataset[config.split]

    episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
    actions = np.array(dataset.hf_dataset["action"], dtype=np.float32)
    n_frames = int(episode_index.shape[0])

    gt_path = config.gt_state_zarr
    if not config.use_vision and gt_path is None:
        candidate = Path(REPO_ROOT) / "state-only" / "pusht 2" / "pusht_cchi_v7_replay.zarr"
        if candidate.is_dir():
            gt_path = str(candidate)
            print(f"Using default pymunk state zarr: {gt_path}")

    if not config.use_vision:
        if gt_path is None:
            states = np.array(dataset.hf_dataset["observation.state"], dtype=np.float32)
            if states.shape[1] != 5:
                raise ValueError(
                    "State-only training needs 5D pymunk state [agent_x, agent_y, block_x, block_y, angle]. "
                    "Pass --gt_state_zarr pointing to a replay .zarr with data/state (N, 5), or use a dataset "
                    "whose observation.state has shape (5,)."
                )
        else:
            import zarr

            zroot = zarr.open(gt_path, mode="r")
            states = np.asarray(zroot["data"]["state"], dtype=np.float32)
            if states.shape[0] != n_frames:
                raise ValueError(
                    f"Zarr state rows {states.shape[0]} != dataset frames {n_frames}; alignment mismatch."
                )
            if states.shape[1] != 5:
                raise ValueError(f"Expected zarr data/state shape (N, 5), got {states.shape}")
    else:
        states = np.array(dataset.hf_dataset["observation.state"], dtype=np.float32)

    state_dim = int(states.shape[1])

    action_chunks, action_is_pad = build_action_chunks_by_episode(actions, episode_index, config.horizon)
    train_idx, val_idx = split_episode_indices(episode_index, config.val_ratio, config.seed)

    state_mean = states[train_idx].mean(axis=0).astype(np.float32)
    state_std = (states[train_idx].std(axis=0) + 1e-6).astype(np.float32)
    action_mean = actions[train_idx].mean(axis=0).astype(np.float32)
    action_std = (actions[train_idx].std(axis=0) + 1e-6).astype(np.float32)

    if config.use_vision:
        train_dataset = ACTStepDataset(dataset, train_idx, action_chunks, action_is_pad)
        val_dataset = ACTStepDataset(dataset, val_idx, action_chunks, action_is_pad)
    else:
        train_dataset = ACTStepDatasetStateOnly(train_idx, states, action_chunks, action_is_pad)
        val_dataset = ACTStepDatasetStateOnly(val_idx, states, action_chunks, action_is_pad)

    # persistent_workers=True keeps workers alive across epochs so val doesn't pay a big
    # "first batch" delay every time (otherwise new worker processes + first collation block
    # before the Val tqdm moves).
    _dl_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **_dl_kw)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **_dl_kw)

    model = ACTPolicy(
        state_dim=state_dim,
        action_dim=2,
        horizon=config.horizon,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        use_vision=config.use_vision,
    ).to(device)

    if config.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Enabled torch.compile()")
        except Exception as exc:
            print(f"torch.compile() unavailable; continuing without it: {exc}")

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
            tags=["act", "train"],
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)
    action_mean_t = torch.tensor(action_mean, dtype=torch.float32, device=device)
    action_std_t = torch.tensor(action_std, dtype=torch.float32, device=device)

    best_val = float("inf")

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_index = epoch - 1
            kl_beta_t = config.kl_beta * min(1.0, epoch / max(1, config.kl_warmup_epochs))
            model.train()
            train_loss_sum = 0.0
            train_recon_sum = 0.0
            train_kl_sum = 0.0
            train_batches = 0

            for batch in tqdm(train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False):
                states_b = batch["state"].to(device)
                action_chunk = batch["action_chunk"].to(device)
                action_is_pad_b = batch["action_is_pad"].to(device)

                if config.use_vision:
                    images = preprocess_image_batch(
                        batch["image"].to(device),
                        image_normalize,
                        random_shift_px=config.image_shift_px,
                    )
                else:
                    images = None

                states_norm = (states_b - state_mean_t) / state_std_t
                target_actions = (action_chunk - action_mean_t.view(1, 1, -1)) / action_std_t.view(1, 1, -1)

                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
                with amp_ctx:
                    pred_actions, mu, logvar = model(images, states_norm, target_actions)
                    recon_loss = masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
                    kl_loss = ACTPolicy.kl_divergence(mu, logvar)
                    if config.kl_free_nats > 0:
                        kl_objective = torch.clamp(kl_loss, min=float(config.kl_free_nats))
                    else:
                        kl_objective = kl_loss
                    loss = recon_loss + kl_beta_t * kl_objective

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                for batch in tqdm(val_loader, desc=f"Val {epoch}/{config.epochs}", leave=False):
                    states_b = batch["state"].to(device)
                    action_chunk = batch["action_chunk"].to(device)
                    action_is_pad_b = batch["action_is_pad"].to(device)

                    if config.use_vision:
                        images = preprocess_image_batch(
                            batch["image"].to(device),
                            image_normalize,
                            random_shift_px=0,
                        )
                    else:
                        images = None

                    states_norm = (states_b - state_mean_t) / state_std_t
                    target_actions = (action_chunk - action_mean_t.view(1, 1, -1)) / action_std_t.view(1, 1, -1)

                    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
                    with amp_ctx:
                        pred_actions, mu, logvar = model(images, states_norm, target_actions)
                        recon_loss = masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
                        kl_loss = ACTPolicy.kl_divergence(mu, logvar)
                        if config.kl_free_nats > 0:
                            kl_objective = torch.clamp(kl_loss, min=float(config.kl_free_nats))
                        else:
                            kl_objective = kl_loss
                        loss = recon_loss + kl_beta_t * kl_objective

                    val_loss_sum += float(loss.item())
                    val_recon_sum += float(recon_loss.item())
                    val_kl_sum += float(kl_loss.item())
                    val_batches += 1

            val_loss = val_loss_sum / max(1, val_batches)
            val_recon = val_recon_sum / max(1, val_batches)
            val_kl = val_kl_sum / max(1, val_batches)

            model_state_dict = get_model_state_dict_for_saving(model)
            checkpoint = {
                "model_state_dict": model_state_dict,
                "state_mean": state_mean.astype(np.float32),
                "state_std": state_std.astype(np.float32),
                "action_mean": action_mean.astype(np.float32),
                "action_std": action_std.astype(np.float32),
                "config": asdict(config),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_loss": val_loss,
                "val_recon": val_recon,
                "val_kl": val_kl,
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
                f"beta={kl_beta_t:.4f} | "
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
                        "train/kl_beta": float(kl_beta_t),
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
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_std=action_std.astype(np.float32),
    )

    print(f"Saved ACT artifacts to: {config.output_dir}")
    print(f"Best validation loss: {best_val:.6f}")


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train ACT on lerobot/pusht")
    parser.add_argument("--dataset_id", type=str, default=defaults.dataset_id)
    parser.add_argument("--split", type=str, default=defaults.split)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--val_ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--horizon", type=int, default=defaults.horizon)
    parser.add_argument("--hidden_dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--latent_dim", type=int, default=defaults.latent_dim)
    parser.add_argument("--nhead", type=int, default=defaults.nhead)
    parser.add_argument("--num_encoder_layers", type=int, default=defaults.num_encoder_layers)
    parser.add_argument("--num_decoder_layers", type=int, default=defaults.num_decoder_layers)
    parser.add_argument("--kl_beta", type=float, default=defaults.kl_beta)
    parser.add_argument("--kl_warmup_epochs", type=int, default=defaults.kl_warmup_epochs)
    parser.add_argument("--kl_free_nats", type=float, default=defaults.kl_free_nats)
    parser.add_argument("--image_shift_px", type=int, default=defaults.image_shift_px)
    parser.add_argument("--ensemble_decay", type=float, default=defaults.ensemble_decay)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=defaults.amp)
    parser.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=defaults.compile_model)
    parser.add_argument(
        "--no_vision",
        action="store_true",
        help="Train on pymunk ground-truth state only (no ResNet). Use --gt_state_zarr or 5D observation.state.",
    )
    parser.add_argument(
        "--gt_state_zarr",
        type=str,
        default=None,
        help="Replay zarr with data/state (N,5) aligned with dataset frames (e.g. state-only/pusht 2/...zarr).",
    )
    parser.add_argument("--wandb", action="store_true", default=defaults.wandb)
    parser.add_argument("--wandb_project", type=str, default=defaults.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=defaults.wandb_entity)
    args = parser.parse_args()
    ns = vars(args)
    ns["use_vision"] = not ns.pop("no_vision")
    return TrainConfig(**ns)


if __name__ == "__main__":
    train(parse_args())
