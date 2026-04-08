#!/usr/bin/env python3
"""Fine-tune ACT on successful DAgger episodes saved under data/act_dagger."""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.build_chunk import build_action_chunks_by_episode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from models.act import ACTPolicy


@dataclass
class FinetuneConfig:
    model_path: str = "models/act/best.pt"
    data_dir: str = "data/act_dagger"
    original_dataset_id: str = "lerobot/pusht"
    original_split: str = "train"
    output_dir: str = "models/act_dagger"
    include_human_intervention: bool = True
    include_rejection_sample: bool = True
    include_failed_autonomous: bool = False
    include_original_data: bool = True
    mix_dagger_ratio: float = 0.5
    mix_original_ratio: float = 0.5
    success_only: bool = True
    seed: int = 42
    val_ratio: float = 0.1
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    kl_beta: float = 0.0
    kl_warmup_epochs: int = 0
    kl_ramp_epochs: int = 0
    image_shift_px: int = 4
    num_workers: int = 4
    horizon: Optional[int] = None
    hidden_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    nhead: Optional[int] = None
    num_decoder_layers: Optional[int] = None
    ensemble_decay: Optional[float] = None


class ACTNPZStepDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        states: np.ndarray,
        step_indices: np.ndarray,
        action_chunks: np.ndarray,
        action_is_pad: np.ndarray,
    ):
        self.images = images
        self.states = states
        self.step_indices = step_indices.astype(np.int64)
        self.action_chunks = action_chunks
        self.action_is_pad = action_is_pad

    def __len__(self) -> int:
        return int(self.step_indices.shape[0])

    def __getitem__(self, idx: int):
        global_idx = int(self.step_indices[idx])
        image = self.images[global_idx]
        if image.ndim == 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        image_t = torch.tensor(image, dtype=torch.float32)
        if image_t.shape[-2:] != (96, 96):
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(96, 96),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        image_t = image_t.contiguous()
        return {
            # torch.tensor(...) forces owned, resizable storage for worker collation.
            "image": image_t,
            "state": torch.tensor(self.states[global_idx], dtype=torch.float32),
            "action_chunk": torch.tensor(self.action_chunks[global_idx], dtype=torch.float32),
            "action_is_pad": torch.tensor(self.action_is_pad[global_idx], dtype=torch.bool),
        }


class ACTLeRobotStepDataset(Dataset):
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
        image = sample["observation.image"]
        state = sample["observation.state"]
        if isinstance(image, torch.Tensor):
            image = image.detach().to(dtype=torch.float32, device="cpu").contiguous().clone()
        else:
            image = torch.tensor(image, dtype=torch.float32)
        if image.shape[-2:] != (96, 96):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(96, 96),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        image = image.contiguous()
        if isinstance(state, torch.Tensor):
            state = state.detach().to(dtype=torch.float32, device="cpu").contiguous().clone()
        else:
            state = torch.tensor(state, dtype=torch.float32)
        return {
            "image": image,
            "state": state,
            "action_chunk": torch.tensor(self.action_chunks[global_idx], dtype=torch.float32),
            "action_is_pad": torch.tensor(self.action_is_pad[global_idx], dtype=torch.bool),
        }


def random_shift_batch(images: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return images

    batch_size, channels, height, width = images.shape
    pad = int(max_shift)
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    shift_y = torch.randint(0, 2 * pad + 1, (batch_size,), device=images.device)
    shift_x = torch.randint(0, 2 * pad + 1, (batch_size,), device=images.device)

    out = torch.empty_like(images)
    for i in range(batch_size):
        y0 = int(shift_y[i].item())
        x0 = int(shift_x[i].item())
        out[i] = padded[i, :, y0 : y0 + height, x0 : x0 + width]
    return out


def preprocess_image_batch(images: torch.Tensor, image_normalize, random_shift_px: int = 0) -> torch.Tensor:
    images = images.to(dtype=torch.float32)
    if images.ndim != 4:
        raise ValueError(f"Expected image batch with shape (B,C,H,W) or (B,H,W,C), got {tuple(images.shape)}")

    if images.shape[1] != 3 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2).contiguous()

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
        raise ValueError("Empty fine-tune train split")
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def get_kl_beta(epoch_index: int, config: FinetuneConfig) -> float:
    if epoch_index < config.kl_warmup_epochs:
        return 0.0
    if config.kl_ramp_epochs <= 0:
        return float(config.kl_beta)

    ramp_step = epoch_index - config.kl_warmup_epochs + 1
    if ramp_step <= config.kl_ramp_epochs:
        progress = ramp_step / float(config.kl_ramp_epochs)
        return float(config.kl_beta) * progress
    return float(config.kl_beta)


def _episode_success(data: np.lib.npyio.NpzFile) -> bool:
    if "success" in data.files:
        return bool(np.asarray(data["success"]).reshape(-1)[0])
    if "next.success" in data.files:
        return bool(np.asarray(data["next.success"]).any())
    raise KeyError("Episode file has no success key")


def _collect_episode_files(config: FinetuneConfig):
    root = Path(config.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {root}")

    folders = []
    if config.include_human_intervention:
        folders.append(root / "human_intervention")
    if config.include_rejection_sample:
        folders.append(root / "rejection_sample")
    if config.include_failed_autonomous:
        folders.append(root / "failed_autonomous")

    files = []
    for folder in folders:
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.npz")):
            if file_path.name.endswith("_images.npz"):
                continue
            files.append(file_path)

    if not files:
        raise ValueError("No candidate episode files found in selected DAgger folders")
    return files


def _load_successful_data(config: FinetuneConfig):
    files = _collect_episode_files(config)

    all_states = []
    all_actions = []
    all_images = []
    all_episode_index = []

    selected_eps = 0
    selected_steps = 0

    for file_path in files:
        with np.load(file_path, allow_pickle=False) as data:
            success = _episode_success(data)
            if config.success_only and not success:
                continue

            image_path = Path(str(file_path).replace(".npz", "_images.npz"))
            if not image_path.exists():
                continue

            with np.load(image_path, allow_pickle=False) as image_data:
                if "images" not in image_data.files:
                    continue
                images = np.array(image_data["images"], dtype=np.uint8)

            states = np.array(data["observation.state"], dtype=np.float32)
            actions = np.array(data["action"], dtype=np.float32)

            if states.ndim != 2 or states.shape[1] != 2:
                continue
            if actions.ndim != 2 or actions.shape[1] != 2:
                continue
            if images.ndim != 4 or images.shape[-1] != 3:
                continue

            steps = states.shape[0]
            if actions.shape[0] != steps or images.shape[0] != steps:
                continue
            if steps == 0:
                continue

            ep_id = selected_eps
            all_states.append(states)
            all_actions.append(actions)
            all_images.append(images)
            all_episode_index.append(np.full((steps,), ep_id, dtype=np.int64))

            selected_eps += 1
            selected_steps += steps

    if selected_eps == 0:
        raise ValueError("No successful episodes found in selected DAgger data")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    images = np.concatenate(all_images, axis=0)
    episode_index = np.concatenate(all_episode_index, axis=0)

    stats = {
        "candidate_episodes": len(files),
        "selected_episodes": selected_eps,
        "selected_steps": selected_steps,
    }
    return images, states, actions, episode_index, stats


def _resolve_arch_from_checkpoint(config: FinetuneConfig, ckpt_config: dict):
    defaults = {
        "horizon": 50,
        "hidden_dim": 512,
        "latent_dim": 32,
        "nhead": 8,
        "num_decoder_layers": 2,
    }

    for key, default in defaults.items():
        user_value = getattr(config, key)
        ckpt_value = ckpt_config.get(key, default)
        if user_value is None:
            setattr(config, key, int(ckpt_value))
        else:
            if int(user_value) != int(ckpt_value):
                raise ValueError(
                    f"{key}={user_value} does not match checkpoint {key}={ckpt_value}. "
                    "Fine-tuning requires the same ACT architecture."
                )
            setattr(config, key, int(user_value))

    if config.ensemble_decay is None:
        config.ensemble_decay = float(ckpt_config.get("ensemble_decay", 0.01))


def _load_original_lerobot_data(config: FinetuneConfig):
    dataset = LeRobotDataset(config.original_dataset_id)
    if hasattr(dataset, "hf_dataset") and hasattr(dataset.hf_dataset, "keys") and config.original_split in dataset.hf_dataset.keys():
        dataset.hf_dataset = dataset.hf_dataset[config.original_split]
    elif hasattr(dataset, "keys") and config.original_split in dataset.keys():
        dataset = dataset[config.original_split]

    episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
    states = np.array(dataset.hf_dataset["observation.state"], dtype=np.float32)
    actions = np.array(dataset.hf_dataset["action"], dtype=np.float32)
    action_chunks, action_is_pad = build_action_chunks_by_episode(actions, episode_index, config.horizon)

    train_idx, val_idx = split_episode_indices(episode_index, config.val_ratio, config.seed)
    train_dataset = ACTLeRobotStepDataset(dataset, train_idx, action_chunks, action_is_pad)
    val_dataset = ACTLeRobotStepDataset(dataset, val_idx, action_chunks, action_is_pad)

    return {
        "states": states,
        "actions": actions,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "total_steps": int(states.shape[0]),
    }


def _build_mixed_train_loader(
    dagger_train_dataset: Dataset,
    original_train_dataset: Dataset,
    config: FinetuneConfig,
):
    if len(dagger_train_dataset) == 0 or len(original_train_dataset) == 0:
        raise ValueError(
            "Cannot build mixed loader with empty train dataset: "
            f"dagger={len(dagger_train_dataset)}, original={len(original_train_dataset)}"
        )

    mixed_dataset = ConcatDataset([dagger_train_dataset, original_train_dataset])

    ratio_sum = float(config.mix_dagger_ratio) + float(config.mix_original_ratio)
    dagger_prob = float(config.mix_dagger_ratio) / ratio_sum
    original_prob = float(config.mix_original_ratio) / ratio_sum

    dagger_weight = dagger_prob / len(dagger_train_dataset)
    original_weight = original_prob / len(original_train_dataset)

    weights = [dagger_weight] * len(dagger_train_dataset)
    weights.extend([original_weight] * len(original_train_dataset))

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(mixed_dataset),
        replacement=True,
    )

    return DataLoader(
        mixed_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def train(config: FinetuneConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    base_checkpoint = torch.load(config.model_path, map_location=device, weights_only=False)
    ckpt_config = base_checkpoint.get("config", {})
    _resolve_arch_from_checkpoint(config, ckpt_config)

    images, states, actions, episode_index, data_stats = _load_successful_data(config)
    print(
        "Loaded successful DAgger data | "
        f"candidates={data_stats['candidate_episodes']} "
        f"selected_episodes={data_stats['selected_episodes']} "
        f"selected_steps={data_stats['selected_steps']}"
    )

    action_chunks, action_is_pad = build_action_chunks_by_episode(actions, episode_index, config.horizon)
    train_idx, val_idx = split_episode_indices(episode_index, config.val_ratio, config.seed)

    train_dataset = ACTNPZStepDataset(images, states, train_idx, action_chunks, action_is_pad)
    val_dataset = ACTNPZStepDataset(images, states, val_idx, action_chunks, action_is_pad)

    if config.include_original_data:
        if not np.isfinite(config.mix_dagger_ratio) or not np.isfinite(config.mix_original_ratio):
            raise ValueError("mix ratios must be finite when include_original_data is enabled")
        if config.mix_dagger_ratio <= 0 or config.mix_original_ratio <= 0:
            raise ValueError(
                "mix ratios must both be > 0 when include_original_data is enabled; "
                f"got dagger={config.mix_dagger_ratio}, original={config.mix_original_ratio}"
            )

    original_data = None
    if config.include_original_data:
        original_data = _load_original_lerobot_data(config)
        print(
            "Loaded original LeRobot data | "
            f"total_steps={original_data['total_steps']} "
            f"train_steps={len(original_data['train_idx'])} "
            f"val_steps={len(original_data['val_idx'])}"
        )

    if original_data is not None:
        train_states = np.concatenate([states[train_idx], original_data["states"][original_data["train_idx"]]], axis=0)
        train_actions = np.concatenate([actions[train_idx], original_data["actions"][original_data["train_idx"]]], axis=0)
    else:
        train_states = states[train_idx]
        train_actions = actions[train_idx]

    state_mean = train_states.mean(axis=0).astype(np.float32)
    state_std = (train_states.std(axis=0) + 1e-6).astype(np.float32)
    action_mean = train_actions.mean(axis=0).astype(np.float32)
    action_std = (train_actions.std(axis=0) + 1e-6).astype(np.float32)

    if original_data is not None:
        train_loader = _build_mixed_train_loader(train_dataset, original_data["train_dataset"], config)
        val_concat = ConcatDataset([val_dataset, original_data["val_dataset"]])
        val_loader = DataLoader(
            val_concat,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    model = ACTPolicy(
        state_dim=2,
        action_dim=2,
        horizon=config.horizon,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        num_decoder_layers=config.num_decoder_layers,
    ).to(device)
    model.load_state_dict(base_checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)
    action_mean_t = torch.tensor(action_mean, dtype=torch.float32, device=device)
    action_std_t = torch.tensor(action_std, dtype=torch.float32, device=device)

    best_val = float("inf")

    for epoch in range(1, config.epochs + 1):
        epoch_kl_beta = get_kl_beta(epoch - 1, config)

        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False):
            images_b = preprocess_image_batch(batch["image"].to(device), image_normalize, random_shift_px=config.image_shift_px)
            states_b = batch["state"].to(device)
            action_chunk = batch["action_chunk"].to(device)
            action_is_pad_b = batch["action_is_pad"].to(device)

            states_norm = (states_b - state_mean_t) / state_std_t
            target_actions = (action_chunk - action_mean_t.view(1, 1, -1)) / action_std_t.view(1, 1, -1)

            pred_actions, mu, logvar = model(images_b, states_norm, target_actions)
            recon_loss = masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
            kl_loss = ACTPolicy.kl_divergence(mu, logvar)
            loss = recon_loss + epoch_kl_beta * kl_loss

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
            for batch in tqdm(val_loader, desc=f"Val {epoch}/{config.epochs}", leave=False):
                images_b = preprocess_image_batch(batch["image"].to(device), image_normalize, random_shift_px=0)
                states_b = batch["state"].to(device)
                action_chunk = batch["action_chunk"].to(device)
                action_is_pad_b = batch["action_is_pad"].to(device)

                states_norm = (states_b - state_mean_t) / state_std_t
                target_actions = (action_chunk - action_mean_t.view(1, 1, -1)) / action_std_t.view(1, 1, -1)

                pred_actions, mu, logvar = model(images_b, states_norm, target_actions)
                recon_loss = masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
                kl_loss = ACTPolicy.kl_divergence(mu, logvar)
                loss = recon_loss + epoch_kl_beta * kl_loss

                val_loss_sum += float(loss.item())
                val_recon_sum += float(recon_loss.item())
                val_kl_sum += float(kl_loss.item())
                val_batches += 1

        val_loss = val_loss_sum / max(1, val_batches)
        val_recon = val_recon_sum / max(1, val_batches)
        val_kl = val_kl_sum / max(1, val_batches)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "config": asdict(config),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_kl": train_kl,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kl": val_kl,
            "source_model_path": config.model_path,
            "data_stats": data_stats,
            "mix_stats": {
                "include_original_data": bool(config.include_original_data),
                "mix_dagger_ratio": float(config.mix_dagger_ratio),
                "mix_original_ratio": float(config.mix_original_ratio),
                "dagger_train_steps": int(len(train_idx)),
                "dagger_val_steps": int(len(val_idx)),
                "original_train_steps": int(len(original_data["train_idx"])) if original_data is not None else 0,
                "original_val_steps": int(len(original_data["val_idx"])) if original_data is not None else 0,
            },
        }

        latest_path = os.path.join(config.output_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(config.output_dir, "best.pt")
            torch.save(checkpoint, best_path)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"kl_beta={epoch_kl_beta:.6f} | "
            f"train={train_loss:.6f} (recon={train_recon:.6f}, kl={train_kl:.6f}) | "
            f"val={val_loss:.6f} (recon={val_recon:.6f}, kl={val_kl:.6f})"
        )

    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    np.savez(
        os.path.join(config.output_dir, "normalization_stats.npz"),
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    print(f"Saved ACT fine-tune artifacts to: {config.output_dir}")
    print(f"Best validation loss: {best_val:.6f}")


def parse_args() -> FinetuneConfig:
    parser = argparse.ArgumentParser(description="Fine-tune ACT on successful DAgger episodes")
    parser.add_argument("--model_path", type=str, default="models/act/best.pt")
    parser.add_argument("--data_dir", type=str, default="data/act_dagger")
    parser.add_argument("--original_dataset_id", type=str, default="lerobot/pusht")
    parser.add_argument("--original_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="models/act_dagger")
    parser.add_argument("--include_human_intervention", action="store_true", default=True)
    parser.add_argument("--no_include_human_intervention", dest="include_human_intervention", action="store_false")
    parser.add_argument("--include_rejection_sample", action="store_true", default=True)
    parser.add_argument("--no_include_rejection_sample", dest="include_rejection_sample", action="store_false")
    parser.add_argument("--include_failed_autonomous", action="store_true", default=False)
    parser.add_argument("--include_original_data", action="store_true", default=True)
    parser.add_argument("--no_include_original_data", dest="include_original_data", action="store_false")
    parser.add_argument("--mix_dagger_ratio", type=float, default=0.5)
    parser.add_argument("--mix_original_ratio", type=float, default=0.5)
    parser.add_argument("--success_only", action="store_true", default=True)
    parser.add_argument("--no_success_only", dest="success_only", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--kl_beta", type=float, default=10)
    parser.add_argument("--kl_warmup_epochs", type=int, default=0)
    parser.add_argument("--kl_ramp_epochs", type=int, default=0)
    parser.add_argument("--image_shift_px", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--ensemble_decay", type=float, default=None)
    args = parser.parse_args()

    return FinetuneConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        original_dataset_id=args.original_dataset_id,
        original_split=args.original_split,
        output_dir=args.output_dir,
        include_human_intervention=args.include_human_intervention,
        include_rejection_sample=args.include_rejection_sample,
        include_failed_autonomous=args.include_failed_autonomous,
        include_original_data=args.include_original_data,
        mix_dagger_ratio=args.mix_dagger_ratio,
        mix_original_ratio=args.mix_original_ratio,
        success_only=args.success_only,
        seed=args.seed,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        kl_beta=args.kl_beta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_ramp_epochs=args.kl_ramp_epochs,
        image_shift_px=args.image_shift_px,
        num_workers=args.num_workers,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        ensemble_decay=args.ensemble_decay,
    )


if __name__ == "__main__":
    train(parse_args())
