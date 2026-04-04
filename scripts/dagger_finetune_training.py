#!/usr/bin/env python3
"""Fine-tune a pretrained vision BC model on collected DAgger NPZ trajectories."""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class FinetuneConfig:
    model_path: str = "models/pretrain/best.pt"
    data_dir: str = "data/dagger"
    original_dataset_id: str = "lerobot/pusht"
    original_split: str = "train"
    output_dir: str = "models/dagger_finetuned"
    include_autonomous: bool = False
    include_failed: bool = False
    only_human_steps: bool = False
    mix_dagger_ratio: float = 0.2
    mix_original_ratio: float = 0.8
    val_ratio: float = 0.1
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    seed: int = 42


class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim: int = 2, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.action_head = nn.Sequential(
            nn.Linear(512 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_features = self.vision_backbone(image).view(image.size(0), -1)
        state_features = self.state_encoder(state)
        return self.action_head(torch.cat([image_features, state_features], dim=1))


class DaggerNPZDataset(Dataset):
    def __init__(
        self,
        state_list: List[np.ndarray],
        image_list: List[np.ndarray],
        action_list: List[np.ndarray],
        is_human_list: List[np.ndarray],
        only_human_steps: bool,
    ):
        states = np.concatenate(state_list, axis=0).astype(np.float32)
        images = np.concatenate(image_list, axis=0).astype(np.float32)
        actions = np.concatenate(action_list, axis=0).astype(np.float32)
        is_human = np.concatenate(is_human_list, axis=0).astype(bool)

        if states.shape[0] != images.shape[0] or states.shape[0] != actions.shape[0]:
            raise ValueError("Mismatched state/image/action lengths after concatenation")

        if only_human_steps:
            keep = is_human
            if not np.any(keep):
                raise ValueError("No human intervention steps found, but only_human_steps=True")
            states = states[keep]
            images = images[keep]
            actions = actions[keep]

        self.states = states
        self.images = images
        self.actions = actions

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H,W,3), got {image.shape}")
        image = np.transpose(image, (2, 0, 1))
        image_t = torch.from_numpy(image).float()
        if image_t.numel() > 0 and torch.max(image_t) > 1.0:
            image_t = image_t / 255.0
        if image_t.shape[-2:] != (96, 96):
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(96, 96),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return {
            "state": torch.from_numpy(self.states[idx]).float(),
            "image": image_t,
            "action": torch.from_numpy(self.actions[idx]).float(),
        }


class LeRobotAdapterDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        image = sample["observation.image"].float()
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W] or [H,W,C], got shape {tuple(image.shape)}")
        if image.shape[0] != 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.numel() > 0 and torch.max(image) > 1.0:
            image = image / 255.0
        if image.shape[-2:] != (96, 96):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(96, 96),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return {
            "state": sample["observation.state"].float(),
            "image": image,
            "action": sample["action"].float(),
        }


def preprocess_image_batch(images: torch.Tensor) -> torch.Tensor:
    images = images.to(dtype=torch.float32)
    if images.numel() > 0 and torch.max(images) > 1.0:
        images = images / 255.0
    if images.shape[-2:] != (96, 96):
        images = F.interpolate(images, size=(96, 96), mode="bilinear", align_corners=False)
    return images


def _collect_episode_files(config: FinetuneConfig) -> List[str]:
    base = Path(config.data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")

    folders = [base / "human_intervention"]
    if config.include_autonomous:
        folders.append(base / "rejection_sample")
    if config.include_failed:
        folders.append(base / "failed_autonomous")

    files: List[str] = []
    for folder in folders:
        if folder.exists():
            files.extend(sorted(str(p) for p in folder.glob("*.npz") if not p.name.endswith("_images.npz")))

    if not files:
        raise ValueError(
            "No episode NPZ files found. Make sure you collected data and set include_* flags correctly."
        )
    return files


def _load_episode_pair(ep_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_file = ep_file.replace(".npz", "_images.npz")
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Missing image NPZ for episode: {image_file}")

    ep = np.load(ep_file)
    img = np.load(image_file)

    required = ["observation.state", "action", "is_human_intervention"]
    missing = [key for key in required if key not in ep]
    if missing:
        raise KeyError(f"Episode {ep_file} missing keys: {missing}")
    if "images" not in img:
        raise KeyError(f"Image NPZ missing `images` key: {image_file}")

    states = np.asarray(ep["observation.state"], dtype=np.float32)
    actions = np.asarray(ep["action"], dtype=np.float32)
    is_human = np.asarray(ep["is_human_intervention"], dtype=bool)
    images = np.asarray(img["images"], dtype=np.uint8)

    if states.ndim != 2 or states.shape[1] != 2:
        raise ValueError(f"Expected states (N,2), got {states.shape} in {ep_file}")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"Expected actions (N,2), got {actions.shape} in {ep_file}")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected images (N,H,W,3), got {images.shape} in {image_file}")

    n = states.shape[0]
    if actions.shape[0] != n or images.shape[0] != n or is_human.shape[0] != n:
        raise ValueError(
            f"Length mismatch in {ep_file}: states={n}, actions={actions.shape[0]}, "
            f"images={images.shape[0]}, is_human={is_human.shape[0]}"
        )

    return states, images, actions, is_human


def _split_episode_files(episode_files: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not episode_files:
        raise ValueError("No episode files to split")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episode_files))
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * val_ratio)) if val_ratio > 0 else 0
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    if train_indices.size == 0:
        raise ValueError("Empty train split")
    if val_indices.size == 0:
        val_indices = train_indices.copy()

    train_files = [episode_files[i] for i in train_indices.tolist()]
    val_files = [episode_files[i] for i in val_indices.tolist()]
    return train_files, val_files


def _build_dataset_from_episode_files(episode_files: List[str], only_human_steps: bool) -> DaggerNPZDataset:
    state_list: List[np.ndarray] = []
    image_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    is_human_list: List[np.ndarray] = []

    for ep_file in episode_files:
        states, images, actions, is_human = _load_episode_pair(ep_file)
        state_list.append(states)
        image_list.append(images)
        action_list.append(actions)
        is_human_list.append(is_human)

    return DaggerNPZDataset(
        state_list=state_list,
        image_list=image_list,
        action_list=action_list,
        is_human_list=is_human_list,
        only_human_steps=only_human_steps,
    )


def _load_lerobot_dataset(dataset_id: str, split: str) -> LeRobotDataset:
    dataset = LeRobotDataset(dataset_id)
    if hasattr(dataset, "hf_dataset") and hasattr(dataset.hf_dataset, "keys"):
        if split in dataset.hf_dataset.keys():
            dataset.hf_dataset = dataset.hf_dataset[split]
    elif hasattr(dataset, "keys") and split in dataset.keys():
        dataset = dataset[split]
    return dataset


def _split_lerobot_episode_indices(dataset: LeRobotDataset, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(dataset, "hf_dataset"):
        raise ValueError("LeRobotDataset missing `hf_dataset`, cannot perform episode-based split")

    episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
    unique_eps = np.unique(episode_index)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)

    n_val_eps = max(1, int(len(unique_eps) * val_ratio)) if val_ratio > 0 else 0
    val_eps = set(unique_eps[:n_val_eps].tolist())

    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]

    if train_idx.size == 0:
        raise ValueError("Empty original-data train split")
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def _build_mixed_loader(
    dagger_dataset: Dataset,
    original_dataset: Dataset,
    dagger_ratio: float,
    original_ratio: float,
    batch_size: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    if dagger_ratio < 0 or original_ratio < 0:
        raise ValueError("Mix ratios must be non-negative")
    if dagger_ratio + original_ratio <= 0:
        raise ValueError("At least one mix ratio must be positive")

    datasets: List[Dataset] = []
    weights: List[float] = []

    if len(dagger_dataset) > 0 and dagger_ratio > 0:
        datasets.append(dagger_dataset)
        weights.extend([dagger_ratio / len(dagger_dataset)] * len(dagger_dataset))

    if len(original_dataset) > 0 and original_ratio > 0:
        datasets.append(original_dataset)
        weights.extend([original_ratio / len(original_dataset)] * len(original_dataset))

    if not datasets:
        raise ValueError("No samples available for mixed loader")

    mixed_dataset = ConcatDataset(datasets)
    num_samples = len(mixed_dataset)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.float64),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )


def train(config: FinetuneConfig) -> None:
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {config.model_path}")

    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.manual_seed(config.seed)

    ratio_sum = config.mix_dagger_ratio + config.mix_original_ratio
    if ratio_sum <= 0:
        raise ValueError("mix_dagger_ratio + mix_original_ratio must be > 0")
    dagger_ratio = config.mix_dagger_ratio / ratio_sum
    original_ratio = config.mix_original_ratio / ratio_sum

    episode_files = _collect_episode_files(config)
    print(f"Found {len(episode_files)} episode files for fine-tuning")

    train_files, val_files = _split_episode_files(episode_files, config.val_ratio, config.seed)
    dagger_train_dataset = _build_dataset_from_episode_files(train_files, config.only_human_steps)
    dagger_val_dataset = _build_dataset_from_episode_files(val_files, config.only_human_steps)

    original_dataset = _load_lerobot_dataset(config.original_dataset_id, config.original_split)
    original_train_idx, original_val_idx = _split_lerobot_episode_indices(original_dataset, config.val_ratio, config.seed)
    original_train_dataset = LeRobotAdapterDataset(Subset(original_dataset, original_train_idx.tolist()))
    original_val_dataset = LeRobotAdapterDataset(Subset(original_dataset, original_val_idx.tolist()))

    train_loader = _build_mixed_loader(
        dagger_dataset=dagger_train_dataset,
        original_dataset=original_train_dataset,
        dagger_ratio=dagger_ratio,
        original_ratio=original_ratio,
        batch_size=config.batch_size,
        seed=config.seed,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = _build_mixed_loader(
        dagger_dataset=dagger_val_dataset,
        original_dataset=original_val_dataset,
        dagger_ratio=dagger_ratio,
        original_ratio=original_ratio,
        batch_size=config.batch_size,
        seed=config.seed + 1,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Mix ratios (effective): DAgger={dagger_ratio:.2f} | Original={original_ratio:.2f} | "
        f"LR={config.learning_rate}"
    )
    print(
        f"Train samples: DAgger={len(dagger_train_dataset)} Original={len(original_train_dataset)} | "
        f"Val samples: DAgger={len(dagger_val_dataset)} Original={len(original_val_dataset)}"
    )

    checkpoint = torch.load(config.model_path, map_location=device, weights_only=False)
    hidden_dim = int(checkpoint.get("config", {}).get("hidden_dim", 256))

    model = BehavioralCloningPolicy(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    state_mean = torch.tensor(checkpoint["state_mean"], dtype=torch.float32, device=device)
    state_std = torch.tensor(checkpoint["state_std"], dtype=torch.float32, device=device)
    action_mean = torch.tensor(checkpoint["action_mean"], dtype=torch.float32, device=device)
    action_std = torch.tensor(checkpoint["action_std"], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            images = preprocess_image_batch(images)
            states = batch["state"].to(device)
            actions = batch["action"].to(device)

            states_norm = (states - state_mean) / state_std
            actions_norm = (actions - action_mean) / action_std

            preds = model(images, states_norm)
            loss = loss_fn(preds, actions_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1

        train_loss = train_loss_sum / max(1, train_batches)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                images = preprocess_image_batch(images)
                states = batch["state"].to(device)
                actions = batch["action"].to(device)

                states_norm = (states - state_mean) / state_std
                actions_norm = (actions - action_mean) / action_std

                preds = model(images, states_norm)
                loss = loss_fn(preds, actions_norm)
                val_loss_sum += float(loss.item())
                val_batches += 1

        val_loss = val_loss_sum / max(1, val_batches)

        save_payload = {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean.detach().cpu().numpy().astype(np.float32),
            "state_std": state_std.detach().cpu().numpy().astype(np.float32),
            "action_mean": action_mean.detach().cpu().numpy().astype(np.float32),
            "action_std": action_std.detach().cpu().numpy().astype(np.float32),
            "config": {**asdict(config), "hidden_dim": hidden_dim, "source_checkpoint": config.model_path},
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": datetime.utcnow().isoformat(),
        }

        latest_path = os.path.join(config.output_dir, "latest.pt")
        torch.save(save_payload, latest_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(config.output_dir, "best.pt")
            torch.save(save_payload, best_path)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, indent=2)

    np.savez(
        os.path.join(config.output_dir, "normalization_stats.npz"),
        state_mean=state_mean.detach().cpu().numpy().astype(np.float32),
        state_std=state_std.detach().cpu().numpy().astype(np.float32),
        action_mean=action_mean.detach().cpu().numpy().astype(np.float32),
        action_std=action_std.detach().cpu().numpy().astype(np.float32),
    )

    print("=" * 72)
    print("DAgger Fine-tuning complete")
    print(f"Data dir: {config.data_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Best val loss: {best_val:.6f}")
    print("=" * 72)


def parse_args() -> FinetuneConfig:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained BC model on DAgger-collected NPZ data")
    parser.add_argument("--model_path", type=str, default="models/pretrain/best.pt")
    parser.add_argument("--data_dir", type=str, default="data/dagger")
    parser.add_argument("--original_dataset_id", type=str, default="lerobot/pusht")
    parser.add_argument("--original_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="models/dagger_finetuned")
    parser.add_argument("--include_autonomous", action="store_true", default=False)
    parser.add_argument("--include_failed", action="store_true", default=False)
    parser.add_argument("--only_human_steps", action="store_true")
    parser.add_argument("--mix_dagger_ratio", type=float, default=0.2)
    parser.add_argument("--mix_original_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return FinetuneConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        original_dataset_id=args.original_dataset_id,
        original_split=args.original_split,
        output_dir=args.output_dir,
        include_autonomous=args.include_autonomous,
        include_failed=args.include_failed,
        only_human_steps=args.only_human_steps,
        mix_dagger_ratio=args.mix_dagger_ratio,
        mix_original_ratio=args.mix_original_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    train(parse_args())
