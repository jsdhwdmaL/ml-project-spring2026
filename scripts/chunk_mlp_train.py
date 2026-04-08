"""Train chunked BC-MLP policy on lerobot/pusht with frame stacking."""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

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
from models.chunk_mlp import BehavioralCloningPolicy


@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    split: str = "train"
    output_dir: str = "models/chunk_mlp"
    seed: int = 42
    val_ratio: float = 0.1
    hidden_dim: int = 256
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    n_frames: int = 2
    frame_gap: int = 9
    horizon: int = 50
    ensemble_decay: float = 0.01


class ChunkMLPStepDataset(Dataset):
    def __init__(self, base_dataset, step_indices: np.ndarray, action_chunks: np.ndarray, action_is_pad: np.ndarray):
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
            "observation.image": sample["observation.image"].float(),
            "observation.state": sample["observation.state"].float(),
            "action_chunk": torch.from_numpy(self.action_chunks[global_idx]).float(),
            "action_is_pad": torch.from_numpy(self.action_is_pad[global_idx]).bool(),
        }


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
        raise ValueError("Empty train split")
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def get_transforms():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess_image_batch(images: torch.Tensor, transform_fn) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"Expected image shape (B,F,C,H,W), got {tuple(images.shape)}")

    batch_size, n_frames, channels, height, width = images.shape
    images = images.to(dtype=torch.float32)
    if images.numel() > 0 and torch.max(images) > 1.0:
        images = images / 255.0

    if (height, width) != (96, 96):
        images = F.interpolate(
            images.view(-1, channels, height, width),
            size=(96, 96),
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, n_frames, channels, 96, 96)

    images_3c = images.view(batch_size * n_frames, channels, 96, 96)
    images_3c = transform_fn(images_3c)
    return images_3c.view(batch_size, n_frames * channels, 96, 96)


def masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    valid = (~is_pad).unsqueeze(-1).float()
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    denom = valid.sum().clamp_min(1.0) * pred.size(-1)
    return (loss * valid).sum() / denom


def train(config: TrainConfig) -> None:
    if config.n_frames != 2:
        raise ValueError("chunk_mlp currently expects n_frames=2 to match frame-gap buffering logic")

    os.makedirs(config.output_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    fps = 10
    delta_indices = [-config.frame_gap, 0]

    print(f"Loading dataset with {config.n_frames} frames (gap: {config.frame_gap})...")
    dataset = LeRobotDataset(
        config.dataset_id,
        delta_timestamps={
            "observation.image": [i / fps for i in delta_indices],
            "observation.state": [i / fps for i in delta_indices],
        },
    )
    if hasattr(dataset, "hf_dataset") and hasattr(dataset.hf_dataset, "keys") and config.split in dataset.hf_dataset.keys():
        dataset.hf_dataset = dataset.hf_dataset[config.split]
    elif hasattr(dataset, "keys") and config.split in dataset.keys():
        dataset = dataset[config.split]

    episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
    all_states = np.array(dataset.hf_dataset["observation.state"], dtype=np.float32)
    all_actions = np.array(dataset.hf_dataset["action"], dtype=np.float32)

    action_chunks, action_is_pad = build_action_chunks_by_episode(all_actions, episode_index, config.horizon)
    train_idx, val_idx = split_episode_indices(episode_index, config.val_ratio, config.seed)

    print("Computing normalization stats...")
    state_mean = all_states[train_idx].mean(axis=0).astype(np.float32)
    state_std = (all_states[train_idx].std(axis=0) + 1e-6).astype(np.float32)
    action_mean = all_actions[train_idx].mean(axis=0).astype(np.float32)
    action_std = (all_actions[train_idx].std(axis=0) + 1e-6).astype(np.float32)

    train_dataset = ChunkMLPStepDataset(dataset, train_idx, action_chunks, action_is_pad)
    val_dataset = ChunkMLPStepDataset(dataset, val_idx, action_chunks, action_is_pad)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = BehavioralCloningPolicy(
        state_dim=2,
        action_dim=2,
        hidden_dim=config.hidden_dim,
        n_frames=config.n_frames,
        horizon=config.horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    s_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    s_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)
    a_mean_t = torch.tensor(action_mean, dtype=torch.float32, device=device)
    a_std_t = torch.tensor(action_std, dtype=torch.float32, device=device)

    transform_fn = get_transforms()
    best_val = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False):
            images = preprocess_image_batch(batch["observation.image"].to(device), transform_fn)
            states = batch["observation.state"].to(device, dtype=torch.float32)
            action_chunk = batch["action_chunk"].to(device)
            action_is_pad_b = batch["action_is_pad"].to(device)

            states_norm = ((states - s_mean_t) / s_std_t).view(states.size(0), -1)
            target_actions = (action_chunk - a_mean_t.view(1, 1, -1)) / a_std_t.view(1, 1, -1)

            pred = model(images, states_norm)
            loss = masked_smooth_l1_loss(pred, target_actions, action_is_pad_b, beta=1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1

        avg_train = train_loss_sum / max(1, train_batches)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch}/{config.epochs}", leave=False):
                images = preprocess_image_batch(batch["observation.image"].to(device), transform_fn)
                states = batch["observation.state"].to(device, dtype=torch.float32)
                action_chunk = batch["action_chunk"].to(device)
                action_is_pad_b = batch["action_is_pad"].to(device)

                states_norm = ((states - s_mean_t) / s_std_t).view(states.size(0), -1)
                target_actions = (action_chunk - a_mean_t.view(1, 1, -1)) / a_std_t.view(1, 1, -1)

                pred = model(images, states_norm)
                loss = masked_smooth_l1_loss(pred, target_actions, action_is_pad_b, beta=1.0)

                val_loss_sum += float(loss.item())
                val_batches += 1

        avg_val = val_loss_sum / max(1, val_batches)
        print(f"Epoch {epoch:03d} | Train Recon: {avg_train:.6f} | Val Recon: {avg_val:.6f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "config": asdict(config),
            "epoch": epoch,
            "train_recon": avg_train,
            "val_recon": avg_val,
        }

        torch.save(checkpoint, os.path.join(config.output_dir, "latest.pt"))

        if avg_val < best_val:
            best_val = avg_val
            torch.save(checkpoint, os.path.join(config.output_dir, "best.pt"))
            print(f"  --> New best model saved (Val Recon: {best_val:.6f})")

    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train chunked BC-MLP with frame stacking")
    parser.add_argument("--dataset_id", type=str, default="lerobot/pusht")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="models/chunk_mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_frames", type=int, default=2)
    parser.add_argument("--frame_gap", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--ensemble_decay", type=float, default=0.01)
    args = parser.parse_args()

    return TrainConfig(
        dataset_id=args.dataset_id,
        split=args.split,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_frames=args.n_frames,
        frame_gap=args.frame_gap,
        horizon=args.horizon,
        ensemble_decay=args.ensemble_decay,
    )


if __name__ == "__main__":
    train(parse_args())