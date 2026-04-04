"""Train a Vision-based Behavioral Cloning policy on lerobot/pusht.

Usage:
python scripts/standard_bc_training.py \
    --output_dir models/standard_bc \
    --epochs 50 \
    --batch_size 64

Small run:
python scripts/standard_bc_training.py \
    --output_dir models/standard_bc \
    --epochs 2 \
    --batch_size 32 \
    --hidden_dim 64 \
    --val_ratio 0.05
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models

# Import LeRobotDataset to safely handle the MP4 video image chunks
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    split: str = "train"
    output_dir: str = "models/standard_bc"
    seed: int = 42
    val_ratio: float = 0.1
    hidden_dim: int = 256
    epochs: int = 50
    batch_size: int = 64  # Lowered default for images! 512 images would OOM your GPU.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5


# ==========================================
# 1. Vision Architecture
# ==========================================
class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256):
        super().__init__()
        
        # 1. Vision Backbone (ResNet18)
        resnet = models.resnet18(weights=None)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512 
        
        # 2. State Encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 3. Action Head (MLP)
        self.action_head = nn.Sequential(
            nn.Linear(vision_feature_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, image, state):
        img_features = self.vision_backbone(image).view(image.size(0), -1) 
        state_features = self.state_encoder(state) 
        combined_features = torch.cat([img_features, state_features], dim=1) 
        return self.action_head(combined_features) 


# ==========================================
# 2. Training Pipeline
# ==========================================
def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    print(f"Loading dataset {config.dataset_id}...")
    dataset = LeRobotDataset(config.dataset_id)
    
    # --- HF Compatibility Fix ---
    if hasattr(dataset, "hf_dataset") and hasattr(dataset.hf_dataset, "keys"):
        if config.split in dataset.hf_dataset.keys():
            dataset.hf_dataset = dataset.hf_dataset[config.split]
    elif hasattr(dataset, "keys") and config.split in dataset.keys():
        dataset = dataset[config.split]

    # --- Your Episode Split Logic ---
    episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
    unique_eps = np.unique(episode_index)
    rng.shuffle(unique_eps)

    n_val_eps = max(1, int(len(unique_eps) * config.val_ratio)) if config.val_ratio > 0 else 0
    val_eps = set(unique_eps[:n_val_eps].tolist())

    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]

    if len(train_idx) == 0:
        raise ValueError("Empty train split after episode-based split")
    if len(val_idx) == 0:
        val_idx = train_idx.copy()

    # --- Your Normalization Logic ---
    # We only pull the low-dimensional state/action numbers into memory to compute stats
    print("Computing normalization stats...")
    all_states = np.array(dataset.hf_dataset["observation.state"])
    all_actions = np.array(dataset.hf_dataset["action"])
    
    train_states = all_states[train_idx]
    train_actions = all_actions[train_idx]

    state_mean = train_states.mean(axis=0)
    state_std = train_states.std(axis=0) + 1e-6
    action_mean = train_actions.mean(axis=0)
    action_std = train_actions.std(axis=0) + 1e-6

    # --- Dataloaders (To handle Images efficiently) ---
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Initialize Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = BehavioralCloningPolicy(hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    # Move normalization stats to GPU for on-the-fly math
    s_mean_t = torch.tensor(state_mean, dtype=torch.float32).to(device)
    s_std_t = torch.tensor(state_std, dtype=torch.float32).to(device)
    a_mean_t = torch.tensor(action_mean, dtype=torch.float32).to(device)
    a_std_t = torch.tensor(action_std, dtype=torch.float32).to(device)

    best_val = float("inf")

    # --- Training Loop ---
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["observation.image"].to(device, dtype=torch.float32)
            states = batch["observation.state"].to(device, dtype=torch.float32)
            actions = batch["action"].to(device, dtype=torch.float32)

            # Apply your normalization on-the-fly
            states_norm = (states - s_mean_t) / s_std_t
            actions_norm = (actions - a_mean_t) / a_std_t

            pred = model(images, states_norm)
            loss = loss_fn(pred, actions_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())

        train_loss = running_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["observation.image"].to(device, dtype=torch.float32)
                states = batch["observation.state"].to(device, dtype=torch.float32)
                actions = batch["action"].to(device, dtype=torch.float32)
                
                states_norm = (states - s_mean_t) / s_std_t
                actions_norm = (actions - a_mean_t) / a_std_t

                val_pred = model(images, states_norm)
                val_loss_sum += float(loss_fn(val_pred, actions_norm).item())
        
        val_loss = val_loss_sum / len(val_loader)

        # --- Checkpoint Saving ---
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": action_std.astype(np.float32),
            "config": asdict(config),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        latest_path = os.path.join(config.output_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(config.output_dir, "best.pt")
            torch.save(checkpoint, best_path)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    # --- Artifact Saving ---
    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    np.savez(
        os.path.join(config.output_dir, "normalization_stats.npz"),
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_std=action_std.astype(np.float32),
    )

    print(f"Saved BC model artifacts to: {config.output_dir}")
    print(f"Best validation loss: {best_val:.6f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train standard Vision-BC on lerobot/pusht")
    parser.add_argument("--dataset_id", type=str, default="lerobot/pusht")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="models/standard_bc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
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
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)