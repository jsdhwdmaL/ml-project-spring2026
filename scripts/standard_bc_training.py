"""Train a Vision-based Behavioral Cloning policy on lerobot/pusht.

Usage:
python scripts/standard_bc_training.py \
    --output_dir models/pretrain2 \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --n_frames 2 \
    --frame_gap 9

Small run:
python scripts/standard_bc_training.py \
    --output_dir models/pretrain \
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

# Import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    split: str = "train"
    output_dir: str = "models/pretrain2"
    seed: int = 42
    val_ratio: float = 0.1
    hidden_dim: int = 256
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    # Frame Stacking Config
    n_frames: int = 2
    frame_gap: int = 9 

# ==========================================
# 1. Improved Vision Architecture
# ==========================================
class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256, n_frames=2):
        super().__init__()
        self.n_frames = n_frames
        resnet = models.resnet18(weights="DEFAULT")
        self.input_channels = 3 * n_frames
        resnet.conv1 = nn.Conv2d(
            self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512 
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * n_frames, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
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

def get_transforms(is_train=True):
    t_list = []
    if is_train:
        t_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    t_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(t_list)

def preprocess_image_batch(images: torch.Tensor, transform_fn) -> torch.Tensor:
    B, F_idx, C, H, W = images.shape
    images = images.to(dtype=torch.float32) / 255.0
    if (H, W) != (96, 96):
        images = F.interpolate(images.view(-1, C, H, W), size=(96, 96), mode="bilinear").view(B, F_idx, C, 96, 96)
    processed_frames = []
    for i in range(F_idx):
        processed_frames.append(transform_fn(images[:, i]))
    return torch.cat(processed_frames, dim=1)

# ==========================================
# 2. Training Pipeline
# ==========================================
def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    fps = 10
    delta_indices = [-config.frame_gap, 0]
    
    print(f"Loading dataset with {config.n_frames} frames (gap: {config.frame_gap})...")
    dataset = LeRobotDataset(
        config.dataset_id, 
        delta_timestamps={
            "observation.image": [i/fps for i in delta_indices],
            "observation.state": [i/fps for i in delta_indices],
        }
    )
    
    episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
    unique_eps = np.unique(episode_index)
    np.random.shuffle(unique_eps)
    n_val_eps = max(1, int(len(unique_eps) * config.val_ratio))
    val_eps = set(unique_eps[:n_val_eps].tolist())

    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]

    all_states = np.array(dataset.hf_dataset["observation.state"])
    all_actions = np.array(dataset.hf_dataset["action"])
    state_mean, state_std = all_states[train_idx].mean(axis=0), all_states[train_idx].std(axis=0) + 1e-6
    action_mean, action_std = all_actions[train_idx].mean(axis=0), all_actions[train_idx].std(axis=0) + 1e-6

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = BehavioralCloningPolicy(n_frames=config.n_frames).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0) 

    s_mean_t = torch.tensor(state_mean, dtype=torch.float32).to(device)
    s_std_t = torch.tensor(state_std, dtype=torch.float32).to(device)
    a_mean_t = torch.tensor(action_mean, dtype=torch.float32).to(device)
    a_std_t = torch.tensor(action_std, dtype=torch.float32).to(device)

    train_tf, val_tf = get_transforms(is_train=True), get_transforms(is_train=False)
    best_val_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            images = preprocess_image_batch(batch["observation.image"].to(device), train_tf)
            states = batch["observation.state"].to(device, dtype=torch.float32)
            actions = batch["action"].to(device, dtype=torch.float32)

            states_norm = ((states - s_mean_t) / s_std_t).view(states.size(0), -1)
            actions_norm = (actions - a_mean_t) / a_std_t

            pred = model(images, states_norm)
            loss = loss_fn(pred, actions_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = preprocess_image_batch(batch["observation.image"].to(device), val_tf)
                states = batch["observation.state"].to(device, dtype=torch.float32)
                actions = batch["action"].to(device, dtype=torch.float32)

                states_norm = ((states - s_mean_t) / s_std_t).view(states.size(0), -1)
                actions_norm = (actions - a_mean_t) / a_std_t

                val_pred = model(images, states_norm)
                val_loss = loss_fn(val_pred, actions_norm)
                val_loss_sum += val_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- SAVE CHECKPOINTS ---
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "config": asdict(config),
            "epoch": epoch
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(config.output_dir, "latest.pt"))

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(config.output_dir, "best.pt"))
            print(f"  --> New best model saved (Val Loss: {best_val_loss:.6f})")

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Vision-BC with Frame Stacking")
    parser.add_argument("--dataset_id", type=str, default="lerobot/pusht")
    parser.add_argument("--output_dir", type=str, default="models/pretrain_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--n_frames", type=int, default=2)
    parser.add_argument("--frame_gap", type=int, default=9)
    args = parser.parse_args()

    return TrainConfig(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_frames=args.n_frames,
        frame_gap=args.frame_gap
    )

if __name__ == "__main__":
    config = parse_args()
    train(config)