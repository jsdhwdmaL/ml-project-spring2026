"""
Train a Vision-based Behavioral Cloning policy on lerobot/pusht with Frame Stacking,
SmoothL1 loss, and pretrained weights from ImageNet, with mlp for states/actions and ResNet for image frames.

Usage:
python scripts/standard_bc_training_new.py \
    --output_dir models/pretrain2 \
    --n_frames 2 \
    --frame_gap 9
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from models.bc_mlp import BehavioralCloningPolicy

@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    split: str = "train"
    output_dir: str = "models/pretrain_stack"
    seed: int = 42
    val_ratio: float = 0.1
    hidden_dim: int = 256
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    # Frame Stacking Config
    n_frames: int = 2
    frame_gap: int = 9 

# ==========================================
# 1. Vision Architecture
# ==========================================
class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256, n_frames=2):
        super().__init__()
        self.n_frames = n_frames
        
        # Load Pre-trained ResNet18
        resnet = models.resnet18(weights="DEFAULT")
        
        # Modify the first layer to accept stacked frames (6 channels instead of 3)
        self.input_channels = 3 * n_frames
        resnet.conv1 = nn.Conv2d(
            self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512 
        
        # Encoder for stacked states (4 dims if n_frames=2)
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
        # image shape: (B, 6, 96, 96) | state shape: (B, 4)
        img_features = self.vision_backbone(image).view(image.size(0), -1) 
        state_features = self.state_encoder(state) 
        combined_features = torch.cat([img_features, state_features], dim=1) 
        return self.action_head(combined_features) 

# ==========================================
# 2. Preprocessing (Bug-Free)
# ==========================================
def get_transforms():
    # Only applying normalization to ensure lighting/colors stay consistent between t-9 and t
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_image_batch(images: torch.Tensor, transform_fn) -> torch.Tensor:
    # images shape from LeRobot: (B, F, C, H, W)
    B, F, C, H, W = images.shape
    images = images.to(dtype=torch.float32) / 255.0
    
    # Resize to 96x96 if necessary
    if (H, W) != (96, 96):
        images = F.interpolate(
            images.view(-1, C, H, W), size=(96, 96), mode="bilinear", align_corners=False
        ).view(B, F, C, 96, 96)

    # Flatten the Frame and Channel dimensions together: (B, 6, 96, 96)
    images = images.view(B, F * C, 96, 96)
    
    # Temporarily reshape to 3-channels to apply standard ImageNet normalization evenly
    images_3c = images.view(B * F, C, 96, 96)
    images_3c = transform_fn(images_3c)
    
    # Reshape back to 6 channels
    return images_3c.view(B, F * C, 96, 96)

# ==========================================
# 3. Training Pipeline
# ==========================================
def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

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
    
    # Episode-based Split
    episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
    unique_eps = np.unique(episode_index)
    np.random.shuffle(unique_eps)
    n_val_eps = max(1, int(len(unique_eps) * config.val_ratio))
    val_eps = set(unique_eps[:n_val_eps].tolist())

    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]

    # Calculate Normalization Stats
    print("Computing normalization stats...")
    all_states = np.array(dataset.hf_dataset["observation.state"])
    all_actions = np.array(dataset.hf_dataset["action"])
    state_mean = all_states[train_idx].mean(axis=0)
    state_std = all_states[train_idx].std(axis=0) + 1e-6
    action_mean = all_actions[train_idx].mean(axis=0)
    action_std = all_actions[train_idx].std(axis=0) + 1e-6

    # DataLoaders
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = BehavioralCloningPolicy(n_frames=config.n_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0) 

    # Move normalizations to device
    s_mean_t = torch.tensor(state_mean, dtype=torch.float32).to(device)
    s_std_t = torch.tensor(state_std, dtype=torch.float32).to(device)
    a_mean_t = torch.tensor(action_mean, dtype=torch.float32).to(device)
    a_std_t = torch.tensor(action_std, dtype=torch.float32).to(device)

    transform_fn = get_transforms()
    best_val = float("inf")

    # Training Loop
    for epoch in range(1, config.epochs + 1):
        # --- TRAIN ---
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            images = preprocess_image_batch(batch["observation.image"].to(device), transform_fn)
            states = batch["observation.state"].to(device, dtype=torch.float32)
            actions = batch["action"].to(device, dtype=torch.float32)

            # Normalize and flatten
            states_norm = ((states - s_mean_t) / s_std_t).view(states.size(0), -1)
            actions_norm = (actions - a_mean_t) / a_std_t

            pred = model(images, states_norm)
            loss = loss_fn(pred, actions_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()

        # --- VALIDATE ---
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = preprocess_image_batch(batch["observation.image"].to(device), transform_fn)
                states = batch["observation.state"].to(device, dtype=torch.float32)
                actions = batch["action"].to(device, dtype=torch.float32)
                
                states_norm = ((states - s_mean_t) / s_std_t).view(states.size(0), -1)
                actions_norm = (actions - a_mean_t) / a_std_t
                
                pred = model(images, states_norm)
                v_loss += loss_fn(pred, actions_norm).item()
        
        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)
        print(f"Epoch {epoch:03d} | Train Loss: {avg_t:.6f} | Val Loss: {avg_v:.6f}")

        # --- SAVE ---
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": action_std.astype(np.float32),
            "config": asdict(config),
            "epoch": epoch,
        }
        
        torch.save(checkpoint, os.path.join(config.output_dir, "latest.pt"))

        if avg_v < best_val:
            best_val = avg_v
            torch.save(checkpoint, os.path.join(config.output_dir, "best.pt"))
            print(f"  --> New best model saved (Val Loss: {best_val:.6f})")

    # Save artifacts
    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Vision-BC with Frame Stacking")
    parser.add_argument("--dataset_id", type=str, default="lerobot/pusht")
    parser.add_argument("--output_dir", type=str, default="models/pretrain_stack")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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