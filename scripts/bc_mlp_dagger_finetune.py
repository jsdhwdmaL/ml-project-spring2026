#!/usr/bin/env python3
"""
Fine-tune a frame-stacking vision BC model on DAgger NPZ trajectories.
Maintains the original mix-ratio and episode-splitting logic.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scripts.bc_mlp_train import BehavioralCloningPolicy

@dataclass
class FinetuneConfig:
    model_path: str = "models/pretrain_stack/best.pt"
    data_dir: str = "data/dagger"
    original_dataset_id: str = "lerobot/pusht"
    original_split: str = "train"
    output_dir: str = "models/dagger_stack"
    n_frames: int = 2
    frame_gap: int = 9
    include_autonomous: bool = False
    include_failed: bool = False
    only_human_steps: bool = False
    mix_dagger_ratio: float = 0.5
    mix_original_ratio: float = 0.5
    val_ratio: float = 0.1
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    seed: int = 42


class DaggerNPZDataset(Dataset):
    """Processes raw NPZ lists into [t-gap, t] stacks."""
    def __init__(self, state_list, image_list, action_list, is_human_list, frame_gap, only_human_steps):
        self.frame_gap = frame_gap
        self.episodes_states = state_list
        self.episodes_images = image_list
        self.episodes_actions = action_list
        
        self.indices = []
        for ep_idx, is_human_mask in enumerate(is_human_list):
            for t in range(len(is_human_mask)):
                if only_human_steps and not is_human_mask[t]:
                    continue
                self.indices.append((ep_idx, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        ep_idx, t = self.indices[idx]
        t_prev = max(0, t - self.frame_gap)

        # Image Stack: (6, 96, 96)
        img_curr = self._prep_img(self.episodes_images[ep_idx][t])
        img_prev = self._prep_img(self.episodes_images[ep_idx][t_prev])
        image_stack = torch.cat([img_prev, img_curr], dim=0)

        # State Stack: (2, 2) -> will be flattened to 4D in training loop
        state_stack = np.stack([self.episodes_states[ep_idx][t_prev], 
                                self.episodes_states[ep_idx][t]], axis=0)

        return {
            "state": torch.from_numpy(state_stack).float(),
            "image": image_stack,
            "action": torch.from_numpy(self.episodes_actions[ep_idx][t]).float(),
        }

    def _prep_img(self, img_np):
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        if img.shape[-2:] != (96, 96):
            img = F.interpolate(img.unsqueeze(0), size=(96, 96), mode="bilinear", align_corners=False).squeeze(0)
        return img

class LeRobotAdapterDataset(Dataset):
    """Maps LeRobot (F, C, H, W) to (F*C, H, W)."""
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        # LeRobot yields (2, 3, 96, 96) -> flatten to (6, 96, 96)
        images = sample["observation.image"].float() / 255.0
        return {
            "state": sample["observation.state"].float(),
            "image": images.view(-1, 96, 96),
            "action": sample["action"].float(),
        }


def _collect_episode_files(config: FinetuneConfig) -> List[str]:
    base = Path(config.data_dir)
    folders = [base / "human_intervention"]
    if config.include_autonomous: folders.append(base / "rejection_sample")
    if config.include_failed: folders.append(base / "failed_autonomous")
    
    files = []
    for folder in folders:
        if folder.exists():
            files.extend(sorted(str(p) for p in folder.glob("*.npz") if not p.name.endswith("_images.npz")))
    if not files: raise ValueError("No NPZ files found.")
    return files

def _load_episode_pair(ep_file: str):
    img_file = ep_file.replace(".npz", "_images.npz")
    ep, img = np.load(ep_file), np.load(img_file)
    return ep["observation.state"], img["images"], ep["action"], ep["is_human_intervention"]

def _split_episode_files(files, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_ratio))
    return [files[i] for i in indices[n_val:]], [files[i] for i in indices[:n_val]]

def _split_lerobot_episode_indices(dataset, val_ratio, seed):
    ep_idx = np.array(dataset.hf_dataset["episode_index"]).flatten()
    unique_eps = np.unique(ep_idx)
    np.random.default_rng(seed).shuffle(unique_eps)
    n_val = max(1, int(len(unique_eps) * val_ratio))
    val_eps = set(unique_eps[:n_val])
    mask = np.array([e in val_eps for e in ep_idx])
    return np.where(~mask)[0], np.where(mask)[0]

def _build_mixed_loader(dagger_ds, original_ds, dagger_ratio, original_ratio, batch_size, seed, pin_memory):
    datasets = [dagger_ds, original_ds]
    # Calculate weights based on ratios and lengths
    w_d = dagger_ratio / len(dagger_ds) if len(dagger_ds) > 0 else 0
    w_o = original_ratio / len(original_ds) if len(original_ds) > 0 else 0
    weights = ([w_d] * len(dagger_ds)) + ([w_o] * len(original_ds))
    
    mixed = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(weights, num_samples=len(mixed), replacement=True)
    return DataLoader(mixed, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)


def train(config: FinetuneConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Load Normalization Stats
    checkpoint = torch.load(config.model_path, map_location=device, weights_only=False)
    s_mean = torch.tensor(checkpoint["state_mean"][:2], device=device)
    s_std = torch.tensor(checkpoint["state_std"][:2], device=device)
    a_mean = torch.tensor(checkpoint["action_mean"], device=device)
    a_std = torch.tensor(checkpoint["action_std"], device=device)
    norm_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 2. Build Datasets
    fps = 10
    deltas = [-config.frame_gap, 0]
    original_dataset = LeRobotDataset(config.original_dataset_id, delta_timestamps={
        "observation.image": [i/fps for i in deltas], "observation.state": [i/fps for i in deltas]
    })
    
    o_train_idx, o_val_idx = _split_lerobot_episode_indices(original_dataset, config.val_ratio, config.seed)
    o_train_ds = LeRobotAdapterDataset(Subset(original_dataset, o_train_idx.tolist()))
    o_val_ds = LeRobotAdapterDataset(Subset(original_dataset, o_val_idx.tolist()))

    files = _collect_episode_files(config)
    tr_files, val_files = _split_episode_files(files, config.val_ratio, config.seed)
    
    def build_dag_ds(fls):
        sl, il, al, hl = [], [], [], []
        for f in fls:
            s, i, a, h = _load_episode_pair(f)
            sl.append(s); il.append(i); al.append(a); hl.append(h)
        return DaggerNPZDataset(sl, il, al, hl, config.frame_gap, config.only_human_steps)

    dagger_train, dagger_val = build_dag_ds(tr_files), build_dag_ds(val_files)

    train_loader = _build_mixed_loader(dagger_train, o_train_ds, config.mix_dagger_ratio, config.mix_original_ratio, config.batch_size, config.seed, True)
    val_loader = _build_mixed_loader(dagger_val, o_val_ds, config.mix_dagger_ratio, config.mix_original_ratio, config.batch_size, config.seed+1, True)

    # 3. Model Setup
    model = BehavioralCloningPolicy(n_frames=config.n_frames, hidden_dim=checkpoint['config']['hidden_dim']).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # Prepare Image: Normalize each frame in stack
            imgs = batch["image"].to(device)
            B, C_stack, H, W = imgs.shape
            imgs = norm_tf(imgs.view(B*2, 3, H, W)).view(B, C_stack, H, W)
            
            # Prepare State/Action
            states_norm = ((batch["state"].to(device) - s_mean) / s_std).view(B, -1)
            actions_norm = (batch["action"].to(device) - a_mean) / a_std
            
            preds = model(imgs, states_norm)
            loss = loss_fn(preds, actions_norm)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                imgs = norm_tf(imgs.view(-1, 3, 96, 96)).view(imgs.shape)
                states_norm = ((batch["state"].to(device) - s_mean) / s_std).view(imgs.shape[0], -1)
                actions_norm = (batch["action"].to(device) - a_mean) / a_std
                val_loss += loss_fn(model(imgs, states_norm), actions_norm).item()
        
        avg_v = val_loss / len(val_loader)
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {avg_v:.6f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({
                "model_state_dict": model.state_dict(), "state_mean": s_mean.cpu().numpy(),
                "state_std": s_std.cpu().numpy(), "action_mean": a_mean.cpu().numpy(),
                "action_std": a_std.cpu().numpy(), "config": checkpoint['config']
            }, os.path.join(config.output_dir, "best.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/bc_mlp/best.pt")
    parser.add_argument("--output_dir", type=str, default="models/bc_mlp_dagger")
    parser.add_argument("--data_dir", type=str, default="data/bc_mlp_dagger")
    parser.add_argument("--mix_dagger_ratio", type=float, default=0.6)
    parser.add_argument("--mix_original_ratio", type=float, default=0.4)
    args = parser.parse_args()
    
    cfg = FinetuneConfig(model_path=args.model_path, output_dir=args.output_dir, 
                         data_dir=args.data_dir, mix_dagger_ratio=args.mix_dagger_ratio, 
                         mix_original_ratio=args.mix_original_ratio)
    train(cfg)