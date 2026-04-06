"""Train a Hybrid Vision-Motor Encoder-Decoder Transformer on lerobot/pusht."""

import argparse
import json
import os
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset

@dataclass
class TrainConfig:
    dataset_id: str = "lerobot/pusht"
    output_dir: str = "models/hybrid_transformer"
    seed: int = 42
    val_ratio: float = 0.1
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 3
    future_steps: int = 1   # How many steps into the future to predict
    fps: int = 10
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, start_idx : start_idx + seq_len, :]

class HybridVisionTransformer(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, config: TrainConfig = None):
        super().__init__()
        self.config = config
        
        # --- ENCODER COMPONENTS ---
        resnet = models.resnet18(weights="DEFAULT")
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512 
        
        # State is just [agent_x, agent_y] (2D)
        self.state_encoder = nn.Linear(state_dim, 64)
        
        self.scene_proj = nn.Linear(vision_feature_dim + 64, config.d_model)
        
        # --- DECODER COMPONENTS ---
        self.action_embed = nn.Linear(action_dim, config.d_model)
        self.pe = PositionalEncoding(config.d_model)
        
        self.transformer = nn.Transformer(
            d_model=config.d_model, 
            nhead=config.nhead, 
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.action_head = nn.Linear(config.d_model, action_dim)

    def forward(self, image, state, target_actions=None):
        B = image.size(0)
        device = image.device
        
        # 1. --- PROCESS SCENE (ENCODER) ---
        img_features = self.vision_backbone(image.squeeze(1)).view(B, -1)
        
        # Extract the 2D current state (Batch, 1, 2) -> (Batch, 2)
        current_state = state[:, 0, :] 
        state_features = F.relu(self.state_encoder(current_state))
        
        scene_combined = torch.cat([img_features, state_features], dim=-1)
        scene_token = self.scene_proj(scene_combined).unsqueeze(1) # (Batch, 1, d_model)
        
        memory = self.transformer.encoder(scene_token)

        # 2. --- GENERATE ACTIONS (DECODER) ---
        if target_actions is not None:
            # TRAINING MODE (Teacher Forcing)
            current_pos = state # (Batch, 1, 2)
            
            decoder_input_actions = torch.cat([current_pos, target_actions[:, :-1, :]], dim=1)
            
            dec_in_emb = self.action_embed(decoder_input_actions)
            dec_in_emb = self.pe(dec_in_emb)
            
            seq_len = dec_in_emb.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            out = self.transformer.decoder(dec_in_emb, memory, tgt_mask=causal_mask)
            return self.action_head(out)
            
        else:
            # INFERENCE MODE (Auto-Regressive Generation)
            current_pos = state # (Batch, 1, 2)
            dec_in_emb = self.pe(self.action_embed(current_pos), start_idx=0)
            
            generated_actions = []
            
            for step in range(self.config.future_steps):
                seq_len = dec_in_emb.size(1)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                
                out = self.transformer.decoder(dec_in_emb, memory, tgt_mask=causal_mask)
                next_action = self.action_head(out[:, -1:, :]) 
                generated_actions.append(next_action)
                
                next_emb = self.pe(self.action_embed(next_action), start_idx=seq_len)
                dec_in_emb = torch.cat([dec_in_emb, next_emb], dim=1)
                
            return torch.cat(generated_actions, dim=1) # (Batch, future_steps, 2)

# ==========================================
# 3. Preprocessing & Training
# ==========================================
def get_transforms():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_image_batch(images: torch.Tensor, transform_fn) -> torch.Tensor:
    images = images.to(dtype=torch.float32) / 255.0
    images = F.interpolate(images.squeeze(1), size=(96, 96), mode="bilinear", align_corners=False)
    images = transform_fn(images).unsqueeze(1)
    return images

def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    # Dataloading: Only 1 current image/state, but 8 future actions!
    future_timestamps = (np.arange(config.future_steps) / config.fps).tolist()
    
    print("Loading dataset...")
    dataset = LeRobotDataset(
        config.dataset_id, 
        delta_timestamps={
            "observation.image": [0.0],  # Current frame
            "observation.state": [0.0],  # Current state
            "action": future_timestamps, # Chunk of 8 future actions
        }
    )
    
    episode_index = np.array(dataset.hf_dataset["episode_index"]).flatten()
    unique_eps = np.unique(episode_index)
    np.random.shuffle(unique_eps)
    val_eps = set(unique_eps[:max(1, int(len(unique_eps) * config.val_ratio))].tolist())

    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_loader = DataLoader(Subset(dataset, np.where(~val_mask)[0]), batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(dataset, np.where(val_mask)[0]), batch_size=config.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = HybridVisionTransformer(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.SmoothL1Loss() 

    # Normalization (computed dynamically as before)
    all_states = np.array(dataset.hf_dataset["observation.state"])
    all_actions = np.array(dataset.hf_dataset["action"])
    state_m, state_s = all_states.mean(axis=0), all_states.std(axis=0) + 1e-6
    act_m, act_s = all_actions.mean(axis=0), all_actions.std(axis=0) + 1e-6

    s_mean_t = torch.tensor(state_m, dtype=torch.float32).to(device)
    s_std_t = torch.tensor(state_s, dtype=torch.float32).to(device)
    a_mean_t = torch.tensor(act_m, dtype=torch.float32).to(device)
    a_std_t = torch.tensor(act_s, dtype=torch.float32).to(device)

    transform_fn = get_transforms()
    best_val = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            images = preprocess_image_batch(batch["observation.image"].to(device), transform_fn)
            states = batch["observation.state"].to(device, dtype=torch.float32)
            actions = batch["action"].to(device, dtype=torch.float32)

            states_norm = (states - s_mean_t) / s_std_t
            actions_norm = (actions - a_mean_t) / a_std_t

            pred_actions = model(images, states_norm, target_actions=actions_norm)
            loss = loss_fn(pred_actions, actions_norm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = preprocess_image_batch(batch["observation.image"].to(device), transform_fn)
                states = batch["observation.state"].to(device, dtype=torch.float32)
                actions = batch["action"].to(device, dtype=torch.float32)
                
                states_norm = (states - s_mean_t) / s_std_t
                actions_norm = (actions - a_mean_t) / a_std_t
                
                # Notice we still pass target_actions in eval to measure Teacher Forced loss for validation parity
                pred_actions = model(images, states_norm, target_actions=actions_norm)
                v_loss += loss_fn(pred_actions, actions_norm).item()
        
        scheduler.step()
        
        avg_t, avg_v = t_loss / len(train_loader), v_loss / len(val_loader)
        print(f"Epoch {epoch:03d} | Train Loss: {avg_t:.6f} | Val Loss: {avg_v:.6f}")

        if avg_v < best_val:
            best_val = avg_v
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "state_mean": state_m.astype(np.float32), "state_std": state_s.astype(np.float32),
                "action_mean": act_m.astype(np.float32), "action_std": act_s.astype(np.float32),
                "config": asdict(config)
            }
            torch.save(checkpoint, os.path.join(config.output_dir, "best.pt"))

if __name__ == "__main__":
    train(TrainConfig())