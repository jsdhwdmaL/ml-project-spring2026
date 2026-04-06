import warnings
from typing import Dict
from collections import deque

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import gymnasium as gym
import pygame
from absl import app, flags

import gym_pusht  # registers environment
from envs.interactive_utils import get_observation_image, draw_status_overlay, ControlState

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/pretrain_stacked/best.pt", "Path to trained model weights")
flags.DEFINE_integer("num_seeds", 5, "Number of episodes to evaluate")
flags.DEFINE_boolean("random_seeds", True, "Sample random seeds instead of using 0..num_seeds-1")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz (Must match training data!)")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")

# ==========================================
# 1. Model Architecture (UPDATED FOR 6 CHANNELS)
# ==========================================
class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256, n_frames=2):
        super().__init__()
        self.n_frames = n_frames
        
        # We don't need pretrained weights for evaluation, we will load them from the checkpoint
        resnet = models.resnet18(weights=None)
        
        # SURGERY: Accept stacked frames (6 channels)
        self.input_channels = 3 * n_frames
        resnet.conv1 = nn.Conv2d(
            self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512 
        
        # Encoder for stacked states (4 dims)
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
        img_features = self.vision_backbone(image)
        img_features = img_features.view(img_features.size(0), -1) 
        state_features = self.state_encoder(state) 
        combined_features = torch.cat([img_features, state_features], dim=1) 
        return self.action_head(combined_features) 

# ==========================================
# 2. Helper Functions
# ==========================================
def get_agent_pos_from_obs(obs: Dict) -> np.ndarray:
    """Extract current agent position [x, y] from observation."""
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    if agent_pos.ndim == 1:
        return agent_pos
    return agent_pos[-1]

# ==========================================
# 3. Main Evaluation Loop
# ==========================================
def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model from {FLAGS.model_path} onto {device}...")
    
    checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    
    # Extract config dynamically to match training setup
    config = checkpoint.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 256))
    n_frames = int(config.get("n_frames", 2))
    frame_gap = int(config.get("frame_gap", 9))
    buffer_size = frame_gap + 1  # If gap is 9, we need 10 frames (0 through 9)

    model = BehavioralCloningPolicy(state_dim=2, action_dim=2, hidden_dim=hidden_dim, n_frames=n_frames).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Normalization Stats
    state_mean = torch.tensor(checkpoint["state_mean"], dtype=torch.float32).to(device)
    state_std = torch.tensor(checkpoint["state_std"], dtype=torch.float32).to(device)
    action_mean = torch.tensor(checkpoint["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(checkpoint["action_std"], dtype=torch.float32).to(device)

    # Image Preprocessors
    base_transform = T.Compose([
        T.ToTensor(),
        T.Resize((96, 96), antialias=True)
    ])
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    window_size = int(512 * FLAGS.window_scale)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="human", 
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    print("\nStarting Evaluation...")
    success_count = 0

    seeds = np.random.randint(0, 2**31, size=FLAGS.num_seeds).tolist() if FLAGS.random_seeds else range(FLAGS.num_seeds)
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        
        # --- INITIALIZE BUFFERS ---
        image_buffer = deque(maxlen=buffer_size)
        state_buffer = deque(maxlen=buffer_size)
        
        # Grab initial frame to populate the buffer
        init_pos = get_agent_pos_from_obs(obs)
        init_img = get_observation_image(env)
        
        init_state_t = torch.tensor(init_pos, dtype=torch.float32)
        init_img_t = base_transform(init_img) # Shape: (3, 96, 96)
        
        for _ in range(buffer_size):
            image_buffer.append(init_img_t)
            state_buffer.append(init_state_t)
            
        step = 0
        terminated = False
        truncated = False
        success = False
        clock = pygame.time.Clock()
        raw_action_min = np.array([np.inf, np.inf], dtype=np.float32)
        raw_action_max = np.array([-np.inf, -np.inf], dtype=np.float32)
        clipped_action_min = np.array([np.inf, np.inf], dtype=np.float32)
        clipped_action_max = np.array([-np.inf, -np.inf], dtype=np.float32)
        clipped_count = 0
        edge_count = 0
        action_count = 0

        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    print("Evaluation aborted by user.")
                    env.close()
                    return

            # --- 1. UPDATE BUFFERS ---
            agent_pos = get_agent_pos_from_obs(obs)
            img_array = get_observation_image(env)
            
            state_buffer.append(torch.tensor(agent_pos, dtype=torch.float32))
            image_buffer.append(base_transform(img_array))
            
            # --- 2. EXTRACT t-9 AND t ---
            state_t9 = state_buffer[0]
            state_t = state_buffer[-1]
            img_t9 = image_buffer[0]
            img_t = image_buffer[-1]
            
            # --- 3. PREPARE INPUTS ---
            # Apply ImageNet normalization to each frame individually
            img_t9_norm = normalize_transform(img_t9)
            img_t_norm = normalize_transform(img_t)
            
            # Concat images on channel dim: (3,96,96) + (3,96,96) -> (6,96,96) -> Add Batch Dim -> (1,6,96,96)
            image_tensor = torch.cat([img_t9_norm, img_t_norm], dim=0).unsqueeze(0).to(device)
            
            # Stack states into shape (2, 2) to match training normalization shape, then flatten to (1, 4)
            state_combined = torch.stack([state_t9, state_t], dim=0).to(device)
            state_tensor_norm = ((state_combined - state_mean) / state_std).view(1, -1)
            
            # --- 4. PREDICT ---
            with torch.no_grad():
                action_tensor_norm = model(image_tensor, state_tensor_norm)
            
            # Un-normalize Action
            action_tensor = (action_tensor_norm * action_std) + action_mean
            raw_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            action = np.clip(raw_action, 0.0, 512.0)

            # Track Stats
            raw_action_min = np.minimum(raw_action_min, raw_action)
            raw_action_max = np.maximum(raw_action_max, raw_action)
            clipped_action_min = np.minimum(clipped_action_min, action)
            clipped_action_max = np.maximum(clipped_action_max, action)
            clipped_count += int(np.any(np.abs(action - raw_action) > 1e-6))
            edge_count += int(np.any((action <= 1.0) | (action >= 511.0)))
            action_count += 1
            
            # --- 5. STEP ENV ---
            obs, reward, terminated, truncated, info = env.step(action)
            step_success = bool(info.get("is_success", terminated)) if isinstance(info, dict) else bool(terminated)
            success = success or step_success

            step += 1
            if step >= FLAGS.max_steps:
                truncated = True

            env.render()
            
            draw_status_overlay(
                env, 
                ControlState.MODEL_CONTROL, 
                seed, 
                0, 
                step, 
                FLAGS.max_steps, 
                agent_pos, 
                False
            )
                
            clock.tick(FLAGS.fps)

        if success:
            success_count += 1
            print(f"Episode {seed + 1}/{FLAGS.num_seeds} - SUCCESS (Took {step} steps)")
        else:
            print(f"Episode {seed + 1}/{FLAGS.num_seeds} - FAILED (Reached {step} steps)")

        if action_count > 0:
            clip_rate = 100.0 * clipped_count / action_count
            edge_rate = 100.0 * edge_count / action_count
            print(
                "  Action stats | "
                f"raw_min={raw_action_min.tolist()} raw_max={raw_action_max.tolist()} | "
                f"clipped_min={clipped_action_min.tolist()} clipped_max={clipped_action_max.tolist()} | "
                f"clip_rate={clip_rate:.1f}% edge_rate={edge_rate:.1f}%"
            )

    print("=" * 60)
    print(f"Evaluation Complete! Success Rate: {success_count}/{FLAGS.num_seeds} ({(success_count/FLAGS.num_seeds)*100:.1f}%)")
    print("=" * 60)
    env.close()

if __name__ == "__main__":
    app.run(main)