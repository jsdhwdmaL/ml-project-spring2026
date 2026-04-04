import warnings
from typing import Dict

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import gymnasium as gym
import pygame
from absl import app, flags

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gym_pusht  # registers environment
from envs.interactive_utils import get_observation_image, draw_status_overlay, ControlState

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/dagger_finetuned/best.pt", "Path to trained model weights")
flags.DEFINE_integer("num_seeds", 20, "Number of episodes to evaluate")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz (Must match training data!)")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")

# ==========================================
# 1. Model Architecture
# ==========================================
class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feature_dim = 512 
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
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
# 2. Helper Functions (Matching Data Collection)
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
    
    # Load checkpoint with weights_only=False to allow numpy arrays (normalization stats)
    checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    
    config = checkpoint.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 256))

    model = BehavioralCloningPolicy(state_dim=2, action_dim=2, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Normalization Stats
    state_mean = torch.tensor(checkpoint["state_mean"], dtype=torch.float32).to(device)
    state_std = torch.tensor(checkpoint["state_std"], dtype=torch.float32).to(device)
    action_mean = torch.tensor(checkpoint["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(checkpoint["action_std"], dtype=torch.float32).to(device)

    # Image Preprocessor
    image_transform = T.Compose([
        T.ToTensor(),
        T.Resize((96, 96), antialias=True)
    ])

    window_size = int(512 * FLAGS.window_scale)

    # Env Setup exactly matching your collection script
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="human", # Change to "rgb_array" if running on headless server
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    print("\nStarting Evaluation...")
    success_count = 0

    for seed in range(FLAGS.num_seeds):
        obs, _ = env.reset(seed=seed)
        step = 0
        terminated = False
        truncated = False
        success = False
        clock = pygame.time.Clock()

        while not (terminated or truncated):
            # Allow quitting via Q key
            quit_eval = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    quit_eval = True
            if quit_eval:
                print("Evaluation aborted by user.")
                env.close()
                return

            # Use your custom util to get the image natively
            agent_pos = get_agent_pos_from_obs(obs)
            img_array = get_observation_image(env)
            
            # Prepare Tensors
            state_tensor = torch.tensor(agent_pos, dtype=torch.float32).unsqueeze(0).to(device)
            image_tensor = image_transform(img_array).unsqueeze(0).to(device)
            
            # Normalize State
            state_tensor_norm = (state_tensor - state_mean) / state_std
            
            # Predict
            with torch.no_grad():
                action_tensor_norm = model(image_tensor, state_tensor_norm)
            
            # Un-normalize Action
            action_tensor = (action_tensor_norm * action_std) + action_mean
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            action = np.clip(action, 0.0, 512.0)
            
            # Step Env
            obs, reward, terminated, truncated, info = env.step(action)
            step_success = bool(info.get("is_success", terminated)) if isinstance(info, dict) else bool(terminated)
            success = success or step_success

            step += 1
            if step >= FLAGS.max_steps:
                truncated = True

            env.render()
            
            # Use your custom overlay drawing
            # Now that we updated the utils, this call will be clean and descriptive
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

    print("=" * 60)
    print(f"Evaluation Complete! Success Rate: {success_count}/{FLAGS.num_seeds} ({(success_count/FLAGS.num_seeds)*100:.1f}%)")
    print("=" * 60)
    env.close()

if __name__ == "__main__":
    app.run(main)