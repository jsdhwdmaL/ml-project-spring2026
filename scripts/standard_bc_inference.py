import os
import sys
import warnings
from typing import Dict

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import gymnasium as gym
import pygame
from absl import app, flags
from tqdm import tqdm

import gym_pusht  # registers environment
from envs.frame_stack_wrapper import FrameStackWrapperEnv

# Optional: If you want the status overlay, import it. Otherwise we just render visually.
try:
    from envs.interactive_utils import draw_status_overlay
except ImportError:
    draw_status_overlay = None

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/standard_bc/best_model.pth", "Path to trained model weights")
flags.DEFINE_integer("num_seeds", 20, "Number of episodes to evaluate")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz (Must match training data!)")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")

# ==========================================
# 1. Model Architecture (Must match training)
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
# 2. Helper Functions
# ==========================================
def get_observation_image_array(env):
    """Grabs the active Pygame screen as an RGB numpy array."""
    screen = pygame.display.get_surface()
    if screen is not None:
        img_array = pygame.surfarray.array3d(screen)
        # Pygame uses (W, H, C), we need (H, W, C) for standard image processing
        return np.transpose(img_array, (1, 0, 2))
    return np.zeros((512, 512, 3), dtype=np.uint8)

def get_agent_pos_from_obs(obs: Dict) -> np.ndarray:
    """Extract current agent position [x, y] from frame-stacked observation."""
    return obs["agent_pos"][-1]

# ==========================================
# 3. Main Evaluation Loop
# ==========================================
def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model from {FLAGS.model_path} onto {device}...")
    
    # Load Model
    model = BehavioralCloningPolicy(state_dim=2, action_dim=2).to(device)
    model.load_state_dict(torch.load(FLAGS.model_path, map_location=device))
    model.eval() # Set to evaluation mode

    # Image Preprocessor (Matches LeRobot dataset format)
    image_transform = T.Compose([
        T.ToTensor(), # Converts [0, 255] numpy to [0.0, 1.0] float tensor and shifts to (C, H, W)
        T.Resize((96, 96), antialias=True) # Shrink from 512x512 down to 96x96
    ])

    window_size = int(512 * FLAGS.window_scale)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="human",
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)
    env = FrameStackWrapperEnv(env, n_frames=2, gap=1)

    print("\nStarting Evaluation...")
    success_count = 0

    # Evaluate over N random seeds
    for seed in range(FLAGS.num_seeds):
        obs, _ = env.reset(seed=seed)
        step = 0
        terminated = False
        truncated = False
        success = False
        clock = pygame.time.Clock()

        # Render first frame so we can capture it
        env.render()

        while not (terminated or truncated):
            # 1. Allow exiting via Pygame UI
            quit_eval = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    quit_eval = True
            if quit_eval:
                print("Evaluation aborted by user.")
                env.close()
                return

            # 2. Extract state and image for the model
            agent_pos = get_agent_pos_from_obs(obs) # Shape: (2,)
            img_array = get_observation_image_array(env)
            
            # Convert to PyTorch tensors and add Batch Dimension (B=1)
            state_tensor = torch.tensor(agent_pos, dtype=torch.float32).unsqueeze(0).to(device)
            image_tensor = image_transform(img_array).unsqueeze(0).to(device)
            
            # 3. Model Prediction
            with torch.no_grad():
                action_tensor = model(image_tensor, state_tensor)
            
            action = action_tensor.squeeze(0).cpu().numpy() # Remove batch dim, back to numpy
            
            # 4. Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_success = bool(info.get("is_success", terminated)) if isinstance(info, dict) else bool(terminated)
            success = success or step_success

            step += 1
            if step >= FLAGS.max_steps:
                truncated = True

            env.render()
            if draw_status_overlay is not None:
                # Mock a "HUMAN_CONTROL" enum just to use your drawing util if you imported it
                class MockState: value = "MODEL_CONTROL"
                draw_status_overlay(env, MockState, seed, 0, step, FLAGS.max_steps, agent_pos, False)
                
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