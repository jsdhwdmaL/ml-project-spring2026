#!/usr/bin/env python3
"""
Collect human-intervention trajectories for DAgger using a pretrained frame-stacking BC model.
"""

import os
import sys
import collections
import numpy as np
import gymnasium as gym
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from absl import app, flags
from tqdm import tqdm

import gym_pusht
from envs.interactive_utils import (
    ControlState,
    InterventionController,
    get_observation_image,
    draw_status_overlay,
)
from data.trajectory_recorder import TrajectoryRecorder
from data.episode_saver import EpisodeSaver

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/pretrain_stack/best.pt", "Path to pretrained model")
flags.DEFINE_string("output_dir", "data/dagger_stacked", "Directory to save data")
flags.DEFINE_integer("fps", 10, "Control frequency")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_float("window_scale", 1.0, "Window scale")
flags.DEFINE_float("activation_radius", 30.0, "Mouse threshold")

# ==========================================
# 1. Model Architecture (Matching your Training Script)
# ==========================================
class BehavioralCloningPolicy(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256, n_frames=2):
        super().__init__()
        self.n_frames = n_frames
        resnet = models.resnet18(weights=None)
        self.input_channels = 3 * n_frames
        resnet.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * n_frames, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.action_head = nn.Sequential(
            nn.Linear(512 + 64, hidden_dim),
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
# 2. Inference Preprocessing
# ==========================================
def preprocess_stacked_input(image_buffer, state_buffer, transform_fn, device):
    """
    Takes list of images/states, stacks them, and normalizes for the model.
    """
    # Image: List of (H,W,3) -> (B, F*C, 96, 96)
    imgs = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in image_buffer]
    imgs = [F.interpolate(img.unsqueeze(0), size=(96, 96), mode="bilinear").squeeze(0) for img in imgs]
    
    # Stack frames along channel dimension: (6, 96, 96)
    stacked_img = torch.cat(imgs, dim=0)
    
    # Apply ImageNet normalization per 3-channel group
    normalized_chunks = []
    for i in range(len(image_buffer)):
        chunk = stacked_img[i*3:(i+1)*3, :, :]
        normalized_chunks.append(transform_fn(chunk))
    
    img_tensor = torch.cat(normalized_chunks, dim=0).unsqueeze(0).to(device)
    
    # State: List of (2,) -> (B, 4)
    state_tensor = torch.from_numpy(np.concatenate(state_buffer)).float().unsqueeze(0).to(device)
    
    return img_tensor, state_tensor

# ==========================================
# 3. DAgger Loop
# ==========================================
def run_dagger_episode(env, model, transform_fn, stats, controller, recorder, env_seed, device, step_pbar, n_frames, frame_gap):
    obs, _ = env.reset(seed=env_seed)
    controller.reset()
    recorder.reset()
    step_pbar.reset()

    # Buffers to hold history for frame stacking
    # We store observations every 'frame_gap' steps or just maintain a queue
    # To match your training script's [-9, 0] gap:
    image_history = collections.deque(maxlen=frame_gap + 1)
    state_history = collections.deque(maxlen=frame_gap + 1)

    terminated = truncated = success = quit_requested = False
    clock = pygame.time.Clock()
    step = 0

    while not (terminated or truncated):
        events = controller.handle_events()
        if events.get("quit", False):
            quit_requested = True; break

        agent_pos = obs["agent_pos"] if isinstance(obs, dict) else obs
        image_array = get_observation_image(env)

        # Update buffers
        image_history.append(image_array)
        state_history.append(agent_pos)

        # We need at least 'frame_gap + 1' frames to have the t-9 and t frames
        if len(image_history) < (frame_gap + 1):
            # During warm-up, we just take expert-ish steps or identity steps
            action = agent_pos 
            is_human = True
        else:
            # 1. Prepare Stacked Input: [history[0], history[-1]] -> frames t-9 and t
            input_imgs = [image_history[0], image_history[-1]]
            input_states = [state_history[0], state_history[-1]]
            
            img_t, state_raw_t = preprocess_stacked_input(input_imgs, input_states, transform_fn, device)

            # 2. Decision Logic
            controller.try_activate_human_control(agent_pos)
            
            if controller.state == ControlState.HUMAN_CONTROL:
                action = controller.get_human_action(agent_pos)
                is_human = True
            else:
                # Normalize state using training stats
                state_norm = (state_raw_t - stats["state_mean"]) / stats["state_std"]
                with torch.no_grad():
                    pred_norm = model(img_t, state_norm)
                
                # Unnormalize action
                action = (pred_norm * stats["action_std"]) + stats["action_mean"]
                action = action.squeeze(0).cpu().numpy().astype(np.float32)
                action = np.clip(action, 0, 512)
                is_human = False

        next_obs, reward, terminated, truncated, info = env.step(action)
        
        recorder.record_step(
            obs_state=agent_pos, raw_action=action, reward=float(reward),
            done=bool(terminated or truncated), success=bool(info.get("is_success", False)),
            is_human=is_human, image=image_array
        )

        obs = next_obs
        step += 1
        step_pbar.update(1)
        if step >= FLAGS.max_steps: truncated = True

        env.render()
        draw_status_overlay(env, controller.state, env_seed, 0, step, FLAGS.max_steps, agent_pos, False)
        clock.tick(FLAGS.fps)

    return terminated, truncated, any(recorder.success), any(recorder.is_human), quit_requested

def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(FLAGS.model_path, map_location=device)
    config = checkpoint["config"]
    
    # Load model with correct stacking config
    model = BehavioralCloningPolicy(
        n_frames=config["n_frames"], 
        hidden_dim=config["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Expand stats to handle concatenated states (4 dims if n_frames=2)
    # We repeat the 2D stats to cover the stacked 4D vector
    s_mean = np.tile(checkpoint["state_mean"], config["n_frames"])
    s_std = np.tile(checkpoint["state_std"], config["n_frames"])
    
    stats = {
        "state_mean": torch.tensor(s_mean, device=device),
        "state_std": torch.tensor(s_std, device=device),
        "action_mean": torch.tensor(checkpoint["action_mean"], device=device),
        "action_std": torch.tensor(checkpoint["action_std"], device=device),
    }

    transform_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    env = gym.make("gym_pusht/PushT-v0", render_mode="human")
    controller = InterventionController(activation_radius=FLAGS.activation_radius)
    recorder = TrajectoryRecorder()
    saver = EpisodeSaver(FLAGS.output_dir)

    seeds = range(10) # Simplified seed loop
    for seed in tqdm(seeds):
        res = run_dagger_episode(
            env, model, transform_fn, stats, controller, recorder, seed, device, 
            tqdm(total=FLAGS.max_steps, leave=False), config["n_frames"], config["frame_gap"]
        )
        if res[4]: break # Quit
        
        data = recorder.finalize(seed, 0, -1, res[0], res[1], res[2])
        saver.save(data, recorder.get_images(), seed, 0, res[2], res[3], True)

if __name__ == "__main__":
    app.run(main)