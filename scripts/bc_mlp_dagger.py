"""
Collect human-intervention trajectories for DAgger using a frame-stacking ResNet BC model.
"""

import os
import sys
import warnings
from typing import Dict, List
from collections import deque

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import gymnasium as gym
import pygame
import torch
import torch.nn as nn
import torchvision.transforms as T
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
from scripts.bc_mlp_train import BehavioralCloningPolicy

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/bc_mlp/best.pt", "Path to pretrained model")
flags.DEFINE_string("output_dir", "data/dagger1", "Directory to save collected data")
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to collect")
flags.DEFINE_integer("fps", 10, "Control frequency")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_float("activation_radius", 30.0, "Mouse threshold")
flags.DEFINE_integer("start_seed", 0, "Starting seed for deterministic sequences")
flags.DEFINE_boolean("random_seeds", True, "Sample random seeds instead of using start_seed sequence")


def get_latest_agent_pos(obs: Dict) -> np.ndarray:
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    return agent_pos if agent_pos.ndim == 1 else agent_pos[-1]


def run_dagger_episode(
    env, model, base_transform, norm_transform, stats, 
    controller, recorder, env_seed, device, step_pbar, buffer_size
):
    obs, _ = env.reset(seed=env_seed)
    controller.reset()
    recorder.reset()
    step_pbar.reset()

    # Buffers to hold history for frame stacking [t-9, ..., t]
    image_buffer = deque(maxlen=buffer_size)
    state_buffer = deque(maxlen=buffer_size)

    # Pre-fill buffers with the initial state to avoid cold-start issues
    init_pos = get_latest_agent_pos(obs)
    init_img = base_transform(get_observation_image(env))
    for _ in range(buffer_size):
        image_buffer.append(init_img)
        state_buffer.append(torch.tensor(init_pos, dtype=torch.float32))

    terminated = truncated = quit_requested = False
    clock = pygame.time.Clock()
    step = 0

    while not (terminated or truncated):
        events = controller.handle_events()
        if events.get("quit", False):
            quit_requested = True; break

        # Allow 'R' to return to model control
        if controller.state == ControlState.HUMAN_CONTROL:
            if pygame.key.get_pressed()[pygame.K_r]:
                controller.state = ControlState.MODEL_CONTROL

        agent_pos = get_latest_agent_pos(obs)
        image_array = get_observation_image(env)
        
        # Update rolling history
        image_buffer.append(base_transform(image_array))
        state_buffer.append(torch.tensor(agent_pos, dtype=torch.float32))

        # Check for intervention
        if controller.state != ControlState.HUMAN_CONTROL:
            controller.try_activate_human_control(agent_pos)

        if controller.state == ControlState.HUMAN_CONTROL:
            action = controller.get_human_action(agent_pos)
            is_human = True
        else:
            # 1. Prepare Stacked Inputs (t-9 and t)
            img_stack = torch.cat([norm_transform(image_buffer[0]), 
                                   norm_transform(image_buffer[-1])], dim=0).unsqueeze(0).to(device)
            
            # 2. Prepare Stacked States (Normalize each 2D state, then flatten to 1x4)
            s_t9_norm = (state_buffer[0].to(device) - stats["s_mean"]) / stats["s_std"]
            s_t_norm = (state_buffer[-1].to(device) - stats["s_mean"]) / stats["s_std"]
            state_stack_norm = torch.cat([s_t9_norm, s_t_norm], dim=0).view(1, -1)

            with torch.no_grad():
                pred_norm = model(img_stack, state_stack_norm)
            
            # 3. Unnormalize Action
            action_tensor = (pred_norm * stats["a_std"]) + stats["a_mean"]
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model from {FLAGS.model_path} onto {device}...")
    
    checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 256))
    n_frames = int(config.get("n_frames", 2))
    frame_gap = int(config.get("frame_gap", 9))
    buffer_size = frame_gap + 1 
    
    model = BehavioralCloningPolicy(
        state_dim=2, action_dim=2, hidden_dim=hidden_dim, n_frames=n_frames
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = {
        "s_mean": torch.tensor(checkpoint["state_mean"][:2], dtype=torch.float32).to(device),
        "s_std": torch.tensor(checkpoint["state_std"][:2], dtype=torch.float32).to(device),
        "a_mean": torch.tensor(checkpoint["action_mean"], dtype=torch.float32).to(device),
        "a_std": torch.tensor(checkpoint["action_std"], dtype=torch.float32).to(device),
    }

    base_transform = T.Compose([T.ToTensor(), T.Resize((96, 96), antialias=True)])
    norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    env = gym.make(
        "gym_pusht/PushT-v0", 
        obs_type="environment_state_agent_pos",
        render_mode="human",
        visualization_width=int(512 * FLAGS.window_scale)
    )
    
    controller = InterventionController(activation_radius=FLAGS.activation_radius, window_scale=FLAGS.window_scale)
    recorder = TrajectoryRecorder()
    saver = EpisodeSaver(FLAGS.output_dir)

    # --- RANDOM SEED LOGIC ---
    if FLAGS.random_seeds:
        # Use high range to avoid overlapping with common 0-100 training seeds
        seeds = np.random.randint(0, 2**31 - 1, size=FLAGS.num_seeds).tolist()
    else:
        seeds = list(range(FLAGS.start_seed, FLAGS.start_seed + FLAGS.num_seeds))

    print(f"Collecting {len(seeds)} episodes. Random mode: {FLAGS.random_seeds}")

    for seed in tqdm(seeds, desc="Episodes"):
        res = run_dagger_episode(
            env, model, base_transform, norm_transform, stats, 
            controller, recorder, int(seed), device, 
            tqdm(total=FLAGS.max_steps, leave=False), buffer_size
        )
        
        if res[4]: # quit_requested
            print("\nCollection aborted by user.")
            break 
        
        # data = recorder.finalize(env_seed, trial_idx, policy_seed, ...)
        data = recorder.finalize(int(seed), 0, -1, res[0], res[1], res[2])
        saver.save(data, recorder.get_images(), int(seed), 0, res[2], res[3], True)
        
    env.close()


if __name__ == "__main__":
    app.run(main)