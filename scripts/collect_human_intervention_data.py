#!/usr/bin/env python3
"""Collect human-intervention trajectories for DAgger using a pretrained BC model.

Behavior:
- Model controls by default.
- Human can intervene by moving/clicking near the agent.
- Press `R` to return to model control after intervening.
- Press `Q` to quit.

Saved data follows the repository per-step schema via TrajectoryRecorder and EpisodeSaver.
"""

import os
import sys
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import gymnasium as gym
import pygame
import torch
import torch.nn as nn
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
from envs.frame_stack_wrapper import FrameStackWrapperEnv
from data.trajectory_recorder import TrajectoryRecorder
from data.episode_saver import EpisodeSaver

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/pretrain/best.pt", "Path to pretrained model checkpoint")
flags.DEFINE_string("output_dir", "data/dagger", "Directory to save collected DAgger data")
flags.DEFINE_integer("start_seed", 0, "Starting seed (ignored if --seeds is provided)")
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to collect (ignored if --seeds is provided)")
flags.DEFINE_string("seeds", None, "Comma-separated explicit seed list (overrides start_seed/num_seeds)")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>=1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")
flags.DEFINE_bool("save_images", True, "Save image observations")
flags.DEFINE_float("activation_radius", 30.0, "Mouse proximity threshold for human activation")
flags.DEFINE_integer("policy_seed", -1, "Policy seed metadata to save")
flags.DEFINE_integer("n_frames", 2, "Frame stack size")
flags.DEFINE_integer("frame_gap", 1, "Frame stack temporal gap")


class BehavioralCloningPolicy(nn.Module):
    """Vision + state BC policy architecture matching standard_bc_training.py."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.action_head = nn.Sequential(
            nn.Linear(512 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_features = self.vision_backbone(image).view(image.size(0), -1)
        state_features = self.state_encoder(state)
        return self.action_head(torch.cat([image_features, state_features], dim=1))


def parse_seed_list() -> List[int]:
    if FLAGS.seeds:
        seeds = [int(seed.strip()) for seed in FLAGS.seeds.split(",") if seed.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integers were parsed")
        return seeds
    return list(range(FLAGS.start_seed, FLAGS.start_seed + FLAGS.num_seeds))


def get_latest_agent_pos(obs: Dict) -> np.ndarray:
    return np.asarray(obs["agent_pos"][-1], dtype=np.float32)


def run_dagger_episode(
    env,
    model: nn.Module,
    preprocess,
    stats: Dict[str, torch.Tensor],
    controller: InterventionController,
    recorder: TrajectoryRecorder,
    env_seed: int,
    device: torch.device,
    step_pbar: tqdm,
):
    obs, _ = env.reset(seed=env_seed)
    controller.reset()
    controller.state = ControlState.MODEL_CONTROL
    recorder.reset()
    step_pbar.reset()

    terminated = False
    truncated = False
    success = False
    quit_requested = False
    clock = pygame.time.Clock()
    step = 0

    while not (terminated or truncated):
        events = controller.handle_events()
        if events.get("quit", False):
            quit_requested = True
            break

        if controller.state == ControlState.HUMAN_CONTROL:
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_r]:
                controller.state = ControlState.MODEL_CONTROL

        agent_pos = get_latest_agent_pos(obs)
        image_array = get_observation_image(env)

        if controller.state != ControlState.HUMAN_CONTROL:
            controller.try_activate_human_control(agent_pos)

        if controller.state == ControlState.HUMAN_CONTROL:
            action = controller.get_human_action(agent_pos)
            is_human = True
        else:
            state_tensor = torch.from_numpy(agent_pos).float().unsqueeze(0).to(device)
            image_tensor = preprocess(image_array).unsqueeze(0).to(device)

            state_norm = (state_tensor - stats["state_mean"]) / stats["state_std"]
            with torch.no_grad():
                pred_norm = model(image_tensor, state_norm)
            pred_action = (pred_norm * stats["action_std"]) + stats["action_mean"]
            action = pred_action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            action = np.clip(action, 0.0, 512.0)
            is_human = False

        if action is None:
            env.render()
            draw_status_overlay(
                env=env,
                state=controller.state,
                env_seed=env_seed,
                trial_idx=0,
                step=step,
                max_steps=FLAGS.max_steps,
                agent_pos=agent_pos,
                is_pure_teleop=False,
            )
            clock.tick(FLAGS.fps)
            continue

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_success = bool(info.get("is_success", terminated)) if isinstance(info, dict) else bool(terminated)
        success = success or step_success

        recorder.record_step(
            obs_state=agent_pos,
            raw_action=action,
            reward=float(reward),
            done=done,
            success=step_success,
            is_human=is_human,
            image=image_array if FLAGS.save_images else None,
        )

        obs = next_obs
        step += 1
        step_pbar.update(1)

        if step >= FLAGS.max_steps and not done:
            truncated = True

        env.render()
        draw_status_overlay(
            env=env,
            state=controller.state,
            env_seed=env_seed,
            trial_idx=0,
            step=step,
            max_steps=FLAGS.max_steps,
            agent_pos=agent_pos,
            is_pure_teleop=False,
        )
        clock.tick(FLAGS.fps)

    had_intervention = bool(any(recorder.is_human))
    return terminated, truncated, success, had_intervention, quit_requested


def main(_):
    if FLAGS.window_scale < 1.0:
        raise ValueError("window_scale must be >= 1.0")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if not os.path.exists(FLAGS.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {FLAGS.model_path}")

    checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 256))

    model = BehavioralCloningPolicy(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = {
        "state_mean": torch.tensor(checkpoint["state_mean"], dtype=torch.float32, device=device),
        "state_std": torch.tensor(checkpoint["state_std"], dtype=torch.float32, device=device),
        "action_mean": torch.tensor(checkpoint["action_mean"], dtype=torch.float32, device=device),
        "action_std": torch.tensor(checkpoint["action_std"], dtype=torch.float32, device=device),
    }

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((96, 96), antialias=True),
        ]
    )

    window_size = int(512 * FLAGS.window_scale)
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="human",
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)
    env = FrameStackWrapperEnv(env, n_frames=FLAGS.n_frames, gap=FLAGS.frame_gap)

    controller = InterventionController(
        activation_radius=FLAGS.activation_radius,
        window_scale=FLAGS.window_scale,
    )
    recorder = TrajectoryRecorder()
    saver = EpisodeSaver(FLAGS.output_dir)

    seeds = parse_seed_list()

    print("=" * 60)
    print("DAgger Human-Intervention Collection")
    print("=" * 60)
    print(f"Model: {FLAGS.model_path}")
    print(f"Output dir: {FLAGS.output_dir}")
    print(f"Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"FPS: {FLAGS.fps} | Device: {device}")
    print("Controls: move/click near agent to intervene, R=resume model, Q=quit")
    print("=" * 60)

    seed_pbar = tqdm(total=len(seeds), desc="Seeds", position=0)
    step_pbar = tqdm(total=FLAGS.max_steps, desc="Steps", position=1, leave=False)

    saved_count = 0
    quit_requested = False

    for env_seed in seeds:
        step_pbar.set_description(f"Seed {env_seed} (dagger)")

        terminated, truncated, success, had_intervention, quit_requested = run_dagger_episode(
            env=env,
            model=model,
            preprocess=preprocess,
            stats=stats,
            controller=controller,
            recorder=recorder,
            env_seed=env_seed,
            device=device,
            step_pbar=step_pbar,
        )

        if quit_requested:
            break

        data = recorder.finalize(
            env_seed=env_seed,
            trial_idx=0,
            policy_seed=FLAGS.policy_seed,
            terminated=terminated,
            truncated=truncated,
            success=success,
        )

        saver.save(
            data=data,
            images=recorder.get_images(),
            env_seed=env_seed,
            trial_idx=0,
            success=success,
            had_intervention=had_intervention,
            save_images=FLAGS.save_images,
        )

        saved_count += 1
        status = "SUCCESS" if success else "FAIL"
        mode = "HUMAN+MODEL" if had_intervention else "MODEL-ONLY"
        seed_pbar.set_postfix_str(f"Last: {status} ({mode})")
        seed_pbar.update(1)

    step_pbar.close()
    seed_pbar.close()
    env.close()

    if quit_requested:
        print("\nQuit requested. Exiting early.")

    print(f"Saved episodes: {saved_count}")


if __name__ == "__main__":
    app.run(main)
