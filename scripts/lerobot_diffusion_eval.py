#!/usr/bin/env python3
"""Evaluate the Hugging Face diffusion policy for PushT (lerobot/diffusion_pusht).

Loads the pretrained policy from the Hub (or a local save_pretrained directory) with
LeRobot preprocessors, and rolls out gym-pusht like scripts/lerobot_act_eval.py.

Training dataset (per model card): lerobot/pusht — optionally print basic dataset stats
at startup (--print_dataset_info).

Requires: lerobot, and for inference the diffusion stack (e.g. diffusers); install
project dependencies first.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import gymnasium as gym
import numpy as np
import pygame
import torch
import torchvision.transforms as T
from absl import app, flags

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gym_pusht  # noqa: F401
from envs.interactive_utils import ControlState, draw_status_overlay, get_observation_image
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_id",
    "lerobot/diffusion_pusht",
    "Hugging Face model repo id or path to a local pretrained_model directory",
)
flags.DEFINE_string(
    "dataset_id",
    "lerobot/pusht",
    "Hugging Face dataset id used to train lerobot/diffusion_pusht (for --print_dataset_info)",
)
flags.DEFINE_boolean(
    "print_dataset_info",
    False,
    "Download/load LeRobotDataset(dataset_id) and print frame/episode counts (first run may download)",
)
flags.DEFINE_integer("num_seeds", 5, "Number of episodes to evaluate")
flags.DEFINE_boolean("random_seeds", True, "Sample random seeds instead of using 0..num_seeds-1")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 400, "Maximum steps per episode")
flags.DEFINE_string("device", "auto", "Device: auto|cuda|mps|cpu")
flags.DEFINE_float("action_min", 0.0, "Minimum action value after postprocessing")
flags.DEFINE_float("action_max", 512.0, "Maximum action value after postprocessing")


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device 'mps' is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name}. Use one of: auto, cuda, mps, cpu.")


def get_agent_pos_from_obs(obs: Dict) -> np.ndarray:
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    if agent_pos.ndim == 1:
        return agent_pos
    return agent_pos[-1]


def set_processor_device(processor, device: str) -> None:
    for step in getattr(processor, "steps", []):
        if hasattr(step, "device"):
            step.device = device


def load_policy_and_processors(model_ref: str, device: torch.device):
    """Load DiffusionPolicy + Hub/local processor JSON (policy_preprocessor.json, etc.)."""
    root = Path(model_ref)
    if root.is_dir():
        policy = DiffusionPolicy.from_pretrained(str(root.resolve()))
        processor_root = str(root.resolve())
    elif root.is_file() and root.suffix == ".pt":
        checkpoint = torch.load(root, map_location="cpu", weights_only=False)
        if "policy_config" not in checkpoint or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint {root} missing policy_config and/or model_state_dict"
            )
        cfg = DiffusionConfig(**checkpoint["policy_config"])
        if hasattr(cfg, "device"):
            cfg.device = device.type
        policy = DiffusionPolicy(cfg)
        policy.load_state_dict(checkpoint["model_state_dict"])
        processor_root = str(root.parent)
    else:
        # Hugging Face hub id (e.g. lerobot/diffusion_pusht); processors load from the same id
        policy = DiffusionPolicy.from_pretrained(model_ref)
        processor_root = model_ref

    required = ("policy_preprocessor.json", "policy_postprocessor.json")
    proc_path = Path(processor_root)
    if proc_path.exists():
        missing = [name for name in required if not (proc_path / name).exists()]
        if missing:
            raise ValueError(
                f"Missing processor files in {processor_root}: {missing}. "
                "Use a Hub model or save_pretrained output that includes policy_*processor.json."
            )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=processor_root,
        preprocessor_overrides={
            "device_processor": {
                "device": device.type,
                "float_dtype": None,
            }
        },
        postprocessor_overrides={
            "device_processor": {
                "device": "cpu",
                "float_dtype": None,
            }
        },
    )
    set_processor_device(preprocessor, device.type)
    policy.to(device)
    policy.eval()
    return policy, preprocessor, postprocessor


def print_lerobot_dataset_summary(dataset_id: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"\nLoading LeRobot dataset {dataset_id!r} (cached after first download)...")
    ds = LeRobotDataset(dataset_id)
    print(f"  len(dataset) = {len(ds)} frames")
    meta = getattr(ds, "meta", None)
    if meta is not None:
        if hasattr(meta, "total_episodes"):
            print(f"  meta.total_episodes = {meta.total_episodes!r}")
        if hasattr(meta, "total_frames"):
            print(f"  meta.total_frames = {meta.total_frames!r}")
        if hasattr(meta, "fps"):
            print(f"  meta.fps = {meta.fps!r}")
    print()


def main(_):
    if FLAGS.window_scale < 1.0:
        raise ValueError("window_scale must be >= 1.0")
    if FLAGS.num_seeds <= 0:
        raise ValueError("num_seeds must be > 0")
    if FLAGS.max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if FLAGS.action_min > FLAGS.action_max:
        raise ValueError("action_min must be <= action_max")

    device = resolve_device(FLAGS.device)

    if FLAGS.print_dataset_info:
        print_lerobot_dataset_summary(FLAGS.dataset_id)

    print(f"Loading Diffusion policy from {FLAGS.model_id!r} onto {device}...")
    policy, preprocessor, postprocessor = load_policy_and_processors(FLAGS.model_id, device)

    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((96, 96), antialias=True),
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

    print("\nStarting LeRobot Diffusion evaluation...")
    success_count = 0

    if FLAGS.random_seeds:
        seeds = np.random.randint(0, 2**31, size=FLAGS.num_seeds).tolist()
    else:
        seeds = list(range(FLAGS.num_seeds))

    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        policy.reset()

        step = 0
        terminated = False
        truncated = False
        success = False
        clock = pygame.time.Clock()

        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    print("Evaluation aborted by user.")
                    env.close()
                    return

            agent_pos = get_agent_pos_from_obs(obs)
            img_array = get_observation_image(env)

            batch = {
                "observation.image": image_transform(img_array),
                "observation.state": torch.tensor(agent_pos, dtype=torch.float32),
            }
            batch = preprocessor(batch)

            with torch.no_grad():
                action_norm = policy.select_action(batch)

            action = postprocessor(action_norm)
            action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            action = np.clip(action, FLAGS.action_min, FLAGS.action_max)

            obs, _, terminated, truncated, info = env.step(action)
            step_success = bool(info.get("is_success", terminated)) if isinstance(info, dict) else bool(terminated)
            success = success or step_success

            step += 1
            if step >= FLAGS.max_steps:
                truncated = True

            env.render()
            draw_status_overlay(
                env,
                ControlState.MODEL_CONTROL,
                int(seed),
                0,
                step,
                FLAGS.max_steps,
                agent_pos,
                False,
            )
            clock.tick(FLAGS.fps)

        if success:
            success_count += 1
            print(f"Episode {i + 1}/{FLAGS.num_seeds} (seed={seed}) - SUCCESS ({step} steps)")
        else:
            print(f"Episode {i + 1}/{FLAGS.num_seeds} (seed={seed}) - FAILED ({step} steps)")

    print("=" * 60)
    rate = (success_count / FLAGS.num_seeds) * 100.0
    print(
        f"LeRobot Diffusion evaluation complete. Success: {success_count}/{FLAGS.num_seeds} ({rate:.1f}%)"
    )
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    app.run(main)
