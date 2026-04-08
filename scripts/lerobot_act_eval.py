#!/usr/bin/env python3
"""Evaluate LeRobot ACT policy on PushT with the same environment setup as act_eval."""

import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import os
import sys

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
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/lerobot_act", "Path to model directory or ckpt_step_*.pt checkpoint")
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


def load_policy_and_processors(model_path: Path, device: torch.device):
	if model_path.is_dir():
		policy = ACTPolicy.from_pretrained(str(model_path))
		processor_root = model_path
	elif model_path.is_file() and model_path.suffix == ".pt":
		checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
		if "policy_config" not in checkpoint or "model_state_dict" not in checkpoint:
			raise ValueError(
				f"Checkpoint {model_path} missing required keys: policy_config and model_state_dict"
			)

		cfg = ACTConfig(**checkpoint["policy_config"])
		if hasattr(cfg, "device"):
			cfg.device = device.type
		policy = ACTPolicy(cfg)
		policy.load_state_dict(checkpoint["model_state_dict"])
		processor_root = model_path.parent
	else:
		raise ValueError(
			f"model_path must be a directory (save_pretrained output) or .pt checkpoint, got: {model_path}"
		)

	required_processor_files = [
		"policy_preprocessor.json",
		"policy_postprocessor.json",
	]
	missing = [name for name in required_processor_files if not (processor_root / name).exists()]
	if missing:
		raise ValueError(
			f"Missing processor files in {processor_root}: {missing}. "
			"Expected artifacts saved by scripts/lerobot_act_train.py"
		)

	preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(processor_root))
	set_processor_device(preprocessor, device.type)

	policy.to(device)
	policy.eval()
	return policy, preprocessor, postprocessor


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
	model_path = Path(FLAGS.model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Model path does not exist: {model_path}")

	print(f"Loading LeRobot ACT artifacts from {model_path} onto {device}...")
	policy, preprocessor, postprocessor = load_policy_and_processors(model_path, device)

	image_transform = T.Compose([
		T.ToTensor(),
		T.Resize((96, 96), antialias=True),
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

	print("\nStarting LeRobot ACT evaluation...")
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
				if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
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
	print(f"LeRobot ACT Evaluation Complete! Success Rate: {success_count}/{FLAGS.num_seeds} ({rate:.1f}%)")
	print("=" * 60)
	env.close()


if __name__ == "__main__":
	app.run(main)
