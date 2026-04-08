"""Collect human-intervention trajectories for DAgger using ACT chunk policy."""

import os
import sys
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

import numpy as np
import gymnasium as gym
import pygame
import torch
import torchvision.transforms as T
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
from models.act import ACTPolicy

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/act/best.pt", "Path to pretrained ACT checkpoint")
flags.DEFINE_string("output_dir", "data/act_dagger", "Directory to save collected data")
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to collect")
flags.DEFINE_integer("fps", 10, "Control frequency")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_float("activation_radius", 30.0, "Mouse threshold")
flags.DEFINE_integer("start_seed", 0, "Starting seed for deterministic sequences")
flags.DEFINE_boolean("random_seeds", True, "Sample random seeds instead of using start_seed sequence")
flags.DEFINE_boolean("save_images", True, "Save image observations")
flags.DEFINE_float("ensemble_decay", -1.0, "Override temporal ensembling decay; <0 uses checkpoint")


def get_latest_agent_pos(obs: Dict) -> np.ndarray:
	agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
	return agent_pos if agent_pos.ndim == 1 else agent_pos[-1]


def ensemble_current_action(
	t_step: int,
	predictions: List[Tuple[int, np.ndarray]],
	horizon: int,
	decay: float,
) -> np.ndarray:
	candidates: List[np.ndarray] = []
	weights: List[float] = []

	for start_step, chunk in predictions:
		offset = t_step - start_step
		if 0 <= offset < horizon:
			candidates.append(chunk[offset])
			weights.append(float(np.exp(-decay * offset)))

	if not candidates:
		raise ValueError("No valid chunk predictions available for temporal ensembling")

	stacked = np.stack(candidates, axis=0)
	w = np.asarray(weights, dtype=np.float32)
	w = w / np.sum(w)
	return np.sum(stacked * w[:, None], axis=0)


def run_dagger_episode(
	env,
	model,
	base_transform,
	norm_transform,
	stats,
	controller,
	recorder,
	env_seed,
	device,
	step_pbar,
	horizon,
	ensemble_decay,
):
	obs, _ = env.reset(seed=env_seed)
	controller.reset()
	recorder.reset()
	step_pbar.reset()

	terminated = False
	truncated = False
	quit_requested = False
	clock = pygame.time.Clock()
	step = 0

	chunk_predictions: List[Tuple[int, np.ndarray]] = []

	while not (terminated or truncated):
		events = controller.handle_events()
		if events.get("quit", False):
			quit_requested = True
			break

		if controller.state == ControlState.HUMAN_CONTROL:
			if pygame.key.get_pressed()[pygame.K_r]:
				controller.state = ControlState.MODEL_CONTROL

		agent_pos = get_latest_agent_pos(obs)
		image_array = get_observation_image(env)

		if controller.state != ControlState.HUMAN_CONTROL:
			controller.try_activate_human_control(agent_pos)

		if controller.state == ControlState.HUMAN_CONTROL:
			action = controller.get_human_action(agent_pos)
			is_human = True
		else:
			image_tensor = base_transform(image_array)
			image_tensor = norm_transform(image_tensor).unsqueeze(0).to(device)

			state_tensor = torch.tensor(agent_pos, dtype=torch.float32, device=device).unsqueeze(0)
			state_tensor_norm = (state_tensor - stats["s_mean"]) / stats["s_std"]

			with torch.no_grad():
				pred_norm_chunk, _, _ = model(image_tensor, state_tensor_norm, action_chunk=None)

			pred_chunk = (pred_norm_chunk * stats["a_std"].view(1, 1, -1)) + stats["a_mean"].view(1, 1, -1)
			pred_chunk_np = pred_chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
			chunk_predictions.append((step, pred_chunk_np))

			action = ensemble_current_action(step, chunk_predictions, horizon=horizon, decay=ensemble_decay)
			action = np.clip(action, 0.0, 512.0)
			is_human = False

		next_obs, reward, terminated, truncated, info = env.step(action)

		recorder.record_step(
			obs_state=agent_pos,
			raw_action=action,
			reward=float(reward),
			done=bool(terminated or truncated),
			success=bool(info.get("is_success", False)),
			is_human=is_human,
			image=image_array if FLAGS.save_images else None,
		)

		obs = next_obs
		step += 1
		step_pbar.update(1)
		if step >= FLAGS.max_steps:
			truncated = True

		env.render()
		draw_status_overlay(env, controller.state, env_seed, 0, step, FLAGS.max_steps, agent_pos, False)
		clock.tick(FLAGS.fps)

	return terminated, truncated, any(recorder.success), any(recorder.is_human), quit_requested


def main(_):
	device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
	print(f"Loading ACT model from {FLAGS.model_path} onto {device}...")

	checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
	config = checkpoint.get("config", {})

	horizon = int(config.get("horizon", 50))
	hidden_dim = int(config.get("hidden_dim", 512))
	latent_dim = int(config.get("latent_dim", 32))
	nhead = int(config.get("nhead", 8))
	num_decoder_layers = int(config.get("num_decoder_layers", 2))
	ckpt_decay = float(config.get("ensemble_decay", 0.01))
	ensemble_decay = ckpt_decay if FLAGS.ensemble_decay < 0 else FLAGS.ensemble_decay

	model = ACTPolicy(
		state_dim=2,
		action_dim=2,
		horizon=horizon,
		hidden_dim=hidden_dim,
		latent_dim=latent_dim,
		nhead=nhead,
		num_decoder_layers=num_decoder_layers,
	).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	stats = {
		"s_mean": torch.tensor(checkpoint["state_mean"], dtype=torch.float32, device=device),
		"s_std": torch.tensor(checkpoint["state_std"], dtype=torch.float32, device=device),
		"a_mean": torch.tensor(checkpoint["action_mean"], dtype=torch.float32, device=device),
		"a_std": torch.tensor(checkpoint["action_std"], dtype=torch.float32, device=device),
	}

	base_transform = T.Compose([
		T.ToTensor(),
		T.Resize((96, 96), antialias=True),
	])
	norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	env = gym.make(
		"gym_pusht/PushT-v0",
		obs_type="environment_state_agent_pos",
		render_mode="human",
		visualization_width=int(512 * FLAGS.window_scale),
		visualization_height=int(512 * FLAGS.window_scale),
	)
	env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

	controller = InterventionController(activation_radius=FLAGS.activation_radius, window_scale=FLAGS.window_scale)
	recorder = TrajectoryRecorder()
	saver = EpisodeSaver(FLAGS.output_dir)

	if FLAGS.random_seeds:
		seeds = np.random.randint(0, 2**31 - 1, size=FLAGS.num_seeds).tolist()
	else:
		seeds = list(range(FLAGS.start_seed, FLAGS.start_seed + FLAGS.num_seeds))

	print(f"Collecting {len(seeds)} episodes. Random mode: {FLAGS.random_seeds}")
	print(f"H={horizon} | ensemble_decay={ensemble_decay}")

	seed_pbar = tqdm(total=len(seeds), desc="Episodes", position=0)
	step_pbar = tqdm(total=FLAGS.max_steps, desc="Steps", position=1, leave=False)

	for seed in seeds:
		step_pbar.set_description(f"Seed {seed}")
		terminated, truncated, success, had_intervention, quit_requested = run_dagger_episode(
			env=env,
			model=model,
			base_transform=base_transform,
			norm_transform=norm_transform,
			stats=stats,
			controller=controller,
			recorder=recorder,
			env_seed=int(seed),
			device=device,
			step_pbar=step_pbar,
			horizon=horizon,
			ensemble_decay=ensemble_decay,
		)

		if quit_requested:
			print("\nCollection aborted by user.")
			break

		data = recorder.finalize(int(seed), 0, -1, terminated, truncated, success)
		saver.save(data, recorder.get_images(), int(seed), 0, success, had_intervention, FLAGS.save_images)
		seed_pbar.update(1)

	step_pbar.close()
	seed_pbar.close()
	env.close()


if __name__ == "__main__":
	app.run(main)
