"""Collect human-intervention DAgger trajectories for the 5-D Push-T zarr ACT model.

Mirrors ``scripts/act_dagger_collect.py`` but targets the policy trained by
``scripts/act_train_pusht.py``: a 5-D state ``[agent_xy, block_xy, block_θ]``
matching the local ``data/pusht/pusht_cchi_v7_replay.zarr`` buffer.
"""

import os
import sys
import warnings
from typing import List, Tuple

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

import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
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

flags.DEFINE_string("model_path", "models/act_og_data_50_epochs/latest.pt", "Path to pretrained 5-D ACT checkpoint")
flags.DEFINE_string("output_dir", "data/act_dagger_og_data", "Directory to save collected data")
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to collect")
flags.DEFINE_integer("fps", 10, "Control frequency")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_float("activation_radius", 30.0, "Mouse threshold")
flags.DEFINE_integer("start_seed", 0, "Starting seed for deterministic sequences")
flags.DEFINE_boolean("random_seeds", True, "Sample random seeds instead of using start_seed sequence")
flags.DEFINE_boolean("save_images", True, "Save image observations")
flags.DEFINE_float("ensemble_decay", 0.01, "Override temporal ensembling decay; <0 uses checkpoint")
flags.DEFINE_float(
	"success_threshold",
	0.9,
	"Coverage fraction in (0, 1] required to count as success. "
	"Note: gym_pusht auto-terminates at 0.95, so values >0.95 are clamped to 0.95.",
)


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
	state_dim,
	success_thresh,
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
	max_coverage = 0.0

	chunk_predictions: List[Tuple[int, np.ndarray]] = []

	while not (terminated or truncated):
		events = controller.handle_events()
		if events.get("quit", False):
			quit_requested = True
			break

		if controller.state == ControlState.HUMAN_CONTROL:
			if pygame.key.get_pressed()[pygame.K_r]:
				controller.state = ControlState.MODEL_CONTROL

		# obs_type="state" returns a flat 5-D vector [agent_xy, block_xy, block_theta]
		state_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
		if state_vec.shape[0] != state_dim:
			raise ValueError(
				f"env returned state of length {state_vec.shape[0]} but model expects {state_dim}"
			)
		agent_pos = state_vec[:2]
		image_array = get_observation_image(env)

		if controller.state != ControlState.HUMAN_CONTROL:
			controller.try_activate_human_control(agent_pos)

		if controller.state == ControlState.HUMAN_CONTROL:
			action = controller.get_human_action(agent_pos)
			is_human = True
		else:
			image_tensor = base_transform(image_array)
			image_tensor = norm_transform(image_tensor).unsqueeze(0).to(device)

			state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
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

		coverage = float(info.get("coverage", 0.0)) if isinstance(info, dict) else 0.0
		if coverage > max_coverage:
			max_coverage = coverage
		step_success = max_coverage >= success_thresh
		if step_success:
			truncated = True  # end episode as soon as user threshold is reached

		recorder.record_step(
			obs_state=state_vec,
			raw_action=action,
			reward=float(reward),
			done=bool(terminated or truncated),
			success=bool(step_success),
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

	episode_success = max_coverage >= success_thresh
	had_intervention = any(recorder.is_human)
	return terminated, truncated, episode_success, had_intervention, quit_requested, max_coverage


def main(_):
	if not (0.0 < FLAGS.success_threshold <= 1.0):
		raise ValueError(f"--success_threshold must be in (0, 1], got {FLAGS.success_threshold}")
	if FLAGS.success_threshold > 0.95:
		print(
			f"[warn] success_threshold {FLAGS.success_threshold} > 0.95; gym_pusht "
			f"auto-terminates at 0.95, clamping to 0.95."
		)
		success_thresh = 0.95
	else:
		success_thresh = float(FLAGS.success_threshold)

	device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
	print(f"Loading ACT model from {FLAGS.model_path} onto {device}...")

	checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
	config = checkpoint.get("config", {})

	horizon = int(config.get("horizon", 20))
	hidden_dim = int(config.get("hidden_dim", 256))
	latent_dim = int(config.get("latent_dim", 32))
	nhead = int(config.get("nhead", 8))
	num_encoder_layers = int(config.get("num_encoder_layers", 4))
	num_decoder_layers = int(config.get("num_decoder_layers", 7))
	ckpt_decay = float(config.get("ensemble_decay", 0.05))
	ensemble_decay = ckpt_decay if FLAGS.ensemble_decay < 0 else FLAGS.ensemble_decay

	state_mean_np = np.asarray(checkpoint["state_mean"], dtype=np.float32)
	state_dim = int(state_mean_np.shape[0])
	if state_dim != 5:
		raise ValueError(
			f"This script targets the local Push-T zarr ACT model (state_dim=5), "
			f"but checkpoint has state_dim={state_dim}. "
			f"Use scripts/act_dagger_collect.py for 2-D agent-only models."
		)

	model = ACTPolicy(
		state_dim=state_dim,
		action_dim=2,
		horizon=horizon,
		hidden_dim=hidden_dim,
		latent_dim=latent_dim,
		nhead=nhead,
		num_encoder_layers=num_encoder_layers,
		num_decoder_layers=num_decoder_layers,
		use_vision=True,
	).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	stats = {
		"s_mean": torch.tensor(state_mean_np, dtype=torch.float32, device=device),
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
		obs_type="state",  # 5-D flat numpy: [agent_xy, block_xy, block_theta]
		render_mode="human",
		visualization_width=int(512 * FLAGS.window_scale),
		visualization_height=int(512 * FLAGS.window_scale),
	)
	env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

	controller = InterventionController(activation_radius=FLAGS.activation_radius, window_scale=FLAGS.window_scale)
	recorder = TrajectoryRecorder(state_dim=state_dim, act_dim=2)
	saver = EpisodeSaver(FLAGS.output_dir)

	if FLAGS.random_seeds:
		seeds = np.random.randint(0, 2**31 - 1, size=FLAGS.num_seeds).tolist()
	else:
		seeds = list(range(FLAGS.start_seed, FLAGS.start_seed + FLAGS.num_seeds))

	print(f"Collecting {len(seeds)} episodes. Random mode: {FLAGS.random_seeds}")
	print(
		f"H={horizon} | ensemble_decay={ensemble_decay} | state_dim={state_dim} | "
		f"success_threshold={success_thresh:.2f}"
	)

	seed_pbar = tqdm(total=len(seeds), desc="Episodes", position=0)
	step_pbar = tqdm(total=FLAGS.max_steps, desc="Steps", position=1, leave=False)

	for seed in seeds:
		step_pbar.set_description(f"Seed {seed}")
		terminated, truncated, success, had_intervention, quit_requested, max_coverage = run_dagger_episode(
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
			state_dim=state_dim,
			success_thresh=success_thresh,
		)

		if quit_requested:
			print("\nCollection aborted by user.")
			break

		data = recorder.finalize(int(seed), 0, -1, terminated, truncated, success)
		saver.save(data, recorder.get_images(), int(seed), 0, success, had_intervention, FLAGS.save_images)
		tag = "SUCCESS" if success else "FAILED"
		intervention_tag = " (intervention)" if had_intervention else ""
		print(
			f"Seed {seed} - {tag}{intervention_tag} "
			f"(coverage={max_coverage:.3f}, threshold={success_thresh:.2f})"
		)
		seed_pbar.update(1)

	step_pbar.close()
	seed_pbar.close()
	env.close()


if __name__ == "__main__":
	app.run(main)
