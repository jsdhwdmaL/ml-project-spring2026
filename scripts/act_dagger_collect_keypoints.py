"""Collect human-intervention DAgger trajectories for the 18-D keypoints ACT model.

Mirrors ``scripts/act_dagger_collect_og_data.py`` but targets the policy
trained by ``scripts/act_train_keypoints.py``: an 18-D state
``concat(agent_pos, environment_state)`` matching gym_pusht's
``environment_state_agent_pos`` observation, with **min-max** normalization
and **no image input**.

Episodes are routed by :class:`EpisodeSaver` into:
  - ``human_intervention/`` — any step had human control
  - ``rejection_sample/``   — fully autonomous + success
  - ``failed_autonomous/``  — fully autonomous + failure
"""

import os
import sys
import warnings
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import gymnasium as gym
import pygame
import torch
from absl import app, flags
from tqdm import tqdm

import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
from envs.interactive_utils import (
    ControlState,
    InterventionController,
    draw_status_overlay,
)
from data.trajectory_recorder import TrajectoryRecorder
from data.episode_saver import EpisodeSaver
from models.act import ACTPolicy

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/act_keypoints/best.pt", "Path to pretrained 18-D keypoints ACT checkpoint")
flags.DEFINE_string("output_dir", "data/act_dagger_keypoints", "Directory to save collected data")
flags.DEFINE_integer("num_seeds", 10, "Number of seeds to collect")
flags.DEFINE_integer("fps", 10, "Control frequency")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_float("activation_radius", 30.0, "Mouse threshold for human takeover (pixels around agent_pos)")
flags.DEFINE_integer("start_seed", 0, "Starting seed for deterministic sequences")
flags.DEFINE_boolean("random_seeds", True, "Sample random seeds instead of using start_seed sequence")
flags.DEFINE_boolean("save_images", False, "Save image observations (default off; keypoints model is state-only)")
flags.DEFINE_boolean("temporal_agg", True, "Enable temporal ensembling (query model every step, blend predictions)")
flags.DEFINE_float("ensemble_decay", 0.01, "Override temporal ensembling decay; <0 uses checkpoint")
flags.DEFINE_integer(
    "query_frequency",
    -1,
    "When --notemporal_agg, how many steps to execute from each predicted chunk before re-querying. "
    "<0 uses the model's full horizon (LeRobot's n_action_steps semantics).",
)
flags.DEFINE_float(
    "success_threshold",
    0.9,
    "Coverage fraction in (0, 1] required to count as success. "
    "Note: gym_pusht auto-terminates at 0.95, so values >0.95 are clamped to 0.95.",
)


def build_state_vector(obs: dict) -> np.ndarray:
    """Compose the 18-D state: concat(agent_pos, environment_state)."""
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
    env_state = np.asarray(obs["environment_state"], dtype=np.float32).reshape(-1)
    return np.concatenate([agent_pos, env_state], axis=0)


def ensemble_current_action(
    t_step: int,
    predictions: List[Tuple[int, np.ndarray]],
    horizon: int,
    decay: float,
) -> np.ndarray:
    """Exponential-decay blend of overlapping chunk predictions covering ``t_step``."""
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


def predict_chunk(
    model: ACTPolicy,
    state_vec: np.ndarray,
    state_min: torch.Tensor,
    state_max: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run a single forward pass and return an un-normalized (H, action_dim) chunk."""
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
    state_norm = 2.0 * (state_tensor - state_min) / (state_max - state_min) - 1.0
    with torch.no_grad():
        pred_norm, _, _ = model(None, state_norm, action_chunk=None)
    pred = (pred_norm + 1.0) * 0.5 * (
        action_max.view(1, 1, -1) - action_min.view(1, 1, -1)
    ) + action_min.view(1, 1, -1)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)


def run_dagger_episode(
    env,
    model,
    stats,
    controller,
    recorder,
    env_seed,
    device,
    step_pbar,
    horizon,
    temporal_agg,
    ensemble_decay,
    query_frequency,
    state_dim,
    action_dim,
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
    cached_chunk: Optional[np.ndarray] = None
    last_query_step = -1

    while not (terminated or truncated):
        events = controller.handle_events()
        if events.get("quit", False):
            quit_requested = True
            break

        if controller.state == ControlState.HUMAN_CONTROL:
            if pygame.key.get_pressed()[pygame.K_r]:
                controller.state = ControlState.MODEL_CONTROL

        state_vec = build_state_vector(obs)
        if state_vec.shape[0] != state_dim:
            raise ValueError(
                f"env produced state of length {state_vec.shape[0]} but model expects {state_dim}"
            )
        agent_pos = state_vec[:2]

        if controller.state != ControlState.HUMAN_CONTROL:
            controller.try_activate_human_control(agent_pos)

        if controller.state == ControlState.HUMAN_CONTROL:
            action = controller.get_human_action(agent_pos)
            is_human = True
        else:
            if temporal_agg:
                pred_chunk_np = predict_chunk(
                    model, state_vec,
                    stats["s_min"], stats["s_max"],
                    stats["a_min"], stats["a_max"],
                    device,
                )
                chunk_predictions.append((step, pred_chunk_np))
                action = ensemble_current_action(
                    step, chunk_predictions, horizon=horizon, decay=ensemble_decay
                )
            else:
                # Open-loop: re-query every `query_frequency` autonomous steps
                if cached_chunk is None or (step - last_query_step) >= query_frequency:
                    cached_chunk = predict_chunk(
                        model, state_vec,
                        stats["s_min"], stats["s_max"],
                        stats["a_min"], stats["a_max"],
                        device,
                    )
                    last_query_step = step
                offset = step - last_query_step
                action = cached_chunk[offset]
            action = np.clip(action, 0.0, 512.0).astype(np.float32)
            is_human = False

        next_obs, reward, terminated, truncated, info = env.step(action)

        coverage = float(info.get("coverage", 0.0)) if isinstance(info, dict) else 0.0
        if coverage > max_coverage:
            max_coverage = coverage
        step_success = max_coverage >= success_thresh
        if step_success:
            truncated = True

        recorder.record_step(
            obs_state=state_vec,
            raw_action=action,
            reward=float(reward),
            done=bool(terminated or truncated),
            success=bool(step_success),
            is_human=is_human,
            image=None,  # keypoints model is state-only
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

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading ACT keypoints model from {FLAGS.model_path} onto {device}...")

    checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    norm_mode = str(config.get("norm_mode", "min_max"))
    if norm_mode != "min_max":
        raise ValueError(
            f"Expected norm_mode='min_max' for keypoints checkpoints, got '{norm_mode}'. "
            f"Use scripts/act_dagger_collect_og_data.py for mean/std checkpoints."
        )
    for key in ("state_min", "state_max", "action_min", "action_max"):
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint missing required key '{key}'. This script is for keypoints "
                f"checkpoints saved by scripts/act_train_keypoints.py."
            )

    horizon = int(config.get("horizon", 16))
    hidden_dim = int(config.get("hidden_dim", 512))
    latent_dim = int(config.get("latent_dim", 32))
    nhead = int(config.get("nhead", 8))
    num_encoder_layers = int(config.get("num_encoder_layers", 4))
    num_decoder_layers = int(config.get("num_decoder_layers", 4))
    ckpt_decay = float(config.get("ensemble_decay", 0.05))
    ensemble_decay = ckpt_decay if FLAGS.ensemble_decay < 0 else FLAGS.ensemble_decay

    state_min_np = np.asarray(checkpoint["state_min"], dtype=np.float32)
    state_max_np = np.asarray(checkpoint["state_max"], dtype=np.float32)
    action_min_np = np.asarray(checkpoint["action_min"], dtype=np.float32)
    action_max_np = np.asarray(checkpoint["action_max"], dtype=np.float32)
    state_dim = int(state_min_np.shape[0])
    action_dim = int(action_min_np.shape[0])
    if state_dim != 18:
        raise ValueError(
            f"This script targets the LeRobot pusht_keypoints ACT model (state_dim=18), "
            f"but checkpoint has state_dim={state_dim}."
        )

    temporal_agg: bool = FLAGS.temporal_agg
    if temporal_agg:
        query_frequency = 1
    elif FLAGS.query_frequency < 0:
        query_frequency = horizon
    else:
        query_frequency = max(1, min(FLAGS.query_frequency, horizon))

    model = ACTPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        use_vision=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = {
        "s_min": torch.tensor(state_min_np, dtype=torch.float32, device=device),
        "s_max": torch.tensor(state_max_np, dtype=torch.float32, device=device),
        "a_min": torch.tensor(action_min_np, dtype=torch.float32, device=device),
        "a_max": torch.tensor(action_max_np, dtype=torch.float32, device=device),
    }

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="human",
        visualization_width=int(512 * FLAGS.window_scale),
        visualization_height=int(512 * FLAGS.window_scale),
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    controller = InterventionController(
        activation_radius=FLAGS.activation_radius, window_scale=FLAGS.window_scale
    )
    recorder = TrajectoryRecorder(state_dim=state_dim, act_dim=action_dim)
    saver = EpisodeSaver(FLAGS.output_dir)

    if FLAGS.random_seeds:
        seeds = np.random.randint(0, 2**31 - 1, size=FLAGS.num_seeds).tolist()
    else:
        seeds = list(range(FLAGS.start_seed, FLAGS.start_seed + FLAGS.num_seeds))

    print(f"Collecting {len(seeds)} episodes. Random mode: {FLAGS.random_seeds}")
    mode_str = (
        f"temporal_agg=ON (decay={ensemble_decay})"
        if temporal_agg
        else f"temporal_agg=OFF (open-loop, query_frequency={query_frequency})"
    )
    print(
        f"H={horizon} | {mode_str} | state_dim={state_dim} | "
        f"action_dim={action_dim} | success_threshold={success_thresh:.2f}"
    )

    seed_pbar = tqdm(total=len(seeds), desc="Episodes", position=0)
    step_pbar = tqdm(total=FLAGS.max_steps, desc="Steps", position=1, leave=False)

    for seed in seeds:
        step_pbar.set_description(f"Seed {seed}")
        terminated, truncated, success, had_intervention, quit_requested, max_coverage = run_dagger_episode(
            env=env,
            model=model,
            stats=stats,
            controller=controller,
            recorder=recorder,
            env_seed=int(seed),
            device=device,
            step_pbar=step_pbar,
            horizon=horizon,
            temporal_agg=temporal_agg,
            ensemble_decay=ensemble_decay,
            query_frequency=query_frequency,
            state_dim=state_dim,
            action_dim=action_dim,
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
