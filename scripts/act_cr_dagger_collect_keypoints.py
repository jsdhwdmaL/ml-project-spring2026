"""CR-DAgger style on-policy delta collector for the 18-D keypoints ACT model.

Forked from ``scripts/act_dagger_collect_keypoints.py`` (which performs hard
takeover on mouse proximity). This script instead implements the *on-policy
delta* half of Compliant Residual DAgger
(https://compliant-residual-dagger.github.io/): the base policy keeps running,
and the human gently nudges it with a soft blend.

Engagement: hold **SHIFT** to engage. Release to return to pure model control.
While engaged, the executed action is

    a_executed = (1 - lambda) * a_base + lambda * a_human

where:
- ``a_base``   is the policy action that would have been executed this step
              (the same temporal-ensembled / open-loop chunk action used in
              autonomous mode).
- ``a_human``  is the current mouse cursor position in env coordinates.
- ``lambda``   is the constant blend weight (--blend_lambda, default 0.5).

Saved trajectory schema is identical to the original collector:
- ``action[i]`` = the executed action a_executed (what the env actually got).
- ``is_human_intervention[i]`` = True for every step SHIFT was held.

This means the existing fine-tune script
(``scripts/act_dagger_finetune_keypoints.py``) consumes the produced npz
without modification, including ``--keep_only_human``.

Episode routing (via :class:`EpisodeSaver`) is unchanged:
  - ``human_intervention/`` — any step had SHIFT engaged
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
from models.act_refactored import ACTLeRobotConfig, ACTLeRobotPolicy

AGENT_POS_DIM = 2
ENV_STATE_DIM = 16

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/act_cr_dagger_keypoints/latest.pt", "Path to pretrained 18-D keypoints ACT checkpoint")
flags.DEFINE_string("output_dir", "data/act_cr_dagger_keypoints", "Directory to save collected data")
flags.DEFINE_integer("num_seeds", 50, "Number of seeds to collect")
flags.DEFINE_integer("fps", 10, "Control frequency")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_integer("start_seed", 0, "Starting seed for deterministic sequences")
flags.DEFINE_boolean("random_seeds", True, "Sample random seeds instead of using start_seed sequence")
flags.DEFINE_boolean("save_images", False, "Save image observations (default off; keypoints model is state-only)")
flags.DEFINE_boolean("temporal_agg", True, "Enable temporal ensembling (query model every step, blend predictions)")
flags.DEFINE_float("ensemble_decay", 0.1, "Override temporal ensembling decay; <0 uses checkpoint")
flags.DEFINE_integer(
    "query_frequency",
    -1,
    "When --notemporal_agg, how many steps to execute from each predicted chunk before re-querying. "
    "<0 uses the model's full horizon (LeRobot's n_action_steps semantics).",
)
flags.DEFINE_float(
    "success_threshold",
    0.95,
    "Coverage fraction in (0, 1] required to count as success. "
    "Note: gym_pusht auto-terminates at 0.95, so values >0.95 are clamped to 0.95.",
)
flags.DEFINE_float(
    "blend_lambda",
    0.5,
    "Blend weight in [0, 1] for CR-DAgger soft correction. "
    "a_executed = (1 - lambda) * a_base + lambda * a_mouse_target. "
    "0 = pure model (engagement is a no-op), 1 = full overwrite.",
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


def split_state_to_obs(state: torch.Tensor) -> dict:
    """Split the 18-D normalized state into the dict ACTLeRobotPolicy expects."""
    return {
        "observation.state": state[..., :AGENT_POS_DIM],
        "observation.environment_state": state[..., AGENT_POS_DIM:],
    }


def predict_chunk(
    model: ACTLeRobotPolicy,
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
        pred_norm, _, _ = model(split_state_to_obs(state_norm), action_chunk=None)
    pred = (pred_norm + 1.0) * 0.5 * (
        action_max.view(1, 1, -1) - action_min.view(1, 1, -1)
    ) + action_min.view(1, 1, -1)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)


def compute_base_action(
    *,
    model,
    state_vec,
    stats,
    device,
    step,
    horizon,
    temporal_agg,
    ensemble_decay,
    query_frequency,
    chunk_predictions,
    open_loop_state,
):
    """Compute the base policy action for the current step.

    Always called (every step), so the blended branch can use a_base too.
    Mutates chunk_predictions / open_loop_state in place to keep
    temporal-ensembling and open-loop caching consistent across steps.
    """
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
        cached_chunk = open_loop_state["cached_chunk"]
        last_query_step = open_loop_state["last_query_step"]
        if cached_chunk is None or (step - last_query_step) >= query_frequency:
            cached_chunk = predict_chunk(
                model, state_vec,
                stats["s_min"], stats["s_max"],
                stats["a_min"], stats["a_max"],
                device,
            )
            last_query_step = step
            open_loop_state["cached_chunk"] = cached_chunk
            open_loop_state["last_query_step"] = last_query_step
        offset = step - last_query_step
        action = cached_chunk[offset]
    return np.clip(action, 0.0, 512.0).astype(np.float32)


def run_cr_dagger_episode(
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
    open_loop_state = {"cached_chunk": None, "last_query_step": -1}

    while not (terminated or truncated):
        events = controller.handle_events()
        if events.get("quit", False):
            quit_requested = True
            break

        state_vec = build_state_vector(obs)
        if state_vec.shape[0] != state_dim:
            raise ValueError(
                f"env produced state of length {state_vec.shape[0]} but model expects {state_dim}"
            )
        agent_pos = state_vec[:2]

        # SHIFT held -> BLENDED_INTERVENTION; released -> MODEL_CONTROL.
        controller.try_activate_blended_control()

        # Always compute a_base every step so blended steps have a valid policy
        # action to blend against, and so temporal ensembling sees uninterrupted
        # observation history.
        a_base = compute_base_action(
            model=model,
            state_vec=state_vec,
            stats=stats,
            device=device,
            step=step,
            horizon=horizon,
            temporal_agg=temporal_agg,
            ensemble_decay=ensemble_decay,
            query_frequency=query_frequency,
            chunk_predictions=chunk_predictions,
            open_loop_state=open_loop_state,
        )

        if controller.state == ControlState.BLENDED_INTERVENTION:
            action = controller.get_blended_action(a_base)
            is_human = True
        else:
            action = a_base
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

    if not (0.0 <= FLAGS.blend_lambda <= 1.0):
        raise ValueError(f"--blend_lambda must be in [0, 1], got {FLAGS.blend_lambda}")

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
            f"Expected norm_mode='min_max' for keypoints checkpoints, got '{norm_mode}'."
        )
    for key in ("state_min", "state_max", "action_min", "action_max"):
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint missing required key '{key}'. This script is for keypoints "
                f"checkpoints saved by scripts/act_train_keypoints.py."
            )

    lerobot_cfg_dict = checkpoint.get("lerobot_cfg")
    if lerobot_cfg_dict is None:
        lerobot_cfg_dict = config.get("lerobot_cfg")
    if lerobot_cfg_dict is None:
        raise KeyError(
            "Checkpoint missing 'lerobot_cfg'. This script only loads ACTLeRobotPolicy "
            "checkpoints saved by the updated scripts/act_train_keypoints.py "
            "(or scripts/act_dagger_finetune_keypoints.py)."
        )

    horizon = int(lerobot_cfg_dict.get("chunk_size", 16))
    ckpt_decay = float(config.get("ensemble_decay", 0.05))
    ensemble_decay = ckpt_decay if FLAGS.ensemble_decay < 0 else FLAGS.ensemble_decay

    state_min_np = np.asarray(checkpoint["state_min"], dtype=np.float32)
    state_max_np = np.asarray(checkpoint["state_max"], dtype=np.float32)
    action_min_np = np.asarray(checkpoint["action_min"], dtype=np.float32)
    action_max_np = np.asarray(checkpoint["action_max"], dtype=np.float32)
    state_dim = int(state_min_np.shape[0])
    action_dim = int(action_min_np.shape[0])
    if state_dim != AGENT_POS_DIM + ENV_STATE_DIM:
        raise ValueError(
            f"This script targets the LeRobot pusht_keypoints ACT model "
            f"(state_dim={AGENT_POS_DIM + ENV_STATE_DIM}), but checkpoint has state_dim={state_dim}."
        )

    temporal_agg: bool = FLAGS.temporal_agg
    if temporal_agg:
        query_frequency = 1
    elif FLAGS.query_frequency < 0:
        query_frequency = horizon
    else:
        query_frequency = max(1, min(FLAGS.query_frequency, horizon))

    model = ACTLeRobotPolicy(ACTLeRobotConfig.from_dict(lerobot_cfg_dict)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state_min = torch.tensor(state_min_np, dtype=torch.float32, device=device)
    state_max = torch.tensor(state_max_np, dtype=torch.float32, device=device)
    action_min = torch.tensor(action_min_np, dtype=torch.float32, device=device)
    action_max = torch.tensor(action_max_np, dtype=torch.float32, device=device)

    stats = {
        "s_min": state_min,
        "s_max": state_max,
        "a_min": action_min,
        "a_max": action_max,
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
        window_scale=FLAGS.window_scale,
        blend_lambda=float(FLAGS.blend_lambda),
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
        f"action_dim={action_dim} | success_threshold={success_thresh:.2f} | "
        f"CR-DAgger blend_lambda={FLAGS.blend_lambda:.2f} (hold SHIFT to engage)"
    )

    seed_pbar = tqdm(total=len(seeds), desc="Episodes", position=0)
    step_pbar = tqdm(total=FLAGS.max_steps, desc="Steps", position=1, leave=False)

    for seed in seeds:
        step_pbar.set_description(f"Seed {seed}")
        terminated, truncated, success, had_intervention, quit_requested, max_coverage = run_cr_dagger_episode(
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
