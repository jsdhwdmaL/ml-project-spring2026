"""Evaluate ACT checkpoints trained on LeRobot ``pusht_keypoints`` (state-only).

Mirrors ``scripts/act_eval_og_data.py`` but for models from
``scripts/act_train_keypoints.py`` where the policy expects an 18-D state
``concat(agent_pos, environment_state)`` and NO image input. The environment
is gym_pusht with ``obs_type="environment_state_agent_pos"`` which returns
``{"agent_pos": (2,), "environment_state": (16,)}`` in pixel coordinates,
matching the training dataset. Normalization is ``min_max`` to/from [-1, 1].

python scripts/act_eval_keypoints.py \
    --model_path models/act_keypoints_150_epochs_3200/latest.pt \
    --num_seeds 10
"""

import warnings
from typing import List, Optional

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym
import pygame
from absl import app, flags
import imageio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
from envs.interactive_utils import draw_status_overlay, ControlState
from models.act_lerobot import ACTLeRobotConfig, ACTLeRobotPolicy

AGENT_POS_DIM = 2
ENV_STATE_DIM = 16

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/act_keypoints/best.pt", "Path to trained ACT keypoints checkpoint")
flags.DEFINE_integer("num_seeds", 50, "Number of episodes to evaluate")
flags.DEFINE_integer("seed", 42, "Global seed for numpy/torch and the env-seed sequence (matches LeRobot's deterministic eval semantics)")
flags.DEFINE_boolean("random_seeds", True, "If True, draw env seeds from the seeded RNG; if False, use sequential seeds [seed, seed+num_seeds)")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")
flags.DEFINE_float("ensemble_decay",0.05, "Override temporal ensembling decay; <0 uses checkpoint")
flags.DEFINE_boolean("save_video", True, "Save episodes as an MP4 video")
flags.DEFINE_string("video_dir", "videos/act_keypoints", "Directory to save episode videos")
flags.DEFINE_boolean(
    "temporal_agg",
    True,
    "Enable temporal ensembling. Set --notemporal_agg to run open-loop chunked "
    "execution (matches LeRobot's temporal_ensemble_momentum=null).",
)
flags.DEFINE_integer(
    "query_frequency",
    -1,
    "When temporal_agg is disabled, how many steps to execute from each predicted "
    "chunk before re-querying. <0 (default) means use the model's full horizon "
    "(LeRobot's n_action_steps semantics).",
)
flags.DEFINE_boolean(
    "on_cuda",
    False,
    "If true, require CUDA and run headless as fast as possible (no realtime clock throttling).",
)
flags.DEFINE_float(
    "success_threshold",
    0.95,
    "Coverage fraction in (0, 1] required to count as success. "
    "Note: gym_pusht auto-terminates at 0.95, so values >0.95 are clamped to 0.95.",
)


def normalize_state_dict_keys_for_eval(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip a leading `_orig_mod.` prefix from torch.compile-saved checkpoints."""
    prefix = "_orig_mod."
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def capture_frame(env) -> Optional[np.ndarray]:
    """Grab the current pygame window surface as an RGB numpy array."""
    surface = pygame.display.get_surface()
    if surface is None:
        return None
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def build_state_vector(obs: dict) -> np.ndarray:
    """Compose the 18-D state: concat(agent_pos, environment_state)."""
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
    env_state = np.asarray(obs["environment_state"], dtype=np.float32).reshape(-1)
    return np.concatenate([agent_pos, env_state], axis=0)


def split_state_to_obs(state: torch.Tensor) -> dict:
    """Split the 18-D normalized state into the dict ACTLeRobotPolicy expects."""
    return {
        "observation.state": state[..., :AGENT_POS_DIM],
        "observation.environment_state": state[..., AGENT_POS_DIM:],
    }


def main(_):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if FLAGS.on_cuda and not torch.cuda.is_available():
        raise ValueError("--on_cuda=true was requested but CUDA is not available.")
    device = torch.device(
        "cuda" if FLAGS.on_cuda else (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )
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

    print(f"Loading ACT keypoints model from {FLAGS.model_path} onto {device}...")
    checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    norm_mode = str(config.get("norm_mode", "min_max"))
    if norm_mode != "min_max":
        raise ValueError(
            f"Expected norm_mode='min_max' for keypoints checkpoints, got '{norm_mode}'. "
            f"Use scripts/act_eval_og_data.py or scripts/act_eval.py for mean/std checkpoints."
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
    model.load_state_dict(normalize_state_dict_keys_for_eval(checkpoint["model_state_dict"]))
    model.eval()

    state_min = torch.tensor(state_min_np, dtype=torch.float32, device=device)
    state_max = torch.tensor(state_max_np, dtype=torch.float32, device=device)
    action_min = torch.tensor(action_min_np, dtype=torch.float32, device=device)
    action_max = torch.tensor(action_max_np, dtype=torch.float32, device=device)

    window_size = int(512 * FLAGS.window_scale)
    fast_mode = bool(FLAGS.on_cuda)
    render_mode = "rgb_array" if fast_mode else "human"
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",  # {'agent_pos': (2,), 'environment_state': (16,)}
        render_mode=render_mode,
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    print("\nStarting ACT (LeRobot pusht_keypoints) Evaluation...")
    mode_str = (
        f"temporal_agg=ON (decay={ensemble_decay})"
        if temporal_agg
        else f"temporal_agg=OFF (open-loop, query_frequency={query_frequency})"
    )
    print(
        f"H={horizon} | {mode_str} | state_dim={state_dim} | "
        f"action_dim={action_dim} | fast_mode={fast_mode}"
    )
    success_count = 0
    coverages: List[float] = []

    seeds = (
        np.random.randint(0, 2**31, size=FLAGS.num_seeds).tolist()
        if FLAGS.random_seeds
        else list(range(FLAGS.seed, FLAGS.seed + FLAGS.num_seeds))
    )
    frames: List[np.ndarray] = []

    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        step = 0
        terminated = False
        truncated = False
        success = False
        max_coverage = 0.0
        clock = pygame.time.Clock() if not fast_mode else None
        latest_render = env.render() if fast_mode else None

        if temporal_agg:
            all_time_actions = torch.zeros(
                (FLAGS.max_steps, FLAGS.max_steps + horizon, action_dim),
                dtype=torch.float32,
                device=device,
            )

        cached_chunk: Optional[np.ndarray] = None

        while not (terminated or truncated):
            if not fast_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        print("Evaluation aborted by user.")
                        env.close()
                        return

            state_vec = build_state_vector(obs)
            if state_vec.shape[0] != state_dim:
                raise ValueError(
                    f"env produced state of length {state_vec.shape[0]} but model expects {state_dim}"
                )
            agent_pos = state_vec[:2]

            state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            state_tensor_norm = 2.0 * (state_tensor - state_min) / (state_max - state_min) - 1.0

            if step % query_frequency == 0:
                with torch.no_grad():
                    obs_dict = split_state_to_obs(state_tensor_norm)
                    pred_norm_chunk, _, _ = model(obs_dict, action_chunk=None)
                pred_chunk = (pred_norm_chunk + 1.0) * 0.5 * (
                    action_max.view(1, 1, -1) - action_min.view(1, 1, -1)
                ) + action_min.view(1, 1, -1)

                if temporal_agg:
                    all_time_actions[[step], step : step + horizon] = pred_chunk
                else:
                    cached_chunk = pred_chunk.squeeze(0).cpu().numpy().astype(np.float32)

            if temporal_agg:
                start = max(0, step - horizon + 1)
                actions_for_step = all_time_actions[start : step + 1, step]

                k = ensemble_decay
                exp_weights = np.exp(-k * np.arange(len(actions_for_step) - 1, -1, -1))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights_t = torch.from_numpy(exp_weights).float().to(device).unsqueeze(1)

                action_np = (actions_for_step * exp_weights_t).sum(dim=0).cpu().numpy().astype(np.float32)
            else:
                offset = step % query_frequency
                action_np = cached_chunk[offset].astype(np.float32)

            action = np.clip(action_np, 0.0, 512.0)

            obs, reward, terminated, truncated, info = env.step(action)
            coverage = float(info.get("coverage", 0.0)) if isinstance(info, dict) else 0.0
            if coverage > max_coverage:
                max_coverage = coverage
            success = max_coverage >= success_thresh
            if success:
                truncated = True

            step += 1
            if step >= FLAGS.max_steps:
                truncated = True

            if fast_mode:
                latest_render = env.render()
                if FLAGS.save_video and latest_render is not None:
                    frames.append(np.asarray(latest_render))
            else:
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
                    reward=float(reward),
                )

                if FLAGS.save_video:
                    frame = capture_frame(env)
                    if frame is not None:
                        frames.append(frame)

                clock.tick(FLAGS.fps)

        coverages.append(max_coverage)
        if success:
            success_count += 1
            tag = "SUCCESS"
        else:
            tag = "FAILED"
        print(
            f"Episode {i + 1}/{FLAGS.num_seeds} (seed={seed}) - {tag} "
            f"({step} steps, coverage={max_coverage:.3f}, threshold={success_thresh:.2f})"
        )

        if FLAGS.save_video and frames:
            for _ in range(FLAGS.fps):
                frames.append(frames[-1])

    print("=" * 60)
    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    max_cov = float(np.max(coverages)) if coverages else 0.0
    print(
        f"ACT (pusht_keypoints) Evaluation Complete! Success Rate: "
        f"{success_count}/{FLAGS.num_seeds} ({(success_count / FLAGS.num_seeds) * 100:.1f}%) "
        f"@ threshold={success_thresh:.2f}"
    )
    print(f"Mean max-coverage: {mean_cov:.3f} | Best max-coverage: {max_cov:.3f}")
    print("=" * 60)

    if FLAGS.save_video and frames:
        os.makedirs(FLAGS.video_dir, exist_ok=True)
        video_path = os.path.join(FLAGS.video_dir, time.strftime("%Y-%m-%d-%H-%M-%S.mp4"))
        imageio.mimwrite(video_path, frames, fps=FLAGS.fps)
        print(f"Saved video to {video_path}")

    env.close()


if __name__ == "__main__":
    app.run(main)
