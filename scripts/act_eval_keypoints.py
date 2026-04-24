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
# Default: --save_video with two-phase eval (all metrics first, then replay for MP4 + future plan overlay).
# Live frame capture instead:  --nodefer_video
python scripts/act_eval_keypoints.py --model_path models/act_keypoints/best.pt
"""

import warnings
from typing import List, Optional, Tuple

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
from envs.interactive_utils import draw_future_plan_on_rgb_frame, draw_status_overlay, ControlState
from models.act_refactored import ACTLeRobotConfig, ACTLeRobotPolicy

AGENT_POS_DIM = 2
ENV_STATE_DIM = 16

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/act_keypoints/best.pt", "Path to trained ACT keypoints checkpoint")
flags.DEFINE_integer("num_seeds", 50, "Number of episodes to evaluate")
flags.DEFINE_integer("seed", 42, "Global seed for numpy/torch (model/eval reproducibility). Used as the first env seed only when --random_seeds=False.")
flags.DEFINE_boolean("random_seeds", True, "If True, draw env seeds from OS entropy each run (truly different across runs). If False, use sequential seeds [seed, seed+num_seeds).")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")
flags.DEFINE_float("ensemble_decay",0.1, "Override temporal ensembling decay; <0 uses checkpoint")
flags.DEFINE_boolean("save_video", True, "Save episodes as an MP4 video")
flags.DEFINE_boolean(
    "defer_video",
    True,
    "With --save_video (default: True), run all rollouts first (no video frames), print metrics, then replay "
    "for one MP4 with future-action-chunk overlay. Use --nodefer_video to record frames during the first pass instead.",
)
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
    0.9,
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


def replay_keypoints_traces_to_frames(
    traces: List[Tuple[int, List[Tuple[np.ndarray, np.ndarray]]]],
    *,
    fast_mode: bool,
    window_size: int,
    max_steps: int,
    fps: int,
) -> List[np.ndarray]:
    """Replay stored trajectories. Each step is (executed action (2,), future plan (H,2) from last query)."""
    render_mode = "rgb_array" if fast_mode else "human"
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode=render_mode,
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    clock = None if fast_mode else pygame.time.Clock()
    frames: List[np.ndarray] = []
    for seed, steps in traces:
        if not steps:
            continue
        obs, _ = env.reset(seed=int(seed))
        plan0 = np.asarray(steps[0][1], dtype=np.float32)
        ag0 = build_state_vector(obs)[:2]
        if fast_mode:
            r0 = env.render()
            if r0 is not None and plan0 is not None and plan0.size:
                frames.append(draw_future_plan_on_rgb_frame(np.asarray(r0), plan0))
        else:
            env.render()
            draw_status_overlay(
                env,
                ControlState.MODEL_CONTROL,
                int(seed),
                0,
                0,
                max_steps,
                ag0,
                False,
                reward=None,
                future_plan_xy=plan0,
            )
            fr = capture_frame(env)
            if fr is not None:
                frames.append(fr)
            if clock is not None:
                clock.tick(fps)

        for step, (act, plan) in enumerate(steps, start=1):
            if not fast_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_q
                    ):
                        env.close()
                        return frames
            agent_pos = build_state_vector(obs)[:2]
            plan_xy = np.asarray(plan, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(
                np.clip(np.asarray(act, dtype=np.float32), 0.0, 512.0)
            )
            if fast_mode:
                latest_render = env.render()
                if latest_render is not None:
                    arr = np.asarray(latest_render)
                    if plan_xy.size:
                        arr = draw_future_plan_on_rgb_frame(arr, plan_xy)
                    frames.append(arr)
            else:
                env.render()
                draw_status_overlay(
                    env,
                    ControlState.MODEL_CONTROL,
                    int(seed),
                    0,
                    step,
                    max_steps,
                    agent_pos,
                    False,
                    reward=float(reward),
                    future_plan_xy=plan_xy,
                )
                frame = capture_frame(env)
                if frame is not None:
                    frames.append(frame)
                if clock is not None:
                    clock.tick(fps)
        if frames:
            for _ in range(fps):
                frames.append(frames[-1])
    env.close()
    return frames


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
    # Two-phase: compute all success metrics + (action, plan) traces first, then optional MP4 replay (bc_mlp style flow).
    use_defer = bool(FLAGS.save_video and FLAGS.defer_video)
    pass1_render_mode = "rgb_array" if (fast_mode or use_defer) else "human"
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",  # {'agent_pos': (2,), 'environment_state': (16,)}
        render_mode=pass1_render_mode,
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
        f"{' | two_phase_eval+replay=ON' if use_defer else ' | two_phase_eval+replay=OFF (live capture)'}"
    )
    success_count = 0
    coverages: List[float] = []

    if FLAGS.random_seeds:
        # Draw env seeds from OS entropy so each run picks a fresh batch.
        # We deliberately avoid np.random here because it was seeded with
        # FLAGS.seed above for model/eval reproducibility — using it would
        # produce the same "random" seeds every run.
        env_seed_rng = np.random.default_rng()
        seeds = env_seed_rng.integers(0, 2**31, size=FLAGS.num_seeds).tolist()
    else:
        seeds = list(range(FLAGS.seed, FLAGS.seed + FLAGS.num_seeds))
    print(f"Eval env seeds ({'random' if FLAGS.random_seeds else 'sequential'}): {seeds}")
    frames: List[np.ndarray] = []
    traces: List[Tuple[int, List[Tuple[np.ndarray, np.ndarray]]]] = []

    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        step = 0
        terminated = False
        truncated = False
        success = False
        max_coverage = 0.0
        clock = pygame.time.Clock() if (not fast_mode and not use_defer) else None
        ep_traj: List[Tuple[np.ndarray, np.ndarray]] = []
        last_plan_vis: Optional[np.ndarray] = None  # (H, 2) denorm chunk from the latest policy query

        if temporal_agg:
            all_time_actions = torch.zeros(
                (FLAGS.max_steps, FLAGS.max_steps + horizon, action_dim),
                dtype=torch.float32,
                device=device,
            )

        cached_chunk: Optional[np.ndarray] = None

        while not (terminated or truncated):
            if not fast_mode and not use_defer:
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
                last_plan_vis = pred_chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)

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
            plan_for_step = last_plan_vis if last_plan_vis is not None else np.zeros(
                (horizon, action_dim), dtype=np.float32
            )
            if use_defer:
                ep_traj.append((action.copy(), plan_for_step.copy()))

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

            if not use_defer:
                if fast_mode:
                    latest_render = env.render()
                    if FLAGS.save_video and latest_render is not None:
                        arr = np.asarray(latest_render)
                        if last_plan_vis is not None and last_plan_vis.size:
                            arr = draw_future_plan_on_rgb_frame(arr, last_plan_vis)
                        frames.append(arr)
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
                        future_plan_xy=last_plan_vis,
                    )

                    if FLAGS.save_video:
                        frame = capture_frame(env)
                        if frame is not None:
                            frames.append(frame)

                    if clock is not None:
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

        if use_defer:
            traces.append((int(seed), ep_traj))
        if FLAGS.save_video and (not use_defer) and frames:
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

    if use_defer and FLAGS.save_video and traces:
        env.close()
        print("Replaying action traces for video encoding...")
        frames = replay_keypoints_traces_to_frames(
            traces,
            fast_mode=fast_mode,
            window_size=window_size,
            max_steps=FLAGS.max_steps,
            fps=FLAGS.fps,
        )
        if frames:
            os.makedirs(FLAGS.video_dir, exist_ok=True)
            video_path = os.path.join(FLAGS.video_dir, time.strftime("%Y-%m-%d-%H-%M-%S.mp4"))
            imageio.mimwrite(video_path, frames, fps=FLAGS.fps)
            print(f"Saved video to {video_path}")
    elif FLAGS.save_video and frames:
        os.makedirs(FLAGS.video_dir, exist_ok=True)
        video_path = os.path.join(FLAGS.video_dir, time.strftime("%Y-%m-%d-%H-%M-%S.mp4"))
        imageio.mimwrite(video_path, frames, fps=FLAGS.fps)
        print(f"Saved video to {video_path}")
        env.close()
    else:
        env.close()


if __name__ == "__main__":
    app.run(main)
