"""Evaluate BC MLP (keypoints, min–max) on gym_pusht.

State is 18-D ``concat(agent_pos, environment_state)`` with ``obs_type=environment_state_agent_pos``,
same as ``scripts/act_eval_keypoints.py`` but a single MLP forward per step (no chunking / temporal ensemble).

  python scripts/bc_mlp_keypoints_eval.py --model_path models/bc_mlp_keypoints/best.pt
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import imageio
import numpy as np
import pygame
import torch
import gymnasium as gym
from absl import app, flags

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gym_pusht  # noqa: F401
from data.dataloader_keypoints import min_max_denormalize, min_max_normalize
from envs.interactive_utils import draw_status_overlay, ControlState
from models.mlp_keypoints import BCKeypointsMLP, DEFAULT_STATE_DIM, DEFAULT_ACTION_DIM

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/bc_mlp_keypoints/best.pt", "Checkpoint from bc_mlp_keypoints_train")
flags.DEFINE_integer("num_seeds", 50, "Number of evaluation episodes")
flags.DEFINE_integer("seed", 42, "Numpy/torch seed; first env seed if --random_seeds=False")
flags.DEFINE_boolean("random_seeds", True, "Draw env seeds from OS entropy")
flags.DEFINE_integer("fps", 10, "Control / render rate (Hz)")
flags.DEFINE_float("window_scale", 1.0, "pygame window scale")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_boolean("save_video", True, "Save MP4 of episodes")
flags.DEFINE_string("video_dir", "videos/bc_mlp_keypoints", "Output video directory")
flags.DEFINE_boolean("on_cuda", False, "CUDA headless fast eval")
flags.DEFINE_float(
    "success_threshold",
    0.9,
    "Coverage in (0,1] for success. gym_pusht stops at 0.95; >0.95 is clamped.",
)


def capture_frame(env) -> Optional[np.ndarray]:
    surface = pygame.display.get_surface()
    if surface is None:
        return None
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def build_state_vector(obs: dict) -> np.ndarray:
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
    env_state = np.asarray(obs["environment_state"], dtype=np.float32).reshape(-1)
    return np.concatenate([agent_pos, env_state], axis=0)


def main(_: list[str]) -> None:
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if FLAGS.on_cuda and not torch.cuda.is_available():
        raise ValueError("--on_cuda but CUDA is not available.")
    device = torch.device(
        "cuda" if FLAGS.on_cuda else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    if not (0.0 < FLAGS.success_threshold <= 1.0):
        raise ValueError(f"--success_threshold must be in (0,1], got {FLAGS.success_threshold}")
    success_thresh = 0.95 if FLAGS.success_threshold > 0.95 else float(FLAGS.success_threshold)
    if FLAGS.success_threshold > 0.95:
        print(f"[warn] success_threshold > 0.95; clamping to 0.95 (gym_pusht)")

    print(f"Loading BC keypoints MLP from {FLAGS.model_path} on {device}...")
    checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    if str(config.get("norm_mode", "min_max")) != "min_max":
        raise ValueError("This eval expects min_max keypoints checkpoints from bc_mlp_keypoints_train.")
    for k in ("state_min", "state_max", "action_min", "action_max"):
        if k not in checkpoint:
            raise KeyError(f"Checkpoint missing '{k}'")

    state_min_np = np.asarray(checkpoint["state_min"], dtype=np.float32)
    state_max_np = np.asarray(checkpoint["state_max"], dtype=np.float32)
    action_min_np = np.asarray(checkpoint["action_min"], dtype=np.float32)
    action_max_np = np.asarray(checkpoint["action_max"], dtype=np.float32)
    state_dim = int(state_min_np.shape[0])
    action_dim = int(action_min_np.shape[0])
    if state_dim != DEFAULT_STATE_DIM or action_dim != DEFAULT_ACTION_DIM:
        raise ValueError(
            f"Expected state_dim={DEFAULT_STATE_DIM}, action_dim={DEFAULT_ACTION_DIM}; "
            f"got {state_dim}, {action_dim}."
        )

    hidden_dim = int(config.get("hidden_dim", 256))
    model = BCKeypointsMLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state_min = torch.tensor(state_min_np, device=device, dtype=torch.float32)
    state_max = torch.tensor(state_max_np, device=device, dtype=torch.float32)
    action_min = torch.tensor(action_min_np, device=device, dtype=torch.float32)
    action_max = torch.tensor(action_max_np, device=device, dtype=torch.float32)

    window_size = int(512 * FLAGS.window_scale)
    fast = bool(FLAGS.on_cuda)
    # rgb_array is reliable for MP4; pygame surface capture often yields no frames with render_mode=human
    if FLAGS.on_cuda or FLAGS.save_video:
        render_mode: str = "rgb_array"
    else:
        render_mode = "human"
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode=render_mode,
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    if FLAGS.random_seeds:
        env_seed_rng = np.random.default_rng()
        seeds = env_seed_rng.integers(0, 2**31, size=FLAGS.num_seeds).tolist()
    else:
        seeds = list(range(FLAGS.seed, FLAGS.seed + FLAGS.num_seeds))
    print(f"Seeds ({'random' if FLAGS.random_seeds else 'sequential'}): {seeds}")

    success_count = 0
    coverages: List[float] = []
    frames: List[np.ndarray] = []

    # human: pygame window exists; rgb_array: no pygame display — never call pygame.event without init
    use_pygame_window = render_mode == "human"

    print(
        f"\nBC MLP (pusht_keypoints) | success_thresh={success_thresh} | "
        f"fast={fast} | render_mode={render_mode} | save_video={FLAGS.save_video}\n"
    )

    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        step = 0
        terminated = False
        truncated = False
        max_coverage = 0.0
        success = False
        clock = pygame.time.Clock() if (use_pygame_window and not fast) else None
        if FLAGS.save_video:
            init_frame = env.render()
            if init_frame is not None:
                frames.append(np.asarray(init_frame))

        while not (terminated or truncated):
            if use_pygame_window and not fast:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        print("Aborted by user.")
                        env.close()
                        return

            state_vec = build_state_vector(obs)
            agent_pos = state_vec[:2]
            s = torch.from_numpy(state_vec).float().to(device).unsqueeze(0)
            s_n = min_max_normalize(s, state_min, state_max)
            with torch.no_grad():
                a_n = model(s_n)
            a = min_max_denormalize(a_n, action_min, action_max)
            action = np.clip(a.squeeze(0).cpu().numpy(), 0.0, 512.0).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            coverage = float(info.get("coverage", 0.0)) if isinstance(info, dict) else 0.0
            if coverage > max_coverage:
                max_coverage = coverage
            if max_coverage >= success_thresh:
                success = True
                truncated = True

            step += 1
            if step >= FLAGS.max_steps:
                truncated = True

            if render_mode == "rgb_array":
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
                    fr = capture_frame(env)
                    if fr is not None:
                        frames.append(fr)
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
            f"({step} steps, max_coverage={max_coverage:.3f})"
        )

        if FLAGS.save_video and frames:
            for _ in range(FLAGS.fps):
                frames.append(frames[-1])

    mean_c = float(np.mean(coverages)) if coverages else 0.0
    max_c = float(np.max(coverages)) if coverages else 0.0
    print("=" * 60)
    print(
        f"BC keypoints MLP: success {success_count}/{FLAGS.num_seeds} "
        f"({100.0 * success_count / FLAGS.num_seeds:.1f}%) @ threshold={success_thresh}"
    )
    print(f"Mean max-coverage: {mean_c:.3f} | best: {max_c:.3f}")
    print("=" * 60)

    if FLAGS.save_video:
        if not frames:
            print(
                "save_video is True but no frames were captured; "
                "use default settings or render_mode=rgb_array (set automatically with --save_video)."
            )
        else:
            os.makedirs(FLAGS.video_dir, exist_ok=True)
            out = os.path.join(FLAGS.video_dir, time.strftime("%Y-%m-%d-%H-%M-%S.mp4"))
            imageio.mimwrite(out, frames, fps=FLAGS.fps)
            print(f"Saved video ({len(frames)} frames): {out}")

    env.close()


if __name__ == "__main__":
    app.run(main)
