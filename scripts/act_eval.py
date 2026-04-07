import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
import gymnasium as gym
import pygame
from absl import app, flags
import imageio
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gym_pusht
from envs.interactive_utils import get_observation_image, draw_status_overlay, ControlState
from models.act import ACTPolicy

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/act/best.pt", "Path to trained ACT checkpoint")
flags.DEFINE_integer("num_seeds", 5, "Number of episodes to evaluate")
flags.DEFINE_boolean("random_seeds", True, "Sample random seeds instead of using 0..num_seeds-1")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")
flags.DEFINE_float("ensemble_decay", -1.0, "Override temporal ensembling decay; <0 uses checkpoint")
flags.DEFINE_boolean("save_video", True, "Save episodes as an MP4 video")
flags.DEFINE_string("video_dir", "videos/act", "Directory to save episode videos")


def get_agent_pos_from_obs(obs: Dict) -> np.ndarray:
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    if agent_pos.ndim == 1:
        return agent_pos
    return agent_pos[-1]


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


def capture_frame(env) -> Optional[np.ndarray]:
    """Grab the current pygame window surface as an RGB numpy array."""
    surface = pygame.display.get_surface()
    if surface is None:
        return None
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))  # (H, W, 3)


def save_episode_video(frames: List[np.ndarray], path: str, fps: int) -> None:
    if not frames:
        return
    print(f"Saved video to {path}")

def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading ACT model from {FLAGS.model_path} onto {device}...")

    checkpoint = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    horizon = int(config.get("horizon", 50))
    hidden_dim = int(config.get("hidden_dim", 512))
    latent_dim = int(config.get("latent_dim", 32))
    nhead = int(config.get("nhead", 8))
    num_encoder_layers = int(config.get("num_encoder_layers", 4))
    num_decoder_layers = int(config.get("num_decoder_layers", 4))
    ckpt_decay = float(config.get("ensemble_decay", 0.01))
    ensemble_decay = ckpt_decay if FLAGS.ensemble_decay < 0 else FLAGS.ensemble_decay

    model = ACTPolicy(
        state_dim=2,
        action_dim=2,
        horizon=horizon,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state_mean = torch.tensor(checkpoint["state_mean"], dtype=torch.float32, device=device)
    state_std = torch.tensor(checkpoint["state_std"], dtype=torch.float32, device=device)
    action_mean = torch.tensor(checkpoint["action_mean"], dtype=torch.float32, device=device)
    action_std = torch.tensor(checkpoint["action_std"], dtype=torch.float32, device=device)

    base_transform = T.Compose([
        T.ToTensor(),
        T.Resize((96, 96), antialias=True),
    ])
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    window_size = int(512 * FLAGS.window_scale)
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="human",
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    print("\nStarting ACT Evaluation...")
    print(f"H={horizon} | ensemble_decay={ensemble_decay}")
    success_count = 0

    seeds = np.random.randint(0, 2**31, size=FLAGS.num_seeds).tolist() if FLAGS.random_seeds else range(FLAGS.num_seeds)
    frames: List[np.ndarray] = [] # for video export

    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        step = 0
        terminated = False
        truncated = False
        success = False
        clock = pygame.time.Clock()

        chunk_predictions: List[Tuple[int, np.ndarray]] = []

        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    print("Evaluation aborted by user.")
                    env.close()
                    return

            agent_pos = get_agent_pos_from_obs(obs)
            img_array = get_observation_image(env)

            image_tensor = base_transform(img_array)
            image_tensor = normalize_transform(image_tensor).unsqueeze(0).to(device)
            state_tensor = torch.tensor(agent_pos, dtype=torch.float32, device=device).unsqueeze(0)
            state_tensor_norm = (state_tensor - state_mean) / state_std

            with torch.no_grad():
                pred_norm_chunk, _, _ = model(image_tensor, state_tensor_norm, action_chunk=None)
            pred_chunk = (pred_norm_chunk * action_std.view(1, 1, -1)) + action_mean.view(1, 1, -1)
            pred_chunk_np = pred_chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
            chunk_predictions.append((step, pred_chunk_np))

            action = ensemble_current_action(step, chunk_predictions, horizon=horizon, decay=ensemble_decay)
            action = np.clip(action, 0.0, 512.0)

            obs, reward, terminated, truncated, info = env.step(action)
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

            if FLAGS.save_video:
                frame = capture_frame(env)
                if frame is not None:
                    frames.append(frame)

            clock.tick(FLAGS.fps)

        if success:
            success_count += 1
            print(f"Episode {i + 1}/{FLAGS.num_seeds} (seed={seed}) - SUCCESS ({step} steps)")
        else:
            print(f"Episode {i + 1}/{FLAGS.num_seeds} (seed={seed}) - FAILED ({step} steps)")

        # pause for 1 second on final state
        if FLAGS.save_video:
            for j in range(FLAGS.fps):
                frames.append(frames[-1])

    print("=" * 60)
    print(f"ACT Evaluation Complete! Success Rate: {success_count}/{FLAGS.num_seeds} ({(success_count/FLAGS.num_seeds)*100:.1f}%)")
    print("=" * 60)

    if FLAGS.save_video and frames:
        video_path = os.path.join(FLAGS.video_dir, time.strftime("%Y-%m-%d-%H-%M-%S.mp4"))
        imageio.mimwrite(video_path, frames, fps=FLAGS.fps)
        print(f"Saved video to {video_path}")

    env.close()


if __name__ == "__main__":
    app.run(main)
