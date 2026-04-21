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
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gym_pusht
from envs.interactive_utils import get_observation_image, draw_status_overlay, ControlState
from models.act import ACTPolicy

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/act_20/best.pt", "Path to trained ACT checkpoint")
flags.DEFINE_integer("num_seeds", 50, "Number of episodes to evaluate")
flags.DEFINE_integer("seed", 42, "Global seed for numpy/torch and the env-seed sequence (matches LeRobot's deterministic eval semantics)")
flags.DEFINE_boolean("random_seeds", False, "If True, draw env seeds from the seeded RNG; if False, use sequential seeds [seed, seed+num_seeds)")
flags.DEFINE_integer("fps", 10, "Control/render frequency in Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale factor (>= 1.0)")
flags.DEFINE_integer("max_steps", 300, "Maximum steps per episode")
flags.DEFINE_float("ensemble_decay", 0.01, "Override temporal ensembling decay; <0 uses checkpoint")
flags.DEFINE_boolean("save_video", True, "Save episodes as an MP4 video")
flags.DEFINE_string("video_dir", "videos/act", "Directory to save episode videos")
flags.DEFINE_boolean("temporal_agg", True, "Enable temporal ensembling (query model every step, blend predictions)")
flags.DEFINE_integer("query_frequency", 1, "When temporal_agg is disabled, how many steps to execute from each predicted chunk before re-querying")
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
flags.DEFINE_boolean(
    "stop_on_success",
    False,
    "If true, end the episode as soon as max_coverage reaches success_threshold. "
    "Default False matches LeRobot semantics (run to env-termination or max_steps).",
)


def get_policy_image(env) -> np.ndarray:
    """Return the env's native 96x96 RGB observation render (uint8 HWC).

    Training data was captured at 96x96; bilinear-downscaling a 512x512
    visualization render produces visibly different pixels and breaks the
    ResNet's input distribution. We bypass the visualization render and use
    the env's internal observation render.
    """
    u = env.unwrapped
    screen = u._draw()
    return u._get_img(screen, width=u.observation_width, height=u.observation_height, render_action=False)


def get_agent_pos_from_obs(obs) -> np.ndarray:
    """Works for dict observations (pixels + agent_pos) or flat state vectors (first two dims = agent)."""
    if isinstance(obs, np.ndarray):
        return np.asarray(obs[:2], dtype=np.float32).reshape(2)
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    if agent_pos.ndim == 1:
        return agent_pos
    return agent_pos[-1]


def policy_state_vector(obs, state_dim: int) -> np.ndarray:
    """Build the normalized-state vector expected by ACT (2D agent or 5D pymunk state)."""
    if isinstance(obs, np.ndarray):
        v = np.asarray(obs, dtype=np.float32).reshape(-1)
        if v.shape[0] != state_dim:
            raise ValueError(f"obs_type=state length {v.shape[0]} != checkpoint state_dim {state_dim}")
        return v
    if state_dim == 2:
        return get_agent_pos_from_obs(obs)
    raise ValueError("5D state eval requires obs_type='state' (flat numpy observation).")


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


def normalize_state_dict_keys_for_eval(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Support checkpoints saved from torch.compile-wrapped modules.

    Compiled models often save parameter keys prefixed with `_orig_mod.` while the
    plain ACTPolicy in eval expects unprefixed keys.
    """
    prefix = "_orig_mod."
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def main(_):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if FLAGS.on_cuda and not torch.cuda.is_available():
        raise ValueError("--on_cuda=true was requested but CUDA is not available.")
    device = torch.device(
        "cuda" if FLAGS.on_cuda else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    use_vision = bool(config.get("use_vision", True))

    # When temporal ensembling is on we must query the model every step.
    # When it is off we query every query_frequency steps and execute that
    # many actions from the chunk before re-querying.
    temporal_agg: bool = FLAGS.temporal_agg
    query_frequency: int = 1 if temporal_agg else max(1, FLAGS.query_frequency)

    state_mean_np = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_dim = int(state_mean_np.shape[0])

    model = ACTPolicy(
        state_dim=state_dim,
        action_dim=2,
        horizon=horizon,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        use_vision=use_vision,
    ).to(device)
    model_state = normalize_state_dict_keys_for_eval(checkpoint["model_state_dict"])
    model.load_state_dict(model_state)
    model.eval()

    state_mean = torch.tensor(state_mean_np, dtype=torch.float32, device=device)
    state_std = torch.tensor(checkpoint["state_std"], dtype=torch.float32, device=device)
    action_mean = torch.tensor(checkpoint["action_mean"], dtype=torch.float32, device=device)
    action_std = torch.tensor(checkpoint["action_std"], dtype=torch.float32, device=device)

    action_dim = int(action_mean.shape[-1])

    # Image is the env's native 96x96 observation render -- no resize needed.
    base_transform = T.ToTensor()
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    window_size = int(512 * FLAGS.window_scale)
    obs_type = "environment_state_agent_pos" if use_vision else "state"
    fast_mode = bool(FLAGS.on_cuda)
    render_mode = "rgb_array" if fast_mode else "human"
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type=obs_type,
        render_mode=render_mode,
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    print("\nStarting ACT Evaluation...")
    print(
        f"H={horizon} | ensemble_decay={ensemble_decay} | use_vision={use_vision} | "
        f"state_dim={state_dim} | fast_mode={fast_mode}"
    )
    if FLAGS.num_seeds < 20:
        print(
            f"[warn] num_seeds={FLAGS.num_seeds} is small; success rate will have "
            f"high variance (one episode = {100.0 / FLAGS.num_seeds:.1f}%)."
        )

    success_count = 0
    per_episode_max_coverage: List[float] = []
    per_episode_max_reward: List[float] = []
    per_episode_sum_reward: List[float] = []

    seeds = (
        np.random.randint(0, 2**31, size=FLAGS.num_seeds).tolist()
        if FLAGS.random_seeds
        else list(range(FLAGS.seed, FLAGS.seed + FLAGS.num_seeds))
    )
    frames: List[np.ndarray] = [] # for video export

    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        step = 0
        terminated = False
        truncated = False
        success = False
        max_coverage = 0.0
        max_reward = 0.0
        sum_reward = 0.0
        clock = pygame.time.Clock() if not fast_mode else None
        latest_render = env.render() if fast_mode else None

        # --------------- temporal ensembling state ---------------
        # all_time_actions[t_query, t_exec] stores the action predicted
        # at query step t_query for execution at absolute step t_exec.
        # Shape: [max_steps, max_steps + horizon, action_dim]
        # Allocated once per episode so indexing stays simple.
        if temporal_agg:
            all_time_actions = torch.zeros(
                (FLAGS.max_steps, FLAGS.max_steps + horizon, action_dim),
                dtype=torch.float32,
                device=device,
            )
 
        # Cache the last predicted chunk when not using temporal ensembling
        cached_chunk: Optional[np.ndarray] = None

        while not (terminated or truncated):
            if not fast_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        print("Evaluation aborted by user.")
                        env.close()
                        return

            agent_pos = get_agent_pos_from_obs(obs)
            state_vec = policy_state_vector(obs, state_dim)
            state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            state_tensor_norm = (state_tensor - state_mean) / state_std

            if use_vision:
                # Policy image: env's native 96x96 obs render (matches training data).
                # The pygame visualization window is for the user only; the policy
                # never sees a downscaled 512 frame.
                img_array = get_policy_image(env)
                image_tensor = base_transform(img_array)
                image_tensor = normalize_transform(image_tensor).unsqueeze(0).to(device)
            else:
                image_tensor = None

            # ---- decide whether to query the model this step ----
            if step % query_frequency == 0:
                with torch.no_grad():
                    pred_norm_chunk, _, _ = model(image_tensor, state_tensor_norm, action_chunk=None)
                # pred_norm_chunk: [1, horizon, action_dim]
                pred_chunk = (pred_norm_chunk * action_std.view(1, 1, -1)) + action_mean.view(1, 1, -1)
                # [1, horizon, action_dim]
 
                if temporal_agg:
                    # Write this chunk into the row for the current step.
                    # Columns t_exec = step .. step+horizon hold the predictions.
                    all_time_actions[[step], step : step + horizon] = pred_chunk
                else:
                    # Store the raw numpy chunk for replay over the next
                    # query_frequency steps.
                    cached_chunk = pred_chunk.squeeze(0).cpu().numpy().astype(np.float32)

            # ---- select the action for this step ----
            if temporal_agg:
                # Column `step` across all query rows gives every prediction
                # ever made for this execution timestep.
                start = max(0, step - horizon + 1)
                actions_for_step = all_time_actions[start : step + 1, step]
 
                # Exponential weights: index 0 = oldest query, index K-1 = newest.
                # We want to weight *newer* predictions more heavily, so assign
                # weight exp(-k * (K-1-j)) ... equivalently, reverse the array
                # and weight exp(-k * offset) where offset=0 is the newest.
                k = ensemble_decay
                exp_weights = np.exp(-k * np.arange(len(actions_for_step) - 1, -1, -1))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights_t = torch.from_numpy(exp_weights).float().to(device).unsqueeze(1)
 
                action_np = (actions_for_step * exp_weights_t).sum(dim=0).cpu().numpy().astype(np.float32)
            else:
                # Execute the cached chunk at the appropriate offset.
                offset = step % query_frequency
                action_np = cached_chunk[offset].astype(np.float32)

            action = np.clip(action_np, 0.0, 512.0)

            obs, reward, terminated, truncated, info = env.step(action)
            reward_f = float(reward)
            sum_reward += reward_f
            if reward_f > max_reward:
                max_reward = reward_f
            coverage = float(info.get("coverage", 0.0)) if isinstance(info, dict) else 0.0
            if coverage > max_coverage:
                max_coverage = coverage
            success = max_coverage >= success_thresh
            if success and FLAGS.stop_on_success:
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

        if success:
            success_count += 1
            tag = "SUCCESS"
        else:
            tag = "FAILED"
        per_episode_max_coverage.append(max_coverage)
        per_episode_max_reward.append(max_reward)
        per_episode_sum_reward.append(sum_reward)
        print(
            f"Episode {i + 1}/{FLAGS.num_seeds} (seed={seed}) - {tag} "
            f"({step} steps, coverage={max_coverage:.3f}, max_reward={max_reward:.3f}, "
            f"sum_reward={sum_reward:.2f}, threshold={success_thresh:.2f})"
        )

        # pause for 1 second on final state
        if FLAGS.save_video and frames:
            for j in range(FLAGS.fps):
                frames.append(frames[-1])

    mean_max_coverage = float(np.mean(per_episode_max_coverage)) if per_episode_max_coverage else 0.0
    mean_max_reward = float(np.mean(per_episode_max_reward)) if per_episode_max_reward else 0.0
    print("=" * 60)
    print(
        f"ACT Evaluation Complete! Success Rate: {success_count}/{FLAGS.num_seeds} "
        f"({(success_count/FLAGS.num_seeds)*100:.1f}%) @ threshold={success_thresh:.2f}"
    )
    print(
        f"  mean max_coverage: {mean_max_coverage:.3f}    "
        f"mean max_reward: {mean_max_reward:.3f}"
    )
    print("=" * 60)

    if FLAGS.save_video and frames:
        os.makedirs(FLAGS.video_dir, exist_ok=True)  # auto-create dir
        video_path = os.path.join(FLAGS.video_dir, time.strftime("%Y-%m-%d-%H-%M-%S.mp4"))
        imageio.mimwrite(video_path, frames, fps=FLAGS.fps)
        print(f"Saved video to {video_path}")

    env.close()


if __name__ == "__main__":
    app.run(main)
