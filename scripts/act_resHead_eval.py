"""Evaluate ACTResHead (keypoints, min/max norm, state-only) in PushT.

Same env / metrics style as ``act_eval_keypoints.py``. Teacher signal is
optional: default is no hint (matches training-time dropout to zeros at test).

  python scripts/act_resHead_eval.py --model_path models/act_resHead_keypoints/best.pt
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

import gym_pusht
from envs.interactive_utils import draw_status_overlay, ControlState
from data.dataloader_keypoints import min_max_normalize
from data.reshead_keypoints_mix import EXPECTED_STATE_DIM
from models.act_reshead import ACTResHeadPolicy

AGENT_POS_DIM = 2
ENV_STATE_DIM = 16

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "models/act_resHead_keypoints/best.pt", "Checkpoint from act_resHead_train")
flags.DEFINE_integer("num_seeds", 50, "Number of episodes")
flags.DEFINE_integer("seed", 42, "Numpy/torch seed; first env seed if --random_seeds=False")
flags.DEFINE_boolean("random_seeds", True, "If True, fresh random env seeds per run (OS entropy)")
flags.DEFINE_integer("fps", 10, "Control/render Hz")
flags.DEFINE_float("window_scale", 1.0, "Window scale")
flags.DEFINE_integer("max_steps", 300, "Max steps per episode")
flags.DEFINE_float("ensemble_decay", 0.1, "Temporal ensemble decay; <0 uses checkpoint")
flags.DEFINE_boolean("save_video", True, "Save MP4")
flags.DEFINE_string("video_dir", "videos/act_resHead_keypoints", "Video output dir")
flags.DEFINE_boolean("temporal_agg", True, "Temporal ensembling of chunk predictions")
flags.DEFINE_integer("query_frequency", -1, "If temporal_agg off: <0 = full horizon else steps per requery")
flags.DEFINE_boolean("on_cuda", False, "Headless fast CUDA")
flags.DEFINE_float("success_threshold", 0.9, "Max coverage to count success; >0.95 clamped to 0.95")
flags.DEFINE_string(
    "teacher_hint",
    "",
    "Optional comma-separated teacher vector (len = checkpoint teacher_signal_dim). "
    "Empty = no hint (zeros). E.g. '1,0' to simulate human flag with no local-NPZ channel.",
)


def normalize_state_dict_keys_for_eval(state_dict: dict) -> dict:
    p = "_orig_mod."
    if state_dict and all(k.startswith(p) for k in state_dict.keys()):
        return {k[len(p):]: v for k, v in state_dict.items()}
    return state_dict


def build_state_vector(obs: dict) -> np.ndarray:
    a = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
    e = np.asarray(obs["environment_state"], dtype=np.float32).reshape(-1)
    return np.concatenate([a, e], axis=0)


def capture_frame(env) -> Optional[np.ndarray]:
    surface = pygame.display.get_surface()
    if surface is None:
        return None
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def parse_teacher_hint(
    raw: str, dim: int, device: torch.device
) -> Optional[torch.Tensor]:
    t = raw.strip()
    if not t:
        return None
    vals = [float(x) for x in t.split(",") if x.strip()]
    if len(vals) != dim:
        raise ValueError(f"--teacher_hint needs {dim} values, got {len(vals)}")
    return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)


def main(_):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if not (0.0 < FLAGS.success_threshold <= 1.0):
        raise ValueError("success_threshold must be in (0,1]")
    success_thresh = 0.95 if FLAGS.success_threshold > 0.95 else float(FLAGS.success_threshold)

    device = torch.device(
        "cuda" if FLAGS.on_cuda else (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )
    if FLAGS.on_cuda and not torch.cuda.is_available():
        raise ValueError("on_cuda but CUDA not available")

    print(f"Loading {FLAGS.model_path} -> {device}")
    ck = torch.load(FLAGS.model_path, map_location=device, weights_only=False)
    config = ck.get("config", {})
    if ck.get("norm_mode", config.get("norm_mode")) != "min_max":
        raise ValueError("This eval expects act_resHead_train checkpoints (norm_mode=min_max).")
    for k in ("state_min", "state_max", "action_min", "action_max"):
        if k not in ck:
            raise KeyError(f"Checkpoint missing {k}")

    horizon = int(config.get("horizon", 16))
    hidden_dim = int(config.get("hidden_dim", 512))
    latent_dim = int(config.get("latent_dim", 32))
    nhead = int(config.get("nhead", 8))
    n_enc = int(config.get("num_encoder_layers", 4))
    n_dec = int(config.get("num_decoder_layers", 4))
    t_dim = int(config.get("teacher_signal_dim", 2))
    tdrop = float(config.get("teacher_dropout_prob", 0.0))
    ens_ck = float(config.get("ensemble_decay", 0.05))
    ensemble_decay = ens_ck if FLAGS.ensemble_decay < 0 else FLAGS.ensemble_decay

    state_min = torch.tensor(ck["state_min"], dtype=torch.float32, device=device)
    state_max = torch.tensor(ck["state_max"], dtype=torch.float32, device=device)
    action_min = torch.tensor(ck["action_min"], dtype=torch.float32, device=device)
    action_max = torch.tensor(ck["action_max"], dtype=torch.float32, device=device)
    if int(state_min.shape[0]) != EXPECTED_STATE_DIM:
        raise ValueError(f"Expected state_dim {EXPECTED_STATE_DIM}, got {state_min.shape[0]}")

    teacher_t = parse_teacher_hint(FLAGS.teacher_hint, t_dim, device)
    if teacher_t is None:
        print("Teacher: none (zeros at inference — matches no-hint deployment)")
    else:
        print(f"Teacher: {teacher_t.cpu().numpy().squeeze().tolist()}")

    model = ACTResHeadPolicy(
        state_dim=EXPECTED_STATE_DIM,
        action_dim=2,
        horizon=horizon,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        nhead=nhead,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        teacher_signal_dim=t_dim,
        teacher_dropout_prob=tdrop,
        use_vision=False,
    ).to(device)
    model.load_state_dict(normalize_state_dict_keys_for_eval(ck["model_state_dict"]))
    model.eval()

    temporal_agg = bool(FLAGS.temporal_agg)
    if temporal_agg:
        qfreq = 1
    elif FLAGS.query_frequency < 0:
        qfreq = horizon
    else:
        qfreq = max(1, min(int(FLAGS.query_frequency), horizon))

    window_size = int(512 * FLAGS.window_scale)
    fast = bool(FLAGS.on_cuda)
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array" if fast else "human",
        visualization_width=window_size,
        visualization_height=window_size,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=FLAGS.max_steps)

    if FLAGS.random_seeds:
        seeds = np.random.default_rng().integers(0, 2**31, size=FLAGS.num_seeds).tolist()
    else:
        seeds = list(range(FLAGS.seed, FLAGS.seed + FLAGS.num_seeds))

    print(
        f"\nACTResHead (keypoints) | H={horizon} | temporal_agg={temporal_agg} | "
        f"decay={ensemble_decay} | success_thresh<={success_thresh}"
    )
    print(f"Seeds: {seeds[:8]}{'...' if len(seeds) > 8 else ''}")

    success_n = 0
    frames: List[np.ndarray] = []
    for ep_i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        step = 0
        term = trunc = False
        max_cov = 0.0
        clock = None if fast else pygame.time.Clock()
        latest = env.render() if fast else None

        if temporal_agg:
            all_t = torch.zeros(
                (FLAGS.max_steps, FLAGS.max_steps + horizon, 2), dtype=torch.float32, device=device
            )
        cache: Optional[np.ndarray] = None

        while not (term or trunc):
            if not fast:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
                        env.close()
                        return
            svec = build_state_vector(obs)
            s_t = torch.tensor(svec, dtype=torch.float32, device=device).unsqueeze(0)
            s_n = min_max_normalize(s_t, state_min, state_max)

            if step % qfreq == 0:
                with torch.no_grad():
                    pred_n, _, _ = model(
                        None, s_n, action_chunk=None, teacher_signal=teacher_t
                    )
                pred = (pred_n + 1.0) * 0.5 * (action_max - action_min).view(1, 1, -1) + action_min.view(1, 1, -1)
                if temporal_agg:
                    all_t[[step], step : step + horizon] = pred
                else:
                    cache = pred.squeeze(0).cpu().numpy().astype(np.float32)

            if temporal_agg:
                afs = all_t[max(0, step - horizon + 1) : step + 1, step]
                w = np.exp(-ensemble_decay * np.arange(afs.size(0) - 1, -1, -1))
                w = w / w.sum()
                wt = torch.from_numpy(w).float().to(device).unsqueeze(1)
                act = (afs * wt).sum(0).cpu().numpy()
            else:
                act = cache[step % qfreq]

            act = np.clip(act.astype(np.float32), 0.0, 512.0)
            obs, reward, term, trunc, info = env.step(act)
            cov = float(info.get("coverage", 0.0)) if isinstance(info, dict) else 0.0
            if cov > max_cov:
                max_cov = cov
            if max_cov >= success_thresh:
                trunc = True
            step += 1
            if step >= FLAGS.max_steps:
                trunc = True
            if fast:
                latest = env.render()
                if FLAGS.save_video and latest is not None:
                    frames.append(np.asarray(latest))
            else:
                env.render()
                draw_status_overlay(
                    env, ControlState.MODEL_CONTROL, int(seed), 0, step, FLAGS.max_steps,
                    svec[:2], False, reward=float(reward),
                )
                if FLAGS.save_video:
                    fr = capture_frame(env)
                    if fr is not None:
                        frames.append(fr)
                clock.tick(FLAGS.fps)

        ok = max_cov >= success_thresh
        if ok:
            success_n += 1
        print(
            f"Ep {ep_i+1}/{len(seeds)} seed={seed} "
            f"{'SUCCESS' if ok else 'FAIL'} steps={step} max_cov={max_cov:.3f}"
        )
        if FLAGS.save_video and frames:
            for _ in range(FLAGS.fps):
                frames.append(frames[-1])

    print("=" * 60)
    print(f"Success {success_n}/{len(seeds)} = {100*success_n/len(seeds):.1f}%")
    print("=" * 60)
    if FLAGS.save_video and frames:
        os.makedirs(FLAGS.video_dir, exist_ok=True)
        p = os.path.join(FLAGS.video_dir, time.strftime("%Y-%m-%d-%H-%M-%S.mp4"))
        imageio.mimwrite(p, frames, fps=FLAGS.fps)
        print(f"Saved video: {p}")
    env.close()


if __name__ == "__main__":
    app.run(main)
