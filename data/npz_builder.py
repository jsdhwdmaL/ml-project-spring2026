"""NPZ dataset builder for per-step PushT trajectory data."""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def _coerce_episode_arrays(episode: Dict) -> Dict[str, np.ndarray]:
    """Normalize one in-memory episode dict to canonical per-step arrays."""
    obs = episode.get("observation.state", episode.get("observations"))
    act = episode.get("action", episode.get("actions"))
    if obs is None or act is None:
        raise KeyError("Each episode must provide observation.state/observations and action/actions")

    observation_state = np.asarray(obs, dtype=np.float32)
    action = np.asarray(act, dtype=np.float32)

    if observation_state.ndim != 2 or observation_state.shape[1] != 2:
        raise ValueError(f"observation.state must be (N,2), got {observation_state.shape}")
    if action.ndim != 2 or action.shape[1] != 2:
        raise ValueError(f"action must be (N,2), got {action.shape}")
    if observation_state.shape[0] != action.shape[0]:
        raise ValueError("observation.state and action must have the same N")

    n_steps = observation_state.shape[0]

    frame_index = np.asarray(episode.get("frame_index", np.arange(n_steps)), dtype=np.int64)
    timestamp = np.asarray(episode.get("timestamp", frame_index.astype(np.float32)), dtype=np.float32)
    next_reward = np.asarray(episode.get("next.reward", episode.get("rewards", np.zeros(n_steps))), dtype=np.float32)
    next_done = np.asarray(episode.get("next.done", np.eye(1, n_steps, n_steps - 1, dtype=bool).reshape(-1) if n_steps > 0 else np.zeros(0, dtype=bool)), dtype=bool)
    next_success = np.asarray(episode.get("next.success", np.zeros(n_steps, dtype=bool)), dtype=bool)
    is_human = np.asarray(episode.get("is_human_intervention", np.zeros(n_steps, dtype=bool)), dtype=bool)

    if n_steps > 0 and next_done.shape[0] == n_steps and not np.any(next_done):
        next_done[-1] = True

    for name, arr in {
        "frame_index": frame_index,
        "timestamp": timestamp,
        "next.reward": next_reward,
        "next.done": next_done,
        "next.success": next_success,
        "is_human_intervention": is_human,
    }.items():
        if arr.ndim != 1 or arr.shape[0] != n_steps:
            raise ValueError(f"{name} must be (N,), got {arr.shape} with N={n_steps}")

    return {
        "observation.state": observation_state,
        "action": action,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "next.reward": next_reward,
        "next.done": next_done,
        "next.success": next_success,
        "is_human_intervention": is_human,
        "env_seed": np.array(int(episode.get("env_seed", -1)), dtype=np.int64),
        "trial_idx": np.array(int(episode.get("trial_idx", -1)), dtype=np.int64),
    }


def build_npz_from_episodes(
    episodes: List[Dict],
    output_path: str,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert in-memory episodes to one merged NPZ in canonical per-step schema."""
    normalized = [_coerce_episode_arrays(episode) for episode in episodes]

    all_obs = []
    all_actions = []
    all_episode_idx = []
    all_frame_idx = []
    all_timestamp = []
    all_reward = []
    all_done = []
    all_success = []
    all_is_human = []
    all_env_seed = []
    all_trial_idx = []

    cursor = 0
    for episode_id, episode in enumerate(normalized):
        n_steps = episode["observation.state"].shape[0]
        all_obs.append(episode["observation.state"])
        all_actions.append(episode["action"])
        all_episode_idx.append(np.full(n_steps, episode_id, dtype=np.int64))
        all_frame_idx.append(episode["frame_index"])
        all_timestamp.append(episode["timestamp"])
        all_reward.append(episode["next.reward"])
        all_done.append(episode["next.done"])
        all_success.append(episode["next.success"])
        all_is_human.append(episode["is_human_intervention"])
        all_env_seed.append(np.full(n_steps, int(episode["env_seed"]), dtype=np.int64))
        all_trial_idx.append(np.full(n_steps, int(episode["trial_idx"]), dtype=np.int64))
        cursor += n_steps

    data = {
        "observation.state": np.concatenate(all_obs, axis=0).astype(np.float32),
        "action": np.concatenate(all_actions, axis=0).astype(np.float32),
        "episode_index": np.concatenate(all_episode_idx, axis=0).astype(np.int64),
        "frame_index": np.concatenate(all_frame_idx, axis=0).astype(np.int64),
        "timestamp": np.concatenate(all_timestamp, axis=0).astype(np.float32),
        "next.reward": np.concatenate(all_reward, axis=0).astype(np.float32),
        "next.done": np.concatenate(all_done, axis=0).astype(bool),
        "next.success": np.concatenate(all_success, axis=0).astype(bool),
        "index": np.arange(cursor, dtype=np.int64),
        "task_index": np.zeros(cursor, dtype=np.int64),
        "is_human_intervention": np.concatenate(all_is_human, axis=0).astype(bool),
        "env_seed": np.concatenate(all_env_seed, axis=0).astype(np.int64),
        "trial_idx": np.concatenate(all_trial_idx, axis=0).astype(np.int64),
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path, **data)

    if verbose:
        print(f"Saved dataset to {output_path}")
        for key, value in data.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    return data


def verify_npz_format(
    collected_path: str,
    reference_path: str,
    expect_all_success: bool = True,
    verbose: bool = True,
) -> bool:
    """Verify collected NPZ matches reference schema (ignoring leading N)."""
    collected = np.load(collected_path)
    reference = np.load(reference_path)

    all_match = True

    for key in reference.keys():
        if key not in collected:
            if verbose:
                print(f"MISSING: {key}")
            all_match = False
            continue

        ref_dtype = reference[key].dtype
        col_dtype = collected[key].dtype
        ref_shape = reference[key].shape[1:]
        col_shape = collected[key].shape[1:]

        if ref_dtype != col_dtype or ref_shape != col_shape:
            if verbose:
                print(f"MISMATCH: {key}")
                print(f"  Reference: dtype={ref_dtype}, shape=(*,{ref_shape})")
                print(f"  Collected: dtype={col_dtype}, shape=(*,{col_shape})")
            all_match = False

    if "next.success" in collected:
        terminal_mask = collected["next.done"]
        terminal_success = collected["next.success"][terminal_mask]
        if expect_all_success and len(terminal_success) > 0 and not terminal_success.all():
            if verbose:
                print("Not all terminal samples are successful")
            all_match = False

    return all_match


def merge_episode_files(
    input_dir: str,
    output_path: str,
    pattern: str = "*.npz",
    exclude_pattern: str = "*_images.npz",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Merge per-episode NPZ files into canonical per-step dataset."""
    input_path = Path(input_dir)
    episode_files = sorted(input_path.glob(pattern))

    if exclude_pattern:
        exclude_files = set(input_path.glob(exclude_pattern))
        episode_files = [file_path for file_path in episode_files if file_path not in exclude_files]

    if len(episode_files) == 0:
        raise ValueError(f"No episode files found in {input_dir} with pattern {pattern}")

    if verbose:
        print(f"Found {len(episode_files)} episode files in {input_dir}")

    episodes = []
    for file_path in episode_files:
        data = np.load(file_path)

        required = ["observation.state", "action", "next.done"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Episode file {file_path} missing keys: {missing}")

        obs = np.asarray(data["observation.state"], dtype=np.float32)
        act = np.asarray(data["action"], dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != 2:
            raise ValueError(f"Episode file {file_path} has invalid observation.state shape: {obs.shape}")
        if act.ndim != 2 or act.shape[1] != 2:
            raise ValueError(f"Episode file {file_path} has invalid action shape: {act.shape}")
        if obs.shape[0] != act.shape[0]:
            raise ValueError(
                f"Episode file {file_path} mismatched N: observation.state={obs.shape[0]}, action={act.shape[0]}"
            )

        n_steps = obs.shape[0]
        frame_index = np.asarray(data["frame_index"], dtype=np.int64) if "frame_index" in data else np.arange(n_steps, dtype=np.int64)
        timestamp = np.asarray(data["timestamp"], dtype=np.float32) if "timestamp" in data else frame_index.astype(np.float32)
        next_reward = np.asarray(data["next.reward"], dtype=np.float32) if "next.reward" in data else np.zeros(n_steps, dtype=np.float32)
        next_done = np.asarray(data["next.done"], dtype=bool)
        next_success = np.asarray(data["next.success"], dtype=bool) if "next.success" in data else np.zeros(n_steps, dtype=bool)
        is_human = np.asarray(data["is_human_intervention"], dtype=bool) if "is_human_intervention" in data else np.zeros(n_steps, dtype=bool)

        for name, arr in {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "next.reward": next_reward,
            "next.done": next_done,
            "next.success": next_success,
            "is_human_intervention": is_human,
        }.items():
            if arr.ndim != 1 or arr.shape[0] != n_steps:
                raise ValueError(
                    f"Episode file {file_path} invalid {name} shape: {arr.shape}, expected ({n_steps},)"
                )

        env_seed = int(np.array(data["env_seed"]).item()) if "env_seed" in data else -1
        trial_idx = int(np.array(data["trial_idx"]).item()) if "trial_idx" in data else -1

        episodes.append(
            {
                "observation.state": obs,
                "action": act,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "next.reward": next_reward,
                "next.done": next_done,
                "next.success": next_success,
                "is_human_intervention": is_human,
                "env_seed": env_seed,
                "trial_idx": trial_idx,
            }
        )

    merged = build_npz_from_episodes(episodes, output_path=output_path, verbose=False)

    if verbose:
        print(f"Loaded {len(episode_files)} episodes")
        print(f"  Total samples: {merged['index'].shape[0]}")
        print(f"Saved merged dataset to {output_path}")

    return merged
