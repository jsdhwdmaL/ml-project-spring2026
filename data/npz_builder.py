"""NPZ dataset builder for trajectory data.
Provides utilities for converting episode lists to merged NPZ datasets.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def build_npz_from_episodes(
    episodes: List[Dict],
    output_path: str,
    horizon: int = 16,
    sparse_reward_scheme: str = "standard",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert episode list to merged NPZ dataset.

    Expected episode dict format:
        - observations: List of (n_obs_steps, state_dim) states
        - action_chunks: List of (horizon, act_dim) action chunks
        - shift_amounts: List of int shift amounts (optional)
        - rewards: List of float raw environment rewards
        - dones: List of bool termination flags
        - success: bool episode success flag

    Output NPZ format:
        observation.state: (N, 2, 18) - frame-stacked states
        action: (N, 16, 2) - 16-step action chunks
        action_is_pad: (N, 16) - padding mask
        episode_index: (N,) - episode IDs
        frame_index: (N,) - frame within episode
        environment.raw_reward: (N,) - raw environment rewards
        next.reward: (N,) - sparse rewards (scheme-dependent)
        next.done: (N,) - terminal flags
        next.success: (N,) - success flags
        timestamp: (N,) - frame index as float
        index: (N,) - global sample index
        task_index: (N,) - all zeros

    Args:
        episodes: List of episode dicts
        output_path: Path to save NPZ file
        horizon: Action chunk horizon (default 16)
        sparse_reward_scheme: Reward scheme for next.reward:
            - "standard": -1 per step, 0 at terminal (legacy)
            - "penalize_failure": -1 per step, -10 at failed terminal, 0 at success
        verbose: Whether to print progress

    Returns:
        Dict of numpy arrays (same as saved to NPZ)
    """
    all_observations = []
    all_actions = []
    all_action_is_pad = []
    all_episode_indices = []
    all_frame_indices = []
    all_raw_rewards = []
    all_sparse_rewards = []
    all_dones = []
    all_success = []
    all_timestamps = []

    global_idx = 0

    for ep_idx, episode in enumerate(episodes):
        num_frames = len(episode['observations'])
        shift_amounts = episode.get('shift_amounts', [0] * num_frames)
        raw_rewards = episode.get('rewards', [0.0] * num_frames)

        for frame_idx in range(num_frames):
            obs = episode['observations'][frame_idx]
            action_chunk = episode['action_chunks'][frame_idx]
            shift_amount = shift_amounts[frame_idx]
            raw_reward = raw_rewards[frame_idx] if frame_idx < len(raw_rewards) else 0.0

            # Compute action padding mask combining:
            # 1. Shift-based padding: positions that shifted out
            # 2. Episode-end padding: positions beyond the episode end
            remaining_steps = num_frames - frame_idx
            action_is_pad = np.zeros(horizon, dtype=bool)

            # Episode-end padding
            if remaining_steps < horizon:
                action_is_pad[remaining_steps:] = True

            # Shift-based padding (mark shifted-in positions as padded)
            if shift_amount > 0:
                action_is_pad[-shift_amount:] = True

            # Only the last frame is terminal
            is_terminal = frame_idx == num_frames - 1

            # Compute sparse reward based on scheme
            if sparse_reward_scheme == "penalize_failure":
                # -1 per step, -10 at failed terminal, 0 at successful terminal
                if is_terminal:
                    sparse_reward = 0.0 if episode['success'] else -10.0
                else:
                    sparse_reward = -1.0
            else:  # "standard"
                # -1 per step, 0 at terminal (legacy behavior)
                sparse_reward = 0.0 if is_terminal else -1.0

            # next.success is True only at terminal step of successful episodes
            step_success = is_terminal and episode['success']

            all_observations.append(obs)
            all_actions.append(action_chunk)
            all_action_is_pad.append(action_is_pad)
            all_episode_indices.append(ep_idx)
            all_frame_indices.append(frame_idx)
            all_raw_rewards.append(raw_reward)
            all_sparse_rewards.append(sparse_reward)
            all_dones.append(is_terminal)
            all_success.append(step_success)
            all_timestamps.append(float(frame_idx))

            global_idx += 1

    # Convert to numpy arrays with correct dtypes (matching original dataset)
    data = {
        'observation.state': np.array(all_observations, dtype=np.float64),
        'action': np.array(all_actions, dtype=np.float64),
        'action_is_pad': np.array(all_action_is_pad, dtype=bool),
        'episode_index': np.array(all_episode_indices, dtype=np.int64),
        'frame_index': np.array(all_frame_indices, dtype=np.int64),
        'environment.raw_reward': np.array(all_raw_rewards, dtype=np.float64),
        'next.reward': np.array(all_sparse_rewards, dtype=np.float64),
        'next.done': np.array(all_dones, dtype=bool),
        'next.success': np.array(all_success, dtype=bool),
        'timestamp': np.array(all_timestamps, dtype=np.float64),
        'index': np.arange(global_idx, dtype=np.int64),
        'task_index': np.zeros(global_idx, dtype=np.int64),
    }

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save
    np.savez(output_path, **data)

    if verbose:
        print(f"\nSaved dataset to {output_path}")
        print(f"  Total samples: {global_idx}")
        print(f"  Total episodes: {len(episodes)}")
        print(f"  Sparse reward scheme: {sparse_reward_scheme}")
        for key, arr in data.items():
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        # Show reward stats
        print(f"\n  Reward stats:")
        print(f"    environment.raw_reward: [{data['environment.raw_reward'].min():.4f}, {data['environment.raw_reward'].max():.4f}]")
        print(f"    next.reward (sparse): {np.unique(data['next.reward'])}")

    return data


def verify_npz_format(
    collected_path: str,
    reference_path: str,
    expect_all_success: bool = True,
    verbose: bool = True,
) -> bool:
    """Verify that collected NPZ matches reference format.

    Args:
        collected_path: Path to collected NPZ file
        reference_path: Path to reference NPZ file
        expect_all_success: If True, verify all episodes are successful
                           (for rejection sampling). If False, report
                           success/failure mix (for diverse data).
        verbose: Whether to print details

    Returns:
        True if format matches, False otherwise
    """
    collected = np.load(collected_path)
    reference = np.load(reference_path)

    if verbose:
        print("\nVerifying NPZ format...")

    all_match = True

    # Check all reference keys exist with matching dtype and shape
    for key in reference.keys():
        if key not in collected:
            if verbose:
                print(f"  MISSING: {key}")
            all_match = False
            continue

        ref_dtype = reference[key].dtype
        col_dtype = collected[key].dtype
        ref_shape = reference[key].shape[1:]  # Compare shapes excluding N
        col_shape = collected[key].shape[1:]

        dtype_match = ref_dtype == col_dtype
        shape_match = ref_shape == col_shape

        if dtype_match and shape_match:
            if verbose:
                print(f"  OK: {key} (dtype={col_dtype}, shape=(*,{col_shape}))")
        else:
            if verbose:
                print(f"  MISMATCH: {key}")
                print(f"    Reference: dtype={ref_dtype}, shape=(*,{ref_shape})")
                print(f"    Collected: dtype={col_dtype}, shape=(*,{col_shape})")
            all_match = False

    # Verify success pattern
    if 'next.success' in collected:
        if expect_all_success:
            # For rejection sampling: all terminal frames should be successful
            terminal_mask = collected['next.done']
            terminal_success = collected['next.success'][terminal_mask]
            all_success = terminal_success.all() if len(terminal_success) > 0 else True
            if verbose:
                print(f"\n  All episodes successful: {all_success}")
            if not all_success:
                all_match = False
        else:
            # For diverse data: report success/failure mix
            if verbose:
                success_count = np.sum(collected['next.success'])
                failure_count = len(collected['next.success']) - success_count
                print(f"\n  Success samples: {success_count}")
                print(f"  Failure samples: {failure_count}")

                # Count unique episodes by success
                if 'episode_index' in collected:
                    unique_eps = np.unique(collected['episode_index'])
                    success_eps = 0
                    failure_eps = 0
                    for ep in unique_eps:
                        mask = collected['episode_index'] == ep
                        # Check terminal frame's success flag
                        ep_done = collected['next.done'][mask]
                        ep_success = collected['next.success'][mask]
                        terminal_idx = np.where(ep_done)[0]
                        if len(terminal_idx) > 0 and ep_success[terminal_idx[0]]:
                            success_eps += 1
                        else:
                            failure_eps += 1
                    print(f"  Success episodes: {success_eps}")
                    print(f"  Failure episodes: {failure_eps}")

    # Verify done pattern: only last frame of each episode should have done=True
    if 'episode_index' in collected and 'frame_index' in collected and 'next.done' in collected:
        if verbose:
            print("\n  Verifying done pattern (only last frame should have done=True)...")

        episode_idx = collected['episode_index']
        frame_idx = collected['frame_index']
        next_done = collected['next.done']

        unique_eps = np.unique(episode_idx)
        done_pattern_ok = True

        for ep in unique_eps[:5]:  # Check first 5 episodes
            mask = episode_idx == ep
            ep_frames = frame_idx[mask]
            ep_done = next_done[mask]

            sorted_idx = np.argsort(ep_frames)
            ep_done_sorted = ep_done[sorted_idx]

            num_done = np.sum(ep_done_sorted)
            last_done = ep_done_sorted[-1]
            others_not_done = np.all(~ep_done_sorted[:-1]) if len(ep_done_sorted) > 1 else True

            if not (num_done == 1 and last_done and others_not_done):
                if verbose:
                    print(f"    Episode {ep}: UNEXPECTED - {num_done} done flags")
                done_pattern_ok = False
            else:
                if verbose:
                    print(f"    Episode {ep}: OK (1 done flag at terminal)")

        if done_pattern_ok:
            if verbose:
                print("  Done pattern verification: PASSED")
        else:
            if verbose:
                print("  Done pattern verification: FAILED")
            all_match = False

    # Verify action chunk shifting (for diverse data)
    if not expect_all_success and 'action' in collected and 'episode_index' in collected:
        if verbose:
            print("\n  Verifying action chunk shifting...")

        episode_idx = collected['episode_index']
        actions = collected['action']

        # Check first episode
        ep0_mask = episode_idx == 0
        ep0_actions = actions[ep0_mask]

        shift_ok = True
        for i in range(min(16, len(ep0_actions) - 1)):
            if np.allclose(ep0_actions[i], ep0_actions[i + 1]):
                shift_ok = False
                if verbose:
                    print(f"    Frame {i} and {i+1} have identical action chunks!")
                break

        if shift_ok:
            if verbose:
                print("  Action chunk shifting verification: PASSED")
        else:
            if verbose:
                print("  Action chunk shifting verification: FAILED")
            all_match = False

    return all_match


def merge_episode_files(
    input_dir: str,
    output_path: str,
    pattern: str = "*.npz",
    exclude_pattern: str = "*_images.npz",
    horizon: int = 16,
    sparse_reward_scheme: str = "penalize_failure",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Merge individual episode NPZ files into a single dataset.

    Useful for merging human intervention data collected as separate files.

    Args:
        input_dir: Directory containing episode NPZ files
        output_path: Path to save merged NPZ file
        pattern: Glob pattern for episode files
        exclude_pattern: Pattern to exclude (e.g., image files)
        horizon: Action chunk horizon
        sparse_reward_scheme: Reward scheme for next.reward:
            - "standard": -1 per step, 0 at terminal
            - "penalize_failure": -1 per step, -10 at failed terminal, 0 at success
        verbose: Whether to print progress

    Returns:
        Dict of merged numpy arrays
    """
    input_path = Path(input_dir)
    episode_files = sorted(input_path.glob(pattern))

    # Exclude image files
    if exclude_pattern:
        exclude_files = set(input_path.glob(exclude_pattern))
        episode_files = [f for f in episode_files if f not in exclude_files]

    if verbose:
        print(f"Found {len(episode_files)} episode files in {input_dir}")

    if len(episode_files) == 0:
        raise ValueError(f"No episode files found in {input_dir} with pattern {pattern}")

    all_observations = []
    all_actions = []
    all_action_is_pad = []
    all_episode_indices = []
    all_frame_indices = []
    all_raw_rewards = []
    all_sparse_rewards = []
    all_dones = []
    all_success = []
    all_timestamps = []
    all_is_human_intervention = []
    all_is_pure_teleop = []
    all_env_seeds = []
    all_trial_idx = []

    global_idx = 0

    for ep_idx, ep_file in enumerate(episode_files):
        data = np.load(ep_file)

        required_keys = ["observation.state", "action", "next.done"]
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(f"Episode file {ep_file} missing keys: {missing}")

        observations = np.array(data["observation.state"], dtype=np.float64)
        actions = np.array(data["action"], dtype=np.float64)
        num_frames = observations.shape[0]

        if actions.shape[0] != num_frames:
            raise ValueError(
                f"Episode file {ep_file} has mismatched first dimension: "
                f"observation.state={observations.shape[0]}, action={actions.shape[0]}"
            )

        action_horizon = actions.shape[1] if actions.ndim >= 3 else horizon
        if "action_is_pad" in data:
            action_is_pad = np.array(data["action_is_pad"], dtype=bool)
        else:
            action_is_pad = np.zeros((num_frames, action_horizon), dtype=bool)

        if action_is_pad.shape[0] != num_frames:
            raise ValueError(
                f"Episode file {ep_file} has mismatched action_is_pad length: "
                f"expected {num_frames}, got {action_is_pad.shape[0]}"
            )

        if "frame_index" in data:
            frame_index = np.array(data["frame_index"], dtype=np.int64)
        else:
            frame_index = np.arange(num_frames, dtype=np.int64)

        if "environment.raw_reward" in data:
            raw_reward = np.array(data["environment.raw_reward"], dtype=np.float64)
        elif "next.reward" in data:
            raw_reward = np.array(data["next.reward"], dtype=np.float64)
        else:
            raw_reward = np.zeros(num_frames, dtype=np.float64)

        if "next.done" in data:
            next_done = np.array(data["next.done"], dtype=bool)
        else:
            next_done = np.zeros(num_frames, dtype=bool)
            if num_frames > 0:
                next_done[-1] = True

        if "next.success" in data:
            next_success = np.array(data["next.success"], dtype=bool)
        else:
            next_success = np.zeros(num_frames, dtype=bool)
            if num_frames > 0:
                episode_success = bool(data["success"]) if "success" in data else False
                next_success[-1] = episode_success

        is_human_intervention = np.array(
            data["is_human_intervention"], dtype=bool
        ) if "is_human_intervention" in data else np.zeros(num_frames, dtype=bool)

        is_pure_teleop = np.array(
            data["is_pure_teleop"], dtype=bool
        ) if "is_pure_teleop" in data else np.zeros(num_frames, dtype=bool)

        if is_human_intervention.shape[0] != num_frames:
            raise ValueError(
                f"Episode file {ep_file} has mismatched is_human_intervention length: "
                f"expected {num_frames}, got {is_human_intervention.shape[0]}"
            )
        if is_pure_teleop.shape[0] != num_frames:
            raise ValueError(
                f"Episode file {ep_file} has mismatched is_pure_teleop length: "
                f"expected {num_frames}, got {is_pure_teleop.shape[0]}"
            )

        if "env_seed" in data:
            env_seed = int(np.array(data["env_seed"]).item())
        else:
            env_seed = -1
        if "trial_idx" in data:
            trial_idx = int(np.array(data["trial_idx"]).item())
        else:
            trial_idx = -1

        if sparse_reward_scheme == "penalize_failure":
            sparse_reward = np.full(num_frames, -1.0, dtype=np.float64)
            if num_frames > 0:
                terminal_success = bool(next_success[-1])
                sparse_reward[-1] = 0.0 if terminal_success else -10.0
        else:
            sparse_reward = np.full(num_frames, -1.0, dtype=np.float64)
            if num_frames > 0:
                sparse_reward[-1] = 0.0

        all_observations.append(observations)
        all_actions.append(actions)
        all_action_is_pad.append(action_is_pad)
        all_episode_indices.append(np.full(num_frames, ep_idx, dtype=np.int64))
        all_frame_indices.append(frame_index)
        all_raw_rewards.append(raw_reward)
        all_sparse_rewards.append(sparse_reward)
        all_dones.append(next_done)
        all_success.append(next_success)
        all_timestamps.append(frame_index.astype(np.float64))
        all_is_human_intervention.append(is_human_intervention)
        all_is_pure_teleop.append(is_pure_teleop)
        all_env_seeds.append(np.full(num_frames, env_seed, dtype=np.int64))
        all_trial_idx.append(np.full(num_frames, trial_idx, dtype=np.int64))
        global_idx += num_frames

    data = {
        "observation.state": np.concatenate(all_observations, axis=0).astype(np.float64),
        "action": np.concatenate(all_actions, axis=0).astype(np.float64),
        "action_is_pad": np.concatenate(all_action_is_pad, axis=0).astype(bool),
        "episode_index": np.concatenate(all_episode_indices, axis=0).astype(np.int64),
        "frame_index": np.concatenate(all_frame_indices, axis=0).astype(np.int64),
        "environment.raw_reward": np.concatenate(all_raw_rewards, axis=0).astype(np.float64),
        "next.reward": np.concatenate(all_sparse_rewards, axis=0).astype(np.float64),
        "next.done": np.concatenate(all_dones, axis=0).astype(bool),
        "next.success": np.concatenate(all_success, axis=0).astype(bool),
        "timestamp": np.concatenate(all_timestamps, axis=0).astype(np.float64),
        "index": np.arange(global_idx, dtype=np.int64),
        "task_index": np.zeros(global_idx, dtype=np.int64),
        "is_human_intervention": np.concatenate(all_is_human_intervention, axis=0).astype(bool),
        "is_pure_teleop": np.concatenate(all_is_pure_teleop, axis=0).astype(bool),
        "env_seed": np.concatenate(all_env_seeds, axis=0).astype(np.int64),
        "trial_idx": np.concatenate(all_trial_idx, axis=0).astype(np.int64),
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path, **data)

    if verbose:
        print(f"Loaded {len(episode_files)} episodes")
        success_count = int(np.sum(data["next.success"]))
        print(f"  Terminal success samples: {success_count}")
        print(f"  Total samples: {global_idx}")
        print(f"Saved merged dataset to {output_path}")

    return data
