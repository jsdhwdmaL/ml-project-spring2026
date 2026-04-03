"""Chunk building utilities for ACT-style training-time targets.

This module is intentionally separate from NPZ building so chunk construction
is done at training/data-loader time from canonical per-step actions.
"""

from typing import Tuple

import numpy as np


def build_action_chunks_from_raw(actions: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build fixed-horizon action chunks from per-step raw actions.

    Args:
        actions: Array with shape (N, 2)
        horizon: Chunk horizon H

    Returns:
        Tuple `(action_chunks, action_is_pad)` where:
          - `action_chunks` has shape (N, H, 2)
          - `action_is_pad` has shape (N, H)
    """
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"actions must be (N,2), got {actions.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    n_steps = actions.shape[0]
    chunks = np.zeros((n_steps, horizon, 2), dtype=np.float32)
    is_pad = np.zeros((n_steps, horizon), dtype=bool)

    for step in range(n_steps):
        for offset in range(horizon):
            future_step = step + offset
            if future_step < n_steps:
                chunks[step, offset] = actions[future_step]
            else:
                chunks[step, offset] = actions[-1]
                is_pad[step, offset] = True

    return chunks, is_pad


def build_action_chunks_by_episode(
	actions: np.ndarray,
	episode_index: np.ndarray,
	horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build action chunks without crossing episode boundaries.

    Args:
        actions: Array with shape (N, 2)
        episode_index: Array with shape (N,) identifying episode ID per step
        horizon: Chunk horizon H

    Returns:
        Tuple `(action_chunks, action_is_pad)` with shapes (N, H, 2) and (N, H).
    """
    if episode_index.ndim != 1:
        raise ValueError(f"episode_index must be (N,), got {episode_index.shape}")
    if actions.shape[0] != episode_index.shape[0]:
        raise ValueError(
            f"actions and episode_index length mismatch: {actions.shape[0]} vs {episode_index.shape[0]}"
        )

    n_steps = actions.shape[0]
    chunks = np.zeros((n_steps, horizon, 2), dtype=np.float32)
    is_pad = np.zeros((n_steps, horizon), dtype=bool)

    unique_episodes = np.unique(episode_index)
    for ep_id in unique_episodes:
        mask = episode_index == ep_id
        ep_actions = actions[mask]
        ep_chunks, ep_pad = build_action_chunks_from_raw(ep_actions, horizon)
        chunks[mask] = ep_chunks
        is_pad[mask] = ep_pad

    return chunks, is_pad

