"""Data loading utilities for the LeRobot ``pusht_keypoints`` dataset.

This module exposes :func:`build_keypoints_dataloaders`, which returns a
:class:`KeypointsDataBundle` with train/val DataLoaders and the **min/max**
normalization stats required at train and inference time.

Dataset (HuggingFace ``lerobot/pusht_keypoints``, LeRobot v3.0):
  observation.state              (N, 2)   float32  agent xy
  observation.environment_state  (N, 16)  float32  T-block keypoints
  action                          (N, 2)   float32  target agent xy

The policy state input is the concatenation
``[observation.state | observation.environment_state]`` of length ``18``,
matching gym_pusht's ``environment_state_agent_pos`` observation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.build_chunk import build_action_chunks_by_episode


def split_episode_indices(
    episode_index: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Episode-level train/val split returning step indices for each split."""
    unique_eps = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)

    n_val_eps = max(1, int(len(unique_eps) * val_ratio)) if val_ratio > 0 else 0
    val_eps = set(unique_eps[:n_val_eps].tolist())

    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]

    if train_idx.size == 0:
        raise ValueError("Empty keypoints train split")
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def min_max_normalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Map x from [lo, hi] to [-1, 1]. Broadcasts on trailing dims."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def min_max_denormalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`min_max_normalize`."""
    return (x + 1.0) * 0.5 * (hi - lo) + lo


class KeypointsStepDataset(Dataset):
    """Torch Dataset over preloaded keypoints arrays.

    Each item returns:
      state:         (state_dim,) float32  (un-normalized)
      action_chunk:  (horizon, 2) float32  (un-normalized)
      action_is_pad: (horizon,)   bool
    """

    def __init__(
        self,
        states: np.ndarray,            # (N, state_dim) float32
        action_chunks: np.ndarray,     # (N, horizon, 2) float32
        action_is_pad: np.ndarray,     # (N, horizon) bool
        step_indices: np.ndarray,      # (M,) int64 indices into N
    ):
        self.states = states
        self.action_chunks = action_chunks
        self.action_is_pad = action_is_pad
        self.step_indices = step_indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.step_indices.shape[0])

    def __getitem__(self, idx: int):
        gi = int(self.step_indices[idx])
        return {
            "state": torch.from_numpy(self.states[gi]).float(),
            "action_chunk": torch.from_numpy(self.action_chunks[gi]).float(),
            "action_is_pad": torch.from_numpy(self.action_is_pad[gi]).bool(),
        }


@dataclass
class KeypointsDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    state_min: np.ndarray
    state_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    state_dim: int
    action_dim: int
    num_train: int
    num_val: int


def _safe_max(lo: np.ndarray, hi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Avoid zero-range channels by ensuring hi >= lo + eps."""
    return np.maximum(hi, lo + eps).astype(np.float32)


def build_keypoints_dataloaders(
    dataset_id: str = "lerobot/pusht_keypoints",
    horizon: int = 16,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> KeypointsDataBundle:
    """Build train/val DataLoaders for the LeRobot pusht_keypoints dataset.

    Returns a :class:`KeypointsDataBundle` containing the loaders and
    per-channel min/max stats computed on the train split only.
    """
    # Local import so importing this module doesn't require lerobot installed.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_id)
    hf = dataset.hf_dataset

    episode_index = np.array(hf["episode_index"], dtype=np.int64).flatten()
    actions = np.array(hf["action"], dtype=np.float32)
    obs_state = np.array(hf["observation.state"], dtype=np.float32)
    obs_env = np.array(hf["observation.environment_state"], dtype=np.float32)

    if not (actions.shape[0] == obs_state.shape[0] == obs_env.shape[0] == episode_index.shape[0]):
        raise ValueError(
            "Inconsistent N across keypoints arrays: "
            f"actions={actions.shape}, state={obs_state.shape}, "
            f"env_state={obs_env.shape}, episode_index={episode_index.shape}"
        )

    states = np.concatenate([obs_state, obs_env], axis=-1).astype(np.float32)  # (N, 18)

    action_chunks, action_is_pad = build_action_chunks_by_episode(
        actions, episode_index, horizon
    )

    train_idx, val_idx = split_episode_indices(episode_index, val_ratio, seed)

    state_min = states[train_idx].min(axis=0).astype(np.float32)
    state_max = _safe_max(state_min, states[train_idx].max(axis=0))
    action_min = actions[train_idx].min(axis=0).astype(np.float32)
    action_max = _safe_max(action_min, actions[train_idx].max(axis=0))

    train_ds = KeypointsStepDataset(states, action_chunks, action_is_pad, train_idx)
    val_ds = KeypointsStepDataset(states, action_chunks, action_is_pad, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return KeypointsDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        state_min=state_min,
        state_max=state_max,
        action_min=action_min,
        action_max=action_max,
        state_dim=int(states.shape[1]),
        action_dim=int(actions.shape[1]),
        num_train=int(train_idx.shape[0]),
        num_val=int(val_idx.shape[0]),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test keypoints dataloader")
    parser.add_argument("--dataset_id", type=str, default="lerobot/pusht_keypoints")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    bundle = build_keypoints_dataloaders(
        dataset_id=args.dataset_id,
        horizon=args.horizon,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=0,
        pin_memory=False,
    )
    print(f"state_dim={bundle.state_dim}, action_dim={bundle.action_dim}")
    print(f"num_train={bundle.num_train}, num_val={bundle.num_val}")
    print(f"state_min shape={bundle.state_min.shape}, state_max shape={bundle.state_max.shape}")
    print(f"action_min={bundle.action_min}, action_max={bundle.action_max}")
    batch = next(iter(bundle.train_loader))
    print(
        f"sample batch -> state {tuple(batch['state'].shape)}, "
        f"action_chunk {tuple(batch['action_chunk'].shape)}, "
        f"action_is_pad {tuple(batch['action_is_pad'].shape)}"
    )
