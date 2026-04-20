"""Data loading utilities for the local Push-T zarr replay buffer.

The dataset lives at ``data/pusht/pusht_cchi_v7_replay.zarr`` and is the
original Diffusion Policy Push-T replay buffer (Chi et al.). It is a flat
buffer (all episodes concatenated) with these arrays:

  data/action       (N, 2)         float32  end-effector xy targets
  data/state        (N, 5)         float32  [agent_xy, block_xy, block_theta]
  data/img          (N, 96, 96, 3) float32  RGB in [0, 255], HWC
  data/keypoint     (N, 9, 2)      float32  (unused)
  data/n_contacts   (N, 1)         float32  (unused)
  meta/episode_ends (E,)           int64    exclusive end indices per episode

This module exposes a ``build_pusht_dataloaders`` one-call helper that the
training script can use without knowing anything about zarr.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.build_chunk import build_action_chunks_by_episode


def _episode_ends_to_index(episode_ends: np.ndarray, n_steps: int) -> np.ndarray:
    """Convert exclusive ``episode_ends`` to a per-step episode id array."""
    episode_index = np.zeros(n_steps, dtype=np.int64)
    start = 0
    for ep_id, end in enumerate(episode_ends):
        end = int(end)
        episode_index[start:end] = ep_id
        start = end
    if start != n_steps:
        raise ValueError(
            f"episode_ends final value {start} does not match n_steps {n_steps}"
        )
    return episode_index


def load_pusht_zarr(path: str) -> Dict[str, np.ndarray]:
    """Load the Push-T replay buffer into in-memory numpy arrays.

    Images are stored as ``uint8`` to keep RAM usage modest (~700 MB for the
    full 25,650-step buffer).
    """
    import zarr  # local import so callers without zarr can still import the module

    root = zarr.open(path, mode="r")
    actions = np.asarray(root["data/action"], dtype=np.float32)
    states = np.asarray(root["data/state"], dtype=np.float32)
    images_f = np.asarray(root["data/img"])  # float32 in [0, 255]
    images = np.clip(images_f, 0.0, 255.0).astype(np.uint8)
    episode_ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)

    if not (actions.shape[0] == states.shape[0] == images.shape[0]):
        raise ValueError(
            f"Inconsistent N across arrays: actions={actions.shape}, "
            f"states={states.shape}, images={images.shape}"
        )

    episode_index = _episode_ends_to_index(episode_ends, n_steps=actions.shape[0])
    return {
        "images": images,            # (N, 96, 96, 3) uint8
        "states": states,            # (N, 5) float32
        "actions": actions,          # (N, 2) float32
        "episode_index": episode_index,  # (N,) int64
    }


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
        raise ValueError("Empty Push-T train split")
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


class PushTZarrDataset(Dataset):
    """Torch Dataset over preloaded Push-T arrays.

    Each item returns:
      image:         (3, 96, 96) float32 in [0, 1]
      state:         (state_dim,) float32
      action_chunk:  (horizon, 2) float32
      action_is_pad: (horizon,)   bool
    """

    def __init__(
        self,
        images: np.ndarray,            # (N, 96, 96, 3) uint8
        states: np.ndarray,            # (N, state_dim) float32
        action_chunks: np.ndarray,     # (N, horizon, 2) float32
        action_is_pad: np.ndarray,     # (N, horizon) bool
        step_indices: np.ndarray,      # (M,) int64 indices into N
    ):
        self.images = images
        self.states = states
        self.action_chunks = action_chunks
        self.action_is_pad = action_is_pad
        self.step_indices = step_indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.step_indices.shape[0])

    def __getitem__(self, idx: int):
        gi = int(self.step_indices[idx])
        img = self.images[gi]  # (96, 96, 3) uint8
        # HWC -> CHW, scale to [0, 1]
        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return {
            "image": img_t,                                                 # (3,96,96) float32 [0,1]
            "state": torch.from_numpy(self.states[gi]).float(),             # (state_dim,)
            "action_chunk": torch.from_numpy(self.action_chunks[gi]).float(),  # (H,2)
            "action_is_pad": torch.from_numpy(self.action_is_pad[gi]).bool(),  # (H,)
        }


@dataclass
class PushTDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    state_dim: int
    action_dim: int
    num_train: int
    num_val:   int


def build_pusht_dataloaders(
    zarr_path: str,
    horizon: int,
    batch_size: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> PushTDataBundle:
    """Build train/val DataLoaders for the local Push-T zarr buffer.

    Returns a ``PushTDataBundle`` containing the loaders, normalization stats
    computed on the train split only, and helpful metadata.
    """
    arrays = load_pusht_zarr(zarr_path)
    images = arrays["images"]
    states = arrays["states"]
    actions = arrays["actions"]
    episode_index = arrays["episode_index"]

    action_chunks, action_is_pad = build_action_chunks_by_episode(
        actions, episode_index, horizon
    )

    train_idx, val_idx = split_episode_indices(episode_index, val_ratio, seed)

    state_mean = states[train_idx].mean(axis=0).astype(np.float32)
    state_std = (states[train_idx].std(axis=0) + 1e-6).astype(np.float32)
    action_mean = actions[train_idx].mean(axis=0).astype(np.float32)
    action_std = (actions[train_idx].std(axis=0) + 1e-6).astype(np.float32)

    train_ds = PushTZarrDataset(images, states, action_chunks, action_is_pad, train_idx)
    val_ds = PushTZarrDataset(images, states, action_chunks, action_is_pad, val_idx)

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

    return PushTDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        state_dim=int(states.shape[1]),
        action_dim=int(actions.shape[1]),
        num_train=int(train_idx.shape[0]),
        num_val=int(val_idx.shape[0]),
    )
