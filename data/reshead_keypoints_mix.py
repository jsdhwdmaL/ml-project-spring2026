"""Load and mix LeRobot keypoints with local NPZ (e.g. act_cr DAgger / res-blend) for ACTResHead.

Teacher signal per step (default 2-D):
  [is_human_intervention, dagger_presence]
  - dagger_presence=1.0 for rows from local NPZ; 0.0 for pure LeRobot rows.
  - is_human=0/1 from NPZ; LeRobot rows use [0, 0] (test-time "no hint" default).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.build_chunk import build_action_chunks_by_episode
from data.dataloader_keypoints import (
    KeypointsStepDataset,
    build_keypoints_dataloaders,
    min_max_normalize,
)

AGENT_POS_DIM = 2
ENV_STATE_DIM = 16
EXPECTED_STATE_DIM = AGENT_POS_DIM + ENV_STATE_DIM
EXPECTED_ACTION_DIM = 2


def _safe_max(lo: np.ndarray, hi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(hi, lo + eps).astype(np.float32)


def _episode_success(data) -> bool:
    if "success" in data.files:
        return bool(np.asarray(data["success"]).reshape(-1)[0])
    if "next.success" in data.files:
        return bool(np.asarray(data["next.success"]).any())
    raise KeyError("Episode file has no success key")


def _contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if mask.size == 0:
        return runs
    in_run = False
    start = 0
    for i, v in enumerate(mask.tolist()):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, int(mask.size)))
    return runs


@dataclass
class DaggerLoadConfig:
    data_dir: str
    success_only: bool
    keep_only_human: bool
    include_human_intervention: bool
    include_rejection_sample: bool
    include_failed_autonomous: bool


def _collect_episode_files(cfg: DaggerLoadConfig) -> List[Path]:
    root = Path(cfg.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {root}")

    folders: List[Path] = []
    if cfg.include_human_intervention:
        folders.append(root / "human_intervention")
    if cfg.include_rejection_sample:
        folders.append(root / "rejection_sample")
    if cfg.include_failed_autonomous:
        folders.append(root / "failed_autonomous")

    files: List[Path] = []
    for folder in folders:
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.npz")):
            if file_path.name.endswith("_images.npz"):
                continue
            files.append(file_path)

    if not files:
        raise ValueError("No candidate episode files found in selected DAgger folders")
    return files


def load_dagger_keypoints_with_teacher(cfg: DaggerLoadConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Load NPZ episodes; return states, actions, episode_index, teacher (N,2), stats."""
    files = _collect_episode_files(cfg)
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_episode_index: List[np.ndarray] = []
    all_teacher: List[np.ndarray] = []

    candidate_episodes = len(files)
    selected_episodes = 0
    selected_steps = 0
    dropped_steps_non_human = 0
    next_pseudo_ep_id = 0

    for file_path in files:
        with np.load(file_path, allow_pickle=False) as data:
            success = _episode_success(data)
            if cfg.success_only and not success:
                continue

            states = np.array(data["observation.state"], dtype=np.float32)
            actions = np.array(data["action"], dtype=np.float32)

            if states.ndim != 2 or states.shape[1] != EXPECTED_STATE_DIM:
                continue
            if actions.ndim != 2 or actions.shape[1] != EXPECTED_ACTION_DIM:
                continue
            t = states.shape[0]
            if actions.shape[0] != t or t == 0:
                continue

            if "is_human_intervention" in data.files:
                is_human = np.asarray(data["is_human_intervention"], dtype=np.float32).reshape(-1)
                if is_human.shape[0] != t:
                    is_human = np.zeros(t, dtype=np.float32)
            else:
                is_human = np.zeros(t, dtype=np.float32)
            is_human = np.clip(is_human, 0.0, 1.0)
            # Column 1: row came from a real local intervention / blend trajectory.
            source_flag = np.ones(t, dtype=np.float32)
            teach = np.stack([is_human, source_flag], axis=1).astype(np.float32)

        if cfg.keep_only_human:
            runs = _contiguous_runs(is_human > 0.5)
            if not runs:
                continue
            for start, stop in runs:
                seg_len = stop - start
                if seg_len == 0:
                    continue
                all_states.append(states[start:stop])
                all_actions.append(actions[start:stop])
                all_episode_index.append(np.full((seg_len,), next_pseudo_ep_id, dtype=np.int64))
                all_teacher.append(teach[start:stop])
                next_pseudo_ep_id += 1
                selected_episodes += 1
                selected_steps += seg_len
            dropped_steps_non_human += int((is_human < 0.5).sum())
        else:
            all_states.append(states)
            all_actions.append(actions)
            all_episode_index.append(np.full((t,), next_pseudo_ep_id, dtype=np.int64))
            all_teacher.append(teach)
            next_pseudo_ep_id += 1
            selected_episodes += 1
            selected_steps += t

    if selected_episodes == 0:
        raise ValueError(
            "No usable DAgger segments. Check folders / success_only / keep_only_human."
        )
    return (
        np.concatenate(all_states, axis=0),
        np.concatenate(all_actions, axis=0),
        np.concatenate(all_episode_index, axis=0),
        np.concatenate(all_teacher, axis=0),
        {
            "candidate_episodes": int(candidate_episodes),
            "selected_segments": int(selected_episodes),
            "selected_steps": int(selected_steps),
            "dropped_steps_non_human": int(dropped_steps_non_human),
        },
    )


class KeypointsResHeadStepDataset(Dataset):
    """Like KeypointsStepDataset with an extra teacher signal vector per step."""

    def __init__(
        self,
        states: np.ndarray,
        action_chunks: np.ndarray,
        action_is_pad: np.ndarray,
        teacher: np.ndarray,
        step_indices: np.ndarray,
    ):
        if teacher.shape[0] != states.shape[0]:
            raise ValueError("teacher and states length mismatch")
        self.states = states
        self.action_chunks = action_chunks
        self.action_is_pad = action_is_pad
        self.teacher = teacher.astype(np.float32)
        self.step_indices = step_indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.step_indices.shape[0])

    def __getitem__(self, idx: int):
        gi = int(self.step_indices[idx])
        return {
            "state": torch.from_numpy(self.states[gi]).float(),
            "action_chunk": torch.from_numpy(self.action_chunks[gi]).float(),
            "action_is_pad": torch.from_numpy(self.action_is_pad[gi]).bool(),
            "teacher_signal": torch.from_numpy(self.teacher[gi]).float(),
        }


def make_lerobot_teacher_tensor(num_steps: int, teacher_dim: int) -> np.ndarray:
    """Pure LeRobot rows: no human hint and presence=0 (matches test-time no-hint)."""
    t = np.zeros((num_steps, teacher_dim), dtype=np.float32)
    return t


def split_episode_indices(episode_index: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique_eps = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    n_val_eps = max(1, int(len(unique_eps) * val_ratio)) if val_ratio > 0 else 0
    val_eps = set(unique_eps[:n_val_eps].tolist())
    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    if train_idx.size == 0:
        raise ValueError("Empty train split")
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx
