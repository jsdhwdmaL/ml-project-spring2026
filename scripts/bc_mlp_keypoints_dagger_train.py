#!/usr/bin/env python3
"""Train the keypoints BC MLP (two-layer state MLP) on DAgger NPZ + optional LeRobot demos.

Data layout and folders match :mod:`scripts.act_dagger_finetune_keypoints` (episodes from
``scripts.act_dagger_collect_keypoints`` by default, under ``data_dir``). Min–max stats are
recomputed on the **train** split (dagger + original) so the checkpoint is consistent with
:mod:`scripts.bc_mlp_keypoints_train` and :mod:`scripts.bc_mlp_keypoints_eval`.

Example:
    python scripts/bc_mlp_keypoints_dagger_train.py \\
        --data_dir data/act_dagger_keypoints \\
        --output_dir models/bc_mlp_keypoints_dagger

    # With offline mix (default ratios 0.2 dagger / 0.8 LeRobot) and a warm-started MLP:
    python scripts/bc_mlp_keypoints_dagger_train.py \\
        --init_checkpoint models/bc_mlp_keypoints/best.pt \\
        --data_dir data/act_dagger_keypoints \\
        --output_dir models/bc_mlp_keypoints_dagger
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.build_chunk import build_action_chunks_by_episode
from data.dataloader_keypoints import (
    KeypointsStepDataset,
    build_keypoints_dataloaders,
    min_max_normalize,
    split_episode_indices,
)
from models.bc_mlp_keypoints import BCKeypointsMLP, DEFAULT_STATE_DIM, DEFAULT_ACTION_DIM

try:
    import wandb
except ImportError:
    wandb = None

AGENT_POS_DIM = 2
ENV_STATE_DIM = 16
EXPECTED_STATE_DIM = AGENT_POS_DIM + ENV_STATE_DIM
EXPECTED_ACTION_DIM = 2


@dataclass
class TrainConfig:
    data_dir: str = "data/act_dagger_keypoints"
    dataset_id: str = "lerobot/pusht_keypoints"
    output_dir: str = "models/bc_mlp_keypoints_dagger"
    init_checkpoint: str | None = None
    include_human_intervention: bool = True
    include_rejection_sample: bool = False
    include_failed_autonomous: bool = False
    include_original_data: bool = True
    mix_dagger_ratio: float = 0.2
    mix_original_ratio: float = 0.8
    success_only: bool = True
    keep_only_human: bool = True
    seed: int = 42
    val_ratio: float = 0.1
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    num_workers: int = 4
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None


def _episode_success(data) -> bool:
    if "success" in data.files:
        return bool(np.asarray(data["success"]).reshape(-1)[0])
    if "next.success" in data.files:
        return bool(np.asarray(data["next.success"]).any())
    raise KeyError("Episode file has no success key")


def _collect_episode_files(config: TrainConfig) -> List[Path]:
    root = Path(config.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {root}")
    folders: List[Path] = []
    if config.include_human_intervention:
        folders.append(root / "human_intervention")
    if config.include_rejection_sample:
        folders.append(root / "rejection_sample")
    if config.include_failed_autonomous:
        folders.append(root / "failed_autonomous")
    files: List[Path] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for file_path in sorted(folder.glob("*.npz")):
            if file_path.name.endswith("_images.npz"):
                continue
            files.append(file_path)
    if not files:
        raise ValueError("No candidate .npz episode files in selected DAgger folders under data_dir")
    return files


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


def _load_dagger_data(config: TrainConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    files = _collect_episode_files(config)
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_episode_index: List[np.ndarray] = []
    candidate_episodes = len(files)
    selected_episodes = 0
    selected_steps = 0
    dropped_steps_non_human = 0
    next_pseudo_ep_id = 0

    for file_path in files:
        with np.load(file_path, allow_pickle=False) as data:
            success = _episode_success(data)
            if config.success_only and not success:
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
                is_human = np.asarray(data["is_human_intervention"], dtype=bool).reshape(-1)
                if is_human.shape[0] != t:
                    is_human = np.zeros(t, dtype=bool)
            else:
                is_human = np.zeros(t, dtype=bool)

        if config.keep_only_human:
            runs = _contiguous_runs(is_human)
            if not runs:
                continue
            for start, stop in runs:
                seg_len = stop - start
                if seg_len == 0:
                    continue
                all_states.append(states[start:stop])
                all_actions.append(actions[start:stop])
                all_episode_index.append(
                    np.full((seg_len,), next_pseudo_ep_id, dtype=np.int64)
                )
                next_pseudo_ep_id += 1
                selected_episodes += 1
                selected_steps += seg_len
            dropped_steps_non_human += int((~is_human).sum())
        else:
            all_states.append(states)
            all_actions.append(actions)
            all_episode_index.append(np.full((t,), next_pseudo_ep_id, dtype=np.int64))
            next_pseudo_ep_id += 1
            selected_episodes += 1
            selected_steps += t

    if selected_episodes == 0:
        raise ValueError(
            "No usable DAgger segments (check folders / success_only / keep_only_human)."
        )
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    episode_index = np.concatenate(all_episode_index, axis=0)
    stats = {
        "candidate_episodes": int(candidate_episodes),
        "selected_segments": int(selected_episodes),
        "selected_steps": int(selected_steps),
        "dropped_steps_non_human": int(dropped_steps_non_human),
        "keep_only_human": bool(config.keep_only_human),
    }
    return states, actions, episode_index, stats


def _safe_max(lo: np.ndarray, hi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(hi, lo + eps).astype(np.float32)


def _build_keypoints_original_bundle(config: TrainConfig):
    val_nw = 1 if config.num_workers > 0 else 0
    bundle = build_keypoints_dataloaders(
        dataset_id=config.dataset_id,
        horizon=1,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
        val_num_workers=val_nw,
        pin_memory=True,
    )
    train_ds: KeypointsStepDataset = bundle.train_loader.dataset
    val_ds: KeypointsStepDataset = bundle.val_loader.dataset
    train_states = train_ds.states[train_ds.step_indices]
    train_a0 = train_ds.action_chunks[train_ds.step_indices, 0, :]
    return {
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "train_states": train_states,
        "train_actions_first_step": train_a0,
        "num_train": int(bundle.num_train),
        "num_val": int(bundle.num_val),
    }


def _build_mixed_train_loader(
    dagger_ds: Dataset, original_ds: Dataset, config: TrainConfig
) -> DataLoader:
    if len(dagger_ds) == 0 or len(original_ds) == 0:
        raise ValueError(
            f"Cannot mix: dagger_len={len(dagger_ds)} original_len={len(original_ds)}"
        )
    mixed = ConcatDataset([dagger_ds, original_ds])
    rsum = float(config.mix_dagger_ratio) + float(config.mix_original_ratio)
    wd = (config.mix_dagger_ratio / rsum) / len(dagger_ds)
    wo = (config.mix_original_ratio / rsum) / len(original_ds)
    w = [wd] * len(dagger_ds) + [wo] * len(original_ds)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(w, dtype=torch.double),
        num_samples=len(mixed),
        replacement=True,
    )
    _pw = config.num_workers > 0
    return DataLoader(
        mixed,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=_pw,
    )


def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    wandb_run = None
    if config.wandb:
        if wandb is None:
            raise ImportError("wandb not installed. pip install wandb or omit --wandb.")
        if not config.wandb_project or not config.wandb_entity:
            raise ValueError("--wandb needs --wandb_project and --wandb_entity")
        run_name = f"{Path(config.output_dir).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=asdict(config),
            tags=["bc_mlp", "keypoints", "dagger"],
        )

    print(f"Loading DAgger keypoints from {config.data_dir} ...")
    states, actions, episode_index, data_stats = _load_dagger_data(config)
    print(
        f"  candidates={data_stats['candidate_episodes']} segments={data_stats['selected_segments']} "
        f"steps={data_stats['selected_steps']}"
    )
    h = 1
    action_chunks, action_is_pad = build_action_chunks_by_episode(actions, episode_index, h)
    train_idx, val_idx = split_episode_indices(episode_index, config.val_ratio, config.seed)
    dagger_train = KeypointsStepDataset(states, action_chunks, action_is_pad, train_idx)
    dagger_val = KeypointsStepDataset(states, action_chunks, action_is_pad, val_idx)

    original: Dict | None = None
    if config.include_original_data:
        if not (config.mix_dagger_ratio > 0 and config.mix_original_ratio > 0):
            raise ValueError("mix_dagger_ratio and mix_original_ratio must be >0 when mixing.")
        print(f"Loading LeRobot {config.dataset_id} (horizon=1) ...")
        original = _build_keypoints_original_bundle(config)
        print(f"  original train/val steps: {original['num_train']}/{original['num_val']}")

    if original is not None:
        st_train = np.concatenate([states[train_idx], original["train_states"]], axis=0)
        ac_train = np.concatenate([actions[train_idx], original["train_actions_first_step"]], axis=0)
    else:
        st_train = states[train_idx]
        ac_train = actions[train_idx]

    state_min = st_train.min(axis=0).astype(np.float32)
    state_max = _safe_max(state_min, st_train.max(axis=0))
    action_min = ac_train.min(axis=0).astype(np.float32)
    action_max = _safe_max(action_min, ac_train.max(axis=0))

    if original is not None:
        train_loader = _build_mixed_train_loader(dagger_train, original["train_dataset"], config)
        val_set = ConcatDataset([dagger_val, original["val_dataset"]])
        _pw = config.num_workers > 0
        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=_pw,
        )
    else:
        train_loader = DataLoader(
            dagger_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )
        val_loader = DataLoader(
            dagger_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )

    state_min_t = torch.tensor(state_min, device=device, dtype=torch.float32)
    state_max_t = torch.tensor(state_max, device=device, dtype=torch.float32)
    action_min_t = torch.tensor(action_min, device=device, dtype=torch.float32)
    action_max_t = torch.tensor(action_max, device=device, dtype=torch.float32)

    model = BCKeypointsMLP(
        state_dim=EXPECTED_STATE_DIM,
        action_dim=EXPECTED_ACTION_DIM,
        hidden_dim=config.hidden_dim,
    ).to(device)
    if config.init_checkpoint is not None:
        ck = torch.load(config.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        print(f"Loaded MLP init from {config.init_checkpoint}")
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    best_val = float("inf")

    if wandb_run is not None:
        wandb.log(
            {
                "data/selected_steps": int(data_stats["selected_steps"]),
                "data/dagger_train_idx": int(train_idx.size),
                "data/original_train": int(original["num_train"]) if original else 0,
            },
            step=0,
        )

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            t_loss = 0.0
            for batch in tqdm(train_loader, desc=f"train {epoch}/{config.epochs}", leave=False):
                s = batch["state"].to(device, non_blocking=True)
                a = batch["action_chunk"][:, 0, :].to(device, non_blocking=True)
                s_n = min_max_normalize(s, state_min_t, state_max_t)
                a_n = min_max_normalize(a, action_min_t, action_max_t)
                pred = model(s_n)
                loss = loss_fn(pred, a_n)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                t_loss += float(loss.item())

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"val {epoch}/{config.epochs}", leave=False):
                    s = batch["state"].to(device, non_blocking=True)
                    a = batch["action_chunk"][:, 0, :].to(device, non_blocking=True)
                    s_n = min_max_normalize(s, state_min_t, state_max_t)
                    a_n = min_max_normalize(a, action_min_t, action_max_t)
                    v_loss += float(loss_fn(model(s_n), a_n).item())

            n_t, n_v = max(len(train_loader), 1), max(len(val_loader), 1)
            avg_t, avg_v = t_loss / n_t, v_loss / n_v
            print(f"Epoch {epoch:03d} | Train: {avg_t:.6f} | Val: {avg_v:.6f}")
            if wandb_run is not None:
                wandb.log({"epoch": epoch, "train/loss": avg_t, "val/loss": avg_v}, step=epoch)

            ckpt = {
                "model_state_dict": model.state_dict(),
                "state_min": state_min,
                "state_max": state_max,
                "action_min": action_min,
                "action_max": action_max,
                "config": {
                    **asdict(config),
                    "norm_mode": "min_max",
                    "state_dim": DEFAULT_STATE_DIM,
                    "action_dim": DEFAULT_ACTION_DIM,
                    "data_dir": config.data_dir,
                },
                "epoch": epoch,
                "data_stats": {**data_stats, "dagger_val_steps": int(val_idx.size)},
            }
            if original is not None:
                ckpt["mix_stats"] = {
                    "include_original_data": True,
                    "mix_dagger_ratio": config.mix_dagger_ratio,
                    "mix_original_ratio": config.mix_original_ratio,
                }
            torch.save(ckpt, os.path.join(config.output_dir, "latest.pt"))
            if avg_v < best_val:
                best_val = avg_v
                torch.save(ckpt, os.path.join(config.output_dir, "best.pt"))
                print(f"  new best val: {best_val:.6f}")
                if wandb_run is not None:
                    wandb.log({"best/val_loss": best_val}, step=epoch)

        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = best_val
    finally:
        if wandb_run is not None:
            wandb.finish()

    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Saved to {config.output_dir} | best val loss = {best_val:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="BC MLP (keypoints) on DAgger NPZ with optional LeRobot mix"
    )
    p.add_argument("--data_dir", type=str, default="data/act_dagger_keypoints")
    p.add_argument("--dataset_id", type=str, default="lerobot/pusht_keypoints")
    p.add_argument("--output_dir", type=str, default="models/bc_mlp_keypoints_dagger")
    p.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional path to bc_mlp_keypoints .pt to initialize weights.",
    )
    p.add_argument("--include_human_intervention", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include_rejection_sample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include_failed_autonomous", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include_original_data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mix_dagger_ratio", type=float, default=0.2)
    p.add_argument("--mix_original_ratio", type=float, default=0.8)
    p.add_argument("--success_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep_only_human", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    a = p.parse_args()
    train(TrainConfig(**{f.name: getattr(a, f.name) for f in fields(TrainConfig)}))


if __name__ == "__main__":
    main()
