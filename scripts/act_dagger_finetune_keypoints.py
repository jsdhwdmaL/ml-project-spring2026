#!/usr/bin/env python3
"""Fine-tune the keypoints ACT (LeRobot-config) model on collected DAgger episodes.

Targets the 18-D state checkpoint trained by
``scripts/act_train_keypoints.py`` using
:class:`models.act_lerobot.ACTLeRobotPolicy`.  Architecture is loaded from the
base checkpoint's ``lerobot_cfg`` and **not overridable** at the CLI (would
break weight loading).

Mixes collected DAgger episodes (saved by
``scripts/act_dagger_collect_keypoints.py``) with the original
LeRobot ``pusht_keypoints`` dataset by default.

``--keep_only_human``: when True, drop autonomous steps from every loaded
DAgger episode and only train on the contiguous human-controlled segments.
Each contiguous human run becomes its own pseudo-episode for chunking, so
action chunks never include policy-generated steps.

Examples:
    python scripts/act_dagger_finetune_keypoints.py \
        --model_path models/act_keypoints_3200/latest.pt \
        --data_dir   data/act_dagger_keypoints \
        --output_dir models/act_dagger_keypoints \
        --wandb --wandb_project introML-proj-graphs \
        --wandb_entity yizhoul2-carnegie-mellon-university
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
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
)
from models.act_lerobot import ACTLeRobotConfig, ACTLeRobotPolicy

try:
    import wandb
except ImportError:
    wandb = None


AGENT_POS_DIM = 2
ENV_STATE_DIM = 16
EXPECTED_STATE_DIM = AGENT_POS_DIM + ENV_STATE_DIM
EXPECTED_ACTION_DIM = 2


@dataclass
class FinetuneConfig:
    model_path: str = "models/act_keypoints_3200/latest.pt"
    data_dir: str = "data/act_dagger_keypoints"
    dataset_id: str = "lerobot/pusht_keypoints"
    output_dir: str = "models/act_dagger_keypoints"
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
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    # kl_weight: if None, inherit lerobot_cfg.kl_weight from the base checkpoint.
    # kl_warmup/ramp scale the (resolved) kl_weight over training.
    kl_weight: float | None = None
    kl_warmup_epochs: int = 0
    kl_ramp_epochs: int = 0
    num_workers: int = 4
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None


def split_state_to_obs(state: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "observation.state": state[..., :AGENT_POS_DIM],
        "observation.environment_state": state[..., AGENT_POS_DIM:],
    }


def split_episode_indices(episode_index: np.ndarray, val_ratio: float, seed: int):
    unique_eps = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    n_val_eps = max(1, int(len(unique_eps) * val_ratio)) if val_ratio > 0 else 0
    val_eps = set(unique_eps[:n_val_eps].tolist())
    val_mask = np.array([ep in val_eps for ep in episode_index], dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    if train_idx.size == 0:
        raise ValueError("Empty fine-tune train split")
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def get_kl_weight(epoch_index: int, base_kl_weight: float, config: FinetuneConfig) -> float:
    if epoch_index < config.kl_warmup_epochs:
        return 0.0
    if config.kl_ramp_epochs <= 0:
        return float(base_kl_weight)
    ramp_step = epoch_index - config.kl_warmup_epochs + 1
    if ramp_step <= config.kl_ramp_epochs:
        progress = ramp_step / float(config.kl_ramp_epochs)
        return float(base_kl_weight) * progress
    return float(base_kl_weight)


def _episode_success(data) -> bool:
    if "success" in data.files:
        return bool(np.asarray(data["success"]).reshape(-1)[0])
    if "next.success" in data.files:
        return bool(np.asarray(data["next.success"]).any())
    raise KeyError("Episode file has no success key")


def _collect_episode_files(config: FinetuneConfig) -> List[Path]:
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
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.npz")):
            if file_path.name.endswith("_images.npz"):
                continue
            files.append(file_path)

    if not files:
        raise ValueError("No candidate episode files found in selected DAgger folders")
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


def _load_dagger_data(config: FinetuneConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
            T = states.shape[0]
            if actions.shape[0] != T or T == 0:
                continue

            if "is_human_intervention" in data.files:
                is_human = np.asarray(data["is_human_intervention"], dtype=bool).reshape(-1)
                if is_human.shape[0] != T:
                    is_human = np.zeros(T, dtype=bool)
            else:
                is_human = np.zeros(T, dtype=bool)

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
            all_episode_index.append(
                np.full((T,), next_pseudo_ep_id, dtype=np.int64)
            )
            next_pseudo_ep_id += 1
            selected_episodes += 1
            selected_steps += T

    if selected_episodes == 0:
        raise ValueError(
            "No usable DAgger segments found. "
            "(keep_only_human=True with no human steps in the selected folders?)"
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


def _load_lerobot_cfg_from_checkpoint(checkpoint: dict) -> ACTLeRobotConfig:
    """Reconstruct the LeRobot ACT config from a base checkpoint.

    Requires checkpoints written by the new act_train_keypoints.py
    (which stores ``lerobot_cfg`` and ``norm_mode='min_max'``).
    """
    raw_cfg = checkpoint.get("lerobot_cfg")
    if raw_cfg is None:
        nested = checkpoint.get("config", {}).get("lerobot_cfg")
        if nested is not None:
            raw_cfg = nested
    if raw_cfg is None:
        raise KeyError(
            "Base checkpoint is missing 'lerobot_cfg'. "
            "This script only fine-tunes ACTLeRobotPolicy checkpoints. "
            "Re-train with the updated act_train_keypoints.py."
        )

    ckpt_config = checkpoint.get("config", {})
    if ckpt_config.get("norm_mode") != "min_max":
        raise ValueError(
            f"Checkpoint norm_mode={ckpt_config.get('norm_mode')!r}, expected 'min_max'."
        )
    for required in ("state_min", "state_max", "action_min", "action_max"):
        if required not in checkpoint:
            raise KeyError(f"Checkpoint missing required key '{required}'.")

    return ACTLeRobotConfig.from_dict(raw_cfg)


def _build_keypoints_original_bundle(config: FinetuneConfig, horizon: int):
    bundle = build_keypoints_dataloaders(
        dataset_id=config.dataset_id,
        horizon=horizon,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    train_dataset = bundle.train_loader.dataset
    val_dataset = bundle.val_loader.dataset
    train_states = train_dataset.states[train_dataset.step_indices]
    train_actions_first_step = train_dataset.action_chunks[train_dataset.step_indices, 0]
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_states": train_states,
        "train_actions_first_step": train_actions_first_step,
        "num_train": int(bundle.num_train),
        "num_val": int(bundle.num_val),
    }


def _build_mixed_train_loader(
    dagger_train_dataset: Dataset,
    original_train_dataset: Dataset,
    config: FinetuneConfig,
):
    if len(dagger_train_dataset) == 0 or len(original_train_dataset) == 0:
        raise ValueError(
            f"Cannot build mixed loader with empty train dataset: "
            f"dagger={len(dagger_train_dataset)}, original={len(original_train_dataset)}"
        )
    mixed_dataset = ConcatDataset([dagger_train_dataset, original_train_dataset])
    ratio_sum = float(config.mix_dagger_ratio) + float(config.mix_original_ratio)
    dagger_prob = float(config.mix_dagger_ratio) / ratio_sum
    original_prob = float(config.mix_original_ratio) / ratio_sum
    dagger_weight = dagger_prob / len(dagger_train_dataset)
    original_weight = original_prob / len(original_train_dataset)
    weights = [dagger_weight] * len(dagger_train_dataset)
    weights.extend([original_weight] * len(original_train_dataset))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(mixed_dataset),
        replacement=True,
    )
    return DataLoader(
        mixed_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def _safe_max(lo: np.ndarray, hi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(hi, lo + eps).astype(np.float32)


def train(config: FinetuneConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    wandb_run = None
    if config.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install dependencies or disable --wandb.")
        if not config.wandb_project or not config.wandb_entity:
            raise ValueError("--wandb requires both --wandb_project and --wandb_entity")
        run_name = f"{Path(config.output_dir).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=asdict(config),
            tags=["act", "dagger_finetune", "keypoints", "lerobot-cfg"],
        )

    base_checkpoint = torch.load(config.model_path, map_location=device, weights_only=False)
    lerobot_cfg = _load_lerobot_cfg_from_checkpoint(base_checkpoint)
    horizon = int(lerobot_cfg.chunk_size)
    base_kl_weight = (
        float(config.kl_weight) if config.kl_weight is not None else float(lerobot_cfg.kl_weight)
    )

    ckpt_state_min = np.asarray(base_checkpoint["state_min"], dtype=np.float32)
    state_dim = int(ckpt_state_min.shape[0])
    if state_dim != EXPECTED_STATE_DIM:
        raise ValueError(
            f"Expected state_dim={EXPECTED_STATE_DIM} (keypoints), got {state_dim}."
        )

    states, actions, episode_index, data_stats = _load_dagger_data(config)
    print(
        "Loaded DAgger data | "
        f"candidates={data_stats['candidate_episodes']} "
        f"selected_segments={data_stats['selected_segments']} "
        f"selected_steps={data_stats['selected_steps']} "
        f"keep_only_human={data_stats['keep_only_human']} "
        f"dropped_non_human_steps={data_stats['dropped_steps_non_human']}"
    )

    action_chunks, action_is_pad = build_action_chunks_by_episode(
        actions, episode_index, horizon
    )
    train_idx, val_idx = split_episode_indices(episode_index, config.val_ratio, config.seed)

    train_dataset = KeypointsStepDataset(states, action_chunks, action_is_pad, train_idx)
    val_dataset = KeypointsStepDataset(states, action_chunks, action_is_pad, val_idx)

    if config.include_original_data:
        if not (config.mix_dagger_ratio > 0 and config.mix_original_ratio > 0):
            raise ValueError(
                "mix ratios must both be > 0 when include_original_data is enabled; "
                f"got dagger={config.mix_dagger_ratio}, original={config.mix_original_ratio}"
            )

    original_data = None
    if config.include_original_data:
        print(f"Loading original keypoints dataset {config.dataset_id}...")
        original_data = _build_keypoints_original_bundle(config, horizon=horizon)
        print(
            "Loaded original keypoints data | "
            f"train_steps={original_data['num_train']} val_steps={original_data['num_val']}"
        )

    if original_data is not None:
        train_states_arr = np.concatenate(
            [states[train_idx], original_data["train_states"]], axis=0
        )
        train_actions_arr = np.concatenate(
            [actions[train_idx], original_data["train_actions_first_step"]], axis=0
        )
    else:
        train_states_arr = states[train_idx]
        train_actions_arr = actions[train_idx]

    state_min = train_states_arr.min(axis=0).astype(np.float32)
    state_max = _safe_max(state_min, train_states_arr.max(axis=0))
    action_min = train_actions_arr.min(axis=0).astype(np.float32)
    action_max = _safe_max(action_min, train_actions_arr.max(axis=0))

    if original_data is not None:
        train_loader = _build_mixed_train_loader(
            train_dataset, original_data["train_dataset"], config
        )
        val_concat = ConcatDataset([val_dataset, original_data["val_dataset"]])
        val_loader = DataLoader(
            val_concat,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    model = ACTLeRobotPolicy(lerobot_cfg).to(device)
    model.load_state_dict(base_checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    state_min_t = torch.tensor(state_min, dtype=torch.float32, device=device)
    state_max_t = torch.tensor(state_max, dtype=torch.float32, device=device)
    action_min_t = torch.tensor(action_min, dtype=torch.float32, device=device)
    action_max_t = torch.tensor(action_max, dtype=torch.float32, device=device)

    if wandb_run is not None:
        wandb.log(
            {
                "data/candidate_episodes": int(data_stats["candidate_episodes"]),
                "data/selected_segments": int(data_stats["selected_segments"]),
                "data/selected_steps": int(data_stats["selected_steps"]),
                "data/dropped_non_human_steps": int(data_stats["dropped_steps_non_human"]),
                "data/dagger_train_steps": int(len(train_idx)),
                "data/dagger_val_steps": int(len(val_idx)),
                "data/original_train_steps": int(original_data["num_train"]) if original_data is not None else 0,
                "data/original_val_steps": int(original_data["num_val"]) if original_data is not None else 0,
            },
            step=0,
        )

    best_val = float("inf")

    def _step(batch, kl_w: float):
        states_b = batch["state"].to(device)
        action_chunk = batch["action_chunk"].to(device)
        action_is_pad_b = batch["action_is_pad"].to(device)
        states_norm = min_max_normalize(states_b, state_min_t, state_max_t)
        target_actions = min_max_normalize(
            action_chunk,
            action_min_t.view(1, 1, -1),
            action_max_t.view(1, 1, -1),
        )
        obs = split_state_to_obs(states_norm)
        pred_actions, mu, logvar = model(obs, target_actions)
        recon_loss = ACTLeRobotPolicy.masked_l1_loss(pred_actions, target_actions, action_is_pad_b)
        if mu is not None and logvar is not None:
            kl_loss = ACTLeRobotPolicy.kl_divergence(mu, logvar)
        else:
            kl_loss = torch.zeros((), device=device)
        loss = recon_loss + kl_w * kl_loss
        return loss, recon_loss, kl_loss

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_kl_weight = get_kl_weight(epoch - 1, base_kl_weight, config)

            model.train()
            train_loss_sum = train_recon_sum = train_kl_sum = 0.0
            train_batches = 0
            for batch in tqdm(train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False):
                loss, recon_loss, kl_loss = _step(batch, epoch_kl_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += float(loss.item())
                train_recon_sum += float(recon_loss.item())
                train_kl_sum += float(kl_loss.item())
                train_batches += 1

            train_loss = train_loss_sum / max(1, train_batches)
            train_recon = train_recon_sum / max(1, train_batches)
            train_kl = train_kl_sum / max(1, train_batches)

            model.eval()
            val_loss_sum = val_recon_sum = val_kl_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val {epoch}/{config.epochs}", leave=False):
                    loss, recon_loss, kl_loss = _step(batch, epoch_kl_weight)
                    val_loss_sum += float(loss.item())
                    val_recon_sum += float(recon_loss.item())
                    val_kl_sum += float(kl_loss.item())
                    val_batches += 1

            val_loss = val_loss_sum / max(1, val_batches)
            val_recon = val_recon_sum / max(1, val_batches)
            val_kl = val_kl_sum / max(1, val_batches)

            checkpoint_config = asdict(config)
            checkpoint_config["lerobot_cfg"] = asdict(lerobot_cfg)
            checkpoint_config["norm_mode"] = "min_max"
            checkpoint_config["model_arch"] = "act_lerobot"

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "state_min": state_min,
                "state_max": state_max,
                "action_min": action_min,
                "action_max": action_max,
                "config": checkpoint_config,
                "lerobot_cfg": asdict(lerobot_cfg),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_loss": val_loss,
                "val_recon": val_recon,
                "val_kl": val_kl,
                "state_dim": EXPECTED_STATE_DIM,
                "action_dim": EXPECTED_ACTION_DIM,
                "source_model_path": config.model_path,
                "source_dataset_id": config.dataset_id,
                "data_stats": data_stats,
                "mix_stats": {
                    "include_original_data": bool(config.include_original_data),
                    "mix_dagger_ratio": float(config.mix_dagger_ratio),
                    "mix_original_ratio": float(config.mix_original_ratio),
                    "dagger_train_steps": int(len(train_idx)),
                    "dagger_val_steps": int(len(val_idx)),
                    "original_train_steps": int(original_data["num_train"]) if original_data is not None else 0,
                    "original_val_steps": int(original_data["num_val"]) if original_data is not None else 0,
                },
            }

            latest_path = os.path.join(config.output_dir, "latest.pt")
            torch.save(checkpoint, latest_path)
            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(config.output_dir, "best.pt")
                torch.save(checkpoint, best_path)
                if wandb_run is not None:
                    wandb.log({"best/val_loss": float(best_val)}, step=epoch)

            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"kl_weight={epoch_kl_weight:.6f} | "
                f"train={train_loss:.6f} (recon={train_recon:.6f}, kl={train_kl:.6f}) | "
                f"val={val_loss:.6f} (recon={val_recon:.6f}, kl={val_kl:.6f})"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": float(train_loss),
                        "val/loss": float(val_loss),
                        "train/recon_loss": float(train_recon),
                        "train/kl_loss": float(train_kl),
                        "val/recon_loss": float(val_recon),
                        "val/kl_loss": float(val_kl),
                        "train/kl_weight": float(epoch_kl_weight),
                    },
                    step=epoch,
                )

        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = float(best_val)
    finally:
        if wandb_run is not None:
            wandb.finish()

    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump({**asdict(config), "lerobot_cfg": asdict(lerobot_cfg)}, file, indent=2)

    np.savez(
        os.path.join(config.output_dir, "normalization_stats.npz"),
        state_min=state_min,
        state_max=state_max,
        action_min=action_min,
        action_max=action_max,
    )

    print(f"Saved ACT keypoints fine-tune artifacts to: {config.output_dir}")
    print(f"Best validation loss: {best_val:.6f}")


def parse_args() -> FinetuneConfig:
    defaults = FinetuneConfig()
    parser = argparse.ArgumentParser(
        description="Fine-tune the keypoints ACT (LeRobot-config) model on DAgger episodes"
    )
    parser.add_argument("--model_path", type=str, default=defaults.model_path)
    parser.add_argument("--data_dir", type=str, default=defaults.data_dir)
    parser.add_argument("--dataset_id", type=str, default=defaults.dataset_id)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--include_human_intervention", action="store_true", default=defaults.include_human_intervention)
    parser.add_argument("--no_include_human_intervention", dest="include_human_intervention", action="store_false")
    parser.add_argument("--include_rejection_sample", action="store_true", default=defaults.include_rejection_sample)
    parser.add_argument("--no_include_rejection_sample", dest="include_rejection_sample", action="store_false")
    parser.add_argument("--include_failed_autonomous", action="store_true", default=defaults.include_failed_autonomous)
    parser.add_argument("--no_include_failed_autonomous", dest="include_failed_autonomous", action="store_false")
    parser.add_argument("--include_original_data", action="store_true", default=defaults.include_original_data)
    parser.add_argument("--no_include_original_data", dest="include_original_data", action="store_false")
    parser.add_argument("--mix_dagger_ratio", type=float, default=defaults.mix_dagger_ratio)
    parser.add_argument("--mix_original_ratio", type=float, default=defaults.mix_original_ratio)
    parser.add_argument("--success_only", action="store_true", default=defaults.success_only)
    parser.add_argument("--no_success_only", dest="success_only", action="store_false")
    parser.add_argument(
        "--keep_only_human",
        action="store_true",
        default=defaults.keep_only_human,
        help=(
            "Keep only steps where is_human_intervention=True; each contiguous "
            "human run becomes its own pseudo-episode for chunking, so action "
            "chunks never include policy-generated steps."
        ),
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--val_ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=defaults.kl_weight,
        help="Override kl_weight for finetuning (default: inherit from base checkpoint's lerobot_cfg.kl_weight)",
    )
    parser.add_argument("--kl_warmup_epochs", type=int, default=defaults.kl_warmup_epochs)
    parser.add_argument("--kl_ramp_epochs", type=int, default=defaults.kl_ramp_epochs)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--wandb", action="store_true", default=defaults.wandb)
    parser.add_argument("--wandb_project", type=str, default=defaults.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=defaults.wandb_entity)
    args = parser.parse_args()
    return FinetuneConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
