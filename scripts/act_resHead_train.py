#!/usr/bin/env python3
"""Train ACTResHead (teacher token in CVAE encoder) on keypoints (18-D state, no image).

By default **mixes**:
  - LeRobot ``lerobot/pusht_keypoints`` offline demos, and
  - Local NPZ from **compliant res-blend** / DAgger collection
    (``data/act_cr_dagger_keypoints/``) where ``is_human_intervention`` marks
    soft-blend (SHIFT) steps. Those labels feed the teacher MLP in the VAE path.

State/action normalization matches ``act_train_keypoints`` (per-channel min/max to [-1, 1]).

Example (defaults target res-blend + original mix):
  python scripts/act_resHead_train.py --output_dir models/act_resHead_keypoints
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.build_chunk import build_action_chunks_by_episode
from data.dataloader_keypoints import build_keypoints_dataloaders, min_max_normalize
from data.reshead_keypoints_mix import (
    DaggerLoadConfig,
    EXPECTED_STATE_DIM,
    KeypointsResHeadStepDataset,
    make_lerobot_teacher_tensor,
    load_dagger_keypoints_with_teacher,
    split_episode_indices,
    _safe_max,
)
from models.act_reshead import ACTResHeadPolicy

try:
    import wandb
except ImportError:
    wandb = None

ACTION_DIM = 2


@dataclass
class TrainConfig:
    # Data
    dataset_id: str = "lerobot/pusht_keypoints"
    data_dir: str = "data/act_cr_dagger_keypoints"
    output_dir: str = "models/act_resHead_keypoints"
    include_dagger_data: bool = True
    include_original_data: bool = True
    mix_dagger_ratio: float = 0.2
    mix_original_ratio: float = 0.8
    include_human_intervention: bool = True
    include_rejection_sample: bool = False
    include_failed_autonomous: bool = False
    success_only: bool = True
    keep_only_human: bool = False
    # If True, only train on DAgger folders (no LeRobot) — set include_original_data=False.
    seed: int = 42
    val_ratio: float = 0.1
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    # Model (keypoints: 18-D state, no vision)
    horizon: int = 16
    hidden_dim: int = 512
    latent_dim: int = 32
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    kl_beta: float = 10.0
    teacher_dropout_prob: float = 0.25
    # Teacher is [is_human, local_npz_flag]; dim fixed at 2
    teacher_signal_dim: int = 2
    ensemble_decay: float = 0.05
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None


def _build_mixed_train_loader(
    dagger_ds: torch.utils.data.Dataset,
    original_ds: torch.utils.data.Dataset,
    config: TrainConfig,
) -> DataLoader:
    if len(dagger_ds) == 0 or len(original_ds) == 0:
        raise ValueError(
            f"Cannot build mixed loader: dagger={len(dagger_ds)} original={len(original_ds)}"
        )
    mixed = ConcatDataset([dagger_ds, original_ds])
    rsum = float(config.mix_dagger_ratio) + float(config.mix_original_ratio)
    d_prob = float(config.mix_dagger_ratio) / rsum
    o_prob = float(config.mix_original_ratio) / rsum
    w = [d_prob / len(dagger_ds)] * len(dagger_ds) + [o_prob / len(original_ds)] * len(original_ds)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(w, dtype=torch.double), num_samples=len(mixed), replacement=True
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
    if config.teacher_signal_dim != 2:
        raise ValueError("This keypoints build uses a fixed 2-D teacher: [is_human, npz_source].")
    if not config.include_dagger_data and not config.include_original_data:
        raise ValueError("Enable at least one of include_dagger_data or include_original_data")
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # Keep worker processes alive across epochs (requires num_workers > 0).
    _pw = config.num_workers > 0

    dagger_stats: Optional[Dict[str, Any]] = None
    states_d, actions_d, ep_d, teach_d, dagger_stats = None, None, None, None, None
    d_train_loader: Optional[DataLoader] = None
    d_val_loader: Optional[DataLoader] = None
    o_train_loader: Optional[DataLoader] = None
    o_val_loader: Optional[DataLoader] = None
    o_bundle: Any = None

    if config.include_dagger_data:
        print(f"Loading res-blend / DAgger NPZ from {config.data_dir} ...")
        ld_cfg = DaggerLoadConfig(
            data_dir=config.data_dir,
            success_only=config.success_only,
            keep_only_human=config.keep_only_human,
            include_human_intervention=config.include_human_intervention,
            include_rejection_sample=config.include_rejection_sample,
            include_failed_autonomous=config.include_failed_autonomous,
        )
        states_d, actions_d, ep_d, teach_d, dagger_stats = load_dagger_keypoints_with_teacher(ld_cfg)
        print(
            f"  DAgger: steps={states_d.shape[0]} | segments={dagger_stats['selected_segments']} | "
            f"stats={dagger_stats}"
        )
        ch_d, pad_d = build_action_chunks_by_episode(actions_d, ep_d, config.horizon)
        tr_d, va_d = split_episode_indices(ep_d, config.val_ratio, config.seed)
        d_train = KeypointsResHeadStepDataset(states_d, ch_d, pad_d, teach_d, tr_d)
        d_val = KeypointsResHeadStepDataset(states_d, ch_d, pad_d, teach_d, va_d)
        d_train_loader = DataLoader(
            d_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=_pw,
        )
        d_val_loader = DataLoader(
            d_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=_pw,
        )

    if config.include_original_data:
        print(f"Loading LeRobot {config.dataset_id} ...")
        o_bundle = build_keypoints_dataloaders(
            dataset_id=config.dataset_id,
            horizon=config.horizon,
            batch_size=config.batch_size,
            val_ratio=config.val_ratio,
            seed=config.seed,
            num_workers=config.num_workers,
        )
        o_ds = o_bundle.train_loader.dataset
        n_full = o_ds.states.shape[0]
        teach_l = make_lerobot_teacher_tensor(n_full, config.teacher_signal_dim)
        o_train = KeypointsResHeadStepDataset(
            o_ds.states, o_ds.action_chunks, o_ds.action_is_pad, teach_l, o_ds.step_indices
        )
        o_val_ds = o_bundle.val_loader.dataset
        n_val_full = o_val_ds.states.shape[0]
        teach_v = make_lerobot_teacher_tensor(n_val_full, config.teacher_signal_dim)
        o_val = KeypointsResHeadStepDataset(
            o_val_ds.states, o_val_ds.action_chunks, o_val_ds.action_is_pad, teach_v, o_val_ds.step_indices
        )
        o_train_loader = DataLoader(
            o_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=_pw,
        )
        o_val_loader = DataLoader(
            o_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=_pw,
        )
        print(f"  LeRobot: train={o_bundle.num_train} val={o_bundle.num_val}")

    # Min/max from combined training rows (align with dagger finetune)
    parts_s: List[np.ndarray] = []
    parts_a0: List[np.ndarray] = []
    if config.include_dagger_data:
        tr_idx = d_train.step_indices
        parts_s.append(states_d[tr_idx])
        parts_a0.append(actions_d[tr_idx])
    if config.include_original_data and o_bundle is not None:
        o_ds = o_bundle.train_loader.dataset
        tix = o_ds.step_indices
        parts_s.append(o_ds.states[tix])
        parts_a0.append(o_ds.action_chunks[tix, 0])
    if not parts_s:
        raise RuntimeError("No data for min/max")
    train_s = np.concatenate(parts_s, axis=0)
    train_a0 = np.concatenate(parts_a0, axis=0)
    state_min = train_s.min(axis=0).astype(np.float32)
    state_max = _safe_max(state_min, train_s.max(axis=0))
    action_min = train_a0.min(axis=0).astype(np.float32)
    action_max = _safe_max(action_min, train_a0.max(axis=0))

    state_min_t = torch.tensor(state_min, device=device, dtype=torch.float32)
    state_max_t = torch.tensor(state_max, device=device, dtype=torch.float32)
    action_min_t = torch.tensor(action_min, device=device, dtype=torch.float32)
    action_max_t = torch.tensor(action_max, device=device, dtype=torch.float32)

    # Train/val loaders (final)
    if config.include_dagger_data and config.include_original_data:
        if not (config.mix_dagger_ratio > 0 and config.mix_original_ratio > 0):
            raise ValueError("mix_dagger_ratio and mix_original_ratio must be >0 when both data sources on")
        assert d_train is not None and o_train is not None
        train_loader = _build_mixed_train_loader(d_train, o_train, config)
        val_merged = ConcatDataset([d_val, o_val])
        val_loader = DataLoader(
            val_merged,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=_pw,
        )
    elif config.include_dagger_data:
        train_loader = d_train_loader
        val_loader = d_val_loader
    else:
        train_loader = o_train_loader
        val_loader = o_val_loader

    model = ACTResHeadPolicy(
        state_dim=EXPECTED_STATE_DIM,
        action_dim=ACTION_DIM,
        horizon=config.horizon,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        teacher_signal_dim=config.teacher_signal_dim,
        teacher_dropout_prob=config.teacher_dropout_prob,
        use_vision=False,
    ).to(device)

    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"ACTResHead keypoints: {nparam} params | H={config.horizon} | use_vision=False | "
        f"teacher_dim={config.teacher_signal_dim}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    wandb_run = None
    if config.wandb:
        if wandb is None:
            raise ImportError("wandb not installed")
        if not config.wandb_project or not config.wandb_entity:
            raise ValueError("--wandb needs --wandb_project and --wandb_entity")
        run_name = f"{Path(config.output_dir).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=config.wandb_project, entity=config.wandb_entity, name=run_name, config=asdict(config),
            tags=["act", "reshead", "keypoints", "train"],
        )

    best_val = float("inf")
    cfg_save = asdict(config)
    cfg_save["model_arch"] = "act_reshead_keypoints"
    cfg_save["norm_mode"] = "min_max"
    cfg_save["use_vision"] = False
    if dagger_stats is not None:
        cfg_save["dagger_load_stats"] = dagger_stats

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            t_loss = t_recon = t_kl = 0.0
            nb = 0
            for batch in tqdm(train_loader, desc=f"Train {epoch}/{config.epochs}", leave=False):
                s = batch["state"].to(device)
                ac = batch["action_chunk"].to(device)
                ap = batch["action_is_pad"].to(device)
                te = batch["teacher_signal"].to(device)

                s_n = min_max_normalize(s, state_min_t, state_max_t)
                t_act = min_max_normalize(
                    ac, action_min_t.view(1, 1, -1), action_max_t.view(1, 1, -1)
                )
                pred, mu, logvar = model(
                    image=None, state=s_n, action_chunk=t_act, teacher_signal=te
                )
                recon = _masked_l1_norm(pred, t_act, ap)
                kl = ACTResHeadPolicy.kl_divergence(mu, logvar)
                loss = recon + config.kl_beta * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                t_loss += float(loss.item())
                t_recon += float(recon.item())
                t_kl += float(kl.item())
                nb += 1
            t_loss /= max(1, nb)
            t_recon /= max(1, nb)
            t_kl /= max(1, nb)

            model.eval()
            v_loss = v_recon = v_kl = 0.0
            vb = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val {epoch}/{config.epochs}", leave=False):
                    s = batch["state"].to(device)
                    ac = batch["action_chunk"].to(device)
                    ap = batch["action_is_pad"].to(device)
                    te = batch["teacher_signal"].to(device)
                    s_n = min_max_normalize(s, state_min_t, state_max_t)
                    t_act = min_max_normalize(
                        ac, action_min_t.view(1, 1, -1), action_max_t.view(1, 1, -1)
                    )
                    pred, mu, logvar = model(
                        image=None, state=s_n, action_chunk=t_act, teacher_signal=te
                    )
                    recon = _masked_l1_norm(pred, t_act, ap)
                    kl = ACTResHeadPolicy.kl_divergence(mu, logvar)
                    loss = recon + config.kl_beta * kl
                    v_loss += float(loss.item())
                    v_recon += float(recon.item())
                    v_kl += float(kl.item())
                    vb += 1
            v_loss /= max(1, vb)
            v_recon /= max(1, vb)
            v_kl /= max(1, vb)

            ck = {
                "model_state_dict": model.state_dict(),
                "config": cfg_save,
                "norm_mode": "min_max",
                "state_min": state_min, "state_max": state_max,
                "action_min": action_min, "action_max": action_max,
                "state_dim": EXPECTED_STATE_DIM, "action_dim": ACTION_DIM,
                "epoch": epoch,
                "train_loss": t_loss, "val_loss": v_loss,
                "train_recon": t_recon, "val_recon": v_recon,
                "train_kl": t_kl, "val_kl": v_kl,
            }
            torch.save(ck, os.path.join(config.output_dir, "latest.pt"))
            if v_loss < best_val:
                best_val = v_loss
                torch.save(ck, os.path.join(config.output_dir, "best.pt"))
            print(
                f"Epoch {epoch:03d}/{config.epochs} | train={t_loss:.6f} (recon={t_recon:.6f} kl={t_kl:.6f}) | "
                f"val={v_loss:.6f} (recon={v_recon:.6f} kl={v_kl:.6f})"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch, "train/loss": t_loss, "val/loss": v_loss,
                        "train/recon": t_recon, "val/recon": v_recon, "train/kl": t_kl, "val/kl": v_kl,
                    },
                    step=epoch,
                )
    finally:
        if wandb_run is not None:
            wandb.finish()

    with open(os.path.join(config.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_save, f, indent=2)
    np.savez(
        os.path.join(config.output_dir, "normalization_stats.npz"),
        state_min=state_min, state_max=state_max, action_min=action_min, action_max=action_max,
    )
    print(f"Saved to {config.output_dir} | best val={best_val:.6f}")


def _masked_l1_norm(
    pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor
) -> torch.Tensor:
    valid = (~is_pad).unsqueeze(-1)
    err = torch.abs(pred - target) * valid.float()
    d = (valid.float().sum() * pred.size(-1)).clamp_min(1.0)
    return err.sum() / d


def parse_args() -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train ACTResHead on keypoints + optional act_cr DAgger NPZ")
    p.add_argument("--dataset_id", type=str, default=d.dataset_id)
    p.add_argument("--data_dir", type=str, default=d.data_dir)
    p.add_argument("--output_dir", type=str, default=d.output_dir)
    p.add_argument("--include_dagger_data", action=argparse.BooleanOptionalAction, default=d.include_dagger_data)
    p.add_argument("--include_original_data", action=argparse.BooleanOptionalAction, default=d.include_original_data)
    p.add_argument("--mix_dagger_ratio", type=float, default=d.mix_dagger_ratio)
    p.add_argument("--mix_original_ratio", type=float, default=d.mix_original_ratio)
    p.add_argument("--include_human_intervention", action=argparse.BooleanOptionalAction, default=d.include_human_intervention)
    p.add_argument("--include_rejection_sample", action=argparse.BooleanOptionalAction, default=d.include_rejection_sample)
    p.add_argument("--include_failed_autonomous", action=argparse.BooleanOptionalAction, default=d.include_failed_autonomous)
    p.add_argument("--success_only", action=argparse.BooleanOptionalAction, default=d.success_only)
    p.add_argument("--keep_only_human", action=argparse.BooleanOptionalAction, default=d.keep_only_human)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--val_ratio", type=float, default=d.val_ratio)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--learning_rate", type=float, default=d.learning_rate)
    p.add_argument("--weight_decay", type=float, default=d.weight_decay)
    p.add_argument("--num_workers", type=int, default=d.num_workers)
    p.add_argument("--horizon", type=int, default=d.horizon)
    p.add_argument("--hidden_dim", type=int, default=d.hidden_dim)
    p.add_argument("--latent_dim", type=int, default=d.latent_dim)
    p.add_argument("--nhead", type=int, default=d.nhead)
    p.add_argument("--num_encoder_layers", type=int, default=d.num_encoder_layers)
    p.add_argument("--num_decoder_layers", type=int, default=d.num_decoder_layers)
    p.add_argument("--kl_beta", type=float, default=d.kl_beta)
    p.add_argument("--teacher_dropout_prob", type=float, default=d.teacher_dropout_prob)
    p.add_argument("--ensemble_decay", type=float, default=d.ensemble_decay)
    p.add_argument("--wandb", action="store_true", default=d.wandb)
    p.add_argument("--wandb_project", type=str, default=d.wandb_project)
    p.add_argument("--wandb_entity", type=str, default=d.wandb_entity)
    a = p.parse_args()
    return TrainConfig(**vars(a))


if __name__ == "__main__":
    train(parse_args())
