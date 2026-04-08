#!/usr/bin/env python3
"""Train LeRobot ACT on PushT with local-only artifact saving.

This script intentionally avoids dataset metadata/feature utility helpers and
instead declares the PushT policy schema explicitly.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def normalize_dataset_id(dataset_id: str) -> str:
    cleaned = dataset_id.strip()
    if cleaned.lower() == "lerobot.pusht":
        return "lerobot/pusht"
    return cleaned


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device 'mps' is not available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_name}. Use one of: auto, cuda, mps, cpu.")


def parse_norm_mode(name: str) -> NormalizationMode:
    token = name.strip().upper()
    if token in NormalizationMode.__members__:
        return NormalizationMode[token]

    for mode in NormalizationMode:
        if mode.value.upper() == token:
            return mode

    raise ValueError(
        f"Unsupported normalization mode '{name}'. "
        f"Valid values: {', '.join(NormalizationMode.__members__.keys())}."
    )


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0.0]
    return [float(i) / float(fps) for i in delta_indices]


def ensure_pusht_schema(dataset: LeRobotDataset) -> None:
    required_features = {
        "observation.image",
        "observation.state",
        "action",
        "episode_index",
        "next.done",
    }

    if not hasattr(dataset, "meta") or not hasattr(dataset.meta, "features"):
        raise ValueError("Dataset does not expose feature metadata through dataset.meta.features")

    features = dataset.meta.features
    missing = [name for name in sorted(required_features) if name not in features]
    if missing:
        raise ValueError(f"Dataset is missing required PushT features: {missing}")

    image_shape = tuple(features["observation.image"].get("shape", []))
    state_shape = tuple(features["observation.state"].get("shape", []))
    action_shape = tuple(features["action"].get("shape", []))

    if image_shape != (96, 96, 3):
        raise ValueError(f"Expected observation.image shape (96,96,3), got {image_shape}")
    if state_shape != (2,):
        raise ValueError(f"Expected observation.state shape (2,), got {state_shape}")
    if action_shape != (2,):
        raise ValueError(f"Expected action shape (2,), got {action_shape}")


def build_checkpoint_payload(
    policy: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: ACTConfig,
    args: argparse.Namespace,
    step: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "policy_config": dataclasses.asdict(cfg),
        "train_args": vars(args).copy(),
    }


def save_step_checkpoint(
    output_directory: Path,
    policy: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: ACTConfig,
    args: argparse.Namespace,
    step: int,
    suffix: str | None = None,
) -> Path:
    if suffix:
        checkpoint_name = f"ckpt_step_{step:08d}_{suffix}.pt"
    else:
        checkpoint_name = f"ckpt_step_{step:08d}.pt"

    checkpoint_path = output_directory / checkpoint_name
    checkpoint_payload = build_checkpoint_payload(policy, optimizer, cfg, args, step)
    torch.save(checkpoint_payload, checkpoint_path)
    return checkpoint_path


def rotate_step_checkpoints(output_directory: Path, max_to_keep: int) -> None:
    checkpoints = sorted(output_directory.glob("ckpt_step_*.pt"))
    overflow = len(checkpoints) - max_to_keep
    if overflow <= 0:
        return

    for old_checkpoint in checkpoints[:overflow]:
        old_checkpoint.unlink(missing_ok=True)


def build_act_config(args: argparse.Namespace, device: torch.device) -> ACTConfig:
    mode_visual = parse_norm_mode(args.norm_visual)
    mode_state = parse_norm_mode(args.norm_state)
    mode_action = parse_norm_mode(args.norm_action)

    dataclass_fields = {field.name for field in dataclasses.fields(ACTConfig)}

    config_kwargs: dict[str, Any] = {
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
    }

    if "device" in dataclass_fields:
        config_kwargs["device"] = device.type
    if "dim_model" in dataclass_fields:
        config_kwargs["dim_model"] = args.dim_model
    if "n_heads" in dataclass_fields:
        config_kwargs["n_heads"] = args.n_heads
    if "n_encoder_layers" in dataclass_fields:
        config_kwargs["n_encoder_layers"] = args.n_encoder_layers
    if "n_decoder_layers" in dataclass_fields:
        config_kwargs["n_decoder_layers"] = args.n_decoder_layers
    if "latent_dim" in dataclass_fields:
        config_kwargs["latent_dim"] = args.latent_dim
    if "optimizer_lr" in dataclass_fields:
        config_kwargs["optimizer_lr"] = args.learning_rate
    if "optimizer_weight_decay" in dataclass_fields:
        config_kwargs["optimizer_weight_decay"] = args.weight_decay

    if "input_features" in dataclass_fields and "output_features" in dataclass_fields:
        config_kwargs["input_features"] = {
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
        config_kwargs["output_features"] = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        }

    if "normalization_mapping" in dataclass_fields:
        config_kwargs["normalization_mapping"] = {
            "VISUAL": mode_visual,
            "STATE": mode_state,
            "ACTION": mode_action,
        }

    if "input_shapes" in dataclass_fields and "output_shapes" in dataclass_fields:
        config_kwargs["input_shapes"] = {
            "observation.image": [3, 96, 96],
            "observation.state": [2],
        }
        config_kwargs["output_shapes"] = {
            "action": [2],
        }

    if "input_normalization_modes" in dataclass_fields:
        config_kwargs["input_normalization_modes"] = {
            "observation.image": mode_visual.name.lower(),
            "observation.state": mode_state.name.lower(),
        }
    if "output_normalization_modes" in dataclass_fields:
        config_kwargs["output_normalization_modes"] = {
            "action": mode_action.name.lower(),
        }

    return ACTConfig(**config_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeRobot ACT on lerobot/pusht with local save")
    parser.add_argument("--dataset_id", type=str, default="lerobot/pusht")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="models/lerobot_act")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--training_steps", type=int, default=10000)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--checkpoint_freq_steps", type=int, default=500)
    parser.add_argument("--max_checkpoints_to_keep", type=int, default=5)

    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--n_action_steps", type=int, default=100)
    parser.add_argument("--dim_model", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_decoder_layers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--norm_visual", type=str, default="MEAN_STD")
    parser.add_argument("--norm_state", type=str, default="MEAN_STD")
    parser.add_argument("--norm_action", type=str, default="MEAN_STD")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.training_steps <= 0:
        raise ValueError("training_steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.log_freq <= 0:
        raise ValueError("log_freq must be > 0")
    if args.checkpoint_freq_steps <= 0:
        raise ValueError("checkpoint_freq_steps must be > 0")
    if args.max_checkpoints_to_keep <= 0:
        raise ValueError("max_checkpoints_to_keep must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if args.n_action_steps <= 0 or args.n_action_steps > args.chunk_size:
        raise ValueError("n_action_steps must be > 0 and <= chunk_size")

    dataset_id = normalize_dataset_id(args.dataset_id)
    if dataset_id != "lerobot/pusht":
        raise ValueError(
            "This script currently supports only lerobot/pusht. "
            "Custom local-data ingestion will be added later."
        )

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    cfg = build_act_config(args, device)

    print(f"Dataset: {dataset_id}")
    print(f"Device: {device}")
    print(f"Output directory: {output_directory}")
    print(
        "Normalization modes: "
        f"visual={args.norm_visual} state={args.norm_state} action={args.norm_action}"
    )

    dataset_base = LeRobotDataset(dataset_id)
    ensure_pusht_schema(dataset_base)

    fps = int(getattr(dataset_base, "fps", 10))
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, fps),
    }

    image_feature_keys = list(getattr(cfg, "image_features", []))
    obs_delta = getattr(cfg, "observation_delta_indices", None)
    if obs_delta is not None:
        for key in image_feature_keys:
            delta_timestamps[key] = make_delta_timestamps(obs_delta, fps)

    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    if hasattr(dataset, "hf_dataset") and hasattr(dataset.hf_dataset, "keys") and args.split in dataset.hf_dataset.keys():
        dataset.hf_dataset = dataset.hf_dataset[args.split]

    if not hasattr(dataset, "meta") or not hasattr(dataset.meta, "stats"):
        raise ValueError("Dataset does not expose stats through dataset.meta.stats for processor normalization")

    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    policy.train()
    policy.to(device)

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=args.num_workers,
    )

    step = 0
    interrupted = False
    try:
        while step < args.training_steps:
            for batch in dataloader:
                batch = preprocessor(batch)
                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if step % args.log_freq == 0:
                    print(f"step: {step} loss: {loss.item():.4f}")

                step += 1

                if step % args.checkpoint_freq_steps == 0:
                    checkpoint_path = save_step_checkpoint(
                        output_directory=output_directory,
                        policy=policy,
                        optimizer=optimizer,
                        cfg=cfg,
                        args=args,
                        step=step,
                    )
                    rotate_step_checkpoints(output_directory, args.max_checkpoints_to_keep)
                    print(f"saved checkpoint: {checkpoint_path.name}")

                if step >= args.training_steps:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted. Saving emergency checkpoint...")
        emergency_path = save_step_checkpoint(
            output_directory=output_directory,
            policy=policy,
            optimizer=optimizer,
            cfg=cfg,
            args=args,
            step=step,
            suffix="interrupt",
        )
        rotate_step_checkpoints(output_directory, args.max_checkpoints_to_keep)
        print(f"saved checkpoint: {emergency_path.name}")

    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    with open(output_directory / "train_args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    print(f"Saved local ACT artifacts to: {output_directory}")
    if interrupted:
        print("Run ended early due to interruption.")


if __name__ == "__main__":
    main()
