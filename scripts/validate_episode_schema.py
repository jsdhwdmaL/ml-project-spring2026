#!/usr/bin/env python3
"""Validate per-episode PushT NPZ files against strict per-step schema.

Expected per-episode schema:
  observation.state: (N, 2), float32
  action: (N, 2), float32
  frame_index: (N,), int64
  timestamp: (N,), float32
  next.reward: (N,), float32
  next.done: (N,), bool
  next.success: (N,), bool
  episode_index: (N,), int64
  index: (N,), int64
  task_index: (N,), int64
  is_human_intervention: (N,), bool
  env_seed: scalar int64
  trial_idx: scalar int64
  success: scalar bool

Usage:
python scripts/validate_episode_schema.py \
    --input data/pretraining/teleop_data_raw/human_intervention
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


REQUIRED_STEP_KEYS = {
    "observation.state",
    "action",
    "frame_index",
    "timestamp",
    "next.reward",
    "next.done",
    "next.success",
    "episode_index",
    "index",
    "task_index",
    "is_human_intervention",
}

REQUIRED_SCALAR_KEYS = {
    "env_seed",
    "trial_idx",
    "success",
}


def _is_scalar_array(arr: np.ndarray) -> bool:
    return arr.shape == () or arr.shape == (1,)


def _check_dtype(arr: np.ndarray, expected: str) -> bool:
    if expected == "float32":
        return np.issubdtype(arr.dtype, np.float32)
    if expected == "int64":
        return np.issubdtype(arr.dtype, np.int64)
    if expected == "bool":
        return np.issubdtype(arr.dtype, np.bool_)
    return False


def validate_episode_file(path: Path, strict: bool = False) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    data = np.load(path)
    keys = set(data.files)

    missing_step = sorted(REQUIRED_STEP_KEYS - keys)
    missing_scalar = sorted(REQUIRED_SCALAR_KEYS - keys)
    if missing_step:
        errors.append(f"Missing step keys: {missing_step}")
    if missing_scalar:
        errors.append(f"Missing scalar keys: {missing_scalar}")

    if errors:
        return False, errors, warnings

    obs = data["observation.state"]
    action = data["action"]

    if obs.ndim != 2 or obs.shape[1] != 2:
        errors.append(f"observation.state must be (N,2), got {obs.shape}")
    if action.ndim != 2 or action.shape[1] != 2:
        errors.append(f"action must be (N,2), got {action.shape}")

    n_steps = obs.shape[0] if obs.ndim == 2 else None
    one_d_fields = {
        "frame_index": data["frame_index"],
        "timestamp": data["timestamp"],
        "next.reward": data["next.reward"],
        "next.done": data["next.done"],
        "next.success": data["next.success"],
        "episode_index": data["episode_index"],
        "index": data["index"],
        "task_index": data["task_index"],
        "is_human_intervention": data["is_human_intervention"],
    }

    if n_steps is not None:
        if action.shape[0] != n_steps:
            errors.append(f"action first dim must match N, got action N={action.shape[0]}, obs N={n_steps}")

        for name, arr in one_d_fields.items():
            if arr.ndim != 1 or arr.shape[0] != n_steps:
                errors.append(f"{name} must be (N,), got {arr.shape} while N={n_steps}")

    dtype_expectations = {
        "observation.state": (obs, "float32"),
        "action": (action, "float32"),
        "frame_index": (data["frame_index"], "int64"),
        "timestamp": (data["timestamp"], "float32"),
        "next.reward": (data["next.reward"], "float32"),
        "next.done": (data["next.done"], "bool"),
        "next.success": (data["next.success"], "bool"),
        "episode_index": (data["episode_index"], "int64"),
        "index": (data["index"], "int64"),
        "task_index": (data["task_index"], "int64"),
        "is_human_intervention": (data["is_human_intervention"], "bool"),
        "env_seed": (data["env_seed"], "int64"),
        "trial_idx": (data["trial_idx"], "int64"),
        "success": (data["success"], "bool"),
    }

    for name, (arr, expected) in dtype_expectations.items():
        if not _check_dtype(arr, expected):
            errors.append(f"{name} dtype should be {expected}, got {arr.dtype}")

    for scalar_key in REQUIRED_SCALAR_KEYS:
        if not _is_scalar_array(data[scalar_key]):
            errors.append(f"{scalar_key} should be a scalar array, got shape {data[scalar_key].shape}")

    if n_steps is not None and n_steps > 0:
        frame_index = data["frame_index"]
        sample_index = data["index"]
        next_done = data["next.done"]
        next_success = data["next.success"]

        if not np.array_equal(frame_index, np.arange(n_steps, dtype=frame_index.dtype)):
            warnings.append("frame_index is not contiguous 0..N-1")

        if not np.array_equal(sample_index, np.arange(n_steps, dtype=sample_index.dtype)):
            warnings.append("index is not contiguous 0..N-1")

        done_count = int(next_done.sum())
        if done_count != 1 or not bool(next_done[-1]):
            warnings.append("next.done is expected to be true only at terminal step (last index)")

        if np.any(next_success & ~next_done):
            warnings.append("next.success has true values where next.done is false")

    if strict and warnings:
        errors.extend([f"[strict] {warning}" for warning in warnings])

    return len(errors) == 0, errors, warnings


def gather_files(path: Path, pattern: str) -> List[Path]:
    if path.is_file():
        return [path]
    return [file_path for file_path in sorted(path.glob(pattern)) if not file_path.name.endswith("_images.npz")]


def main():
    parser = argparse.ArgumentParser(description="Validate per-episode NPZ schema")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to one episode .npz file or a directory containing episodes",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="Glob pattern for episode files when --input is a directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args()

    target = Path(args.input)
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")

    files = gather_files(target, args.pattern)
    if not files:
        raise ValueError(f"No episode files found in {target} with pattern {args.pattern}")

    print(f"Validating {len(files)} episode file(s)...")

    ok_count = 0
    warn_count = 0
    failed_files = []

    for file_path in files:
        ok, errors, warnings = validate_episode_file(file_path, strict=args.strict)
        if ok:
            ok_count += 1
        else:
            failed_files.append(file_path)

        if warnings:
            warn_count += len(warnings)

        status = "OK" if ok else "FAIL"
        print(f"[{status}] {file_path}")

        for warning in warnings:
            print(f"  WARN: {warning}")
        for error in errors:
            print(f"  ERROR: {error}")

    print("-" * 72)
    print(f"Passed: {ok_count}/{len(files)}")
    print(f"Warnings: {warn_count}")

    if failed_files:
        print("Failed files:")
        for file_path in failed_files:
            print(f"  - {file_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
