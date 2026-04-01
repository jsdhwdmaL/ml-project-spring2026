#!/usr/bin/env python3
"""Validate per-episode PushT NPZ files against the unified schema.

Checks required keys, dtypes, shapes, and per-step consistency for files produced
by teleop/intervention collectors.

Expected per-episode schema:
  observation.state: (N, 2, 18), float64
  action: (N, H, 2), float64
  action_is_pad: (N, H), bool
  frame_index: (N,), int64
  environment.raw_reward: (N,), float64
  next.done: (N,), bool
  next.success: (N,), bool
  is_human_intervention: (N,), bool
  is_pure_teleop: (N,), bool
  env_seed: scalar int64
  trial_idx: scalar int64
  success: scalar bool

Usage:
python scripts/validate_episode_schema.py \
    --input path/to/episode.npz
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


REQUIRED_STEP_KEYS = {
    "observation.state",
    "action",
    "action_is_pad",
    "frame_index",
    "environment.raw_reward",
    "next.done",
    "next.success",
    "is_human_intervention",
    "is_pure_teleop",
}

REQUIRED_SCALAR_KEYS = {
    "env_seed",
    "trial_idx",
    "success",
}


def _is_scalar_array(arr: np.ndarray) -> bool:
    return arr.shape == () or arr.shape == (1,)


def _check_dtype(arr: np.ndarray, expected: str) -> bool:
    if expected == "float64":
        return np.issubdtype(arr.dtype, np.float64)
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
    action_is_pad = data["action_is_pad"]
    frame_index = data["frame_index"]
    raw_reward = data["environment.raw_reward"]
    next_done = data["next.done"]
    next_success = data["next.success"]
    is_human = data["is_human_intervention"]
    is_pure_teleop = data["is_pure_teleop"]

    if obs.ndim != 3 or obs.shape[1:] != (2, 18):
        errors.append(f"observation.state must be (N,2,18), got {obs.shape}")

    if action.ndim != 3 or action.shape[2] != 2:
        errors.append(f"action must be (N,H,2), got {action.shape}")

    if action_is_pad.ndim != 2:
        errors.append(f"action_is_pad must be (N,H), got {action_is_pad.shape}")

    if action.ndim == 3 and action_is_pad.ndim == 2:
        if action.shape[:2] != action_is_pad.shape:
            errors.append(
                f"action/action_is_pad mismatch: action{action.shape[:2]} vs action_is_pad{action_is_pad.shape}"
            )

    N = obs.shape[0] if obs.ndim == 3 else None
    one_d_fields = {
        "frame_index": frame_index,
        "environment.raw_reward": raw_reward,
        "next.done": next_done,
        "next.success": next_success,
        "is_human_intervention": is_human,
        "is_pure_teleop": is_pure_teleop,
    }

    if N is not None:
        for name, arr in one_d_fields.items():
            if arr.ndim != 1 or arr.shape[0] != N:
                errors.append(f"{name} must be (N,), got {arr.shape} while N={N}")

        if action.ndim == 3 and action.shape[0] != N:
            errors.append(f"action first dim must match N, got action N={action.shape[0]}, obs N={N}")

    dtype_expectations = {
        "observation.state": (obs, "float64"),
        "action": (action, "float64"),
        "action_is_pad": (action_is_pad, "bool"),
        "frame_index": (frame_index, "int64"),
        "environment.raw_reward": (raw_reward, "float64"),
        "next.done": (next_done, "bool"),
        "next.success": (next_success, "bool"),
        "is_human_intervention": (is_human, "bool"),
        "is_pure_teleop": (is_pure_teleop, "bool"),
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

    if N is not None and N > 0:
        if not np.array_equal(frame_index, np.arange(N, dtype=frame_index.dtype)):
            warnings.append("frame_index is not contiguous 0..N-1")

        done_count = int(next_done.sum())
        if done_count != 1 or not bool(next_done[-1]):
            warnings.append("next.done is expected to be true only at terminal step (last index)")

        if np.any(next_success & ~next_done):
            warnings.append("next.success has true values where next.done is false")

        if action_is_pad.ndim == 2:
            pad_violation = False
            for row in action_is_pad:
                first_true = np.argmax(row) if np.any(row) else -1
                if first_true != -1 and not np.all(row[first_true:]):
                    pad_violation = True
                    break
            if pad_violation:
                warnings.append("action_is_pad rows are not suffix-style masks")

    if strict and warnings:
        errors.extend([f"[strict] {w}" for w in warnings])

    return len(errors) == 0, errors, warnings


def gather_files(path: Path, pattern: str) -> List[Path]:
    if path.is_file():
        return [path]
    files = [f for f in sorted(path.glob(pattern)) if not f.name.endswith("_images.npz")]
    return files


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

        for w in warnings:
            print(f"  WARN: {w}")
        for e in errors:
            print(f"  ERROR: {e}")

    print("-" * 72)
    print(f"Passed: {ok_count}/{len(files)}")
    print(f"Warnings: {warn_count}")

    if failed_files:
        print("Failed files:")
        for fp in failed_files:
            print(f"  - {fp}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
