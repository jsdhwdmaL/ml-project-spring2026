#!/usr/bin/env python3
"""Read PushT replay zarr under state-only/pusht 2/ and write a sample PNG + episode MP4.

Expects layout like pusht_cchi_v7_replay.zarr:
  data/img     (T, 96, 96, 3) float32
  meta/episode_ends  (n_episodes,) int64 — cumulative exclusive frame indices (last == T).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZARR = REPO_ROOT / "state-only" / "pusht 2" / "pusht_cchi_v7_replay.zarr"


def _rgb_float_to_u8(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    m = float(np.nanmax(x)) if x.size else 0.0
    if m <= 1.0 + 1e-3:
        x = np.clip(x, 0.0, 1.0) * 255.0
    else:
        x = np.clip(x, 0.0, 255.0)
    return np.round(x).astype(np.uint8)


def _episode_slices(ends: np.ndarray, total_frames: int) -> list[tuple[int, int]]:
    """Return [(start, end), ...] half-open intervals covering [0, total_frames)."""
    ends = np.asarray(ends, dtype=np.int64).ravel()
    if ends.size == 0:
        return []
    if int(ends[-1]) == total_frames:
        starts = np.concatenate([np.array([0], dtype=np.int64), ends[:-1]])
        return [(int(s), int(e)) for s, e in zip(starts.tolist(), ends.tolist())]
    # Fallback: treat as per-episode lengths
    cum = np.cumsum(ends)
    if int(cum[-1]) != total_frames:
        raise ValueError(
            f"Cannot interpret episode_ends: last={ends[-1]!r}, cum_last={cum[-1]!r}, T={total_frames}"
        )
    starts = np.concatenate([np.array([0], dtype=np.int64), cum[:-1]])
    return [(int(s), int(e)) for s, e in zip(starts.tolist(), cum.tolist())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sample frame + episode from pusht 2 zarr.")
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ZARR, help="Path to *.zarr directory")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "state-only" / "samples_pusht2",
        help="Where to write PNG and MP4",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index for the sample MP4")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for exported MP4")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open outputs with the OS default viewer (macOS: open)",
    )
    args = parser.parse_args()

    if not args.zarr.is_dir():
        raise SystemExit(f"Zarr store not found: {args.zarr}")

    try:
        import zarr
    except ImportError as e:
        raise SystemExit("Install zarr: pip install zarr") from e
    try:
        import cv2
    except ImportError as e:
        raise SystemExit("opencv-python required") from e

    # zarr.open works for zarr v2 directory stores (Group); open_group is an alias on some versions.
    root = zarr.open(str(args.zarr), mode="r")
    img = root["data"]["img"]
    ends = np.asarray(root["meta"]["episode_ends"])

    total = int(img.shape[0])
    slices = _episode_slices(ends, total)
    if not slices:
        raise RuntimeError("Empty episode_ends or could not parse boundaries.")
    if args.episode < 0 or args.episode >= len(slices):
        raise SystemExit(f"--episode must be in [0, {len(slices) - 1}], got {args.episode}")

    args.out_dir.mkdir(parents=True)

    # First timestep in the dataset (sample image)
    rgb0 = _rgb_float_to_u8(np.asarray(img[0]))
    png_path = args.out_dir / "pusht2_sample_frame.png"
    cv2.imwrite(str(png_path), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))

    start, end = slices[args.episode]
    mp4_path = args.out_dir / "pusht2_sample_episode.mp4"
    h, w = rgb0.shape[0], rgb0.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, float(args.fps), (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {mp4_path}")

    for t in range(start, end):
        rgb = _rgb_float_to_u8(np.asarray(img[t]))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    n_frames = end - start
    print(f"Wrote {png_path} (timestep 0, {w}×{h})")
    print(f"Wrote {mp4_path} (episode {args.episode}, frames [{start}, {end}), {n_frames} frames @ {args.fps} fps)")
    print(f"  zarr: {args.zarr.resolve()}")

    if args.open:
        path = png_path.resolve()
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
            subprocess.run(["open", str(mp4_path.resolve())], check=False)
        elif sys.platform.startswith("linux") and shutil.which("xdg-open"):
            subprocess.run(["xdg-open", str(path)], check=False)
            subprocess.run(["xdg-open", str(mp4_path.resolve())], check=False)
        elif sys.platform == "win32":
            subprocess.run(["start", "", str(path)], shell=True, check=False)
            subprocess.run(["start", "", str(mp4_path.resolve())], shell=True, check=False)


if __name__ == "__main__":
    main()
