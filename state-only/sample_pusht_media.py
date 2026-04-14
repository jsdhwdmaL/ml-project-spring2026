#!/usr/bin/env python3
"""Decode one sample frame from lerobot/pusht and copy one episode MP4 to disk.

Uses the same Hub dataset as in meta.json (observation.image video, 96x96, fps 10).
First run may download the dataset cache under ~/.cache/huggingface/.

Example:
  python state-only/sample_pusht_media.py --out_dir state-only/samples
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _tensor_to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """LeRobot may return CHW or HWC, float in [0,1] or uint8; normalize to HWC RGB uint8."""
    if img.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {img.shape}")
    # HWC (e.g. H,W,3)
    if img.shape[-1] == 3 or img.shape[-1] == 1:
        out = img
        if out.shape[-1] == 1:
            out = np.repeat(out, 3, axis=-1)
    # CHW
    elif img.shape[0] in (1, 3):
        out = np.transpose(img, (1, 2, 0))
        if out.shape[-1] == 1:
            out = np.repeat(out, 3, axis=-1)
    else:
        raise ValueError(f"Unrecognized layout for shape {img.shape}")

    if out.dtype == np.uint8:
        return out
    x = out.astype(np.float32)
    if np.nanmax(x) <= 1.0 + 1e-3:
        x = np.clip(x, 0.0, 1.0) * 255.0
    else:
        x = np.clip(x, 0.0, 255.0)
    return np.round(x).astype(np.uint8)


def _save_png_hwc_rgb(path: Path, hwc_rgb_u8: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, hwc_rgb_u8)
    except ImportError:
        import cv2

        bgr = cv2.cvtColor(hwc_rgb_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)


def _resolve_episode_video_path(dataset, episode_index: int) -> Path | None:
    """Return path to an MP4 for this episode, using metadata API or filesystem fallback."""
    root = Path(dataset.root)
    meta = dataset.meta
    keys = list(getattr(meta, "video_keys", []) or [])
    if not keys:
        return None
    vid_key = keys[0]

    if hasattr(meta, "get_video_file_path"):
        try:
            rel = meta.get_video_file_path(episode_index, vid_key)
            p = root / rel
            if p.is_file():
                return p
        except (KeyError, IndexError, ValueError, TypeError):
            pass

    videos = root / "videos"
    if not videos.is_dir():
        return None

    matches = sorted(videos.rglob("*.mp4"))
    if not matches:
        return None

    needle = f"{episode_index:06d}"
    for p in matches:
        if needle in p.stem or needle in p.name:
            return p
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Save one PNG frame and one MP4 from lerobot/pusht.")
    parser.add_argument("--dataset_id", type=str, default="lerobot/pusht", help="HF dataset id")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "state-only" / "samples",
        help="Directory for sample_frame.png and sample_episode.mp4",
    )
    parser.add_argument("--frame_index", type=int, default=0, help="Global frame index for the PNG")
    parser.add_argument("--episode_index", type=int, default=0, help="Which episode's MP4 to copy")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args.out_dir.mkdir(parents=True)

    print(f"Loading {args.dataset_id!r} (may download on first use)...")
    ds = LeRobotDataset(args.dataset_id, download_videos=True)

    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset is empty.")
    fi = int(np.clip(args.frame_index, 0, n - 1))

    sample = ds[fi]
    if "observation.image" not in sample:
        raise KeyError(
            f"No 'observation.image' in frame {fi}. Keys: {list(sample.keys())}"
        )

    img_t = sample["observation.image"]
    if hasattr(img_t, "detach"):
        img_t = img_t.detach().cpu().numpy()
    else:
        img_t = np.asarray(img_t)

    hwc = _tensor_to_hwc_uint8(np.asarray(img_t, dtype=np.float32))
    png_path = args.out_dir / "sample_frame.png"
    _save_png_hwc_rgb(png_path, hwc)
    print(f"Wrote {png_path} (dataset frame index {fi}, shape {hwc.shape})")

    ep_idx = args.episode_index
    vid_src = _resolve_episode_video_path(ds, ep_idx)
    if vid_src is None or not vid_src.is_file():
        print(
            "Could not locate an MP4 under the dataset root (videos/ missing or empty?). "
            "PNG still saved.",
            file=sys.stderr,
        )
        return

    mp4_dst = args.out_dir / "sample_episode.mp4"
    shutil.copy2(vid_src, mp4_dst)
    print(f"Copied episode video to {mp4_dst}")
    print(f"  source: {vid_src}")


if __name__ == "__main__":
    main()
