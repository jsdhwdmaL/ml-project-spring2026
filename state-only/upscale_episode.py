#!/usr/bin/env python3
"""Upscale a PushT episode MP4 (e.g. 96x96), optionally denoise frames, then open it (macOS: open)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "state-only" / "samples" / "sample_episode.mp4"
DEFAULT_OUT = REPO_ROOT / "state-only" / "samples" / "sample_episode_upscaled.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upscale an MP4 with Lanczos, denoise (non-local means), optionally open it."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_IN, help="Source MP4")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output MP4 path")
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Integer or float scale factor (e.g. 4 → 96×96 becomes 384×384)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="If set, output width (height follows aspect ratio; overrides --scale)",
    )
    parser.add_argument("--no-view", action="store_true", help="Do not open the file after writing")
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip denoising (default: fast non-local means after upscale)",
    )
    parser.add_argument(
        "--denoise-strength",
        type=float,
        default=6.0,
        help="Denoise filter strength (h / hColor for fastNlMeansDenoisingColored); higher = smoother",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    try:
        import cv2
    except ImportError as e:
        raise SystemExit("opencv-python is required (pip install opencv-python).") from e

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.width is not None:
        out_w = args.width
        out_h = max(1, int(round(src_h * (out_w / src_w))))
    else:
        s = float(args.scale)
        out_w = max(1, int(round(src_w * s)))
        out_h = max(1, int(round(src_h * s)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {args.output}")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        up = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        if not args.no_denoise:
            h = float(args.denoise_strength)
            up = cv2.fastNlMeansDenoisingColored(up, None, h, h, 7, 21)
        writer.write(up)
        n += 1

    cap.release()
    writer.release()

    denoise_note = "off" if args.no_denoise else f"strength={args.denoise_strength}"
    print(f"Wrote {args.output} ({n} frames, {out_w}×{out_h} @ {fps:.1f} fps, denoise {denoise_note})")

    if args.no_view:
        return

    path = args.output.resolve()
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux") and shutil.which("xdg-open"):
        subprocess.run(["xdg-open", str(path)], check=False)
    elif sys.platform == "win32" and shutil.which("explorer"):
        subprocess.run(["explorer", str(path)], check=False)
    else:
        print("Could not find a viewer; open the file manually:", path)


if __name__ == "__main__":
    main()
