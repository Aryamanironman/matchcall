"""Regenerate the tracking overlay with Aryaman's trained YOLOv5lu.

The FootballVision ball path is raw low-confidence detections plus offline
linear interpolation (``trackers/tracker.py: ball_interpolation``), with no gap
limit. This renders that path on centres with a bounded, teleport-rejecting
version in ``app/vision/fv_tracker.py`` and draws predicted frames as a ring,
using the Aryaman V1 fine-tune once it exists.

It does not re-run commentary or TTS: the goal commentary is taken from the
existing ``annotated_goal.mp4`` when present, else the source audio.

    .venv\\Scripts\\python scripts\\annotate_tracking.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.vision.fv_tracker import DEFAULT_WEIGHTS, FootballVisionTracker
from scripts.make_commented_highlights import _annotate, _track_clip

VIDEO = r"C:\Users\asus\Downloads\3 - 3 ARG vs FRA - Second-Half - Bein.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--t0", type=float, default=2682.0,
                    help="goal at 2687 s; start 5 s before")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--out", default=str(
        settings.out_root / "commented_highlights" / "annotated_goal.mp4"))
    ap.add_argument("--weights", default=None,
                    help="default: Aryaman best.pt if present, else the "
                         "FootballVision checkpoint")
    ap.add_argument("--home", default="Argentina")
    ap.add_argument("--away", default="France")
    ap.add_argument("--audio-from", default=None,
                    help="clip to take audio from (default: current "
                         "annotated_goal.mp4, else the source)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_from = Path(args.audio_from) if args.audio_from else out
    work = settings.out_root / "_annotate_work"
    work.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights) if args.weights else DEFAULT_WEIGHTS
    if not weights.exists():
        raise SystemExit(f"weights missing: {weights}")
    is_aryaman = "aryaman" in str(weights).lower()
    label = ("Aryaman YOLOv5lu · V1 fine-tune" if is_aryaman
             else "FootballVision yolov5su")
    print(f"weights: {weights} ({label})", flush=True)

    tracker = FootballVisionTracker(weights=weights, conf=0.15)
    t0 = time.time()
    stats = _track_clip(Path(args.video), args.t0, args.duration, work,
                        tracker, fps=4.0, width=960)
    rows = stats["rows"]
    ball_frames = sum(1 for r in rows if r.get("ball"))
    interp = sum(1 for r in rows if r.get("ball_interpolated"))
    print(f"frames {len(rows)} · ball {ball_frames} · interpolated {interp}",
          flush=True)

    raw = out.with_name("annotated_goal_raw.mp4")
    _annotate(rows, stats["images"], raw, stats["fps"], args.home, args.away,
              weights_label=label)

    src_audio = audio_from if audio_from.exists() else Path(args.video)
    src_is_video = src_audio.resolve() == Path(args.video).resolve()
    seek = (["-ss", f"{args.t0:.3f}"] if src_is_video else [])
    # never write onto an input file: mux to a temp, then replace
    tmp = out.with_name(out.stem + ".tmp.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(raw), *seek, "-i", str(src_audio),
           "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
           "-shortest", "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(out)
    raw.unlink(missing_ok=True)

    meta = {
        "created": time.time(),
        "model": label,
        "weights": str(weights),
        "weights_mtime": weights.stat().st_mtime if weights.exists() else None,
        "detector": "YOLOv5lu (Aryaman V1 fine-tune)" if is_aryaman
                    else "FootballVision yolov5su",
        "ball_tracking": (
            "raw ball detections + bounded linear interpolation, following "
            "FootballVision trackers/tracker.py; max gap 16 frames, "
            "single-frame teleport rejection, predicted frames marked"),
        "frames": len(rows),
        "ball_frames": ball_frames,
        "interpolated_frames": interp,
        "audio_from": str(src_audio),
        "clip": {"video": args.video, "t0": args.t0,
                 "duration": args.duration},
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out.parent / "tracking_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
