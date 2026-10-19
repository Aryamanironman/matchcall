"""Motion-gated smart frame sampling.

Strategy: one fast ffmpeg pass extracts a small proxy (low fps, small width),
then frame-diff motion scoring gates how much downstream work each moment gets.
Dead time (graphics, crowd close-ups, replays) is sampled sparsely; active
play gets the full detector cadence.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.config import settings


def probe_video(video: Path) -> dict:
    cmd = [
        settings.ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", str(video),
    ]
    import json
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den),
        "duration": float(data["format"]["duration"]),
    }


def has_audio(video: Path) -> bool:
    cmd = [
        settings.ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return out == "audio"


def extract_proxy_frames(video: Path, out_dir: Path, t0: float = 0.0,
                         duration: float | None = None) -> list[Path]:
    """Extract the analysis proxy: settings.analysis_fps fps, small width JPEGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # purge stale frames FIRST: a reused work dir (webapp re-runs, second
    # slices) would otherwise glob old + new frames together and stamp every
    # one of them with fresh timestamps — silently corrupting all features.
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cmd = [
        settings.ffmpeg, "-y", "-v", "error",
        "-ss", f"{t0:.3f}",
    ]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-i", str(video),
        "-vf", f"fps={settings.analysis_fps},scale={settings.analysis_width}:-2",
        "-q:v", "5",
        str(out_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(out_dir.glob("*.jpg"))
    return frames


def frame_timestamps(frames: list[Path], t0: float) -> list[float]:
    """Analysis frame i covers time t0 + i / analysis_fps (1-indexed files)."""
    step = 1.0 / settings.analysis_fps
    return [t0 + i * step for i in range(len(frames))]


def motion_scores(frames: list[Path]) -> np.ndarray:
    """Mean abs frame-diff per frame (first frame gets score of second)."""
    if not frames:
        return np.zeros(0)
    prev = None
    scores = np.zeros(len(frames), dtype=np.float32)
    for i, p in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        small = cv2.resize(img, (128, 72))
        if prev is not None:
            scores[i] = float(np.abs(small.astype(np.int16) - prev).mean())
        prev = small
    scores[0] = scores[1] if len(scores) > 1 else 0.0
    return scores


def detection_schedule(motion: np.ndarray, n_frames: int) -> list[int]:
    """Indices of frames the detector should run on.

    Active frames (motion >= gate): every det_interval frame.
    Dead frames: 4x sparser, but never fully skipped (keeps context for filler).
    """
    step_active = max(1, int(round(settings.det_interval * settings.analysis_fps)))
    step_dead = step_active * 4
    gate = settings.motion_dead_gate
    out: list[int] = []
    i = 0
    while i < n_frames:
        out.append(i)
        active = i + 1 < len(motion) and motion[i + 1] >= gate
        i += step_active if active else step_dead
    return out
