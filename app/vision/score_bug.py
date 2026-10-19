"""Score-bug reading: gold goal anchors from the broadcast's own scoreboard.

Why: on loudness-normalized feeds the audio gate dies, celebrations hide the
goalmouth, and tracker noise floods ball speed — every heuristic discriminator
fails. But a broadcast score bug flips EXACTLY when a goal goes in, replays
do NOT flip it, and it never flips anywhere else. A persistent digit-region
change is therefore the highest-precision goal evidence available, it names
the scoring team (home box vs away box), and it survives camera cuts.

Method:
  1. locate the score-digit regions (auto: stable-overlay candidate boxes;
     anchored: known boxes from a one-time calibration, stored in work dir)
  2. sample those regions at ~2 fps for the whole clip
  3. a "flip" = region level leaves its stabilized band and the new level
     holds for FLIP_HOLD_S. Graphics wipes last 1-2 s and never pass the
     hold; a score change holds for the rest of the match.
  4. each surviving flip = ScoreAnchor(t, side) -> one goal by that side.

The module never guesses teams from pixels: "home"/"away" mapping comes from
the box order the calibration wrote down (left digit = home on most bugs).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# flip = new digit level persists this long (graphics wipes are 1-2 s)
FLIP_HOLD_S = 20.0
# change in the digit region's mean level that counts as a candidate flip
FLIP_DELTA = 18.0
SAMPLE_FPS = 2.0
BAND_H = 200            # px of the top of frame to search (full width)


@dataclass
class ScoreAnchor:
    t: float            # seconds (video-absolute)
    side: str           # "home" | "away"
    minute: int         # broadcast minute estimate (t / 60)

    def to_dict(self) -> dict:
        return asdict(self)


def _sample_region(video: str, box: tuple[int, int, int, int], t0: float,
                   dur: float, fps: float = SAMPLE_FPS) -> np.ndarray:
    """Mean gray level of the box over time (and its std within the box)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(dur), "-i", video,
           "-vf", f"fps={fps},crop={w}:{h}:{x1}:{y1}",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    n = len(raw) // (w * h)
    frames = np.frombuffer(raw[:n * w * h], np.uint8).reshape(n, h, w)
    return frames.astype(np.float32)


def find_flips(level: np.ndarray, fps: float = SAMPLE_FPS,
               delta: float = FLIP_DELTA, hold_s: float = FLIP_HOLD_S
               ) -> list[tuple[float, float]]:
    """(time_s, new_level) for each persistent level change in the series."""
    hold = max(int(hold_s * fps), 1)
    flips: list[tuple[float, float]] = []
    base = float(np.median(level[: max(len(level) // 8, 5)]))
    i, n = 0, len(level)
    while i < n - hold:
        if abs(level[i] - base) > delta and abs(level[i + hold] - base) > delta:
            # candidate start: walk back to the first sample off the base
            j = i
            while j > 0 and abs(level[j - 1] - base) > delta * 0.5:
                j -= 1
            flips.append((j / fps, float(np.median(level[i:i + hold]))))
            base = float(level[i + hold])
            i += hold
        else:
            i += 1
    return flips


def detect_goal_anchors(video: str, box_home: tuple[int, int, int, int],
                        box_away: tuple[int, int, int, int], t0: float = 0.0,
                        duration: float | None = None) -> list[ScoreAnchor]:
    """Read both digit boxes; every persistent flip is one goal by that side."""
    from app.vision.sampling import probe_video
    dur = duration or (probe_video(video)["duration"] - t0)
    anchors: list[ScoreAnchor] = []
    for side, box in (("home", box_home), ("away", box_away)):
        level = _sample_region(video, box, t0, dur).mean(axis=(1, 2))
        for t_sec, _lvl in find_flips(level):
            anchors.append(ScoreAnchor(t=t0 + t_sec, side=side,
                                       minute=int((t0 + t_sec) / 60)))
    anchors.sort(key=lambda a: a.t)
    return anchors


def save_calibration(path: Path, box_home, box_away) -> None:
    path.write_text(json.dumps({"box_home": list(box_home),
                                "box_away": list(box_away)}), encoding="utf-8")


def load_calibration(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return tuple(d["box_home"]), tuple(d["box_away"])


def validate_anchors(anchors: list[ScoreAnchor], known_goals: list[float],
                     tol: float = 10.0) -> dict:
    """How many known goals were caught, and how many flips were spurious."""
    caught, extra = [], []
    for a in anchors:
        if any(abs(a.t - g) <= tol for g in known_goals):
            caught.append(a)
        else:
            extra.append(a)
    return {"known": len(known_goals), "caught": len(caught),
            "missed": len(known_goals) - len(caught),
            "spurious": len(extra), "anchors": [a.to_dict() for a in anchors]}
