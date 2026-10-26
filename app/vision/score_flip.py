"""Score-bug reader: goal anchors from the broadcast scoreboard.

The beIN feed keeps a persistent score bug in the top-left corner. We sample
the two score-digit boxes at 1 fps and classify each digit deterministically
from its binary shape:

    one enclosed hole          -> 0
    no hole, tall/narrow       -> 1
    no hole, wide              -> 2

(0/1/2 cover the ARG-FRA demo; higher digits would need a template set.)
A goal is a *sustained* transition to expected+1 (>= 5 consecutive seconds).
This is far more reliable than OCR on this font, which regularly read a "0"
as "1" or a "1" as "2" during graphics wipes.

The anchor is placed 3 s before the sustained transition because the score
bug updates a few seconds after the ball crosses the line.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from app.config import settings

# (x1, y1, x2, y2) at 1920x1080, top-left score bug.
BEIN_1080_BOXES = {
    "home": (425, 56, 468, 98),   # left score digit (ARG)
    "away": (538, 56, 575, 98),   # right score digit (FRA)
}
BEIN_1080_CLOCK = (115, 50, 240, 100)
_CROP_1080 = (0, 45, 800, 110)     # whole bug strip, y 45..110


@dataclass
class ScoreAnchor:
    t: float           # video-absolute seconds
    side: str          # "home" | "away"
    minute: int        # broadcast-minute estimate
    score_before: int | None = None
    score_after: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _boxes_for(width: int, height: int) -> tuple[dict, tuple, tuple]:
    sx, sy = width / 1920.0, height / 1080.0
    boxes = {k: (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
             for k, (x1, y1, x2, y2) in BEIN_1080_BOXES.items()}
    clock = tuple(int(v * (sx if i % 2 == 0 else sy))
                  for i, v in enumerate(BEIN_1080_CLOCK))
    crop = (int(_CROP_1080[0] * sx), int(_CROP_1080[1] * sy),
            int(_CROP_1080[2] * sx), int(_CROP_1080[3] * sy))
    return boxes, clock, crop


def classify_digit(crop: np.ndarray) -> int | None:
    """Classify a white score digit from its binary shape.

    Returns 0, 1 or 2, or None when the digit is occluded/unreadable.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray > 150).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    # The score-bug border is a long, thin white line; the digit is tall.
    min_h = max(8, int(0.45 * crop.shape[0]))
    max_h = int(0.82 * crop.shape[0])
    candidates = [i for i in range(1, n)
                  if stats[i, cv2.CC_STAT_HEIGHT] >= min_h
                  and stats[i, cv2.CC_STAT_AREA] >= 20]
    if not candidates:
        return None
    # A neighbouring competition logo can share the away-digit box and is
    # taller than the digit; prefer components at digit height.
    digit_like = [i for i in candidates
                  if stats[i, cv2.CC_STAT_HEIGHT] <= max_h]
    if digit_like:
        candidates = digit_like
    idx = max(candidates, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[idx]
    if area < 25 or h < 12:
        return None
    comp = (labels[y:y + h, x:x + w] == idx).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for k, row in enumerate(hierarchy[0]):
            if row[3] != -1 and cv2.contourArea(contours[k]) > 8:
                holes += 1
    if holes >= 1:
        return 0
    # "2" has a wide bottom bar; "1" is a narrow stem (even with its top flag).
    bottom = comp[int(h * 0.72):, :]
    bottom_run = max((int(r.sum()) for r in bottom), default=0)
    return 2 if bottom_run >= 0.80 * w else 1


def read_start_score(video, t0: float = 0.0, samples: int = 3) -> dict:
    """Read the score already on the board at the start of a clip."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes, _clock, _crop = _boxes_for(width, height)
    if t0 > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    rows = []
    for _ in range(max(1, samples)):
        ok, frame = cap.read()
        if not ok:
            break
        rows.append({side: classify_digit(frame[y1:y2, x1:x2])
                     for side, (x1, y1, x2, y2) in boxes.items()})
    cap.release()
    out = {}
    for side in ("home", "away"):
        vals = [r[side] for r in rows if r[side] is not None]
        if vals:
            out[side] = max(set(vals), key=vals.count)
    return out


def detect_score_flips(video, t0: float = 0.0, duration: float | None = None,
                       sample_fps: float = 1.0,
                       minute_offset: float = 0.0,
                       start_score: dict | None = None,
                       use_ocr: bool = False, **kwargs) -> list[ScoreAnchor]:
    """Return score-change anchors for the clip, one per goal.

    ``start_score`` is the score already on the board at ``t0``; transitions
    are matched against it, so an occluded before-window cannot hide a goal.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    boxes, _clock, (cx1, cy1, cx2, cy2) = _boxes_for(width, height)
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw <= 0 or ch <= 0:
        return []

    cmd = [settings.ffmpeg, "-v", "error"]
    if t0 > 0:
        cmd += ["-ss", f"{t0:.3f}"]
    cmd += ["-i", str(video)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"fps={sample_fps},crop={cw}:{ch}:{cx1}:{cy1}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    frame_bytes = cw * ch * 3
    digits: list[dict[str, int | None]] = []
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(ch, cw, 3)
            row = {}
            for side, (x1, y1, x2, y2) in boxes.items():
                row[side] = classify_digit(
                    frame[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1])
            digits.append(row)
    finally:
        try:
            proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        proc.kill()
        proc.wait()

    debug = os.getenv("FC_SCORE_DEBUG") == "1"
    if debug:
        print(f"  [score] first rows {digits[:6]}")

    def _mode(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return 0
        return max(set(vals), key=vals.count)

    provided = start_score or {}
    supplied = any(int(provided.get(s, 0) or 0) > 0
                   for s in ("home", "away"))
    if supplied:
        expected = {"home": 0, "away": 0, **provided}
    else:
        # A short clip can start mid-match. Read the score already on the
        # board from the first few samples instead of assuming 0-0.
        probe = digits[:3]
        expected = {s: _mode([row.get(s) for row in probe])
                    for s in ("home", "away")}
        if debug:
            print(f"  [score] inferred start {expected}")
    state = {s: {"run": 0, "start": None} for s in ("home", "away")}
    anchors: list[ScoreAnchor] = []
    for i, row in enumerate(digits):
        for side in ("home", "away"):
            want = expected[side] + 1
            if row.get(side) == want:
                if state[side]["run"] == 0:
                    state[side]["start"] = i
                state[side]["run"] += 1
                if state[side]["run"] >= 5:
                    start = state[side]["start"]
                    before, after = expected[side], want
                    expected[side] = want
                    state[side] = {"run": 0, "start": None}
                    t = t0 + max(0.0, start / sample_fps - 3.0)
                    minute = int(t / 60.0 + minute_offset)
                    if debug:
                        print(f"  [score] {t:.0f}s {side} {before}->{after} "
                              f"(minute {minute})")
                    anchors.append(ScoreAnchor(
                        t=t, side=side, minute=minute,
                        score_before=before, score_after=after))
            else:
                state[side] = {"run": 0, "start": None}
    anchors.sort(key=lambda a: a.t)
    return anchors


if __name__ == "__main__":
    import sys
    for a in detect_score_flips(sys.argv[1]):
        print(a)
