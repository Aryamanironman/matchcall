"""Score-bug reader: goal anchors from the broadcast scoreboard.

Cheap glyph spikes propose candidate moments; OCR on the two score digits
confirms which side changed and filters graphics wipes; the broadcast clock
supplies the match minute. This is the "CNN action spotting" gold signal for
goals: it survives replays, celebrations and camera cuts.

The boxes below are calibrated for the 1920x1080 beIN Sports feed used by the
ARG-FRA demo and scale with the frame size. OCR is optional: without EasyOCR
the reader falls back to glyph-only anchors (noisier).
"""
from __future__ import annotations

import os
import subprocess
from collections import Counter
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


def _glyph(gray: np.ndarray) -> np.ndarray:
    """Binary 20x32 glyph signature, stable across compression noise."""
    mask = (gray > 150).astype(np.float32)
    mask = cv2.resize(mask, (20, 32), interpolation=cv2.INTER_AREA)
    return (mask > 0.35).astype(np.uint8)


def _boxes_for(width: int, height: int) -> tuple[dict, tuple, tuple]:
    sx, sy = width / 1920.0, height / 1080.0
    boxes = {k: (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
             for k, (x1, y1, x2, y2) in BEIN_1080_BOXES.items()}
    clock = tuple(int(v * (sx if i % 2 == 0 else sy))
                  for i, v in enumerate(BEIN_1080_CLOCK))
    crop = (int(_CROP_1080[0] * sx), int(_CROP_1080[1] * sy),
            int(_CROP_1080[2] * sx), int(_CROP_1080[3] * sy))
    return boxes, clock, crop


def _read_digit(reader, crop: np.ndarray) -> int | None:
    up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    try:
        res = reader.readtext(up, allowlist="0123456789", detail=1,
                              paragraph=False)
    except Exception:  # noqa: BLE001
        return None
    best = None
    for box, text, conf in res:
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            continue
        # the crop is centred on one digit; neighbouring text can leak in at
        # the edges, so keep the rightmost digit of each OCR result
        cand = (float(conf), int(digits[-1]))
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1] if best else None


def _read_clock(reader, crop: np.ndarray) -> float | None:
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try:
        res = reader.readtext(up, allowlist="0123456789:", detail=1,
                              paragraph=False)
    except Exception:  # noqa: BLE001
        return None
    text = "".join(t for _b, t, _c in res)
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 4:
        return None
    # "79:28" or "7928" -> 79 + 28/60
    mm, ss = int(digits[:-2]), int(digits[-2:])
    if ss >= 60 or mm > 130:
        return None
    return mm + ss / 60.0


def detect_score_flips(video, t0: float = 0.0, duration: float | None = None,
                       sample_fps: float = 1.0, min_diff: int = 70,
                       min_gap_s: float = 20.0, use_ocr: bool = True,
                       minute_offset: float = 0.0,
                       start_score: dict | None = None) -> list[ScoreAnchor]:
    """Return score-change anchors for the clip, one per goal.

    ``min_diff`` is the glyph-mask Hamming distance that proposes a candidate.
    OCR then verifies that the changed digit equals the expected score + 1.
    The expected score is carried forward from ``start_score`` (the score
    already on the board at ``t0``), so an occluded before-window cannot hide
    a goal. The two ARG-FRA goals score 99 and 81 against an 18-45 noise band.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    boxes, clock_box, (cx1, cy1, cx2, cy2) = _boxes_for(width, height)
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
    sigs: list[dict[str, np.ndarray]] = []
    crops: list[dict[str, np.ndarray]] = []
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(ch, cw, 3)
            row, crp = {}, {}
            for side, (x1, y1, x2, y2) in boxes.items():
                g = cv2.cvtColor(frame[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1],
                                 cv2.COLOR_BGR2GRAY)
                row[side] = _glyph(g)
                crp[side] = frame[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1].copy()
            row["clock"] = None
            x1, y1, x2, y2 = clock_box
            crp["clock"] = frame[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1].copy()
            sigs.append(row)
            crops.append(crp)
    finally:
        try:
            proc.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        proc.kill()
        proc.wait()

    if len(sigs) < 3:
        return []

    # -- candidates from cheap glyph spikes ----------------------------------
    cands: list[tuple[int, str]] = []
    last_i = -10 ** 9
    sides = ("home", "away")
    for i in range(1, len(sigs)):
        diffs = {s: int(np.count_nonzero(sigs[i - 1][s] != sigs[i][s]))
                 for s in sides}
        side = max(sides, key=lambda s: diffs[s])
        if diffs[side] < min_diff:
            continue
        lo, hi = max(1, i - 2), min(len(sigs), i + 3)
        peak = max(int(np.count_nonzero(sigs[j - 1][side] != sigs[j][side]))
                   for j in range(lo, hi))
        if diffs[side] < peak:
            continue
        gap_frames = int(round(min_gap_s * sample_fps))
        if i - last_i < gap_frames:
            continue
        cands.append((i, side))
        last_i = i

    # -- OCR confirmation ----------------------------------------------------
    reader = None
    if use_ocr:
        try:
            import easyocr
            use_gpu = os.getenv("FC_SCORE_OCR_GPU", "1") != "0"
            reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        except Exception as e:  # noqa: BLE001
            print(f"      [score] OCR unavailable ({type(e).__name__}); "
                  "using glyph-only anchors")

    anchors: list[ScoreAnchor] = []
    expected = {"home": 0, "away": 0, **(start_score or {})}
    debug = os.getenv("FC_SCORE_DEBUG") == "1"
    for i, side in cands:
        if reader is None:
            # glyph-only fallback: trust the spike and advance the score
            before_digit = expected[side]
            after_digit = before_digit + 1
            expected[side] = after_digit
        else:
            after = [_read_digit(reader, crops[j][side])
                     for j in range(i + 4, min(len(sigs), i + 36))]
            after = [d for d in after if d is not None]
            ac = Counter(after).most_common(1)[0] if after else (None, 0)
            if debug:
                print(f"  [score] cand t={t0 + i / sample_fps:.0f} side={side} "
                      f"expected={expected[side]} after={Counter(after).most_common(4)}")
            if len(after) < 3 or ac[1] / len(after) < 0.50:
                continue
            if ac[0] != expected[side] + 1:
                continue
            other = "away" if side == "home" else "home"
            oa = [_read_digit(reader, crops[j][other])
                  for j in range(i + 4, min(len(sigs), i + 36))]
            oa = [d for d in oa if d is not None]
            if len(oa) >= 3 and Counter(oa).most_common(1)[0][0] != expected[other]:
                continue
            before_digit, after_digit = expected[side], ac[0]
            expected[side] = after_digit

        t = t0 + i / sample_fps
        minute = None
        if reader is not None:
            for j in (i, min(len(sigs) - 1, i + 2), max(0, i - 2)):
                minute = _read_clock(reader, crops[j]["clock"])
                if minute is not None:
                    break
        if minute is None:
            minute = t / 60.0 + minute_offset
        # clock OCR can latch onto a different on-screen number (e.g. a stat
        # graphic); reject reads far from the video-time estimate
        est = t / 60.0 + minute_offset
        if abs(minute - est) > 3.0:
            minute = est
        anchors.append(ScoreAnchor(t=t, side=side, minute=int(round(minute)),
                                   score_before=before_digit,
                                   score_after=after_digit))
    if reader is not None:
        del reader
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return anchors


if __name__ == "__main__":
    import sys
    for a in detect_score_flips(sys.argv[1]):
        print(a)
