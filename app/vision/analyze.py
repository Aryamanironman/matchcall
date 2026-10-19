"""Per-video analysis: motion + detector features -> scored windows.

Runs YOLO on the detector-scheduled frames only (motion-gated smart sampling),
computes ball speed / goal-zone occupancy, then pools features into windows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.events import Window
from app.vision.detector import FootballDetector
from app.vision.sampling import (
    extract_proxy_frames, frame_timestamps, motion_scores, detection_schedule,
    probe_video,
)


def goal_zones(w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Ball-centroid zones that indicate danger/goalmouth (fractions of frame)."""
    return [
        (0.00, 0.30, 0.12, 0.85),   # left goalmouth
        (0.88, 0.30, 1.00, 0.85),   # right goalmouth
    ]


def _in_zone(cx_frac: float, cy_frac: float, zone) -> bool:
    x1, y1, x2, y2 = zone
    return x1 <= cx_frac <= x2 and y1 <= cy_frac <= y2


def analyze_video(video: Path, work_dir: Path, detector: FootballDetector,
                  t0: float = 0.0, duration: float | None = None,
                  log_every: int = 200, on_detect=None) -> dict:
    info = probe_video(video)
    frames_dir = work_dir / "proxy_frames"
    frames = extract_proxy_frames(video, frames_dir, t0=t0, duration=duration)
    ts = frame_timestamps(frames, t0)
    motion = motion_scores(frames)
    sched = detection_schedule(motion, len(frames))
    zones = goal_zones(settings.analysis_width, settings.analysis_width * 9 // 16)

    t_start = time.time()
    imgs = [cv2.imread(str(frames[i])) for i in sched]
    imgs = [im for im in imgs if im is not None]
    # keep alignment: re-pair in case of imread failures
    sched_ok = [i for i in sched if frames[i].exists()]
    imgs = [cv2.imread(str(frames[i])) for i in sched_ok]
    det_start = time.time()
    dets = detector.infer(imgs, on_chunk=on_detect)
    det_time = time.time() - det_start

    # scatter frame-indexed features
    n = len(frames)
    ball_x = np.full(n, np.nan); ball_y = np.full(n, np.nan)
    has_ball = np.zeros(n); n_players = np.zeros(n)
    in_goal_zone = np.zeros(n)
    for k, i in enumerate(sched_ok):
        d = dets[k]
        n_players[i] = len(d["player"]) + len(d["referee"])
        if d["ball"] is not None:
            has_ball[i] = 1
            cx, cy, _ = d["ball"]
            h, w = imgs[k].shape[:2]
            fx, fy = cx / w, cy / h
            ball_x[i] = fx; ball_y[i] = fy
            if any(_in_zone(fx, fy, z) for z in zones):
                in_goal_zone[i] = 1

    # ball speed (px/s in analysis-res space): measured between consecutive
    # DETECTED frames with their real dt. The detector schedule is sparse, so
    # consecutive proxy frames almost never both carry a ball detection —
    # pairing by frame index i-1 zeroed ball_speed for entire matches.
    W, H = settings.analysis_width, settings.analysis_width * 9 // 16
    ball_speed = np.zeros(n)
    prev_i = None
    for i in sched_ok:
        if not np.isfinite(ball_x[i]):
            continue
        if prev_i is not None:
            dt = ts[i] - ts[prev_i]
            if dt > 0:
                dx = (ball_x[i] - ball_x[prev_i]) * W
                dy = (ball_y[i] - ball_y[prev_i]) * H
                ball_speed[i] = float(np.hypot(dx, dy) / dt)
        prev_i = i
    step = 1.0 / settings.analysis_fps
    # forward-fill has_ball fraction smoothing
    ball_seen = has_ball.copy()
    for i in range(1, n):
        ball_seen[i] = max(ball_seen[i], ball_seen[i - 1] * 0.5)

    # pool into windows
    win_len = settings.window_len
    win_frames = max(1, int(round(win_len * settings.analysis_fps)))
    windows: list[Window] = []
    for s in range(0, n, win_frames):
        e = min(s + win_frames, n)
        if e - s < win_frames * 0.4:
            break
        w = Window(
            t0=ts[s], t1=ts[e - 1] + step,
            motion=float(np.mean(motion[s:e])),
            ball_speed=float(np.nanmean(np.where(ball_speed[s:e] > 0, ball_speed[s:e], np.nan)))
            if np.any(ball_speed[s:e] > 0) else 0.0,
            n_players=float(np.mean(n_players[s:e])),
            has_ball=float(np.mean(ball_seen[s:e])),
            goal_zone_time=float(np.mean(in_goal_zone[s:e])),
        )
        windows.append(w)

    return {
        "video": str(video), "duration": info["duration"], "fps": info["fps"],
        "width": info["width"], "height": info["height"],
        "n_proxy_frames": n, "n_detector_frames": len(sched_ok),
        "detector_time_s": det_time,
        "analysis_time_s": time.time() - t_start,
        "windows": [w.to_dict() for w in windows],
    }


if __name__ == "__main__":
    import sys
    analyze_video(Path(sys.argv[1]), Path(sys.argv[2]), FootballDetector())
