"""End-to-end orchestrator.

video -> motion-gated sampling -> YOLOv8 features -> auto-label -> ResNet18
-> commentary (Gemini/templates) -> filler planning -> edge-tts -> ffmpeg mix.
"""
from __future__ import annotations

import bisect
import json
import os
import time
from pathlib import Path

import numpy as np

from app.config import settings, RESNET_WEIGHTS
from app.events import Segment, Window
from app.vision.sampling import probe_video, has_audio
from app.vision.audio import audio_rms_profile, crowd_rise, audio_evidence
from app.vision.detector import FootballDetector
from app.vision.fv_tracker import make_detector
from app.vision.analyze import analyze_video
from app.vision.auto_label import (label_windows, merge_events, enforce_caps,
                                   ROAR_FLOOR, ROAR_STRIKE, BALL_SPEED_SHOT)
from app.vision import event_net
from app.commentary import make_announcer
from app.duo import Duo, CHAT_GAP
from app.filler import plan_fillers
from app import tts as tts_mod
from app import render as render_mod


def _fmt_ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def run_pipeline(video: Path, home: str, away: str, t0: float = 0.0,
                 duration: float | None = None, out_dir: Path | None = None,
                 orig_audio: bool = True,
                 start_score: dict | None = None,
                 minute_offset: float = 0.0,
                 progress=None, on_transcript=None, on_segment=None,
                 _cancel_check=None) -> dict:
    progress = progress or (lambda stage, frac, msg="": None)
    _chk = _cancel_check or (lambda: False)
    import app.config as _cfg
    _cfg.SCORE_ANCHORS[:] = []          # fresh state for every job
    t_wall = time.time()
    t_cpu = time.process_time()
    video = Path(video)
    out_dir = out_dir or (settings.out_root / f"{video.stem}_{int(t0)}s")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)

    progress("probe", 0.01, f"probing {video.name}")
    info = probe_video(video)
    dur = duration or max(0.0, info["duration"] - t0)

    progress("sampling", 0.04, f"extracting {dur:.0f}s @ {settings.analysis_fps}fps proxy frames")
    detector = make_detector()

    # crowd-energy profile: goals are confirmed by a crowd surge in ANY feed,
    # vision alone over-fires on every attack (feeds are loudness-normalized,
    # so we track *rise over the local bed*, not absolute loudness)
    progress("sampling", 0.05, "profiling crowd energy from the audio track")
    rms = audio_rms_profile(video, t0=t0, duration=dur)
    rise = crowd_rise(rms) if len(rms) else None
    audio_usable = len(rms) >= 30 and float(np.median(rms)) > 0.0

    # gold goal anchors from the broadcast score bug: the bug flips exactly
    # on goals, survives replays/celebration cuts, and names the scoring
    # side. The glyph reader needs no calibration; the older brightness
    # reader remains a fallback when a calibration file exists.
    anchors: list[dict] = []
    try:
        from app.vision.score_flip import detect_score_flips, read_start_score
        progress("sampling", 0.06, "reading score-bug digits")
        if not start_score or all(
                int(start_score.get(s, 0) or 0) == 0
                for s in ("home", "away")):
            inferred = read_start_score(video, t0=t0)
            if inferred:
                start_score = inferred
                print(f"      score bug: inferred start {start_score}")
        anchors = [a.to_dict() for a in detect_score_flips(
            video, t0=t0, duration=dur, minute_offset=minute_offset,
            start_score=start_score)]
        _cfg.SCORE_ANCHORS[:] = anchors
        if anchors:
            print(f"      score bug: {len(anchors)} goal anchor(s) "
                  f"{[a['side'] + ' ' + _fmt_ts(a['t']) for a in anchors]}")
    except Exception as sb_err:  # noqa: BLE001 - bug reading is optional gold
        print(f"      score-bug flip pass skipped: {type(sb_err).__name__}: {sb_err}")
    if not anchors:
        try:
            from app.vision import score_bug
            calib_path = Path(os.getenv("SCORE_BUG_CALIB", "")) \
                if os.getenv("SCORE_BUG_CALIB") else \
                settings.models_dir / "score_bug_calibration.json"
            calib = score_bug.load_calibration(calib_path)
            if calib:
                box_home, box_away = calib
                anchors = [a.to_dict() for a in score_bug.detect_goal_anchors(
                    str(video), box_home, box_away, t0=t0, duration=dur)]
                for a in anchors:
                    if not a.get("minute"):
                        a["minute"] = int(round(a["t"] / 60.0 + minute_offset))
                _cfg.SCORE_ANCHORS[:] = anchors
                print(f"      score bug (calibrated): {len(anchors)} anchor(s)")
        except Exception as sb_err:  # noqa: BLE001
            print(f"      calibrated score-bug pass skipped: "
                  f"{type(sb_err).__name__}: {sb_err}")

    analysis = analyze_video(
        video, work, detector, t0=t0, duration=dur,
        on_detect=lambda done, total: progress(
            "detect", 0.10 + 0.50 * done / max(total, 1),
            f"YOLO {done}/{total} frames"))
    _chk()
    windows = [Window(**wd) for wd in analysis.pop("windows")]
    progress("detect", 0.60, f"{analysis['n_proxy_frames']} proxy frames, "
             f"{analysis['n_detector_frames']} detector frames "
             f"({analysis['detector_time_s']:.1f}s yolo)")

    progress("labeling", 0.62, "auto-labeling windows (weak supervision + crowd audio)")
    label_windows(windows, rise=rise, clip_t0=t0, audio_usable=audio_usable)
    heuristic_events = merge_events(windows)

    net_used = False
    if not RESNET_WEIGHTS.exists():
        progress("classifier", 0.64, "training ResNet18 event classifier on this video's auto-labels")
        try:
            from app.train_resnet import train
            train(work / "proxy_frames", [w.to_dict() for w in windows], t0,
                  RESNET_WEIGHTS,
                  on_epoch=lambda ep, n, va: progress(
                      "classifier", 0.64 + 0.08 * ep / n,
                      f"ResNet epoch {ep}/{n} (val_acc {va:.2f})"))
        except Exception as re_err:  # noqa: BLE001
            print(f"      ResNet training skipped: {type(re_err).__name__}: {re_err}")
    if RESNET_WEIGHTS.exists():
        progress("classifier", 0.73, "ResNet18 classifying windows")
        try:
            event_net.classify_windows(work / "proxy_frames", windows, t0,
                                       RESNET_WEIGHTS)
            net_used = True
        except Exception as e:  # noqa: BLE001
            print(f"      ResNet inference failed: {type(e).__name__}: {e}")
    _chk()
    # ResNet trained on raw heuristic labels, so its vote is advisory: it may
    # only PROMOTE quiet windows to goal/shot, and only with converging
    # evidence (goal still requires the crowd surge). Gated labels stand.
    if net_used:
        for w in windows:
            if getattr(w, "net_conf", 0.0) < 0.55 or w.label != "pass":
                continue
            roar = audio_evidence(rise, w.mid() - t0) \
                if (rise is not None and audio_usable) else None
            if w.net_label == "goal":
                if w.goal_zone_time >= 0.15 and (roar is None
                                                 or roar >= ROAR_FLOOR):
                    w.label, w.conf = "goal", 0.6 + 0.3 * w.net_conf
                elif (w.ball_speed >= BALL_SPEED_SHOT
                      and roar is not None and roar >= ROAR_STRIKE):
                    w.label, w.conf = "goal", 0.6 + 0.3 * w.net_conf
            elif w.net_label == "shot":
                # The tracker loses the ball on broadcast cuts, so ball_speed
                # alone rejects real chances (the ARG-FRA goals had ResNet
                # shot confidence 0.89-0.99 with ball_speed 0-120).
                strong = w.net_conf >= 0.80
                if (w.ball_speed >= BALL_SPEED_SHOT or strong) and \
                        (roar is None or roar >= ROAR_FLOOR or strong):
                    w.label, w.conf = "shot", max(0.6, 0.6 + 0.2 * w.net_conf)
        enforce_caps(windows, rise, t0)
    events_final = merge_events(windows)

    progress("writing", 0.75, "writing commentary lines")
    teams = {"home": home, "away": away}
    announcer = make_announcer(teams)
    duo = Duo(teams, announcer)
    # match-context layer: rolling score/minute/phase facts + goal-side truth
    from app.match_context import MatchState
    goals_for_state = [{"t": a["t"], "team": a.get("side", "home"),
                        "minute": a.get("minute", int(a["t"] // 60)),
                        "index": i + 1} for i, a in enumerate(anchors)]
    mstate = MatchState(teams, goals=goals_for_state,
                        start_score=start_score)
    segments: list[Segment] = []
    for ev in events_final:
        feats = {
            "motion": ev.motion, "ball_speed": ev.ball_speed,
            "n_players": ev.n_players, "has_ball": ev.has_ball,
            "goal_zone_time": ev.goal_zone_time,
            "ball_x_frac": getattr(ev, "ball_x_frac", 0.5),
            "goal_side": getattr(ev, "goal_side", None),
            "possession_home": getattr(ev, "possession_home", -1.0),
            "possession_away": getattr(ev, "possession_away", -1.0),
            "n_tracks": getattr(ev, "n_tracks", 0.0),
            "minute": ev.t0 / 60.0 + minute_offset, "t0": ev.t0,
            "crowd_rise": audio_evidence(rise, ev.mid() - t0)
            if rise is not None else 1.0,
        }
        mstate.observe(ev.label, ev.t0, feats)
        feats = mstate.context_at(ev.t0, feats)
        segments.extend(duo.event_pair(ev.label, feats, feats["crowd_rise"]))
    # normalize to clip-relative time (filler planner + renderer work in clip space)
    for s in segments:
        s.t0 -= t0
    print(f"      {len(segments)} event lines "
          f"({sum(1 for s in segments if s.event_label == 'goal')} goal, "
          f"{sum(1 for s in segments if s.event_label == 'shot')} shot, "
          f"{sum(1 for s in segments if s.event_label == 'tackle')} tackle)")

    def feats_at(t: float) -> dict:
        mids = [w.mid() - t0 for w in windows]  # windows are absolute; t is clip-relative
        i = bisect.bisect_left(mids, t)
        i = min(max(i, 0), len(windows) - 1)
        w = windows[i]
        raw = {"motion": w.motion, "ball_speed": w.ball_speed,
               "n_players": w.n_players, "has_ball": w.has_ball,
               "goal_zone_time": w.goal_zone_time, "ball_x_frac": 0.5,
               "possession_home": getattr(w, "possession_home", -1.0),
               "possession_away": getattr(w, "possession_away", -1.0),
               "n_tracks": getattr(w, "n_tracks", 0.0)}
        return mstate.context_at(t + t0, raw)

    def transcript_lines(segs: list[Segment]) -> list[str]:
        return [f"[{_fmt_ts(max(0.0, s.t0))}] ({s.event_label or 'ambience'} | "
                f"{s.excitement} | {s.speaker}) {s.text}"
                for s in sorted(segs, key=lambda s: s.t0)]

    progress("writing", 0.78, f"{len(segments)} event lines written")
    if on_transcript:
        on_transcript(transcript_lines(segments))
    progress("voice", 0.80, "planning fillers + synthesizing speech")
    segments = plan_fillers(segments, dur, announcer, feats_at)

    # B-led chat exchanges in long quiet gaps (the duo fills the air, grounded)
    from app.duo import chat_insertions, resolve_overlaps
    for t, minute in chat_insertions(segments, dur, min_gap=CHAT_GAP):
        segments.extend(duo.chat_pair(t, {"minute": minute / 60.0}))
    segments = resolve_overlaps(segments)

    if on_transcript:
        on_transcript(transcript_lines(segments))
    n_total = max(len(segments), 1)
    n_done = 0

    def on_line(seg=None):
        nonlocal n_done
        n_done += 1
        progress("voice", 0.80 + 0.14 * n_done / n_total,
                 f"voicing line {n_done}/{n_total}")
        if on_segment is not None and seg is not None and seg.audio_path:
            on_segment(seg)

    segments = tts_mod.synthesize(segments, out_dir / "tts", on_line=on_line)
    n_ok = sum(1 for s in segments if s.audio_path)
    print(f"      {n_ok}/{len(segments)} segments voiced, "
          f"{sum(s.audio_dur for s in segments):.0f}s of speech")

    _chk()
    progress("render", 0.96, "mixing audio onto video with ffmpeg")
    final = out_dir / "final_commentary.mp4"
    render_mod.render(video, segments, final, t0=t0, duration=dur,
                      orig_audio=orig_audio and has_audio(video))

    # transcript + artifacts
    lines = transcript_lines(segments)
    (out_dir / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "transcript.json").write_text(json.dumps(
        [s.to_dict() for s in sorted(segments, key=lambda s: s.t0)], indent=2),
        encoding="utf-8")
    (out_dir / "analysis.json").write_text(json.dumps(
        {**analysis, "windows": [w.to_dict() for w in windows],
         "net_classifier_used": net_used}, indent=2, default=str), encoding="utf-8")

    elapsed = time.process_time() - t_cpu  # immune to system suspend/resume
    wall = time.time() - t_wall
    rtx = elapsed / dur if dur else 0
    summary = {
        "video": str(video), "output": str(final),
        "clip_t0": t0, "clip_duration_s": round(dur, 1),
        "segments": len(segments), "voiced": n_ok,
        "speech_seconds": round(sum(s.audio_dur for s in segments), 1),
        "process_seconds": round(elapsed, 1),
        "wall_seconds": round(wall, 1),
        "realtime_ratio": round(rtx, 3),
        "faster_than_realtime": rtx < 1.0,
        "net_classifier_used": net_used,
        "detector_weights": str(detector.weights_used),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
