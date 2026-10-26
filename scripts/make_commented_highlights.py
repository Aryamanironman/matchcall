"""Commented highlights: the five-stage pipeline from the design diagram.

  1. action spotting      score-bug OCR (gold goals) + YOLO/ResNet window scan
  2. visual enrichment    10-second clips re-analysed with FootballVision
                          (yolov5su detector + ByteTrack + team possession)
  3. action description   grounded facts from tracking + score state
  4. textual commentary   fine-tuned GPT-2 LoRA, validated, template fallback
  5. audio commentary     Fish Audio s2.1-pro-free when keyed, edge-tts fallback

Output: a 10-second-per-moment commented reel, an annotated tracking clip,
and JSON/transcript artifacts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_V4_DIR = ROOT / "models" / "gpt2_lora_v4"
os.environ.setdefault(
    "GPT2_LORA_DIR",
    str(_V4_DIR if _V4_DIR.exists() else ROOT / "models" / "gpt2_lora_v3"))

import cv2
import numpy as np

from app.config import settings
from app.events import Segment, Window
from app.match_context import MatchState
from app.vision.audio import audio_rms_profile, crowd_rise
from app.vision.auto_label import label_windows
from app.vision.detector import FootballDetector
from app.vision.fv_tracker import FootballVisionTracker
from app.vision.sampling import probe_video
from app.vision.score_flip import detect_score_flips
from app import render as render_mod
from app import tts as tts_mod
from app.commentary import make_announcer


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ffmpeg(args: list[str]) -> None:
    subprocess.run([settings.ffmpeg, "-y", "-v", "error", *args], check=True)


def _extract_clip_frames(video: Path, start: float, length: float,
                         out_dir: Path, fps: float, width: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.jpg"):
        p.unlink()
    _ffmpeg(["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(video),
             "-vf", f"fps={fps},scale={width}:-2", "-q:v", "4",
             str(out_dir / "%04d.jpg")])
    return sorted(out_dir.glob("*.jpg"))


def _load_analysis(video: Path, out: Path, reuse: bool) -> tuple[dict, str]:
    candidates = [out / "analysis.json",
                  Path(r"C:\Users\asus\football-commentator\output_match_full\analysis.json")]
    if reuse:
        for c in candidates:
            if c.exists():
                return json.loads(c.read_text(encoding="utf-8")), str(c)
    _log("stage 1: scanning the match with the project YOLO detector")
    work = out / "work" / "action_spot"
    from app.vision.analyze import analyze_video
    analysis = analyze_video(video, work, FootballDetector())
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str),
                                       encoding="utf-8")
    return analysis, str(out / "analysis.json")


def _score_anchors(video: Path, out: Path, t0: float, dur: float,
                   start_score: dict, minute_offset: float,
                   reuse: bool) -> list[dict]:
    cache = out / "score_anchors.json"
    if reuse and cache.exists():
        anchors = json.loads(cache.read_text(encoding="utf-8"))
    else:
        _log("stage 1: reading the broadcast score bug (glyph spikes + OCR)")
        anchors = [a.to_dict() for a in detect_score_flips(
            video, t0=t0, duration=dur, start_score=start_score,
            minute_offset=minute_offset)]
        cache.write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    import app.config as _cfg
    _cfg.SCORE_ANCHORS[:] = anchors
    return anchors


def _select_moments(windows: list[Window], anchors: list[dict],
                    max_clips: int, minute_offset: float,
                    max_t: float | None = None) -> list[dict]:
    picked: list[dict] = []

    def far(t: float, gap: float = 40.0) -> bool:
        # Non-goal moments need a wide berth around goals: the ResNet fires
        # on celebration cuts 40-60 s after the score, which are not new
        # highlights and would duplicate the goal clip.
        if any(p["kind"] == "goal" for p in picked):
            gap = max(gap, 90.0)
        return all(abs(t - p["t"]) >= gap for p in picked)

    for a in sorted(anchors, key=lambda x: x["t"]):
        picked.append({"kind": "goal", "t": float(a["t"]),
                       "side": a.get("side"),
                       "minute": a.get("minute"),
                       "score_after": a.get("score_after")})

    def add_from(cands, kind: str, score_fn, limit: int, floor: float = 0.0):
        n = 0
        for w in sorted(cands, key=score_fn, reverse=True):
            if n >= limit:
                break
            if score_fn(w) < floor:
                continue
            t = w.mid()
            if max_t is not None and t > max_t:
                continue
            if not far(t):
                continue
            picked.append({"kind": kind, "t": t, "side": None,
                           "minute": int(t / 60.0 + minute_offset),
                           "score_after": None})
            n += 1

    # 4 shots, 2 tackles, 2 corners; goals were already added
    add_from(windows, "shot",
             lambda w: float(w.cls_probs.get("shot", 0.0)), 4, 0.45)
    add_from(windows, "tackle",
             lambda w: float(w.cls_probs.get("tackle", 0.0)) * 0.7 +
             float(w.motion) / 100.0, 2, 0.45)
    add_from([w for w in windows if getattr(w, "corner_time", 0.0) > 0.0],
             "corner", lambda w: float(getattr(w, "corner_time", 0.0)), 2, 0.25)

    # fill any shortfall with the strongest remaining shot candidates
    if len(picked) < max_clips:
        add_from(windows, "shot",
                 lambda w: float(w.cls_probs.get("shot", 0.0)),
                 max_clips - len(picked), 0.20)
    picked.sort(key=lambda p: p["t"])
    return picked[:max_clips]


def _frame_possession(row: dict, width: int) -> str | None:
    """Nearest tracked player to the ball, with a frame-scaled radius.

    FootballVision's 70 px bottom-corner rule is tuned for 1920 px frames; at
    960 px it almost never matches in open play (one clip sampled only 5 of 40
    frames). Scale the radius with the frame, then fall back to the tracker's
    own label.
    """
    ball = row.get("ball")
    tracked = row.get("players_tracked") or {}
    if ball is None or not tracked:
        return row.get("possession_side")
    bx, by = ball[0], ball[1]
    best = None
    for bbox, _team, side in tracked.values():
        d1 = ((bbox[0] - bx) ** 2 + (bbox[3] - by) ** 2) ** 0.5
        d2 = ((bbox[2] - bx) ** 2 + (bbox[3] - by) ** 2) ** 0.5
        dist = min(d1, d2)
        if best is None or dist < best[0]:
            best = (dist, side)
    if best is not None and best[0] <= max(70.0, 0.26 * max(1, width)):
        return best[1]
    return row.get("possession_side")


def _track_clip(video: Path, start: float, length: float, work: Path,
                tracker: FootballVisionTracker, fps: float = 4.0,
                width: int = 960) -> dict:
    frames_dir = work / f"clip_{int(start)}"
    paths = _extract_clip_frames(video, start, length, frames_dir, fps, width)
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [im for im in imgs if im is not None]
    rows = tracker.infer(imgs) if imgs else []
    frame_w = imgs[0].shape[1] if imgs else width
    poss = [p for p in (_frame_possession(r, frame_w) for r in rows) if p]
    n_tracks = [len(r.get("players_tracked", {})) for r in rows]
    home_n = sum(1 for p in poss if p == "home")
    away_n = sum(1 for p in poss if p == "away")
    return {
        "rows": rows,
        "images": imgs,
        "fps": fps,
        "poss_home": home_n / max(1, len(poss)),
        "poss_away": away_n / max(1, len(poss)),
        "n_tracks": float(np.mean(n_tracks)) if n_tracks else 0.0,
        "frames": len(rows),
    }


def _ball_at(rows: list[dict], idx: int) -> tuple[float | None, float | None]:
    for j in range(max(0, idx), min(len(rows), idx + 6)):
        b = rows[j].get("ball")
        if b is not None:
            return b[0], b[1]
    for j in range(min(len(rows) - 1, idx), max(-1, idx - 6), -1):
        b = rows[j].get("ball")
        if b is not None:
            return b[0], b[1]
    return None, None


def _describe(kind: str, team: str, minute: int, score: tuple[int, int],
              stats: dict, ball_x: float | None, ball_y: float | None,
              width: int) -> str:
    poss_team = "the home side"
    ph, pa = stats["poss_home"], stats["poss_away"]
    poss_pct = int(round(100 * max(ph, pa))) if (ph + pa) > 0 else 50
    zone = "middle third"
    if ball_x is not None and width:
        fx = ball_x / width
        zone = ("left channel" if fx < 0.34 else
                "right channel" if fx > 0.66 else "central area")
        if ball_y is not None and ball_y > 0.62:
            zone += ", final third" if fx < 0.25 or fx > 0.75 else ", high"
    return (f"{kind} by {team} at {minute}'. Score {score[0]}-{score[1]}. "
            f"Ball in the {zone}. {stats['n_tracks']:.0f} tracked players. "
            f"Possession {poss_pct}% to one side over the clip.")


def _annotate(rows: list[dict], imgs: list[np.ndarray], out_path: Path,
              fps: float, home: str, away: str,
              weights_label: str = "") -> None:
    if not imgs:
        return
    h, w = imgs[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    home_col, away_col = (255, 255, 255), (255, 120, 0)  # BGR
    sides = [_frame_possession(r, w) for r in rows]
    poss = [p for p in sides if p]
    home_n = sum(1 for p in poss if p == "home")
    away_n = sum(1 for p in poss if p == "away")
    total_poss = max(1, home_n + away_n)
    home_pct = 100.0 * home_n / total_poss
    away_pct = 100.0 - home_pct
    trail: list[tuple[int, int]] = []
    for i, (row, img) in enumerate(zip(rows, imgs)):
        frame = img.copy()
        for tid, (bbox, team, side) in row.get("players_tracked", {}).items():
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = home_col if side == "home" else away_col
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{tid}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        b = row.get("ball")
        if b is not None:
            cx, cy = int(b[0]), int(b[1])
            interp = bool(row.get("ball_interpolated"))
            trail.append((cx, cy))
            trail = trail[-14:]
            for k in range(1, len(trail)):
                cv2.line(frame, trail[k - 1], trail[k], (0, 255, 0), 1,
                         cv2.LINE_AA)
            if interp:
                # predicted position from the bounded interpolation: ring
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                label = "ball (interp)"
            else:
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                label = "ball"
            cv2.putText(frame, label, (cx + 9, cy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1,
                        cv2.LINE_AA)
        if weights_label:
            cv2.putText(frame, weights_label, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (201, 255, 79), 2,
                        cv2.LINE_AA)
        cv2.putText(frame, "ball: dot = detected · ring = interpolated",
                    (10, 46 if weights_label else 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 240, 225), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, f"{home} (white)  vs  {away} (blue)",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # possession: clip aggregate bar + current possession
        bar_x, bar_y, bar_w, bar_h = 10, h - 46, w - 20, 6
        split = bar_x + int(bar_w * home_pct / 100.0)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (min(max(split, bar_x), bar_x + bar_w), bar_y + bar_h),
                      home_col, -1)
        cv2.rectangle(frame, (split, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      away_col, -1)
        cv2.putText(frame, f"{home} {home_pct:.0f}%", (bar_x, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, home_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{away} {away_pct:.0f}%",
                    (bar_x + bar_w - 100, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1,
                    cv2.LINE_AA)
        side = sides[i] if i < len(sides) else row.get("possession_side")
        if side:
            owner = home if side == "home" else away
            cv2.putText(frame, f"possession: {owner}", (bar_x, bar_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1,
                        cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"C:\Users\asus\Downloads\3 - 3 ARG vs FRA - Second-Half - Bein.mp4")
    ap.add_argument("--home", default="Argentina")
    ap.add_argument("--away", default="France")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", default=str(settings.out_root / "commented_highlights"))
    ap.add_argument("--clip-len", type=float, default=10.0)
    ap.add_argument("--pre", type=float, default=4.0)
    ap.add_argument("--max-clips", type=int, default=10)
    ap.add_argument("--start-score-home", type=int, default=2)
    ap.add_argument("--start-score-away", type=int, default=0)
    ap.add_argument("--minute-offset", type=float, default=35.33,
                    help="official match-minute at video t0 (ARG-FRA: clock "
                         "34:38, official ~35:20)")
    ap.add_argument("--orig-vol", type=float, default=0.35)
    ap.add_argument("--reuse-analysis", action="store_true", default=True)
    ap.add_argument("--reuse-score", action="store_true", default=True)
    ap.add_argument("--annotate", action="store_true", default=True)
    ap.add_argument("--tracker-weights", default=None,
                    help="detector used for clip tracking (default: Aryaman "
                         "YOLOv5lu if present); pass the YOLO26 best.pt to "
                         "rerun the pipeline with the new detector")
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    settings.original_audio_volume = args.orig_vol
    t_start = time.time()
    info = probe_video(video)
    t0 = args.t0
    dur = args.duration or max(0.0, info["duration"] - t0)
    _log(f"video {video.name}: {dur:.0f}s, {info['width']}x{info['height']}")

    # ---- stage 1: action spotting ------------------------------------------
    anchors = _score_anchors(video, out, t0, dur,
                             {"home": args.start_score_home,
                              "away": args.start_score_away},
                             args.minute_offset, args.reuse_score)
    _log(f"score anchors: {[(round(a['t'], 1), a['side'], a.get('minute')) for a in anchors]}")
    analysis, analysis_path = _load_analysis(video, out, args.reuse_analysis)
    windows = [Window(**wd) for wd in analysis["windows"]]
    rms = audio_rms_profile(video, t0=0.0, duration=None)
    rise = crowd_rise(rms) if len(rms) else None
    audio_usable = len(rms) >= 30 and float(np.median(rms)) > 0.0
    label_windows(windows, rise=rise, clip_t0=0.0, audio_usable=audio_usable)
    moments = _select_moments(windows, anchors, args.max_clips,
                              args.minute_offset,
                              max_t=max(0.0, info["duration"] - 6.0))
    _log(f"selected {len(moments)} moments: "
         f"{[(m['kind'], round(m['t'], 1)) for m in moments]}")

    # ---- stages 2-5: per-moment enrichment, commentary, voice, render -------
    tracker = FootballVisionTracker(
        weights=Path(args.tracker_weights) if args.tracker_weights else None)
    teams = {"home": args.home, "away": args.away}
    announcer = make_announcer(teams)
    tts_dir = out / "tts"
    tts_dir.mkdir(exist_ok=True)
    parts: list[Path] = []
    records: list[dict] = []
    running = {"home": args.start_score_home, "away": args.start_score_away}
    annotated_done = False

    for n, m in enumerate(moments, 1):
        kind = m["kind"]
        center = m["t"]
        start = max(0.0, center - args.pre)
        length = min(args.clip_len, max(1.0, info["duration"] - start))
        _log(f"[{n}/{len(moments)}] {kind} @ {center:.1f}s "
             f"(clip {start:.1f}+{length:.1f})")
        stats = _track_clip(video, start, length, work, tracker,
                            fps=4.0, width=960)
        event_idx = int(round((center - start) * stats["fps"]))
        event_idx = min(max(0, event_idx), max(0, len(stats["rows"]) - 1))
        bx, by = _ball_at(stats["rows"], event_idx)
        width = stats["images"][0].shape[1] if stats["images"] else 960

        # side: goals from the score bug; otherwise the tracked possession side
        if kind == "goal":
            side = m.get("side") or (
                "home" if stats["poss_home"] >= stats["poss_away"] else "away")
            before = dict(running)
            running[side] = running.get(side, 0) + 1
            score = (running["home"], running["away"])
        else:
            side = ("home" if stats["poss_home"] > stats["poss_away"]
                    else "away" if stats["poss_away"] > stats["poss_home"]
                    else None)
            if side is None:
                side = ("home" if (bx / width if bx is not None else 0.5) >= 0.5
                        else "away")
            score = (running["home"], running["away"])
        team = teams.get(side, args.home)
        minute = m.get("minute") or int(center / 60.0 + args.minute_offset)
        scoreline = f"{args.home} {score[0]}-{score[1]} {args.away}"
        poss = ("home" if stats["poss_home"] > stats["poss_away"]
                else "away" if stats["poss_away"] > stats["poss_home"]
                else "even")
        features = {
            "minute": minute, "score_home": score[0], "score_away": score[1],
            "scoreline": scoreline, "possession": poss,
            "goal_side": side if kind == "goal" else None,
            "ball_x_frac": (bx / width) if bx is not None else 0.5,
            "n_players": stats["n_tracks"], "n_tracks": stats["n_tracks"],
            "has_ball": 1.0 if bx is not None else 0.0,
            "goal_zone_time": 0.0, "motion": 0.0, "ball_speed": 0.0,
        }
        desc = _describe(kind, team, minute, score, stats, bx, by, width)
        try:
            line, exc = announcer.event_line(kind, features)
        except Exception as e:  # noqa: BLE001
            _log(f"  commentary failed ({type(e).__name__}: {e}) — template")
            from app.commentary import TemplateAnnouncer
            line, exc = TemplateAnnouncer(teams).event_line(kind, features)
        _log(f"  line ({exc}): {line}")

        seg = Segment(t0=2.0, kind="event", event_label=kind,
                      excitement=exc, text=line, speaker="A")
        tts_mod.synthesize([seg], tts_dir)
        clip_path = out / f"clip_{n:02d}_{kind}.mp4"
        if seg.audio_path:
            render_mod.render(video, [seg], clip_path, t0=start,
                              duration=length, orig_audio=True)
        else:
            _ffmpeg(["-ss", f"{start:.3f}", "-t", f"{length:.3f}",
                     "-i", str(video), "-c", "copy", str(clip_path)])
        parts.append(clip_path)

        # annotated tracking artefact for the first goal (pipeline showcase)
        if args.annotate and not annotated_done and kind == "goal" and stats["rows"]:
            ann_raw = out / "annotated_goal_raw.mp4"
            _annotate(stats["rows"], stats["images"], ann_raw,
                      stats["fps"], args.home, args.away)
            ann_final = out / "annotated_goal.mp4"
            if seg.audio_path:
                render_mod.render(ann_raw, [Segment(t0=2.0, kind="event",
                                                    event_label=kind,
                                                    excitement=exc, text=line,
                                                    audio_path=seg.audio_path,
                                                    audio_dur=seg.audio_dur,
                                                    speaker="A")],
                                  ann_final, t0=0.0, duration=length,
                                  orig_audio=False)
            annotated_done = True

        records.append({
            "n": n, "kind": kind, "t_video": round(center, 2),
            "t_clip": round(start, 2), "minute": minute,
            "team": team, "side": side, "score": list(score),
            "possession_home": round(stats["poss_home"], 3),
            "possession_away": round(stats["poss_away"], 3),
            "n_tracks": round(stats["n_tracks"], 1),
            "ball_x": round(bx, 1) if bx is not None else None,
            "ball_y": round(by, 1) if by is not None else None,
            "description": desc, "commentary": line, "excitement": exc,
            "clip": str(clip_path),
        })

    # ---- concat with fades --------------------------------------------------
    final = out / "commented_highlights.mp4"
    if parts:
        _log(f"concatenating {len(parts)} clips -> {final.name}")
        cmd = [settings.ffmpeg, "-y", "-v", "error"]
        for p in parts:
            cmd += ["-i", str(p)]
        fc = []
        for i in range(len(parts)):
            fc.append(f"[{i}:v]fps=25,scale=1280:720:force_original_aspect_ratio=decrease,"
                      f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                      f"fade=t=in:st=0:d=0.35,fade=t=out:st={args.clip_len - 0.35:.2f}:d=0.35[v{i}]")
            fc.append(f"[{i}:a]afade=t=in:st=0:d=0.25,"
                      f"afade=t=out:st={args.clip_len - 0.30:.2f}:d=0.30[a{i}]")
        fc.append("".join(f"[v{i}][a{i}]" for i in range(len(parts))) +
                  f"concat=n={len(parts)}:v=1:a=1[v][a]")
        cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                str(final)]
        subprocess.run(cmd, check=True)

    summary = {
        "video": str(video), "output": str(final),
        "analysis_cache": analysis_path,
        "score_anchors": anchors,
        "moments": records,
        "clip_len": args.clip_len,
        "clips": len(parts),
        "reel_seconds": round(len(parts) * args.clip_len, 1),
        "fish_model": os.getenv("FISH_MODEL", "s2.1-pro-free"),
        "fish_keyed": bool(os.getenv("FISH_API_KEY", "").strip()
                           and os.getenv("FISH_API_KEY") != "***"),
        "gpt2_dir": str(settings.gpt2_lora_dir),
        "gpt2_lines": getattr(announcer, "gpt2_lines", None),
        "template_lines": getattr(announcer, "fallback_lines", None),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "highlights.json").write_text(json.dumps(summary, indent=2),
                                         encoding="utf-8")
    (out / "transcript.txt").write_text(
        "\n".join(f"[{r['minute']:>3}'] ({r['kind']}) {r['commentary']}"
                  for r in records) + "\n", encoding="utf-8")
    _log(json.dumps({k: v for k, v in summary.items()
                     if k not in ("moments", "score_anchors")}, indent=2))
    _log(f"done -> {final}")


if __name__ == "__main__":
    main()
