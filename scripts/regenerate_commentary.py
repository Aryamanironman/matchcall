"""Regenerate commentary, TTS and the reel from an existing highlights.json.

The vision pass (action spotting, tracking) is expensive and does not change
when we retrain the commentator, so this reuses the recorded moments and only
re-runs stages 3-5: grounded description, GPT-2 commentary and Fish TTS.

    python scripts/regenerate_commentary.py --clip-len 14

Writes ``highlights_v4.json``, ``transcript_v4.txt`` and
``commented_highlights_v4.mp4`` next to the source highlights file; the old
reel is left untouched.
"""
from __future__ import annotations

import argparse
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

from app import render as render_mod  # noqa: E402
from app import tts as tts_mod  # noqa: E402
from app.commentary import make_announcer  # noqa: E402
from app.config import settings  # noqa: E402
from app.events import Segment  # noqa: E402
from app.vision.sampling import probe_video  # noqa: E402
from scripts.make_commented_highlights import _ffmpeg, _log  # noqa: E402


def _teams(moments: list[dict], home: str | None, away: str | None):
    found = {}
    for m in moments:
        side, team = m.get("side"), m.get("team")
        if side in ("home", "away") and team:
            found.setdefault(side, team)
    return {"home": home or found.get("home", "Home"),
            "away": away or found.get("away", "Away")}


def _features(m: dict, teams: dict, width: float) -> dict:
    hs, as_ = m.get("score") or [0, 0]
    bx = m.get("ball_x")
    frac = (float(bx) / width) if bx is not None else 0.5
    ph = float(m.get("possession_home") or 0.0)
    pa = float(m.get("possession_away") or 0.0)
    poss = ("home" if ph > pa else "away" if pa > ph
            else "home" if frac >= 0.5 else "away")
    return {
        "minute": int(m.get("minute") or 0),
        "score_home": int(hs), "score_away": int(as_),
        "scoreline": f"{teams['home']} {hs}-{as_} {teams['away']}",
        "possession": poss,
        "goal_side": m.get("side") if m.get("kind") == "goal" else None,
        "ball_x_frac": frac,
        "n_players": m.get("n_tracks") or 0,
        "n_tracks": m.get("n_tracks") or 0,
        "has_ball": 1.0 if bx is not None else 0.0,
        "goal_zone_time": 0.0, "motion": 0.0, "ball_speed": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlights",
                    default=str(settings.out_root / "commented_highlights" /
                                "highlights.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--clip-len", type=float, default=14.0)
    ap.add_argument("--width", type=float, default=960.0,
                    help="tracking frame width used for ball_x_frac")
    ap.add_argument("--home", default=None)
    ap.add_argument("--away", default=None)
    ap.add_argument("--orig-vol", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0, help="first N moments only")
    ap.add_argument("--dry-run", action="store_true",
                    help="write commentary only; no TTS or video")
    args = ap.parse_args()

    src = Path(args.highlights)
    data = json.loads(src.read_text(encoding="utf-8"))
    out = Path(args.out_dir) if args.out_dir else src.parent
    out.mkdir(parents=True, exist_ok=True)
    tts_dir = out / "tts_v4"
    tts_dir.mkdir(exist_ok=True)
    moments = data.get("moments") or []
    video = Path(data["video"])
    teams = _teams(moments, args.home, args.away)
    info = probe_video(video)
    settings.original_audio_volume = args.orig_vol
    announcer = make_announcer(teams)
    t_start = time.time()

    parts: list[Path] = []
    records: list[dict] = []
    moments = moments[:args.limit] if args.limit else moments
    for m in moments:
        n = int(m.get("n") or len(records) + 1)
        kind = m.get("kind") or "shot"
        features = _features(m, teams, args.width)
        before = getattr(announcer, "gpt2_lines", 0)
        line, exc = announcer.event_line(kind, features)
        source = "gpt2" if getattr(announcer, "gpt2_lines", 0) > before \
            else "template"
        _log(f"[{n:02d}] {kind} {m.get('minute')}' ({source}, "
             f"{len(line.split())}w): {line}")
        if args.dry_run:
            records.append({**m, "commentary": line, "excitement": exc,
                            "commentary_source": source})
            continue

        seg = Segment(t0=2.0, kind="event", event_label=kind,
                      excitement=exc, text=line, speaker="A")
        tts_mod.synthesize([seg], tts_dir)
        clip = out / f"v4_clip_{n:02d}_{kind}.mp4"
        start = float(m.get("t_clip") or 0.0)
        length = min(args.clip_len, max(1.0, info["duration"] - start))
        if seg.audio_path:
            render_mod.render(video, [seg], clip, t0=start,
                              duration=length, orig_audio=True)
        else:
            _ffmpeg(["-ss", f"{start:.3f}", "-t", f"{length:.3f}",
                     "-i", str(video), "-c", "copy", str(clip)])
        parts.append(clip)

        rec = dict(m)
        rec.update({"commentary": line, "excitement": exc,
                    "clip": str(clip), "commentary_source": source})
        records.append(rec)

    final = out / "commented_highlights_v4.mp4"
    if parts:
        _log(f"concatenating {len(parts)} clips -> {final.name}")
        cmd = [settings.ffmpeg, "-y", "-v", "error"]
        for p in parts:
            cmd += ["-i", str(p)]
        fc = []
        for i in range(len(parts)):
            fc.append(
                f"[{i}:v]fps=25,scale=1280:720:force_original_aspect_ratio=decrease,"
                f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                f"fade=t=in:st=0:d=0.35,"
                f"fade=t=out:st={args.clip_len - 0.35:.2f}:d=0.35[v{i}]")
            fc.append(f"[{i}:a]afade=t=in:st=0:d=0.25,"
                      f"afade=t=out:st={args.clip_len - 0.30:.2f}:d=0.30[a{i}]")
        fc.append("".join(f"[v{i}][a{i}]" for i in range(len(parts))) +
                  f"concat=n={len(parts)}:v=1:a=1[v][a]")
        cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                str(final)]
        subprocess.run(cmd, check=True)

    summary = dict(data)
    summary.update({
        "output": str(final), "moments": records, "clip_len": args.clip_len,
        "clips": len(parts), "reel_seconds": round(len(parts) * args.clip_len, 1),
        "gpt2_dir": str(settings.gpt2_lora_dir),
        "gpt2_lines": getattr(announcer, "gpt2_lines", None),
        "template_lines": getattr(announcer, "fallback_lines", None),
        "teams": teams,
        "wall_seconds": round(time.time() - t_start, 1),
    })
    (out / "highlights_v4.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    (out / "transcript_v4.txt").write_text(
        "\n".join(f"[{r['minute']:>3}'] ({r['kind']}) {r['commentary']}"
                  for r in records) + "\n", encoding="utf-8")
    _log(json.dumps({k: v for k, v in summary.items()
                     if k not in ("moments", "score_anchors")}, indent=2))
    _log(f"done -> {final}")


if __name__ == "__main__":
    main()
