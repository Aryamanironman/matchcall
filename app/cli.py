"""CLI: python -m app.cli --video MATCH.mp4 --home Argentina --away France"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="AI football commentary dubber")
    p.add_argument("--video", required=True)
    p.add_argument("--home", default="Home")
    p.add_argument("--away", default="Away")
    p.add_argument("--t0", type=float, default=0.0, help="start offset in the video (s)")
    p.add_argument("--duration", type=float, default=None, help="clip length (s)")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--no-orig-audio", action="store_true",
                   help="drop the original broadcast audio bed")
    p.add_argument("--score-home", type=int, default=0,
                   help="home score already on the board at t0")
    p.add_argument("--score-away", type=int, default=0,
                   help="away score already on the board at t0")
    p.add_argument("--minute-offset", type=float, default=0.0,
                   help="broadcast match-minute at video t0 "
                        "(e.g. 34.6 for the ARG-FRA second-half file)")
    a = p.parse_args()
    run_pipeline(
        video=Path(a.video), home=a.home, away=a.away, t0=a.t0,
        duration=a.duration, out_dir=Path(a.out) if a.out else None,
        orig_audio=not a.no_orig_audio,
        start_score={"home": a.score_home, "away": a.score_away},
        minute_offset=a.minute_offset,
    )


if __name__ == "__main__":
    main()
