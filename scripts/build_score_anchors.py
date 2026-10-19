"""Precompute score-bug goal anchors for the full match (cached JSON)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.vision.score_flip import detect_score_flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"C:\Users\asus\Downloads\3 - 3 ARG vs FRA - Second-Half - Bein.mp4")
    ap.add_argument("--out", default=r"C:\Users\asus\football-commentator\output\commented_highlights\score_anchors.json")
    ap.add_argument("--start-home", type=int, default=2)
    ap.add_argument("--start-away", type=int, default=0)
    ap.add_argument("--minute-offset", type=float, default=34.63)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["FC_SCORE_OCR_GPU"] = "1" if args.gpu else "0"
    anchors = [a.to_dict() for a in detect_score_flips(
        args.video,
        start_score={"home": args.start_home, "away": args.start_away},
        minute_offset=args.minute_offset)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    print("anchors:", anchors)
    print("wrote", out)


if __name__ == "__main__":
    main()
