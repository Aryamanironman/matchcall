"""Retest goal generation after case-variant blocking and goal prefill."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from app.commentary import Gpt2Announcer

RESULTS = []


def test(name, teams, features):
    ann = Gpt2Announcer(teams)
    a, b = ann.gpt2_lines, ann.fallback_lines
    line, exc = ann.event_line("goal", features)
    source = "gpt2" if ann.gpt2_lines > a else "template"
    RESULTS.append({"case": name, "source": source, "line": line,
                    "excitement": exc})
    print(f"[{source}] {name}: {line}")


test(
    "api_alajuelense_90", {"home": "LD Alajuelense", "away": "Perez Zeledon"},
    {"goal_side": "home", "score_home": 1, "score_away": 0,
     "scoreline": "LD Alajuelense 1–0 Perez Zeledon", "minute": 90,
     "player": "A. Hernandez",
     "ball_x_frac": 0.5, "possession_home": 0.5, "possession_away": 0.5,
     "n_tracks": 8.0, "crowd_rise": 1.6},
)
test(
    "arg_fra_80", {"home": "Argentina", "away": "France"},
    {"goal_side": "away", "score_home": 2, "score_away": 1,
     "scoreline": "Argentina 2–1 France", "minute": 80,
     "ball_x_frac": 0.8, "possession_home": 0.2, "possession_away": 0.8,
     "n_tracks": 8.0, "crowd_rise": 1.6},
)
test(
    "arg_fra_81", {"home": "Argentina", "away": "France"},
    {"goal_side": "away", "score_home": 2, "score_away": 2,
     "scoreline": "Argentina 2–2 France", "minute": 81,
     "ball_x_frac": 0.8, "possession_home": 0.2, "possession_away": 0.8,
     "n_tracks": 8.0, "crowd_rise": 1.6},
)

out = REPO / "output" / "experiments" / "gpt2_goal_retest.json"
out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
print("wrote", out)
