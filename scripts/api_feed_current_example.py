"""Run a real API-Football event feed through the commentary + TTS stack."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from app.commentary import Gpt2Announcer
from app.events import Segment
from app.tts import synthesize
OUT = REPO / "output" / "experiments"
DATA = OUT / "api_football_current.json"
JOB = OUT / "full_match_job.json"
TTS_DIR = OUT / "api_feed_tts"

d = json.loads(DATA.read_text(encoding="utf-8"))
fixture = d["picked"]
events = d["events"]
home = fixture["teams"]["home"]["name"]
away = fixture["teams"]["away"]["name"]
hg = fixture["goals"]["home"]
ag = fixture["goals"]["away"]

# 85:00-90:00 window: 89' card, 90' goal, 90' card.
window = [e for e in events if 85 <= int(e.get("time", {}).get("elapsed") or 0) <= 90]
print("window events:", [(e["time"]["elapsed"], e["team"]["name"], e["type"], e["detail"])
                         for e in window])

goal = next(e for e in window if e["type"] == "Goal" and "Disallowed" not in e["detail"])
team = goal["team"]["name"]
side = "home" if team == home else "away"
minute = int(goal["time"]["elapsed"])
score_before_home = hg - (1 if side == "home" else 0)
score_before_away = ag - (1 if side == "away" else 0)
score_home, score_away = hg, ag
features = {
    "goal_side": side,
    "score_home": score_home,
    "score_away": score_away,
    "scoreline": f"{home} {score_home}–{score_away} {away}",
    "minute": minute,
    "ball_x_frac": 0.5,
    "possession_home": 0.5,
    "possession_away": 0.5,
    "n_tracks": 8.0,
    "crowd_rise": 1.6,
    "player": goal.get("player", {}).get("name"),
}

announcer = Gpt2Announcer({"home": home, "away": away})
a, b = announcer.gpt2_lines, announcer.fallback_lines
t0 = time.perf_counter()
line, exc = announcer.event_line("goal", features)
gen_s = time.perf_counter() - t0
source = "gpt2" if announcer.gpt2_lines > a else "template"

seg = Segment(t0=0.0, kind="event", event_label="goal", excitement=exc,
              text=line, speaker="A")
t0 = time.perf_counter()
voiced = synthesize([seg], TTS_DIR)
tts_s = time.perf_counter() - t0
audio_s = voiced[0].audio_dur

# Pre-recorded ARG-FRA lines from 41:00-46:00, for the system comparison.
job = json.loads(JOB.read_text(encoding="utf-8-sig"))
LINE = re.compile(r"^\[(\d+):(\d+)\]\s+\(([^|]+)\|")
pre = []
for text in job.get("transcript", []):
    m = LINE.match(text)
    if not m:
        continue
    clip = int(m.group(1)) + int(m.group(2)) / 60.0
    if 41.0 <= clip <= 46.0:
        pre.append(text)

out = {
    "match": f"{home} {hg}-{ag} {away}",
    "fixture_id": fixture["fixture"]["id"],
    "date": fixture["fixture"]["date"],
    "window": "85:00-90:00",
    "api_window_events": [
        {"minute": e["time"]["elapsed"], "team": e["team"]["name"],
         "type": e["type"], "detail": e["detail"],
         "player": (e.get("player") or {}).get("name")}
        for e in window
    ],
    "api_goal_event": {
        "minute": minute, "team": team, "player": goal.get("player", {}).get("name"),
        "score_before": f"{score_before_home}-{score_before_away}",
        "score_after": f"{hg}-{ag}",
    },
    "api_commentary": {
        "line": line, "source": source, "excitement": exc,
        "generation_s": round(gen_s, 2), "tts_s": round(tts_s, 2),
        "audio_s": round(audio_s, 2), "audio_path": voiced[0].audio_path,
    },
    "skipped_window_events": [
        {"minute": e["time"]["elapsed"], "type": e["type"], "detail": e["detail"],
         "reason": "no card label in the current commentary template set"}
        for e in window if e is not goal
    ],
    "pre_recorded_arg_fra_5min": pre,
    "note": ("Real API-Football event feed for Alajuelense-Perez Zeledon; the "
             "pre-recorded segment is ARG-FRA and is a system-level comparison, "
             "not the same match."),
}
(OUT / "api_feed_current_example.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
