"""Find an API-Football fixture available on the current plan and fetch its events."""
from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
OUT = ROOT.parents[2] / "output" / "experiments" / "api_football_current.json"


def load_env(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


KEY = load_env(ENV).get("API_FOOTBALL_KEY", "")
if not KEY:
    raise SystemExit("no API_FOOTBALL_KEY in .env")
BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": KEY}
ctx = ssl.create_default_context()


def get(path: str, params: dict):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", errors="replace")[:300]}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"}


fixtures = []
for date in ("2026-10-26", "2026-10-27", "2026-10-28"):
    status, data = get("/fixtures", {"date": date})
    print(date, "HTTP", status, "results", data.get("results"), "errors", data.get("errors"))
    fixtures.extend(data.get("response", []))

print("fixtures found:", len(fixtures))
candidates = []
for f in fixtures:
    st = (f.get("fixture", {}).get("status", {}) or {}).get("short", "")
    if st in ("FT", "AET", "PEN", "1H", "2H", "HT", "ET", "LIVE", "INT"):
        candidates.append(f)
print("live/finished:", len(candidates))

picked = None
picked_events = []
for f in candidates[:60]:
    fid = f["fixture"]["id"]
    status, data = get("/fixtures/events", {"fixture": str(fid)})
    events = data.get("response") or []
    if events:
        picked = f
        picked_events = events
        print("picked", fid, f["league"]["name"],
              f["teams"]["home"]["name"], "vs", f["teams"]["away"]["name"],
              f["fixture"]["status"]["short"], "events", len(events))
        break

if not picked:
    # fall back to the first candidate, even without events
    picked = candidates[0] if candidates else (fixtures[0] if fixtures else None)
    print("no event-bearing candidate; picked", picked)

if picked:
    print("fixture", picked["fixture"]["id"],
          picked["teams"]["home"]["name"], "vs", picked["teams"]["away"]["name"],
          picked["fixture"]["date"], picked["fixture"]["status"]["short"])
    for ev in picked_events[:30]:
        tm = ev.get("time", {}) or {}
        print(f"  {tm.get('elapsed')}' {ev.get('team',{}).get('name')} "
              f"{ev.get('type')}/{ev.get('detail')} {ev.get('player',{}).get('name')}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"fixtures": fixtures, "picked": picked,
                           "events": picked_events}, indent=2), encoding="utf-8")
print("wrote", OUT)
