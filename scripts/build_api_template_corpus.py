"""Build an API-style slot corpus from SoccerReplay-1988 raw records.

Input examples look like:

  [HISTORY] <HOME> vs <AWAY>, score <SCORE>, <MINUTE> minute.
  [EVENT] goal by <TEAM>, scorer <PLAYER>
  [INTENSITY] huge [COMMENTARY] GOAL! <PLAYER> (<TEAM>) ...

The model learns the slots, not the club/player names. At inference the API
facts are substituted into the slots, so unseen teams/players cannot drift.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVENT_MAP = {
    "goal": "goal",
    "shot off target": "shot",
    "saved by goal-keeper": "save",
    "corner": "corner",
    "lead to corner": "corner",
    "clearance": "tackle",
    "foul with no card": "tackle",
    "free kick": "tackle",
    "throw in": "tackle",
    "ball out of play": "dead_time",
    "ball possession": "pass",
    "start of half game": "dead_time",
    "end of half game": "dead_time",
    "show added time": "dead_time",
}
BLOCKED = ("penalty", "red card", "yellow card", "offside", "off side",
           "var", "substitution", "injury", "disallowed", "own goal")
INTENSITY = {"goal": "huge", "shot": "high", "save": "high",
             "corner": "medium", "tackle": "medium", "pass": "low",
             "dead_time": "calm"}
TIME_RE = re.compile(r"^(\d{1,3}):(\d{2})$")
SCORE_RE = re.compile(r"(\d{1,2})\s*[:–-]\s*(\d{1,2})")


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def event_side(text: str, home: str, away: str) -> str | None:
    for side, team in (("home", home), ("away", away)):
        if re.search(r"\(\s*%s\s*\)" % re.escape(team), text, re.I):
            return side
    for side, team in (("home", home), ("away", away)):
        if re.search(r"\b%s\b" % re.escape(team), text, re.I):
            return side
    return None


def convert_match(match: dict, rng: random.Random) -> list[str]:
    info = match.get("match_info") or {}
    home, away = info.get("home_team"), info.get("away_team")
    if not home or not away:
        return []
    comments = [c for c in match.get("comments") or [] if c.get("time_stamp")]
    comments.sort(key=lambda c: (int(c.get("half") or 0), str(c["time_stamp"])))
    score = [0, 0]
    records = []
    for c in comments:
        kind = EVENT_MAP.get(str(c.get("comments_type", "")).lower())
        if not kind:
            continue
        raw = (c.get("comments_text") or "").strip()
        anon = (c.get("comments_text_anonymized") or "").strip()
        if not raw or not anon:
            continue
        low = raw.lower()
        if any(b in low for b in BLOCKED):
            continue
        m = TIME_RE.match(str(c["time_stamp"]))
        if not m:
            continue
        minute = int(m.group(1)) + (45 if int(c.get("half") or 0) == 2 else 0)
        if minute > 95:
            continue
        side = event_side(raw, home, away)
        if side is None:
            continue
        # Preserve the template slots; do not fill them with real names.
        text = (anon.replace("[PLAYER]", "<PLAYER>")
                    .replace("[TEAM]", "<TEAM>")
                    .replace("[REFEREE]", "the referee")
                    .replace("[COACH]", "the manager"))
        if "<TEAM>" not in text:
            continue
        text = re.sub(r"\s*\(\s*%s\s*\)" % re.escape(home), "", text, flags=re.I)
        text = re.sub(r"\s*\(\s*%s\s*\)" % re.escape(away), "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        if kind == "goal":
            sm = SCORE_RE.search(raw)
            if sm:
                score = [int(sm.group(1)), int(sm.group(2))]
            else:
                score[0 if side == "home" else 1] += 1
        digits = [d for d in re.findall(r"\d+", text)
                  if d not in {str(score[0]), str(score[1]),
                               str(minute), str(minute + 1)}]
        if digits:
            continue
        words = text.split()
        if not (8 <= len(words) <= 70):
            continue
        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        records.append({"kind": kind, "side": side, "minute": minute,
                        "sec": minute * 60 + int(m.group(2)),
                        "text": text, "score": list(score),
                        "has_player": "<PLAYER>" in text})

    rows = []
    for i, rec in enumerate(records):
        text = rec["text"]
        if i + 1 < len(records) and rng.random() < 0.4:
            nxt = records[i + 1]
            gap = nxt["sec"] - rec["sec"]
            joined = f"{text} {nxt['text']}"
            if (0 <= gap <= 120 and "<TEAM>" in joined
                    and len(joined.split()) <= 70):
                text = joined
        if "<TEAM>" not in text:
            continue
        player = ", scorer <PLAYER>" if "<PLAYER>" in text else ""
        history = (f"<HOME> vs <AWAY>, score <SCORE>, <MINUTE> minute.")
        rows.append(
            f"[HISTORY] {history} [EVENT] {rec['kind']} by <TEAM>{player} "
            f"[INTENSITY] {INTENSITY[rec['kind']]} [COMMENTARY] {text}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "soccerreplay"))
    ap.add_argument("--out", default=str(ROOT / "data" / "commentator_train_v5.jsonl"))
    ap.add_argument("--valid-out", default=str(ROOT / "data" / "commentator_valid_v5.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = Path(args.raw)
    train_rows, valid_rows = [], []
    for split, dest in (("train", train_rows), ("valid", valid_rows)):
        files = sorted((raw / split).glob("*.json"))
        print(split, len(files), "files", flush=True)
        for i, path in enumerate(files):
            try:
                match = json.loads(path.read_text(encoding="utf-8"))
                dest.extend(convert_match(match, random.Random(args.seed * 100003 + i)))
            except Exception as e:  # noqa: BLE001
                print("skip", path.name, type(e).__name__, flush=True)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)} -> {len(dest)}", flush=True)

    rng.shuffle(train_rows)
    Path(args.out).write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in train_rows)
        + "\n", encoding="utf-8")
    Path(args.valid_out).write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in valid_rows)
        + "\n", encoding="utf-8")
    print("train", len(train_rows), "->", args.out)
    print("valid", len(valid_rows), "->", args.valid_out)
    for line in train_rows[:3]:
        print(" ", line[:260])


if __name__ == "__main__":
    main()
