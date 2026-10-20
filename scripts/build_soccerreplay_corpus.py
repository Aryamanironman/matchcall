"""Build a longer, grounded commentary corpus from SoccerReplay-1988.

Homie0609/SoccerReplay-1988 is a gated dataset of 1,988 European matches with
per-event commentary (``comments``), rosters (``players_data``) and match info.
Its text is licensed CC-BY-NC-SA and is used here locally for training only.

The runtime pipeline only observes team-level facts, so this converter:

* keeps only in-play comments with a timestamp,
* maps the dataset's 24 event types onto the pipeline's seven,
* replaces player names with role nouns ("the forward") and drops lines whose
  facts (cards, penalties, VAR, offside) the vision pipeline cannot ground,
* attributes the event to a team from the roster or the ``(Team)`` tag,
* reconstructs the running score from goal comments and the minute from the
  timestamp, then writes the same markup the adapter was trained on.

Targets stay at their natural 2-3 sentence length, which is what makes the
adapter produce longer lines than the old synthetic one-sentence corpus.

    python scripts/build_soccerreplay_corpus.py --train-matches 500

Token: HF_TOKEN from the environment or the repo ``.env`` (never printed).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DATASET = "Homie0609/SoccerReplay-1988"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"

# dataset comments_type -> pipeline event
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

# the grounding validator at runtime rejects these claims, so never train them
BLOCKED = ("penalty", "red card", "yellow card", "offside", "off side",
           "var", "substitution", "injury", "disallowed", "own goal")

INTENSITY = {"goal": "huge", "shot": "high", "save": "high",
             "corner": "medium", "tackle": "medium", "pass": "low",
             "dead_time": "calm"}

ROLE_WORD = {"Goalkeeper": "the goalkeeper", "Defender": "the defender",
             "Midfielder": "the midfielder", "Forward": "the forward",
             "coach": "the manager"}

_SCORE_RE = re.compile(r"(\d{1,2})\s*[:–-]\s*(\d{1,2})")
_TIME_RE = re.compile(r"^(\d{1,3}):(\d{2})$")
_PLACEHOLDER_RE = re.compile(r"\[(?:PLAYER|TEAM|REFEREE|COACH)\]")
_BIGRAM_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
# competitions and other proper nouns that are safe to keep in a line
_GENERIC_CAPS = {
    "premier league", "champions league", "europa league", "fa cup",
    "league cup", "serie a", "la liga", "bundesliga", "ligue 1",
    "world cup", "european championship", "copa del rey", "dfb pokal",
    "coppa italia", "coupe de france", "efl cup",
}


def has_unknown_name(text: str, teams: dict[str, str]) -> bool:
    """True when a proper-noun pair is neither a team nor a competition."""
    allowed = [t.lower() for t in teams.values() if t]
    for m in _BIGRAM_RE.finditer(text):
        low = m.group(1).lower()
        if low in _GENERIC_CAPS or any(t in low for t in allowed):
            continue
        return True
    return False


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def read_token() -> str:
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (ROOT / ".env", ROOT.parents[2] / ".env"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            continue
    raise SystemExit("HF_TOKEN missing: put it in .env or the environment")


def list_matches(split: str, token: str) -> list[str]:
    # the tree endpoint paginates (and the first page can be all directories),
    # so use the unpaginated repo file listing instead
    from huggingface_hub import HfApi
    files = HfApi(token=token).list_repo_files(DATASET, repo_type="dataset")
    prefix = split.rstrip("/") + "/"
    return sorted(f for f in files
                  if f.startswith(prefix) and f.endswith(".json"))


def download(session: requests.Session, path: str, dest: Path) -> dict | None:
    if dest.exists():
        try:
            return json.loads(dest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            dest.unlink(missing_ok=True)
    r = session.get(BASE + quote(path), timeout=90)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return r.json()


def roster(match: dict) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """(name -> (side, role phrase), [team names]) from players_data."""
    names: dict[str, tuple[str, str]] = {}
    info = match.get("match_info", {})
    teams = {"home": info.get("home_team", ""),
             "away": info.get("away_team", "")}
    for side in ("home", "away"):
        for key, players in (match.get("players_data") or {}).items():
            if not key.startswith(side) or not isinstance(players, list):
                continue
            for p in players:
                role = ROLE_WORD.get(str(p.get("Role", "")), "the player")
                for n in (p.get("Full Name"), p.get("players_name")):
                    if n and len(str(n)) > 2:
                        names[str(n)] = (side, role)
    return names, [teams["home"], teams["away"]]


def clean_text(text: str, names: dict[str, tuple[str, str]],
               teams: dict[str, str]) -> tuple[str, str | None]:
    """Neutralise player names, return (text, event side)."""
    side = None
    # "(Manchester City)" after a player name is a reliable team tag
    for s, team in teams.items():
        tag = re.compile(r"\s*\(%s\)" % re.escape(team), re.I)
        if tag.search(text):
            side = side or s
        text = tag.sub("", text)
    # longest names first so "Kevin De Bruyne" wins over "De Bruyne K."
    for name in sorted(names, key=len, reverse=True):
        s, role = names[name]
        pat = re.compile(r"\b%s\b" % re.escape(name), re.I)
        if pat.search(text):
            if side is None:
                side = s
            text = pat.sub(role, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, side


def align_text(c: dict, names: dict[str, tuple[str, str]],
               teams: dict[str, str]) -> tuple[str, str | None]:
    """Use the anonymized text, then fill placeholders from the original.

    This removes every player and referee name while keeping real team names:
    the original text still contains them, so the k-th ``[TEAM]`` maps to the
    k-th team mention and the k-th ``[PLAYER]`` maps to the k-th roster name.
    """
    orig = (c.get("comments_text") or "").strip()
    anon = (c.get("comments_text_anonymized") or "").strip()
    if not anon:
        return clean_text(orig, names, teams)

    spans: list[tuple[int, tuple[str, str]]] = []
    taken: list[tuple[int, int]] = []
    for name in sorted(names, key=len, reverse=True):
        for m in re.finditer(r"\b%s\b" % re.escape(name), orig, re.I):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            spans.append((m.start(), names[name]))
    spans.sort(key=lambda s: s[0])
    roles = [role for _, (_, role) in spans]
    side = spans[0][1][0] if spans else None

    n_ph = anon.count("[PLAYER]")
    if roles and n_ph == len(roles):
        it = iter(roles)
        anon = re.sub(r"\[PLAYER\]", lambda _: next(it), anon)
    else:
        anon = anon.replace("[PLAYER]", "the player")

    team_mentions = [m.group(0) for m in re.finditer(
        r"\b(?:%s|%s)\b" % (re.escape(teams["home"]),
                            re.escape(teams["away"])), orig, re.I)]
    if side is None and team_mentions:
        first = team_mentions[0].lower()
        side = "home" if first == teams["home"].lower() else "away"
    n_t = anon.count("[TEAM]")
    if team_mentions and n_t == len(team_mentions):
        it = iter(team_mentions)
        anon = re.sub(r"\[TEAM\]", lambda _: next(it), anon)
    else:
        anon = anon.replace("[TEAM]", teams.get(side or "home"))
    anon = (anon.replace("[REFEREE]", "the referee")
                .replace("[COACH]", "the manager"))
    anon = re.sub(r"\s*\((%s|%s)\)" % (re.escape(teams["home"]),
                                       re.escape(teams["away"])), "", anon, flags=re.I)
    return re.sub(r"\s+", " ", anon).strip(), side


def convert_match(match: dict, rng: random.Random) -> list[str]:
    info = match.get("match_info") or {}
    home, away = info.get("home_team"), info.get("away_team")
    if not home or not away:
        return []
    teams = {"home": home, "away": away}
    names, _ = roster(match)
    comments = [c for c in match.get("comments") or []
                if c.get("time_stamp")]
    comments.sort(key=lambda c: (int(c.get("half") or 0),
                                 _TIME_RE.match(str(c["time_stamp"])).group(0)
                                 if _TIME_RE.match(str(c["time_stamp"])) else ""))
    score = [0, 0]
    records: list[dict] = []
    for c in comments:
        kind = EVENT_MAP.get(str(c.get("comments_type", "")).lower())
        if not kind:
            continue
        text = (c.get("comments_text") or "").strip()
        if not text:
            continue
        low = text.lower()
        if any(b in low for b in BLOCKED):
            continue
        m = _TIME_RE.match(str(c["time_stamp"]))
        if not m:
            continue
        minute = int(m.group(1)) + (45 if int(c.get("half") or 0) == 2 else 0)
        if minute > 95:
            continue
        text, side = align_text(c, names, teams)
        if (_PLACEHOLDER_RE.search(text) or side is None
                or has_unknown_name(text, teams)):
            continue
        # lines with ungrounded numbers are rejected at runtime; only keep
        # score-like digits (the running/final score)
        if kind == "goal":
            sm = _SCORE_RE.search(text)
            if sm:
                hs, as_ = int(sm.group(1)), int(sm.group(2))
                if 0 <= hs <= 9 and 0 <= as_ <= 9:
                    score = [hs, as_]
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
        records.append({
            "kind": kind, "side": side, "minute": minute,
            "sec": minute * 60 + int(m.group(2)),
            "text": text, "score": [score[0], score[1]],
        })

    rows: list[str] = []
    for i, rec in enumerate(records):
        text = rec["text"]
        # a second moment from the same passage teaches 2-sentence flow
        if i + 1 < len(records) and rng.random() < 0.4:
            nxt = records[i + 1]
            gap = nxt["sec"] - rec["sec"]
            joined = f"{text} {nxt['text']}"
            if (0 <= gap <= 120 and not re.search(r"\d", nxt["text"])
                    and len(joined.split()) <= 65):
                text = joined
        team = teams[rec["side"]]
        hs, as_ = rec["score"]
        poss_side = rec["side"] if rec["kind"] in ("pass", "tackle") \
            else rng.choice(["home", "away"])
        poss_team = teams[poss_side]
        scoreline = f"{home} {hs}-{as_} {away}"
        minute = rec["minute"]
        history = rng.choice([
            f"{scoreline}, {ordinal(minute)} minute, {poss_team} in possession.",
            f"{ordinal(minute)} minute at {home} v {away}; {poss_team} on the ball.",
            f"{scoreline}, {minute} minutes gone, {poss_team} with the ball.",
        ])
        rows.append(
            f"[HISTORY] {history} [EVENT] {rec['kind']} by {team} "
            f"[INTENSITY] {INTENSITY[rec['kind']]} [COMMENTARY] {text}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-matches", type=int, default=500)
    ap.add_argument("--valid-matches", type=int, default=40)
    ap.add_argument("--raw", default=str(ROOT / "data" / "soccerreplay"))
    ap.add_argument("--out", default=str(ROOT / "data" / "commentator_train_v4.jsonl"))
    ap.add_argument("--valid-out", default=str(ROOT / "data" / "commentator_valid_v4.jsonl"))
    ap.add_argument("--synthetic", default=str(ROOT / "data" / "commentator_train_v2.jsonl"))
    ap.add_argument("--synthetic-mix", type=int, default=3000,
                    help="how many short synthetic examples to blend in")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = read_token()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {tok}"
    rng = random.Random(args.seed)
    raw = Path(args.raw)

    def run_split(split: str, n: int) -> list[str]:
        files = list_matches(split, tok)
        rng.shuffle(files)
        files = files[:n]
        dest = raw / split
        dest.mkdir(parents=True, exist_ok=True)
        print(f"{split}: {len(files)} matches", flush=True)

        def one(i_path):
            i, path = i_path
            try:
                match = download(session, path, dest / f"{i:05d}.json")
                local_rng = random.Random(args.seed * 100003 + i)
                return convert_match(match, local_rng) if match else []
            except Exception as e:  # noqa: BLE001
                print(f"  skip {Path(path).name}: {type(e).__name__}", flush=True)
                return []

        rows: list[str] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, chunk in enumerate(pool.map(one, enumerate(files))):
                rows.extend(chunk)
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(files)} matches, {len(rows)} examples",
                          flush=True)
        return rows

    train = run_split("train", args.train_matches)
    valid = run_split("valid", args.valid_matches)

    synth = Path(args.synthetic)
    if synth.exists() and args.synthetic_mix > 0:
        short = [ln for ln in synth.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        rng.shuffle(short)
        for ln in short[:args.synthetic_mix]:
            try:
                train.append(json.loads(ln)["text"])
            except (ValueError, KeyError):
                continue

    rng.shuffle(train)
    Path(args.out).write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in train)
        + "\n", encoding="utf-8")
    Path(args.valid_out).write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in valid)
        + "\n", encoding="utf-8")
    print(f"train {len(train)} examples -> {args.out}")
    print(f"valid {len(valid)} examples -> {args.valid_out}")
    for line in train[:3]:
        print(" ", line[:220])


if __name__ == "__main__":
    main()
