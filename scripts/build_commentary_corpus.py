"""Build a grounded GPT-2 commentary corpus.

The original adapter was trained on nine hand-written examples, so it
continued with unrelated internet text. This generator creates a few thousand
examples in the exact markup format the adapter expects, with every fact
(team, scoreline, minute, phase, possession) coming from the same context the
runtime prompt will contain. Team-level only: the runtime vision pipeline does
not identify players, so the model is never taught to invent them.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEAM_PAIRS = [
    ("Argentina", "France"),
    ("Brazil", "Germany"),
    ("Liverpool", "Manchester City"),
    ("Spain", "Italy"),
    ("Home", "Away"),
]

SCORES = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1),
          (2, 2), (3, 1), (3, 2), (4, 2)]

MINUTES = [3, 8, 12, 17, 23, 29, 34, 41, 45, 46, 52, 58, 61,
           66, 71, 76, 79, 80, 81, 84, 88, 90]

EVENTS = ["goal", "shot", "tackle", "pass", "dead_time", "corner", "save"]
INTENSITY = {
    "goal": ["huge", "huge", "high"],
    "shot": ["high", "high", "medium"],
    "tackle": ["medium", "medium", "high"],
    "pass": ["low", "low", "medium"],
    "dead_time": ["calm", "calm", "low"],
    "corner": ["medium", "medium", "high"],
    "save": ["high", "high", "huge"],
}
PHASES = ["opening", "midfield", "closing"]

LINES = {
    "goal": [
        "GOAL! {team} have it — {scoreline} and this place erupts!",
        "It's in! {team} strike — {scoreline}!",
        "What a finish from {team}! The net bulges and the crowd goes wild!",
        "{team} score! {scoreline} — this match has come alive!",
        "Buried! {team} find the net and the stadium erupts!",
        "A huge goal for {team}! They've done it — {scoreline}!",
        "Oh, that is spectacular! {team} make it count!",
        "Clinical from {team}! The keeper had no chance!",
        "The breakthrough! {team} finally find the net!",
        "Drama! {team} score and the noise is deafening!",
        "{team} in the {minute_ord} minute — {scoreline}! What a moment!",
        "They've only gone and scored! {team} — {scoreline}!",
        "The deadlock is broken by {team}! {scoreline}!",
        "A goal out of nothing for {team}! The crowd can't believe it!",
        "Sensational from {team}! {scoreline} and the momentum swings!",
        "That's the goal {team} have been waiting for — {scoreline}!",
        "{team} punish them! {scoreline} and the place is rocking!",
        "No doubt about it — {team} score, {scoreline}!",
        "A moment of magic from {team}! {scoreline}!",
        "The net ripples and {team} celebrate — {scoreline}!",
    ],
    "shot": [
        "Ooh, {team} let fly — that's a dangerous effort!",
        "Big chance for {team}!",
        "{team} pull the trigger — the defence scrambles!",
        "What a strike from {team} — the keeper is tested!",
        "{team} try their luck from distance!",
        "Half a yard of space and {team} shoot!",
        "{team} work an opening and get the shot away!",
        "A powerful effort from {team} — just wide!",
        "{team} are turning the screw here!",
        "The crowd rises as {team} shoot!",
        "Good hit from {team}, but the keeper is equal to it!",
        "{team} force a save with a fierce drive!",
        "That's close from {team}! Just past the post!",
        "{team} have a sight of goal — and they take it!",
        "A shot on the turn from {team}!",
        "{team} unleash one from the edge of the box!",
        "The keeper is called into action by {team}!",
        "Nearly a breakthrough for {team}!",
        "A real chance for {team} — blocked at the last moment!",
        "{team} keep the pressure on with another shot!",
    ],
    "tackle": [
        "Crunching challenge — {team} win it back!",
        "Superb defending from {team}!",
        "Strong tackle! {team} snap out of trouble!",
        "{team} throw bodies in and come away with it!",
        "Won cleanly by {team} — that sets the tone!",
        "A perfectly timed challenge from {team}!",
        "{team} refuse to give an inch!",
        "Great recovery from {team} to snuff out the danger!",
        "{team} win the ball back in midfield!",
        "The tackle comes in and {team} break forward!",
        "{team} stand firm under pressure!",
        "A big challenge from {team} — the crowd loves it!",
        "{team} dig in and clear the danger!",
        "No way through — {team} shut the door!",
        "{team} muscle their way back into possession!",
    ],
    "pass": [
        "{team} knock it around nicely, probing for an opening.",
        "Neat build-up from {team}.",
        "{team} keep the ball moving, searching for space.",
        "Patient stuff from {team}, switching the play smartly.",
        "{team} work it wide and look to stretch the defence.",
        "{team} string the passes together in the {phase} stages.",
        "It's the {minute_ord} minute and {team} are dictating the rhythm.",
        "{team} circulate it patiently.",
        "Discipline from {team} — keeping the ball when it matters.",
        "Little triangle from {team} near the touchline, neat and tidy.",
        "{team} switch the point of attack, hunting for a gap.",
        "A composed spell from {team} in midfield.",
        "{team} are seeing plenty of the ball.",
        "The ball keeps finding its way to {team}.",
        "{team} look to build from the back.",
        "Patient probing from {team} — waiting for the opening.",
        "{team} are in no hurry here.",
        "Good control from {team} under pressure.",
        "{team} recycle possession and reset the attack.",
        "The tempo is being set by {team}.",
    ],
    "dead_time": [
        "A brief pause in play as {team} reset their shape.",
        "The tempo drops for a moment.",
        "A quiet spell — {team} content to keep the ball.",
        "Play settles down, {team} taking a breath.",
        "The game catches its breath.",
        "A stoppage in play; both sides regroup.",
        "Not much happening at the moment.",
        "{team} slow it down and keep possession.",
        "The intensity dips briefly.",
        "A calm passage in the {phase}.",
        "The referee has a word; play resumes shortly.",
        "A short break in the action.",
        "{team} take their time over the restart.",
        "The crowd settles as the players reset.",
        "A moment of calm in a frantic contest.",
    ],
    "corner": [
        "Corner to {team} — the box is packed!",
        "{team} win a corner. Everyone forward!",
        "A chance to swing one in for {team}.",
        "{team} have a corner — the keeper is under pressure!",
        "The delivery comes in for {team}!",
        "{team} send the big men up for the corner.",
        "A dangerous corner for {team} here.",
        "The ball is whipped in — {team} attack it!",
        "{team} force a corner and the crowd rises.",
        "A set-piece opportunity for {team}.",
    ],
    "save": [
        "What a save! The keeper denies {team}!",
        "Denied! {team} are kept out by a brilliant stop!",
        "Incredible goalkeeping — {team} can't believe it!",
        "The keeper stands tall and {team} are frustrated!",
        "A superb stop to keep {team} out!",
        "That's a match-winning save!",
        "The goalkeeper reacts brilliantly!",
        "{team} are denied by the woodwork!",
        "Somehow it stays out — what a save!",
        "The keeper gets a strong hand to it!",
    ],
}


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def history_variants(home, away, hs, a_s, minute, poss_team, phase):
    scoreline = f"{home} {hs}-{a_s} {away}"
    mo = ordinal(minute)
    return [
        f"{scoreline}, {mo} minute. {poss_team} in possession.",
        f"{mo} minute at {home} v {away}; {poss_team} on the ball.",
        f"{scoreline}. {poss_team} in possession in the {phase}.",
        f"The {phase} stages, {scoreline}, {poss_team} on the ball.",
        f"{scoreline}, {minute} minutes gone.",
        f"{minute}' — {home} {hs}-{a_s} {away}.",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "commentator_train_v2.jsonl"))
    ap.add_argument("--limit", type=int, default=6000)
    args = ap.parse_args()
    rng = random.Random(42)
    rows = []
    seen = set()

    # keep the original hand-written examples
    old = ROOT / "data" / "commentator_train.jsonl"
    if old.exists():
        for line in old.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(line)
                seen.add(line)

    while len(rows) < args.limit:
        home, away = rng.choice(TEAM_PAIRS)
        hs, as_ = rng.choice(SCORES)
        minute = rng.choice(MINUTES)
        event = rng.choice(EVENTS)
        team = rng.choice([home, away])
        poss = rng.choice([home, away])
        phase = rng.choice(PHASES)
        scoreline = f"{home} {hs}-{as_} {away}"
        intensity = rng.choice(INTENSITY[event])
        hist = rng.choice(history_variants(home, away, hs, as_, minute,
                                           poss, phase))
        line = rng.choice(LINES[event]).format(
            team=team, home=home, away=away, minute=minute,
            minute_ord=ordinal(minute),
            scoreline=scoreline, phase=phase, poss_team=poss)
        text = (f"[HISTORY] {hist} [EVENT] {event} by {team} "
                f"[INTENSITY] {intensity} [COMMENTARY] {line}")
        if text in seen:
            continue
        seen.add(text)
        rows.append(json.dumps({"text": text}, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} examples -> {out}")
    print("sample:")
    for line in rows[9:12]:
        print(" ", line)


if __name__ == "__main__":
    main()
