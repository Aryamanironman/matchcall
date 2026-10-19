"""Match-state context: the facts every commentary line must agree with.

Built once from the labeled windows (+ score-bug anchors when available),
then passed to the announcer and the duo through the features dict, so any
line can reference the scoreline, the run of play and recent history
without ever inventing them. Also owns the shuffle-bag sampler that keeps
line pools from repeating until they are exhausted.
"""
from __future__ import annotations

import random
from collections import deque


class ShuffleBag:
    """Sample from a pool without replacement; reshuffle only when empty.

    With a pool of N lines the first N uses are guaranteed distinct — this
    is what makes v5 not repeat "work it wide" twice in five minutes.
    """

    def __init__(self, pool: list[str], seed: int = 0):
        self._pool = list(pool)
        self._rng = random.Random(seed)
        self._bag: list[str] = []

    def draw(self) -> str:
        if not self._bag:
            self._bag = self._pool[:]
            self._rng.shuffle(self._bag)
        return self._bag.pop()

    def draw_many(self, k: int) -> list[str]:
        return [self.draw() for _ in range(k)]


def speed_word(speed: float) -> str:
    """Grounded speed phrasing — never claims pace the window didn't show."""
    if speed >= 260:
        return "electric"
    if speed >= 170:
        return "quick"
    if speed >= 90:
        return "measured"
    return "patient"


def possession_word(frac: float) -> str:
    if frac >= 0.66:
        return "dominant"
    if frac >= 0.55:
        return "confident"
    return "even"


class MatchState:
    """Rolling facts about the match, updated per labeled event."""

    def __init__(self, teams: dict, goals: list[dict] | None = None,
                 start_score: dict | None = None):
        self.teams = teams
        self.home = teams.get("home", "the home side")
        self.away = teams.get("away", "the visitors")
        # goals: [{"t": sec, "team": "home"|"away", "minute": 34, "index": 1}]
        self.goals = sorted(goals or [], key=lambda g: g["t"])
        # score already on the board when the clip starts (e.g. 2-0 at the
        # beginning of a second-half recording)
        self.start_score = {"home": 0, "away": 0, **(start_score or {})}
        self.score = dict(self.start_score)
        self.shots = {"home": 0, "away": 0}
        self.last_event_t: float | None = None
        self.last_event_label: str | None = None
        self.last_goal_t: float | None = None
        self._pos = deque(maxlen=12)          # recent ball_x_frac
        self._event_ts: deque = deque(maxlen=30)
        self.minute = 46

    # -- bookkeeping ---------------------------------------------------------

    def observe(self, label: str, t: float, features: dict) -> None:
        self.minute = int(features.get("minute", self.minute) or self.minute)
        side = features.get("goal_side")
        if side not in ("home", "away"):
            side = "home" if (features.get("ball_x_frac", 0.5) or 0.5) >= 0.5 else "away"
        self._pos.append(features.get("ball_x_frac", 0.5) or 0.5)
        self._event_ts.append(t)
        if label == "shot":
            self.shots[side] = self.shots.get(side, 0) + 1
        if label == "goal":
            self.score[side] = self.score.get(side, 0) + 1
            self.last_goal_t = t
        self.last_event_t, self.last_event_label = t, label

    # -- queries -------------------------------------------------------------

    def goals_at(self, t: float) -> int:
        return sum(1 for g in self.goals if g["t"] <= t)

    def scoreline_at(self, t: float) -> tuple[int, int]:
        h = self.start_score["home"] + sum(
            1 for g in self.goals if g["t"] <= t and g["team"] == "home")
        a = self.start_score["away"] + sum(
            1 for g in self.goals if g["t"] <= t and g["team"] == "away")
        return h, a

    def phase_at(self, t: float) -> str:
        for g in self.goals:
            if 0 <= t - g["t"] <= 90:
                return "post_goal"
        m = self.minute
        if m <= 50:
            return "opening"
        if m >= 85:
            return "closing"
        return "midfield"

    def possession_side(self) -> str:
        if not self._pos:
            return "even"
        f = sum(self._pos) / len(self._pos)
        return "home" if f >= 0.55 else ("away" if f <= 0.45 else "even")

    def event_density(self, t: float, span: float = 120.0) -> int:
        return sum(1 for e in self._event_ts if t - span <= e <= t)

    def context_at(self, t: float, features: dict) -> dict:
        """The dict every template may draw from. All facts, zero invention."""
        h, a = self.scoreline_at(t)
        phase = self.phase_at(t)
        margin = h - a
        # Tracker possession is more specific than the ball-x proxy; use it
        # when the window actually observed tracked players.
        ph = features.get("possession_home")
        if ph is not None and float(ph) >= 0.0:
            possession = "home" if float(ph) >= 0.55 else (
                "away" if float(ph) <= 0.45 else "even")
        else:
            possession = self.possession_side()
        ctx = dict(features or {})
        ctx.update({
            "minute": max(1, int(features.get("minute", self.minute) or self.minute)),
            "score_home": h, "score_away": a,
            "scoreline": f"{self.home} {h}–{a} {self.away}",
            "margin": margin,
            "leader": self.home if margin > 0 else (self.away if margin < 0 else None),
            "phase": phase,
            "possession": possession,
            "shots_home": self.shots.get("home", 0),
            "shots_away": self.shots.get("away", 0),
            "goals_so_far": self.goals_at(t),
            "speed_word": speed_word(float(features.get("ball_speed", 0) or 0)),
            "since_event": (t - self.last_event_t) if self.last_event_t else None,
            "last_label": self.last_event_label,
        })
        return ctx
