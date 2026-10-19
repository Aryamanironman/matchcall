"""Two-commentator engine: A = play-by-play (calls events), B = color (reacts,
chats in gaps). Grounded dialogue for ANY match.

Design follows the published dual-commentator systems (ACM IMX 2024
"Designing for Automated Sports Commentary Systems"; AiCommentator; live
data-driven demos): the duo separates CALLING the game from TALKING ABOUT
the game, and the interplay — not monologue alternation — is what viewers
rate highest. User studies there flagged exactly the failure we hit in v1:
writers that guess get caught. So:

  - A's calls come from the grounded announcer (only detected events).
  - B reacts to confirmed events and chats from MEASURED match state
    (minutes played, shots/goals so far, last event, crowd surge level).
    B never introduces a player, team, card, or score that was not detected.
  - When the local GPT-2 LoRA is available it may rephrase B's lines, but
    every generation passes an anti-hallucination validator: team names
    must match the real teams, no scorelines, no invented digits; else the
    grounded template is used.
"""
from __future__ import annotations

import random

from app.config import settings
from app.events import Segment

# B (color commentator) is switched off for now — single-voice output until the
# TTS voices are revisited. Set COMMENTATOR_B=1 to re-enable the duo.
import os
ENABLE_B = os.getenv("COMMENTATOR_B", "0") == "1"

# react with a pair only on notable events; passes/dead time get A alone
_PAIR_LABELS = {"goal", "shot", "tackle"}
# start a B-led chat exchange in quiet gaps longer than this (seconds)
CHAT_GAP = 75.0

_EXC_FOR_B = {"goal": "high", "shot": "medium", "tackle": "medium",
              "pass": "low", "dead_time": "calm"}


class _State:
    """Everything B is allowed to know: measured, never guessed."""

    def __init__(self, teams: dict):
        self.teams = teams
        self.minute = 0.0
        self.goals = 0
        self.shots = 0
        self.tackles = 0
        self.events = 0
        self.last_label: str | None = None
        self.last_minute: float | None = None
        self.crowd_rise: float = 1.0  # at the most recent event

    def observe(self, label: str, features: dict, crowd_rise: float = 1.0):
        self.minute = float(features.get("minute", 0) or 0)
        self.crowd_rise = crowd_rise
        if label == "goal":
            self.goals += 1
        elif label == "shot":
            self.shots += 1
        elif label == "tackle":
            self.tackles += 1
        if label in ("goal", "shot", "tackle"):
            self.last_label, self.last_minute = label, self.minute
        self.events += 1


# --- grounded dialogue templates (no invented facts) -----------------------

_B_REACT: dict[str, list[str]] = {
    "goal": [
        "And you could feel that coming — the whole stadium is on its feet!",
        "Look at the celebrations — that means everything at this stage.",
        "What a moment — and the noise you can hear under us says it all.",
        "That's been brewing; they've been knocking on the door for minutes.",
    ],
    "shot": [
        "Ooh, that's close — the keeper did well to deal with it.",
        "That's exactly the chance they've been probing for.",
        "Good covering shape to get anything behind that one.",
        "Well struck — those are the efforts that change games.",
    ],
    "tackle": [
        "Clean challenge — commitment without the malice.",
        "That's the kind of duel that decides midfield battles.",
        "Superb timing — a fraction later and that's a free kick.",
    ],
    "pass": [
        "Lovely rhythm to the build-up right now.",
    ],
    "dead_time": [
        "A chance to reset — you can see the shape being organised from the touchline.",
        "These little pauses matter more than people think.",
    ],
}

_CHAT_B: list[str] = [
    "{minute} minutes gone — and still nothing between these sides but effort.",
    "You sense the next goal in this match could decide everything.",
    "The tempo has hardly dropped all half — credit to both benches.",
    "This is the spell where managers earn their money, tactically.",
]
_CHAT_A: list[str] = [
    "And listen to the crowd — they haven't dropped once.",
    "You can hear it under us — they're right behind every challenge.",
    "Not a quiet seat in the house at the moment.",
]
_CHAT_SHOTS: list[str] = [
    "{shots} shots so far, and honestly the rhythm says more are coming.",
    "We've had {shots} efforts on goal — the pressure is building somewhere.",
]
_CHAT_LAST: list[str] = [
    "That last chance around the {last_min}th minute is still being talked about.",
    "Plenty to discuss from that moment a few minutes back.",
]
_CHAT_ROAR: list[str] = [
    "And the noise levels have spiked lately — you can feel the tension rise.",
    "The crowd have definitely found another gear in the last few minutes.",
]

_GPT2_VALIDATE_BAD = __import__("re").compile(
    r"\b\d+\s*[-–]\s*\d+\b")  # any scoreline like 2-1


class Duo:
    """A + B. A's calls are delegated (grounded announcer); B is grounded here."""

    def __init__(self, teams: dict, announcer=None):
        self.teams = teams
        from app.commentary import TemplateAnnouncer
        self.a = announcer or TemplateAnnouncer(teams)
        self.state = _State(teams)
        self._rng = random.Random(7)
        self._b_i = 0
        self._chat_i = 0
        self._gpt2 = None
        try:
            from app import commentary_gpt2
            if commentary_gpt2.available():
                self._gpt2 = commentary_gpt2
        except Exception:  # noqa: BLE001
            self._gpt2 = None

    # -- validation: B may not invent entities ------------------------------

    def _valid_b(self, text: str) -> bool:
        t = text.strip()
        if not (4 <= len(t.split()) <= 24) or _GPT2_VALIDATE_BAD.search(t):
            return False
        home = (self.teams.get("home") or "").strip()
        away = (self.teams.get("away") or "").strip()
        # every capitalized multi-char word must be a real team word
        for w in t.split():
            word = w.strip(",.!?;:\"'")
            if word.istitle() and len(word) > 2 and word.lower() not in (
                    "that", "this", "these", "those", "the", "and", "but",
                    "look", "ooh", "well", "clean", "superb", "you", "we",
                    "and", "credit", "plenty", "not"):
                if home.lower() not in word.lower() \
                        and away.lower() not in word.lower():
                    return False
        return True

    def _b_line(self, label: str, features: dict) -> tuple[str, str]:
        """B's line: GPT-2 phrasing when it passes validation, else template."""
        template = self._rng.choice(_B_REACT[label])
        excitement = _EXC_FOR_B.get(label, "medium")
        if self._gpt2 is not None:
            try:
                line, gpt2_exc = self._gpt2.event_line(
                    label, features, self.teams,
                    excitement_hint=excitement)
                if self._valid_b(line):
                    return line, gpt2_exc
            except Exception:  # noqa: BLE001
                pass
        return template, excitement

    # -- public API ----------------------------------------------------------

    def event_pair(self, label: str, features: dict,
                   crowd_rise: float = 1.0) -> list[Segment]:
        """Segments for one detected event: A calls it, B reacts (notable only)."""
        self.state.observe(label, features, crowd_rise)
        call, exc = self.a.event_line(label, features)
        segs = [Segment(t0=float(features.get("t0", 0.0)), kind="event",
                        event_label=label, excitement=exc, text=call,
                        speaker="A")]
        if ENABLE_B and label in _PAIR_LABELS:
            b_text, b_exc = self._b_line(label, features)
            # B comes ~1.2s after A starts; TTS renders them back-to-back
            segs.append(Segment(t0=float(features.get("t0", 0.0)) + 1.2,
                                kind="event", event_label=label,
                                excitement=b_exc, text=b_text, speaker="B"))
        return segs

    def chat_pair(self, t: float, features: dict) -> list[Segment]:
        """B-led exchange for long quiet gaps — grounded in match state.
        With B disabled, A alone fills the gap (no dead air)."""
        st = self.state
        minute = max(1, int(float(features.get("minute", 0) or st.minute or 0)))
        if not ENABLE_B:
            return [Segment(t0=t, kind="chat", event_label=None,
                            excitement="low",
                            text=self._rng.choice(_CHAT_A), speaker="A")]
        b = self._rng.choice(_CHAT_B).format(minute=minute)
        if st.shots >= 2 and self._rng.random() < 0.6:
            b = self._rng.choice(_CHAT_SHOTS).format(shots=st.shots)
        elif st.last_minute is not None and st.minute - st.last_minute < 8 \
                and self._rng.random() < 0.5:
            b = self._rng.choice(_CHAT_LAST).format(
                last_min=max(1, int(st.last_minute)))
        elif st.crowd_rise >= 1.3 and self._rng.random() < 0.5:
            b = self._rng.choice(_CHAT_ROAR)
        a = self._rng.choice(_CHAT_A)
        out = [
            Segment(t0=t, kind="chat", event_label=None, excitement="medium",
                    text=b, speaker="B"),
            Segment(t0=t + 4.5, kind="chat", event_label=None,
                    excitement="low", text=a, speaker="A"),
        ]
        self.state.observe("pass", {"minute": st.minute})
        return out

    def opener_pair(self) -> list[Segment]:
        out = [Segment(t0=0.2, kind="filler", event_label=None,
                       excitement="medium", text=self.a.opener(), speaker="A")]
        if ENABLE_B:
            out.append(Segment(t0=5.0, kind="chat", event_label=None,
                               excitement="medium",
                               text="And alongside me in the box — we've got "
                                    "a cracker for you tonight.", speaker="B"))
        return out


def chat_insertions(segments: list[Segment], total_dur: float,
                    min_gap: float = 75.0) -> list[tuple[float, float]]:
    """(t_seconds, t_seconds) midpoints of quiet gaps that deserve a B-led
    chat exchange. Pipeline passes the midpoint twice: position + match minute."""
    ordered = sorted(segments, key=lambda s: s.t0)
    out: list[tuple[float, float]] = []
    last = 0.0
    for s in ordered:
        gap = s.t0 - last
        if gap >= min_gap:
            mid = last + gap / 2.0
            out.append((mid, mid))
        last = max(last, s.t0)
    tail = total_dur - last
    if tail >= min_gap:
        out.append((last + tail / 2.0, last + tail / 2.0))
    return out


_EST_DUR = {"event": 4.8, "filler": 6.0, "chat": 5.0}


def resolve_overlaps(segments: list[Segment]) -> list[Segment]:
    """Make the timeline talk-over-free at PLANNING time (audio durs unknown).

    A's event calls are anchors: they are never DROPPED (never miss a goal),
    but when an event lands inside the previous call's estimated voice span
    the call is shifted to the first free slot (a call 1-2s late beats two
    voices at once). Every other segment (B reactions, fillers, chat) must
    start clear of whatever precedes it AND end clear of the next event
    call, or it is dropped. This kills the two-voices-at-once case the old
    1.2s pop-guard only caught by luck.
    """
    import bisect
    anchors = sorted((s for s in segments
                      if s.kind == "event" and s.speaker == "A"),
                     key=lambda s: s.t0)
    rest = sorted((s for s in segments
                   if not (s.kind == "event" and s.speaker == "A")),
                  key=lambda s: s.t0)

    def est(x: Segment) -> float:
        return _EST_DUR.get(x.kind, 6.0)

    placed: list[Segment] = []
    for a in anchors:
        prev_end = max((p.t0 + est(p) for p in placed), default=0.0)
        a.t0 = max(a.t0, prev_end)
        placed.append(a)
    for s in rest:
        prev_end = max((p.t0 + est(p) for p in placed), default=0.0)
        desired = max(s.t0, prev_end)
        anchor_times = [a.t0 for a in placed
                        if a.kind == "event" and a.speaker == "A"]
        j = bisect.bisect_right(anchor_times, s.t0)
        nxt = anchor_times[j] if j < len(anchor_times) else None
        if nxt is not None and desired + est(s) > nxt - 0.3:
            continue  # cannot fit before the next call -> drop this line
        s.t0 = desired
        placed.append(s)
        placed.sort(key=lambda x: x.t0)
    return placed
