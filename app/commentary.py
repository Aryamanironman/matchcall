"""Commentary generation: Gemini 1.5 Flash when keyed, rich templates otherwise.

Both paths produce present-tense, energetic lines sized for broadcast gaps.
"""
from __future__ import annotations

import json
import re

from app.config import settings
from app.match_context import ShuffleBag

EXCITE_ORDER = ["calm", "low", "medium", "high", "huge"]

TEMPLATE: dict[str, list[str]] = {
    "goal": [
        "IT'S IN! {team} have it — {scoreline} and this place erupts!",
        "GOAL! A thunderous finish from {team} — {scoreline}!",
        "OH THAT IS SPECTACULAR! {team} find the net and the crowd explodes!",
        "IT'S A GOAL! Clinical from {team} — the keeper had no chance!",
        "GOAL {team}! Buried it — and listen to this stadium!",
        "It's in! {team} strike — the net bulges and it's {scoreline}!",
        "GOAL! {team} make it count — sheer precision at a vital time!",
    ],
    "shot": [
        "Ooh, {team} let fly — that's a dangerous effort on goal!",
        "Big chance! {team} strike it towards goal!",
        "{team} pull the trigger — the defence scrambles!",
        "What a strike attempt from {team}, testing the keeper!",
        "{team} try their luck from distance — it's charged down!",
        "Half a yard of space and {team} shoot — the defence throws bodies at it!",
    ],
    "tackle": [
        "Crunching challenge in there — {team} win it back!",
        "Superb defending from {team}, the ball is turned over!",
        "Strong tackle! {team} snap out of trouble and break forward!",
        "{team} throw bodies in and come away with it — brilliant commitment!",
        "Won cleanly by {team} — that challenge sets the tone!",
    ],
    "pass": [
        "{team} knock it around nicely, probing for an opening.",
        "Neat build-up from {team} as they move it through midfield.",
        "{team} keep the ball moving, searching for space.",
        "Patient stuff from {team}, switching the play smartly.",
        "{team} work it wide and look to stretch the defence.",
        "{team} string the passes together in the {phase_word} stages.",
        "It's the {minute}th minute and {team} are dictating the rhythm.",
        "{team} circulate it patiently — {speed_word} possession.",
        "Discipline from {team} — keeping the ball when it matters.",
        "{team} look to release someone in behind — overcooked, goal kick.",
        "Little triangle from {team} near the touchline, neat and tidy.",
        "{team} switch the point of attack, hunting for a gap.",
    ],
    "dead_time": [
        "A brief pause in play as {team} reset their shape.",
        "The tempo drops for a moment with {team} keeping composed.",
        "Play settles down, {team} taking a breath.",
        "A quiet spell — {team} content to keep the ball.",
    ],
    "corner": [
        "Corner to {team} — the box is packed!",
        "{team} win a corner. Everyone forward!",
        "A chance to swing one in for {team}.",
        "{team} have a corner — the keeper is under pressure!",
        "The delivery comes in for {team}!",
    ],
    "save": [
        "What a save! The keeper denies {team}!",
        "Denied! {team} are kept out by a brilliant stop!",
        "Incredible goalkeeping — {team} can't believe it!",
        "The keeper stands tall and {team} are frustrated!",
        "A superb stop to keep {team} out!",
    ],
}

FILLER: list[str] = [
    "Plenty of intensity in this contest, {home} and {away} refusing to give an inch.",
    "The crowd here are right up for it — a cracking atmosphere.",
    "You can sense the tension every time possession changes hands.",
    "{side} look the sharper side at the moment, pressing high up the pitch.",
    "Both benches are screaming for intensity — nobody taking a backward step.",
    "The pitch is lightning fast tonight and both sides are using it.",
    "A real tactical chess match developing in the middle third.",
    "Every 50-50 ball is being contested like a cup final out there.",
    "{side} are knocking on the door here — the momentum is building.",
    "The tempo is relentless — this is top-class football.",
    "{minute} minutes gone and this still feels finely balanced.",
    "Whatever the coaches said at the break, {side} seem to have listened.",
    "Shots standing at {shots_home} for {home} and {shots_away} for {away} — chances are coming.",
    "{possession_sentence}",
    "Neither keeper has been seriously worked yet — but it feels close.",
    "You feel the next goal in this match could decide it.",
]

_POSSESSION_SENTENCES = [
    "Right now it's {possessor} in the ascendant.",
    "The ball keeps finding its way to {possessor}.",
    "{possessor} are seeing plenty of it at the moment.",
]

PLAYER_COLORS = ["skip"]  # reserved for future per-player lines

_openers = [
    "We're underway in the second half — {home} against {away}, what a match this promises to be!",
    "Here we go at a raucous stadium — {home} and {away} set for battle!",
]


def _fmt(line: str, teams: dict, ctx: dict | None = None) -> str:
    """Fill placeholders; anything missing falls back to safe wording."""
    ctx = dict(ctx or {})
    possessor = {"home": teams.get("home", "the home side"),
                 "away": teams.get("away", "the visitors"),
                 "even": "both sides"}.get(ctx.get("possession", "even"),
                                           "both sides")
    base = {
        "home": teams.get("home", "the home side"),
        "away": teams.get("away", "the visitors"),
        "team": teams.get("_last_team", teams.get("home", "the home side")),
        "side": teams.get("_last_side", teams.get("home", "the home side")),
        "minute": ctx.get("minute", 46),
        "scoreline": ctx.get("scoreline", ""),
        "speed_word": ctx.get("speed_word", "measured"),
        "phase_word": {"opening": "early", "midfield": "mid",
                       "closing": "late", "post_goal": "post-goal"}
                    .get(ctx.get("phase", "midfield"), "mid"),
        "shots_home": ctx.get("shots_home", 0),
        "shots_away": ctx.get("shots_away", 0),
        "possession_sentence": _POSSESSION_SENTENCES[
            ctx.get("_pos_i", 0) % len(_POSSESSION_SENTENCES)].format(
            possessor=possessor),
    }
    base.update(ctx)
    return line.format(**base)


class TemplateAnnouncer:
    """Context-aware lines drawn from shuffle bags: grounded and unrepeatable."""

    def __init__(self, teams: dict):
        self.teams = teams
        self._bags = {label: ShuffleBag(pool, seed=i)
                      for i, (label, pool) in enumerate(TEMPLATE.items())}
        self._filler_bag = ShuffleBag(FILLER, seed=99)
        self._opener_bag = ShuffleBag(_openers, seed=5)
        self._pos_i = 0

    def event_line(self, label: str, features: dict) -> tuple[str, str]:
        excitement = {"goal": "huge", "shot": "high", "tackle": "medium",
                      "pass": "low", "dead_time": "calm",
                      "corner": "medium", "save": "high"}.get(label, "medium")
        side = features.get("goal_side")
        if side not in ("home", "away"):
            side = "home" if (features.get("ball_x_frac", 0.5) or 0.5) >= 0.5 else "away"
        self.teams["_last_team"] = self.teams.get(side, "the home side")
        self.teams["_last_side"] = self.teams.get(side, "the home side")
        features = dict(features or {})
        features["_pos_i"] = self._pos_i
        self._pos_i += 1
        line = self._bags[label].draw()
        return _fmt(line, self.teams, features), excitement

    def filler_line(self, features: dict) -> str:
        self.teams["_last_side"] = self.teams.get(
            "home" if (features.get("ball_x_frac", 0.5) or 0.5) >= 0.5 else "away",
            "the home side")
        features = dict(features or {})
        features["_pos_i"] = self._pos_i
        self._pos_i += 1
        return _fmt(self._filler_bag.draw(), self.teams, features)

    def opener(self) -> str:
        return _fmt(self._opener_bag.draw(), self.teams, {})


class GeminiAnnouncer:
    """gemini-1.5-flash: one vivid present-tense line per event, JSON out."""

    SYSTEM = (
        "You are a passionate football play-by-play commentator. Given the match "
        "context and vision features of a 2.5-second window, write ONE short "
        "present-tense commentary line (max {words} words). Excitement must match "
        "the event. Reply ONLY with JSON: "
        '{{"line": "...", "excitement": "huge|high|medium|low|calm"}}'
    )

    def __init__(self, teams: dict):
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            settings.gemini_model,
            system_instruction=self.SYSTEM.format(words=settings.max_commentary_words))
        self.fallback = TemplateAnnouncer(teams)
        self.teams = teams

    def event_line(self, label: str, features: dict) -> tuple[str, str]:
        import google.generativeai as genai
        prompt = (
            f"Event type: {label}. Teams: {self.teams.get('home')} vs {self.teams.get('away')}. "
            f"Match context: minute={features.get('minute')}, score={features.get('scoreline')}, "
            f"phase={features.get('phase')}. "
            f"Vision features: motion={features.get('motion', 0):.1f}, "
            f"ball_speed_px_s={features.get('ball_speed', 0):.0f}, "
            f"players_in_frame={features.get('n_players', 0):.0f}, "
            f"ball_seen={features.get('has_ball', 0):.2f}, "
            f"time_in_goalmouth={features.get('goal_zone_time', 0):.2f}."
        )
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.9, response_mime_type="application/json"),
            )
            data = json.loads(re.sub(r"^```json|```$", "", resp.text.strip(), flags=re.M))
            line = str(data.get("line", "")).strip()
            exc = str(data.get("excitement", "medium")).lower()
            if line and exc in EXCITE_ORDER:
                return line, exc
        except Exception as e:  # noqa: BLE001 - degrade gracefully per event
            print(f"  [gemini] falling back for '{label}': {type(e).__name__}")
        return self.fallback.event_line(label, features)

    def filler_line(self, features: dict) -> str:
        return self.fallback.filler_line(features)

    def opener(self) -> str:
        return self.fallback.opener()


class Gpt2Announcer:
    """GPT-2 (LoRA) supplies the match opener ONLY.

    Its event lines hallucinate teams/players/scores it was never told (it
    inherits the statistical style of its training data), so moment-level
    lines come from the grounded TemplateAnnouncer, which only ever claims
    what the vision/audio pipeline actually detected.
    """

    def __init__(self, teams: dict):
        from app import commentary_gpt2
        self._gpt2 = commentary_gpt2
        self.teams = teams
        self.fallback = TemplateAnnouncer(teams)
        self.gpt2_lines = 0
        self.fallback_lines = 0
        self._used: set[str] = set()

    def event_line(self, label: str, features: dict) -> tuple[str, str]:
        # fine-tuned GPT-2 first; the validator rejects invented facts and
        # the template announcer is the per-line fallback
        try:
            line = self._gpt2.event_line(label, features, self.teams)
            key = line.lower()
            if key in self._used:
                raise RuntimeError("duplicate line")
            self._used.add(key)
            self.gpt2_lines += 1
            return line
        except Exception:  # noqa: BLE001
            self.fallback_lines += 1
            return self.fallback.event_line(label, features)

    def filler_line(self, features: dict) -> str:
        return self.fallback.filler_line(features)

    def opener(self) -> str:
        try:
            return self._gpt2.opener(self.teams)
        except Exception:
            return self.fallback.opener()


def make_announcer(teams: dict):
    if settings.gemini_api_key:
        try:
            a = GeminiAnnouncer(teams)
            print("[llm] using Gemini 1.5 Flash commentary")
            return a
        except Exception as e:  # noqa: BLE001
            print(f"[llm] Gemini unavailable ({type(e).__name__}), trying GPT-2 LoRA")
    try:
        from app import commentary_gpt2
        if commentary_gpt2.available():
            print("[llm] using fine-tuned GPT-2 (LoRA) commentary")
            return Gpt2Announcer(teams)
        print("[llm] no GPT-2 LoRA adapter found — using energetic template announcer")
    except Exception as e:  # noqa: BLE001
        print(f"[llm] GPT-2 LoRA unavailable ({type(e).__name__}: {e}) — using template announcer")
    return TemplateAnnouncer(teams)
