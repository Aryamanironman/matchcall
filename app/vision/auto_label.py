"""Weak-supervision auto-labeler: turns heuristic window scores into labels.

Design rule for ANY match: a label the writer will shout about must have
converging evidence, not just "top quartile of this video".

  goal       <- ball in goalmouth zone + sustained presence + high motion
                AND (when audio exists) a crowd surge near the window
                (broadcast feeds are loudness-normalized; goals raise the
                crowd only ~1.3-2x, but they DO raise it)
  shot       <- real ball speed (absolute floor) + goalmouth involvement,
                capped globally per match
  tackle     <- motion spike + ball speed dip + players clustered
  dead_time  <- motion below gate
  pass       <- everything else

Global caps keep event classes honest: vision heuristics over-fire on busy
matches, so the strongest N candidates win and the rest are demoted.
"""
from __future__ import annotations

import numpy as np

from app.config import settings
from app.events import Window
from app.vision.audio import audio_evidence

GOAL_CAP = 6       # a match has 0-6 goals; accept at most that many
SHOT_CAP = 24      # ~top 24 chances max, even in a chaotic game
CORNER_CAP = 16    # corners are frequent; keep the strongest ones only
# Calibrated on real broadcast lulls vs the goal-laden slice of this feed:
# quiet passages top out at ~155 px/s ball speed, genuine strikes run 147-315.
BALL_SPEED_SHOT = 140.0
ROAR_FLOOR = 1.25  # crowd rise confirming a goalmouth goal (zone shown)
ROAR_STRIKE = 1.7  # stricter: strike + roar -> goal when celebration cuts hide the zone


def _aud_ev(rise, mid_abs: float, clip_t0: float) -> float:
    if rise is None:
        return 1.0
    return audio_evidence(rise, mid_abs - clip_t0)


def enforce_caps(windows: list[Window], rise=None, clip_t0: float = 0.0) -> None:
    """Keep only the strongest GOAL_CAP / SHOT_CAP candidates; demote the rest."""
    goals = [w for w in windows if w.label == "goal"]
    shots = [w for w in windows if w.label == "shot"]
    corners = [w for w in windows if w.label == "corner"]

    def gscore(w: Window) -> float:
        return (w.goal_zone_time * 2.0 + w.has_ball
                + max(0.0, _aud_ev(rise, w.mid(), clip_t0) - 1.0))

    def sscore(w: Window) -> float:
        return (w.ball_speed / 1000.0 + w.goal_zone_time * 1.5
                + max(0.0, _aud_ev(rise, w.mid(), clip_t0) - 1.0))

    def cscore(w: Window) -> float:
        return w.corner_time * 2.0 + w.has_ball + w.motion / 100.0

    for w in sorted(goals, key=gscore)[GOAL_CAP:]:
        w.label, w.conf = "pass", 0.4
    for w in sorted(shots, key=sscore)[SHOT_CAP:]:
        w.label, w.conf = "pass", 0.4
    for w in sorted(corners, key=cscore)[CORNER_CAP:]:
        w.label, w.conf = "pass", 0.4


def label_windows(windows: list[Window], rise=None, clip_t0: float = 0.0,
                  audio_usable: bool = False) -> list[Window]:
    if not windows:
        return windows
    motion = np.array([w.motion for w in windows])
    m_hi = float(np.quantile(motion, 0.75))
    gate = settings.motion_dead_gate
    # score-bug anchors (gold evidence): windows within +-15s of a bug flip get
    # promoted to goal; windows with NO nearby flip may only stay goal if they
    # pass the strict no-bug evidence chain (zone + roar). Flips outrank audio.
    from app.config import SCORE_ANCHORS
    anchors = SCORE_ANCHORS or []

    def near_flip(t_abs: float, span: float = 15.0):
        for a in anchors:
            if a["t"] - span <= t_abs <= a["t"] + span:
                return a
        return None

    for i, w in enumerate(windows):
        zone = w.goal_zone_time
        roar = _aud_ev(rise, w.mid(), clip_t0) if audio_usable else 1.0
        flip = near_flip(w.mid())
        if w.motion < gate:
            w.label, w.conf = "dead_time", min(0.95, 0.6 + (gate - w.motion) / 10)
        elif flip is not None:
            # gold: the score bug flipped here — call it, name the side
            w.label = "goal"
            w.conf = max(0.75, min(0.9, 0.55 + zone))
            w.goal_side = flip.get("side")
        elif (zone >= 0.20 and w.has_ball >= 0.5 and w.motion >= m_hi
              and (not audio_usable or roar >= ROAR_FLOOR)):
            w.label, w.conf = "goal", min(0.9, 0.55 + zone)
        elif (w.ball_speed >= BALL_SPEED_SHOT and roar >= ROAR_STRIKE
              and w.has_ball >= 0.15):
            # celebration cuts hide the goalmouth zone on real broadcasts:
            # a genuine strike + a crowd surge IS a goal even without zone time
            w.label, w.conf = "goal", min(0.9, 0.55 + roar / 10.0)
        elif (w.ball_speed >= BALL_SPEED_SHOT
              and (zone >= 0.05 or w.has_ball >= 0.2)
              and (not audio_usable or roar >= ROAR_FLOOR)):
            w.label, w.conf = "shot", min(0.8, 0.5 + zone)
        elif w.corner_time >= 0.30 and w.has_ball >= 0.20 and w.motion >= gate:
            # ball camping in a corner with the box populated -> corner
            w.label, w.conf = "corner", min(0.75, 0.45 + w.corner_time)
        elif w.motion >= m_hi and w.ball_speed < BALL_SPEED_SHOT * 0.5 \
                and w.n_players >= 6:
            w.label, w.conf = "tackle", 0.55
        else:
            w.label, w.conf = "pass", 0.5
    # when anchors exist, vision-only goals far from any flip are demoted to
    # shots (the bug is ground truth; tracker noise is not)
    if anchors:
        for w in windows:
            if w.label == "goal" and w.goal_side is None \
                    and near_flip(w.mid()) is None:
                w.label, w.conf = "shot", 0.7
    # local smoothing: isolated labels inside a run of another class get merged
    labels = [w.label for w in windows]
    for i in range(1, len(labels) - 1):
        if labels[i] != labels[i - 1] and labels[i] != labels[i + 1] \
                and labels[i - 1] == labels[i + 1] and labels[i] not in ("goal",):
            windows[i].label = labels[i - 1]
    # goal + replay dedupe (AFTER smoothing, so smoothing can't undo it):
    # a replayed strike lands 20-40s after its goal with the roar still
    # ringing. Real restarts take >45s, so a goal starting within 45s of the
    # previous goal's end is a replay -> shot.
    last_goal_end = None
    for w in windows:
        if w.label != "goal":
            continue
        if last_goal_end is not None and w.t0 - last_goal_end < 45.0:
            w.label, w.conf = "shot", 0.7
        else:
            last_goal_end = w.t1
    enforce_caps(windows, rise, clip_t0)
    return windows


def merge_events(windows: list[Window], max_gap: float = 4.0) -> list[Window]:
    """Collapse consecutive same-label windows into single events."""
    events: list[Window] = []
    for w in windows:
        if events and events[-1].label == w.label and (w.t0 - events[-1].t1) <= max_gap:
            prev = events[-1]
            prev.t1 = w.t1
            prev.motion = max(prev.motion, w.motion)
            prev.ball_speed = max(prev.ball_speed, w.ball_speed)
            prev.goal_zone_time = max(prev.goal_zone_time, w.goal_zone_time)
            prev.conf = max(prev.conf, w.conf)
            if getattr(w, "goal_side", None):
                prev.goal_side = w.goal_side
        else:
            events.append(Window(**{**w.to_dict()}))
    return events
