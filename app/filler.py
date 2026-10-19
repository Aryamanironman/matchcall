"""Filler engine: no dead air. Rotating ambient lines during quiet stretches."""
from __future__ import annotations

from app.config import settings
from app.events import Segment


def plan_fillers(events: list[Segment], total_dur: float, announcer,
                 features_lookup) -> list[Segment]:
    """Insert filler segments so narration gaps never exceed min_gap_for_filler.

    events: already-sorted event segments (only those with times).
    features_lookup: fn(t) -> feature dict of nearest window.
    """
    out: list[Segment] = []
    min_gap = settings.min_gap_for_filler
    # opening line
    if not events or events[0].t0 > 1.5:
        out.append(Segment(t0=0.2, kind="filler", event_label=None,
                           excitement="medium", text=announcer.opener()))
    anchors = [(e.t0, e) for e in events]
    last_t = 0.2 if out else 0.0

    def add_fillers_between(a: float, b: float):
        gap = b - a
        if gap < min_gap:
            return
        # one filler around the middle of long quiet stretches; two for huge gaps
        n = 1 if gap < min_gap * 2.5 else 2
        for k in range(n):
            t = a + gap * (k + 1) / (n + 1)
            feats = features_lookup(t)
            out.append(Segment(t0=t, kind="filler", event_label=None,
                               excitement="medium", text=announcer.filler_line(feats)))

    for t, _e in anchors:
        add_fillers_between(last_t, t)
        last_t = t
    if total_dur - last_t > min_gap:
        add_fillers_between(last_t, total_dur)
    out.extend(events)
    out.sort(key=lambda s: (s.t0, 0 if s.kind == "event" else 1))
    # avoid overlaps: if two segments collide within 1.2s, drop the filler
    cleaned: list[Segment] = []
    for s in out:
        if cleaned and s.t0 - cleaned[-1].t0 < 1.2:
            if s.kind == "filler":
                continue
            if cleaned[-1].kind == "filler":
                cleaned.pop()
        cleaned.append(s)
    return cleaned
