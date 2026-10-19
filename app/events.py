"""Shared data structures for windows, events and commentary segments."""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Window:
    """A short slice of match time scored by vision features."""
    t0: float
    t1: float
    motion: float = 0.0            # mean frame-diff score (0..255-ish)
    ball_speed: float = 0.0        # px/s of ball centroid in analysis frames
    n_players: float = 0.0         # mean player detections
    has_ball: float = 0.0          # fraction of frames with a ball detection
    goal_zone_time: float = 0.0    # fraction of frames with ball in a goal zone
    cls_probs: dict = field(default_factory=dict)  # from ResNet18
    label: str = "pass"
    conf: float = 0.0
    goal_side: Optional[str] = None    # "home"/"away" when score bug confirmed

    def mid(self) -> float:
        return (self.t0 + self.t1) / 2.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    """One piece of commentary audio to place on the timeline."""
    t0: float
    kind: str                      # "event" | "filler" | "chat"
    event_label: Optional[str]
    excitement: str                # huge | high | medium | low | calm
    text: str
    audio_path: Optional[str] = None
    audio_dur: float = 0.0
    speaker: str = "A"             # A = play-by-play, B = color

    def to_dict(self) -> dict:
        return asdict(self)
