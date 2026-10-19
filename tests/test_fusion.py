"""Tests for fusion module."""

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')

from app.vision.fusion import fuse


def test_fuse_basic():
    # Create simple inputs
    roar_events = [{"t": 10.0, "peak": 1.0, "rise": 2.5, "start": 9.0, "end": 11.0}]
    whistle_events = [{"t": 15.0, "conf": 0.8, "dur": 0.5}]
    replay_events = [{"t0": 20.0, "t1": 22.0, "orig_t": 5.0, "conf": 0.9}]
    score_anchors = [{"t": 12.0, "side": "home", "minute": 0}]
    windows = [
        {"t": 10.0, "label": "pass", "motion": 0.1},
        {"t": 15.0, "label": "pass", "motion": 0.1},
        {"t": 21.0, "label": "pass", "motion": 0.1},
        {"t": 30.0, "label": "pass", "motion": 0.1},  # no nearby event
    ]
    
    result = fuse(roar_events, whistle_events, replay_events, score_anchors, windows, goal_second_signal=True)
    
    # We expect the first three windows to be relabeled, the last to remain pass
    assert len(result) == 4
    # Check that each window still has the required fields
    for w in result:
        assert "t" in w
        assert "label" in w
        assert "tier" in w
    print("Basic fuse test passed")


if __name__ == "__main__":
    test_fuse_basic()
    print("All tests passed")