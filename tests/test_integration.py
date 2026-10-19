"""Integration test for whistle, replay, and fusion detectors."""

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')

from app.vision.whistle import detect_whistles
from app.vision.replay import frame_hash, detect_replays
from app.vision.fusion import fuse


def test_integration():
    """Test that all three detectors work together."""
    print("Testing integration of whistle, replay, and fusion detectors...")
    
    # 1. Test whistle detector with a clear whistle signal
    sr = 8000
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), False)
    # 3 kHz whistle tone
    whistle_signal = np.sin(2 * np.pi * 3000 * t) + 0.05 * np.random.randn(len(t))
    whistle_signal = whistle_signal.astype(np.float32)
    
    whistle_detections = detect_whistles(whistle_signal, sr)
    print(f"Whistle detections: {len(whistle_detections)}")
    assert len(whistle_detections) > 0, "Should detect whistle in 3kHz tone"
    assert whistle_detections[0]["conf"] > 0.5, "Whistle detection should have good confidence"
    
    # 2. Test replay detector with a simple repeating pattern
    # Create a simple hash pattern: ABCABC where A,B,C are different hashes
    hash_a = bytes([0]*8)
    hash_b = bytes([85]*8)  # 01010101 pattern
    hash_c = bytes([170]*8) # 10101010 pattern
    
    hashes = [hash_a, hash_b, hash_c, hash_a, hash_b, hash_c]  # ABCABC
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # 1 second apart
    
    replay_events = detect_replays(hashes, times, min_gap_s=2.0, max_lag_s=4.0, min_matches=2)
    print(f"Replay events: {len(replay_events)}")
    # Should detect the replay of ABC starting at position 3
    assert len(replay_events) > 0, "Should detect replay in ABCABC pattern"
    assert replay_events[0]["conf"] > 0.5, "Replay detection should have good confidence"
    
    # 3. Test fusion with some dummy data
    # Create some dummy events for fusion
    roar_events = [{"t": 10.0, "peak": 1.0, "rise": 3.0, "start": 9.0, "end": 11.0}]
    # Use the whistle detection we already have
    # Use the replay detection we already have
    score_anchors = [{"t": 12.0, "side": "home", "minute": 0}]
    # Create some dummy windows
    windows = [
        {"t": 10.0, "label": "pass", "motion": 0.1, "mid": 10.0},
        {"t": 15.0, "label": "pass", "motion": 0.1, "mid": 15.0},
        {"t": 20.0, "label": "pass", "motion": 0.1, "mid": 20.0},
    ]
    
    fused_results = fuse(roar_events, whistle_detections, replay_events, score_anchors, windows, goal_second_signal=True)
    print(f"Fused results: {len(fused_results)} windows")
    assert len(fused_results) == len(windows), "Should return same number of windows"
    # Each result should have a tier
    for result in fused_results:
        assert "tier" in result, "Each result should have a tier"
        assert result["tier"] in ["confirmed", "probable"], "Tier should be valid"
    
    print("All integration tests passed!")


if __name__ == "__main__":
    test_integration()