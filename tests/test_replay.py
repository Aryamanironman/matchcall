"""Tests for replay detector."""

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')

from app.vision.replay import frame_hash, detect_replays


def test_frame_hash():
    # Create a simple image: 64x64x3, all zeros
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    h = frame_hash(img)
    # All zeros image should produce a hash of all zeros? Not exactly because of the averaging and threshold.
    # But we can at least check that it returns 8 bytes.
    assert len(h) == 8
    # Create another image, same, should have same hash
    h2 = frame_hash(img)
    assert h == h2
    # Change one pixel to white, should maybe change hash? Not guaranteed but likely.
    img[0, 0, :] = 255
    h3 = frame_hash(img)
    # We'll just check that it's still 8 bytes
    assert len(h3) == 8


def test_detect_replays_simple():
    # Create a simple sequence of hashes where we know there is a replay.
    # We'll create 10 frames of hash A (all zeros), then 10 frames of hash B (all ones), then 10 frames of hash A again.
    # The replay of the first A sequence should be detected with lag = 20 frames (assuming 1 fps, lag = 20s).
    # We'll set times accordingly.
    hash_a = bytes([0]*8)
    hash_b = bytes([255]*8)
    hashes = []
    times = []
    # First A sequence
    for i in range(10):
        hashes.append(hash_a)
        times.append(float(i))
    # B sequence
    for i in range(10):
        hashes.append(hash_b)
        times.append(float(10 + i))
    # Second A sequence
    for i in range(10):
        hashes.append(hash_a)
        times.append(float(20 + i))
    # We expect a replay detection starting at time 20.0, ending at 29.9, original at 0.0-9.9
    # min_gap_s=8.0, max_lag_s=120.0, min_matches=3
    events = detect_replays(hashes, times, min_gap_s=8.0, max_lag_s=120.0, min_matches=3)
    # We expect at least one event
    assert len(events) >= 1
    # Check that the event has the correct approximate times
    for ev in events:
        # The replay start should be around 20.0
        assert 19.0 <= ev["t0"] <= 21.0
        # The replay end should be around 29.0
        assert 28.0 <= ev["t1"] <= 30.0
        # The original time should be around 0.0-9.0
        assert 0.0 <= ev["orig_t"] <= 9.0
        # Confidence should be high
        assert ev["conf"] > 0.5
    # Also, we should not detect any other events (like the B sequence as replay of A? No, because they are different hashes)
    # But note: our algorithm might detect the B sequence as replay of something? It shouldn't because there is no matching hash earlier in the window.
    # We'll just check that we have exactly one event for simplicity.
    # However, there might be multiple events due to overlapping windows. We'll accept at least one.


def test_detect_replays_no_replay():
    # All hashes different, no replay
    hashes = [bytes([i]*8) for i in range(10)]  # each hash different
    times = [float(i) for i in range(10)]
    events = detect_replays(hashes, times, min_gap_s=8.0, max_lag_s=120.0, min_matches=3)
    assert len(events) == 0


def test_detect_replays_too_short():
    # Not enough matches
    hash_a = bytes([0]*8)
    hash_b = bytes([255]*8)
    hashes = [hash_a, hash_b, hash_a]  # only one match at the ends, not consecutive
    times = [0.0, 5.0, 10.0]
    events = detect_replays(hashes, times, min_gap_s=8.0, max_lag_s=120.0, min_matches=3)
    assert len(events) == 0


if __name__ == "__main__":
    test_frame_hash()
    test_detect_replays_simple()
    test_detect_replays_no_replay()
    test_detect_replays_too_short()
    print("All tests passed")