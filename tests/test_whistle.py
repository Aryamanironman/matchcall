import numpy as np
import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')

from app.vision.whistle import detect_whistles

def test_whistle_detection_basic():
    # Create a 3 kHz tone for 0.5 seconds at 8 kHz sample rate
    sr = 8000
    t = np.linspace(0, 0.5, int(sr * 0.5), False)
    tone = np.sin(2 * np.pi * 3000 * t)
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    signal = tone + noise
    # Ensure it's float32
    signal = signal.astype(np.float32)
    # Call detect_whistles
    detections = detect_whistles(signal, sr)
    # We expect at least one detection
    assert len(detections) >= 1, f"Expected at least one whistle detection, got {detections}"
    # Check that the detection is around 0.25 seconds (midpoint)
    # We'll just check that the time is within the signal range
    for det in detections:
        assert 0 <= det["t"] <= 0.5
        assert det["conf"] > 0
        assert det["dur"] > 0

def test_no_whistle():
    # White noise
    sr = 8000
    signal = np.random.randn(sr).astype(np.float32)  # 1 second of noise
    detections = detect_whistles(signal, sr)
    assert len(detections) == 0, f"Expected no detections, got {detections}"

def test_chant_not_detected():
    # Create a sawtooth wave at 400 Hz (chant-like)
    sr = 8000
    t = np.linspace(0, 1, sr, False)
    # Sawtooth: 2 * (t * freq - np.floor(t * freq + 0.5))
    signal = 2 * (t * 400 - np.floor(t * 400 + 0.5))
    signal = signal.astype(np.float32)
    detections = detect_whistles(signal, sr)
    assert len(detections) == 0, f"Chant should not be detected as whistle, got {detections}"

def test_merge_close_whistles():
    # Two whistles 0.3 seconds apart should merge into one
    sr = 8000
    # First whistle: 0.4 seconds of 3 kHz tone starting at 0.2s
    # Second whistle: 0.4 seconds of 3 kHz tone starting at 0.5s (gap 0.3s)
    total_duration = 1.0  # seconds
    t = np.linspace(0, total_duration, int(sr * total_duration), False)
    signal = np.zeros_like(t)
    # First whistle
    start1 = int(0.2 * sr)
    end1 = int((0.2 + 0.4) * sr)
    t1 = np.linspace(0, 0.4, end1 - start1, False)
    signal[start1:end1] += np.sin(2 * np.pi * 3000 * t1)
    # Second whistle
    start2 = int(0.5 * sr)
    end2 = int((0.5 + 0.4) * sr)
    t2 = np.linspace(0, 0.4, end2 - start2, False)
    signal[start2:end2] += np.sin(2 * np.pi * 3000 * t2)
    # Add a little noise
    signal += 0.01 * np.random.randn(len(t))
    signal = signal.astype(np.float32)
    detections = detect_whistles(signal, sr)
    # Expect exactly one detection (merged)
    assert len(detections) == 1, f"Expected one merged detection, got {detections}"
    # The detection time should be somewhere between the two whistles
    assert 0.2 <= detections[0]["t"] <= 0.9
    assert detections[0]["conf"] > 0
    assert detections[0]["dur"] > 0

if __name__ == "__main__":
    test_whistle_detection_basic()
    test_no_whistle()
    test_chant_not_detected()
    test_merge_close_whistles()
    print("All tests passed")