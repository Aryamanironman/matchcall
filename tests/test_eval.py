import sys
sys.path.insert(0, r'C:\Users\asus\football-commentator')

from app.eval import match_events, repetition_rate, minute_error

def test_match_events_basic():
    # Two predictions: one correct (within 5s), one far off
    pred = [{"t": 100.0, "label": "goal"}, {"t": 1000.0, "label": "goal"}]
    truth = [{"t": 102.0, "label": "goal"}]  # within 5s of first pred
    result = match_events(pred, truth, tol_s=5.0)
    assert result["precision"] == 0.5  # 1 TP, 1 FP -> 1/(1+1)=0.5
    assert result["recall"] == 1.0     # 1 TP, 0 FN -> 1/1=1.0
    assert result["f1"] == 2 * 0.5 * 1.0 / (0.5 + 1.0)  # 0.666...
    assert len(result["matches"]) == 1
    assert result["matches"][0] == (0, 0)  # first pred matches first truth

def test_match_events_no_match():
    pred = [{"t": 0.0, "label": "goal"}]
    truth = [{"t": 100.0, "label": "goal"}]
    result = match_events(pred, truth, tol_s=5.0)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["matches"] == []

def test_repetition_rate():
    texts = ["x y.", "x y.", "z."]
    # normalized: we lower and strip? The function should define normalization.
    # We assume the function does: lower, strip, and remove extra spaces.
    # For simplicity, we test with the given example.
    # The repetition rate is 1 - (unique / total)
    # Here we have 2 unique ("x y.", "z.") out of 3 -> 1 - 2/3 = 1/3
    assert abs(repetition_rate(texts) - 1.0/3.0) < 1e-9

def test_minute_error():
    pred_minutes = [61.0]  # 1 minute and 1 second -> 61 seconds
    truth_minutes = [63.0] # 1 minute and 3 seconds -> 63 seconds
    assert minute_error(pred_minutes, truth_minutes) == 2.0

if __name__ == "__main__":
    test_match_events_basic()
    test_match_events_no_match()
    test_repetition_rate()
    test_minute_error()
    print("All tests passed")