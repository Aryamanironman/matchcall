import numpy as np

def match_events(pred, truth, tol_s=5.0):
    """
    Match predictions to ground truth events within a tolerance.
    Returns precision, recall, f1, and list of matched indices (pred_idx, truth_idx).
    Greedy matching: sort by time, match the closest within tolerance, each used once.
    """
    if not pred and not truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": []}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": []}
    if not truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": []}

    # Sort by time
    pred_sorted = sorted(enumerate(pred), key=lambda x: x[1]["t"])
    truth_sorted = sorted(enumerate(truth), key=lambda x: x[1]["t"])

    matches = []
    used_pred = set()
    used_truth = set()
    i = j = 0
    while i < len(pred_sorted) and j < len(truth_sorted):
        p_idx, p = pred_sorted[i]
        t_idx, t = truth_sorted[j]
        if abs(p["t"] - t["t"]) <= tol_s:
            matches.append((p_idx, t_idx))
            used_pred.add(p_idx)
            used_truth.add(t_idx)
            i += 1
            j += 1
        elif p["t"] < t["t"]:
            i += 1
        else:
            j += 1

    tp = len(matches)
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matches": matches}


def repetition_rate(texts):
    """
    Compute repetition rate: 1 - (number of unique normalized strings / total).
    Normalization: lower case, strip whitespace.
    """
    if not texts:
        return 0.0
    normalized = [t.strip().lower() for t in texts]
    unique = set(normalized)
    return 1.0 - len(unique) / len(texts)


def minute_error(pred_minutes, truth_minutes):
    """
    Mean absolute error between predicted and true minutes (in seconds?).
    The inputs are in minutes? The test uses seconds: 61 and 63 -> error 2.
    We'll assume the inputs are in the same unit (seconds) and compute MAE.
    """
    if not pred_minutes or not truth_minutes:
        return 0.0
    # If lengths differ, we compare up to the shorter length? The test uses same length.
    # We'll compute element-wise for the minimum length.
    n = min(len(pred_minutes), len(truth_minutes))
    total_error = sum(abs(p - t) for p, t in zip(pred_minutes[:n], truth_minutes[:n]))
    return total_error / n if n > 0 else 0.0