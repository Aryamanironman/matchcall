"""Fusion layer for combining detector outputs."""

from __future__ import annotations

import numpy as np
from typing import List, Dict


def fuse(
    roar_events: List[Dict],
    whistle_events: List[Dict],
    replay_events: List[Dict],
    score_anchors: List[Dict],
    windows: List[Dict],
    *,
    goal_second_signal: bool = True,
) -> List[Dict]:
    """
    Fuse detector outputs to produce refined event labels for windows.

    Parameters
    ----------
    roar_events : list of dict
        Each dict has keys: t, peak, rise, start, end (from audio_rms_profile and roar_events).
    whistle_events : list of dict
        Each dict has keys: t, conf, dur (from detect_whistles).
    replay_events : list of dict
        Each dict has keys: t0, t1, orig_t, conf (from detect_replays).
    score_anchors : list of dict
        Each dict has keys: t, side, minute (from detect_goal_anchors).
    windows : list of dict
        Each dict represents a time window with keys: t (mid time), label (current label), etc.
    goal_second_signal : bool
        If True, require a second signal for goal (precision-first). If False, recall-first.

    Returns
    -------
    list of dict
        Each dict has keys: t, label, tier, side, replay_of, etc.
        label: "goal", "shot", "whistle", "replay", "ambient"
        tier: "confirmed" or "probable"
        side: for goal, the scoring side ("home" or "away")
        replay_of: for replay events, the time of the original event
    """
    # We'll process each window and assign a label based on the detectors.
    # For simplicity, we'll assign labels to windows based on the nearest event in time.
    # We'll create a list of all events with their times and labels, then for each window,
    # pick the closest event within a reasonable threshold (e.g., 2 seconds).
    # If no event is close, we keep the window's current label.

    # Build a list of candidate events from all detectors
    candidates = []  # each element: (time, label, info)

    # Add roar events as potential goal/shot/ambient
    for ev in roar_events:
        t = ev["t"]
        # We'll decide later if it's goal, shot, or ambient based on evidence
        candidates.append((t, "roar_candidate", ev))

    # Add whistle events
    for ev in whistle_events:
        candidates.append((ev["t"], "whistle", ev))

    # Add replay events
    for ev in replay_events:
        # For replay, we label the entire replay interval as replay
        # We'll add the start and end times? For simplicity, we'll add the midpoint.
        t_mid = (ev["t0"] + ev["t1"]) / 2.0
        candidates.append((t_mid, "replay", ev))

    # Add score anchors as goal evidence
    for ev in score_anchors:
        candidates.append((ev["t"], "score_anchor", ev))

    # Sort candidates by time
    candidates.sort(key=lambda x: x[0])

    # For each window, find the closest candidate within 2 seconds
    labeled_windows = []
    for window in windows:
        w_t = window.get("t", window.get("mid", 0.0))  # assume window has 't' or 'mid'
        # Find the index of the closest candidate time
        times = [c[0] for c in candidates]
        import bisect
        pos = bisect.bisect_left(times, w_t)
        # Check the candidate at pos and pos-1
        best_dist = float("inf")
        best_label = None
        best_info = None
        if pos < len(times):
            dist = abs(times[pos] - w_t)
            if dist < best_dist:
                best_dist = dist
                best_label = candidates[pos][1]
                best_info = candidates[pos][2]
        if pos > 0:
            dist = abs(times[pos-1] - w_t)
            if dist < best_dist:
                best_dist = dist
                best_label = candidates[pos-1][1]
                best_info = candidates[pos-1][2]
        # If we found a candidate within 2 seconds, use its label; otherwise keep window's label
        if best_dist < 2.0:
            # Now we need to convert the candidate label to a final label
            # For roar_candidate, we need to decide if it's goal, shot, or ambient
            if best_label == "roar_candidate":
                # Check for supporting evidence within time windows
                # We'll look for score anchors, replay events, or whistle events near this roar
                t_roar = best_info["t"]
                # Goal if: roar exists AND (score anchor within ±15s OR replay within 30s after OR whistle within 10-25s after)
                has_score = any(abs(sa["t"] - t_roar) <= 15.0 for sa in score_anchors)
                # Replay burst within 30s after: replay event with t0 > t_roar and t0 - t_roar <= 30.0
                has_replay = any(
                    ev["t0"] > t_roar and (ev["t0"] - t_roar) <= 30.0 for ev in replay_events
                )
                # Whistle 10-25s after: whistle event with t > t_roar and 10.0 <= (t - t_roar) <= 25.0
                has_whistle = any(
                    ev["t"] > t_roar and 10.0 <= (ev["t"] - t_roar) <= 25.0 for ev in whistle_events
                )
                if goal_second_signal:
                    # Need at least one of the three
                    if has_score or has_replay or has_whistle:
                        # Determine side from score anchor if available, else unknown
                        side = None
                        for sa in score_anchors:
                            if abs(sa["t"] - t_roar) <= 15.0:
                                side = sa.get("side")
                                break
                        labeled_windows.append(
                            {
                                **window,
                                "t": t_roar,
                                "label": "goal",
                                "tier": "confirmed",
                                "side": side,
                            }
                        )
                    else:
                        # Not enough evidence for goal, treat as shot or ambient based on motion?
                        # We'll keep the window's original label for now, but we could set to shot if motion high.
                        labeled_windows.append(
                            {
                                **window,
                                "t": t_roar,
                                "label": window.get("label", "ambient"),
                                "tier": "probable",
                            }
                        )
                else:
                    # Recall-first: treat as goal if roar exists (we already have roar)
                    side = None
                    for sa in score_anchors:
                        if abs(sa["t"] - t_roar) <= 15.0:
                            side = sa.get("side")
                            break
                    labeled_windows.append(
                        {
                            **window,
                            "t": t_roar,
                            "label": "goal",
                            "tier": "probable",
                            "side": side,
                        }
                    )
            elif best_label == "whistle":
                labeled_windows.append(
                    {
                        **window,
                        "t": best_info["t"],
                        "label": "whistle",
                        "tier": "confirmed",
                        "conf": best_info.get("conf", 0.0),
                    }
                )
            elif best_label == "replay":
                labeled_windows.append(
                    {
                        **window,
                        "t": best_info["t0"],  # use start time
                        "label": "replay",
                        "tier": "confirmed",
                        "replay_of": best_info["orig_t"],
                        "conf": best_info.get("conf", 0.0),
                    }
                )
            elif best_label == "score_anchor":
                # Score anchor alone is not enough for goal without roar? We'll treat as goal only if we have roar nearby?
                # But we already processed roar_candidate separately. This case might not happen because we gave roar_candidate a label.
                # For safety, we'll treat as ambient.
                labeled_windows.append(
                    {
                        **window,
                        "t": best_info["t"],
                        "label": "ambient",
                        "tier": "probable",
                    }
                )
            else:
                # Fallback
                labeled_windows.append({**window, "t": best_info["t"], "label": best_label})
        else:
            # No close candidate, keep the window as is but add a default tier
            labeled_windows.append({**window, "tier": "probable"})

    return labeled_windows