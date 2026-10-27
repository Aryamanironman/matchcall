"""Evaluate event spotting against the curated ARG-FRA reference moments.

This is a preliminary, reference-based evaluation, not a full SoccerNet-style
event-spotting benchmark. The reference is the 10 curated moments in
highlights_v4.json; the detected events come from the corrected full-match
run. Goal precision/recall is checked separately against the known goal
minutes (80', 81') in this recording.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
JOB = REPO / "output" / "experiments" / "full_match_job.json"
REF = REPO / "output" / "commented_highlights" / "highlights_v4.json"
YOLO_METRICS = REPO / "models" / "yolo26" / "yolo26n_v1_640" / "val_metrics.json"
OUT = REPO / "output" / "experiments" / "event_spotting.json"

MINUTE_OFFSET = 35.33
TOLERANCE_MIN = 1.0
GOAL_MINUTES = [80, 81]

LINE = re.compile(r"^\[(\d+):(\d+)\]\s+\(([^|]+)\|")
KINDS = ("goal", "shot", "tackle", "corner", "pass", "dead_time")


def match_minute(mm: int, ss: int) -> float:
    return MINUTE_OFFSET + mm + ss / 60.0


def main() -> None:
    job = json.loads(JOB.read_text(encoding="utf-8-sig"))
    detected = []
    for line in job.get("transcript", []):
        m = LINE.match(line)
        if not m:
            continue
        kind = m.group(3).strip()
        if kind not in KINDS:
            continue
        detected.append({
            "kind": kind,
            "clip_minute": int(m.group(1)) + int(m.group(2)) / 60.0,
            "match_minute": match_minute(int(m.group(1)), int(m.group(2))),
            "text": line,
        })

    ref = json.loads(REF.read_text(encoding="utf-8"))
    reference = []
    for moment in ref.get("moments", []):
        minute = moment.get("minute")
        kind = moment.get("kind")
        if minute is None or kind not in KINDS:
            continue
        reference.append({"kind": kind, "minute": float(minute),
                          "team": moment.get("team")})

    used = set()
    matches = []
    unmatched = []
    for r in sorted(reference, key=lambda x: x["minute"]):
        best = None
        best_dt = None
        for i, d in enumerate(detected):
            if i in used or d["kind"] != r["kind"]:
                continue
            dt = abs(d["match_minute"] - r["minute"])
            if best_dt is None or dt < best_dt:
                best, best_dt = i, dt
        if best is not None and best_dt is not None and best_dt <= TOLERANCE_MIN:
            used.add(best)
            matches.append({
                "reference_minute": r["minute"],
                "reference_kind": r["kind"],
                "detected_minute": round(detected[best]["match_minute"], 2),
                "delta_s": round(best_dt * 60.0, 1),
            })
        else:
            unmatched.append(r)

    by_kind_ref = Counter(r["kind"] for r in reference)
    by_kind_det = Counter(d["kind"] for d in detected)
    by_kind_match = Counter(m["reference_kind"] for m in matches)
    recall = {
        kind: round(by_kind_match[kind] / by_kind_ref[kind], 3)
        for kind in KINDS if by_kind_ref[kind]
    }

    goal_detected = [d for d in detected if d["kind"] == "goal"]
    goal_tp = sum(
        1 for g in GOAL_MINUTES
        if any(abs(d["match_minute"] - g) <= TOLERANCE_MIN
               for d in goal_detected))
    goal_precision = (goal_tp / len(goal_detected)) if goal_detected else 0.0
    goal_recall = goal_tp / len(GOAL_MINUTES)

    yolo = {}
    if YOLO_METRICS.exists():
        yolo = json.loads(YOLO_METRICS.read_text(encoding="utf-8"))

    out = {
        "source_job": job.get("id"),
        "source_video": job.get("result", {}).get("video"),
        "tolerance_minutes": TOLERANCE_MIN,
        "minute_offset": MINUTE_OFFSET,
        "reference_moments": len(reference),
        "detected_event_lines": len(detected),
        "reference_by_kind": dict(by_kind_ref),
        "detected_by_kind": dict(by_kind_det),
        "matched_by_kind": dict(by_kind_match),
        "recall_by_kind": recall,
        "recall_overall": round(len(matches) / max(1, len(reference)), 3),
        "timing_mae_s": round(statistics.mean(
            [m["delta_s"] for m in matches]), 1) if matches else None,
        "matches": matches,
        "unmatched_reference": unmatched,
        "goal_spotting": {
            "gold_minutes": GOAL_MINUTES,
            "detected_minutes": [round(d["match_minute"], 2)
                                 for d in goal_detected],
            "true_positives": goal_tp,
            "precision": round(goal_precision, 3),
            "recall": round(goal_recall, 3),
        },
        "object_detection_yolo26n": {
            "mAP50": yolo.get("mAP50"),
            "mAP50_95": yolo.get("mAP50_95"),
            "precision": yolo.get("precision"),
            "recall": yolo.get("recall"),
            "per_class_mAP50_95": yolo.get("per_class_mAP50_95"),
        },
        "limitations": (
            "Reference moments come from the curated v4 highlight reel, not "
            "an independent exhaustive annotation. Recall is therefore a "
            "consistency check against those moments; event precision for "
            "shots/tackles/corners is not measured. Goal precision/recall uses "
            "the known 80' and 81' goals in this recording. Object-detection "
            "mAP is a separate Player/Ball/Referee metric, not event spotting."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
