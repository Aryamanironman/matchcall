"""Live training, demo and ClearML state for the web demo and report.

Reads files only - no network, no GPU. The web demo polls this while the
Aryaman YOLOv5lu run is in progress; ``scripts/build_report.py`` uses the same
functions for the static showcase page.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from app.config import settings

PROJECT = "Football Commentator"
PROJECT_ID = "ef18bd575abc408591532be2ac8f39b8"
PROJECT_URL = f"https://app.clear.ml/projects/{PROJECT_ID}"
YOLO_TASK_ID = "f020de2dc69443b585b2d28d56f9994a"
TASK_URL = f"{PROJECT_URL}/tasks/{YOLO_TASK_ID}/output/log"

MODELS = settings.models_dir
OUT = settings.out_root
YOLO_ROOT = MODELS / "aryaman_yolo"
YOLO_RUN = YOLO_ROOT / "aryaman_yolov5lu_football"
# The original run wrote train.log; the resumed run appends to
# train_resume.log. Follow whichever was written last so a live run never
# reads as "stalled".
_LOG_CANDIDATES = [YOLO_ROOT / "train_resume.log", YOLO_ROOT / "train.log"]
YOLO_LOG = max((p for p in _LOG_CANDIDATES if p.exists()),
               key=lambda p: p.stat().st_mtime,
               default=YOLO_ROOT / "train.log")
YOLO_ERR = YOLO_ROOT / ("train_resume.err.log"
                        if YOLO_LOG.name.startswith("train_resume")
                        else "train.err.log")
_APP_ROOT = Path(__file__).resolve().parents[1]


def _v4_states() -> list[Path]:
    """Newest v4 trainer state in the main checkout or the worktree."""
    states: list[tuple[int, Path]] = []
    for root in (MODELS / "gpt2_lora_v4", _APP_ROOT / "models" / "gpt2_lora_v4"):
        if not root.exists():
            continue
        for p in root.glob("checkpoint-*/trainer_state.json"):
            try:
                step = int(p.parent.name.split("-", 1)[1])
            except (IndexError, ValueError):
                step = 0
            states.append((step, p))
        states.append((10 ** 9, root / "trainer_state.json"))
    states.sort(key=lambda s: s[0], reverse=True)
    return [p for _s, p in states]


# The v4 adapter is trained on SoccerReplay-1988; fall back to v3 until its
# first checkpoint exists. The checkpoint may live in the worktree while
# settings.models_dir points at the main checkout, so try both.
_LLM_STATES = _v4_states() + [
    MODELS / "gpt2_lora_v3" / "checkpoint-1125" / "trainer_state.json",
    _APP_ROOT / "models" / "gpt2_lora_v3" / "checkpoint-1125" / "trainer_state.json",
    settings.gpt2_lora_dir / "checkpoint-1125" / "trainer_state.json",
]
LLM_STATE = next((p for p in _LLM_STATES if p.exists()), _LLM_STATES[0])
LLM_DIR = LLM_STATE.parents[1]
_CLEARML_CANDIDATES = [
    LLM_DIR / "clearml_task.json",
    MODELS / "gpt2_lora_v4" / "clearml_task.json",
    _APP_ROOT / "models" / "gpt2_lora_v4" / "clearml_task.json",
    MODELS / "gpt2_lora_v3" / "clearml_task.json",
    _APP_ROOT / "models" / "gpt2_lora_v3" / "clearml_task.json",
]
LLM_CLEARML = next((p for p in _CLEARML_CANDIDATES if p.exists()),
                   _CLEARML_CANDIDATES[0])
DEMO_DIR = OUT / "commented_highlights"
SNAPSHOT = OUT / "clearml_snapshot.json"
LLM_EVAL = OUT / "llm_eval.json"

# files the demo may serve from the training run directory
ASSETS = {
    "results.csv", "results.png", "confusion_matrix.png",
    "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png",
    "labels.jpg", "labels_correlogram.jpg", "train_batch0.jpg",
    "train_batch1.jpg", "train_batch2.jpg", "val_batch0_pred.jpg",
    "val_batch0_labels.jpg", "args.yaml",
}

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _tail(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return _ANSI.sub("", "\n".join(lines[-n:]))


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(row: dict, key: str):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def yolo_status() -> dict:
    rows: list[dict] = []
    csv_path = YOLO_RUN / "results.csv"
    if csv_path.exists():
        try:
            with csv_path.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    rows.append({
                        "epoch": int(float(r["epoch"])),
                        "box_loss": _num(r, "train/box_loss"),
                        "cls_loss": _num(r, "train/cls_loss"),
                        "dfl_loss": _num(r, "train/dfl_loss"),
                        "precision": _num(r, "metrics/precision(B)"),
                        "recall": _num(r, "metrics/recall(B)"),
                        "mAP50": _num(r, "metrics/mAP50(B)"),
                        "mAP50_95": _num(r, "metrics/mAP50-95(B)"),
                        "val_box_loss": _num(r, "val/box_loss"),
                        "val_cls_loss": _num(r, "val/cls_loss"),
                        "val_dfl_loss": _num(r, "val/dfl_loss"),
                    })
        except (OSError, ValueError):
            rows = []

    log = _tail(YOLO_LOG, 140)
    err = _tail(YOLO_ERR, 16)
    args = _yaml(YOLO_RUN / "args.yaml")
    total = int(args.get("epochs") or 100)
    best = max(rows, key=lambda r: r["mAP50_95"] or -1) if rows else None
    last = rows[-1] if rows else None
    done = any(k in log for k in ("done ->", "epochs completed",
                                  "Stopping training early"))
    best_pt = YOLO_RUN / "weights" / "best.pt"
    mtime = YOLO_LOG.stat().st_mtime if YOLO_LOG.exists() else 0
    recent = bool(mtime) and (time.time() - mtime) < 1200
    if done:
        status = "done"
    elif recent:
        status = "running" if rows or "Starting training" in log else "starting"
    elif mtime:
        status = "stalled"
    else:
        status = "queued"

    epoch_hits = re.findall(r"^\s*(\d+)/(\d+)\s", log, re.M)
    current = len(rows)
    if epoch_hits:
        current = max(current, int(epoch_hits[-1][0]))

    data_yaml = _yaml(Path(str(args.get("data", ""))))
    names = data_yaml.get("names") or ["Player", "Ball", "Referee"]

    def count(split: str):
        p = Path(str(data_yaml.get("path", ""))) / split
        try:
            return sum(1 for _ in p.iterdir())
        except OSError:
            return None

    assets = {n: f"/api/training/asset/{n}" for n in sorted(ASSETS)
              if (YOLO_RUN / n).exists()}
    return {
        "status": status,
        "task_url": TASK_URL,
        "project_url": PROJECT_URL,
        "run_dir": str(YOLO_RUN),
        "model": str(args.get("model") or "yolov5lu.pt"),
        "architecture": "YOLOv5lu (FootballVision main.py architecture)",
        "epochs_total": total,
        "epochs_done": len(rows),
        "current_epoch": min(current, total) if status != "done" else len(rows),
        "progress": round(min(1.0, len(rows) / total) if total else 0.0, 4),
        "best_epoch": best["epoch"] if best else None,
        "best_metrics": best,
        "last_metrics": last,
        "rows": rows,
        "weights_ready": best_pt.exists(),
        "weights_bytes": best_pt.stat().st_size if best_pt.exists() else 0,
        "dataset": {
            "name": "Roboflow football-players V1",
            "classes": list(names) if isinstance(names, (list, tuple)) else names,
            "train": count("images/train"),
            "val": count("images/test"),
            "source": "data/V1",
        },
        "args": {k: args.get(k) for k in
                 ("model", "epochs", "batch", "imgsz", "patience", "device",
                  "optimizer", "seed") if k in args},
        "assets": assets,
        "log_tail": log,
        "err_tail": err,
        "updated": mtime or None,
    }


def llm_status() -> dict:
    state = _json(LLM_STATE) or {}
    hist = [{"step": h.get("step"), "epoch": h.get("epoch"),
             "loss": h.get("loss"), "lr": h.get("learning_rate")}
            for h in state.get("log_history", []) if "loss" in h]
    final_epoch = max((h["epoch"] for h in hist if h["epoch"] is not None),
                      default=None)
    tail_losses = [h["loss"] for h in hist if h["loss"] is not None
                   and h["epoch"] is not None and final_epoch is not None
                   and h["epoch"] >= final_epoch - 0.99]
    final_loss = sum(tail_losses) / len(tail_losses) if tail_losses else None
    task = _json(LLM_CLEARML) or {}
    eval_ = _json(LLM_EVAL) or {}
    adapter_cfg = _json(LLM_DIR / "adapter_config.json") or {}
    rank = adapter_cfg.get("r")
    is_v4 = "v4" in str(LLM_DIR)
    corpora = [
        _APP_ROOT / "data" / "commentator_train_v4.jsonl",
        settings.data_dir / "commentator_train_v4.jsonl",
        settings.data_dir / "commentator_train_v2.jsonl",
    ]
    corpus = next((p for p in corpora if p.exists()), corpora[-1])
    lines = None
    if corpus.exists():
        try:
            with corpus.open(encoding="utf-8", errors="replace") as f:
                lines = sum(1 for _ in f)
        except OSError:
            lines = None
    adapter = LLM_DIR / "adapter_model.safetensors"
    max_steps = state.get("max_steps")
    if not state:
        status = "not-found"
    elif max_steps and (state.get("global_step") or 0) < max_steps:
        status = "training"
    else:
        status = "done"
    return {
        "status": status,
        "base_model": adapter_cfg.get("base_model") or "gpt2",
        "method": (f"LoRA r={rank}, completion-only loss" if rank
                   else "LoRA r=16, completion-only loss"),
        "dataset": ("SoccerReplay-1988 (500 matches) + synthetic grounding"
                    if is_v4 else "grounded synthetic corpus"),
        "epochs": state.get("num_train_epochs") or 3,
        "steps": state.get("global_step") or (hist[-1]["step"] if hist else None),
        "final_loss": round(final_loss, 4) if final_loss is not None else None,
        "last_loss": hist[-1]["loss"] if hist else None,
        "loss_history": hist,
        "corpus": {"path": str(corpus), "lines": lines,
                   "examples": 6000 if lines is None else lines},
        "adapter": {"path": str(adapter),
                    "bytes": adapter.stat().st_size if adapter.exists() else 0},
        "eval": eval_,
        "task_url": task.get("url"),
        "task_id": task.get("id"),
        "logs": hist[-16:],
        "trainer_state": str(LLM_STATE),
    }


def demo_status() -> dict:
    v4 = DEMO_DIR / "highlights_v4.json"
    src = v4 if v4.exists() else DEMO_DIR / "highlights.json"
    hl = _json(src) or {}
    if not hl:
        return {"available": False}
    clip_len = float(hl.get("clip_len") or 10.0)
    moments = []
    for m in hl.get("moments", []):
        mm = dict(m)
        mm["reel_start"] = round((int(mm.get("n", 1)) - 1) * clip_len, 2)
        moments.append(mm)
    tpath = DEMO_DIR / ("transcript_v4.txt" if v4.exists() else "transcript.txt")
    transcript = ""
    if tpath.exists():
        transcript = tpath.read_text(encoding="utf-8", errors="replace")
    return {
        "available": True,
        "video_url": ("/media/commented_highlights/commented_highlights_v4.mp4"
                      if v4.exists()
                      else "/media/commented_highlights/commented_highlights.mp4"),
        "tracking_url": "/media/commented_highlights/annotated_goal_possession.mp4",
        "moments": moments,
        "transcript": transcript,
        "clip_len": clip_len,
        "clips": hl.get("clips"),
        "reel_seconds": hl.get("reel_seconds"),
        "fish_model": hl.get("fish_model"),
        "fish_keyed": hl.get("fish_keyed"),
        "gpt2_lines": hl.get("gpt2_lines"),
        "template_lines": hl.get("template_lines"),
        "wall_seconds": hl.get("wall_seconds"),
        "score_anchors": hl.get("score_anchors"),
        "match": Path(str(hl.get("video", ""))).stem,
        "updated": tpath.stat().st_mtime if tpath.exists() else None,
    }


def tracking_status() -> dict:
    best = YOLO_RUN / "weights" / "best.pt"
    meta = _json(DEMO_DIR / "tracking_meta.json") or {}
    trained = best.exists()
    return {
        "weights_ready": trained,
        "weights": str(best) if trained else None,
        "weights_bytes": best.stat().st_size if trained else 0,
        "model": "Aryaman YOLOv5lu" if trained else "FootballVision yolov5su",
        "detector": "YOLOv5lu fine-tuned on V1 (3 classes incl. Ball)",
        "ball_tracking": ("ByteTrack ball track + constant-velocity gap fill + "
                          "possession from the smoothed ball"),
        "annotated_url": "/media/commented_highlights/annotated_goal_possession.mp4",
        "meta": meta,
        "updated": meta.get("created"),
    }


def training_bundle() -> dict:
    snap = _json(SNAPSHOT) or {}
    snap.setdefault("project_url", PROJECT_URL)
    snap.setdefault("tasks", [])
    snap.setdefault("models", [])
    return {
        "yolo": yolo_status(),
        "llm": llm_status(),
        "demo": demo_status(),
        "tracking": tracking_status(),
        "clearml": snap,
        "server_time": time.time(),
    }
