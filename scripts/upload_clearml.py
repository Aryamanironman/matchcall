"""Replay past training runs into ClearML + register trained models as artifacts.

One-time setup (browser, needs your ClearML account):
    .venv\\Scripts\\clearml-init     # paste credentials -> creates ~/clearml.conf

Then run:
    .venv\\Scripts\\python scripts/upload_clearml.py

Creates 5 tasks under project "Football Commentator":
  1. YOLOv8n detector training replay   (full curves from results.csv, 50 epochs)
  2. ResNet18 event classifier          (final metrics from checkpoint)
  3. GPT-2 LoRA commentary adapter      (config + weights provenance)
  4. WC-final validation audit          (findings + accuracy metrics)
  5. Live-mode design (planned)         (roadmap doc)
and registers 3 output models linked to local weight files.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from clearml import Task, OutputModel
from clearml import TaskTypes

# ClearML 2.x renamed the enum members (train -> training, development -> custom)
TASK_TRAIN = getattr(TaskTypes, "training", None) or TaskTypes.custom
TASK_DEV = getattr(TaskTypes, "custom", None) or TaskTypes.development


def _summary(logger, values: dict) -> None:
    for k, v in values.items():
        if v is not None:
            logger.report_scalar(title="summary", series=k, value=float(v), iteration=0)


def replay_yolo():
    task = Task.init(
        project_name="Football Commentator",
        task_name="YOLOv8n player detector — training replay (run1)",
        task_type=TASK_TRAIN,
        reuse_last_task_id=False,
    )
    run_dir = ROOT / "models" / "yolov8_football" / "run1"
    rows = list(csv.DictReader((run_dir / "results.csv").read_text().splitlines()))
    args_text = (run_dir / "args.yaml").read_text()

    # hyperparameters from the ultralytics args.yaml
    props = {}
    for line in args_text.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k and v:
                props[f"ultralytics/{k}"] = v
    task.set_user_properties(**props)

    logger = task.get_logger()
    series = [
        ("train/box_loss", "loss", "box_train"),
        ("train/cls_loss", "loss", "cls_train"),
        ("train/dfl_loss", "loss", "dfl_train"),
        ("val/box_loss", "loss", "box_val"),
        ("val/cls_loss", "loss", "cls_val"),
        ("val/dfl_loss", "loss", "dfl_val"),
        ("metrics/precision(B)", "metrics", "precision"),
        ("metrics/recall(B)", "metrics", "recall"),
        ("metrics/mAP50(B)", "metrics", "mAP50"),
        ("metrics/mAP50-95(B)", "metrics", "mAP50-95"),
    ]
    last = {}
    for i, row in enumerate(rows):
        clean = {k.strip(): v for k, v in row.items()}
        ep = int(float(clean.get("epoch", i)))
        for csv_key, title, sname in series:
            if csv_key in clean:
                val = float(clean[csv_key])
                logger.report_scalar(title=title, series=sname, value=val, iteration=ep)
                last[csv_key] = val

    # debug samples: curves + confusion matrix + sample batches
    for png in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                "BoxPR_curve.png", "BoxF1_curve.png", "labels.jpg",
                "val_batch0_pred.jpg", "val_batch1_pred.jpg", "train_batch0.jpg"]:
        p = run_dir / png
        if p.exists():
            logger.report_image(title="training artifacts", series=png, local_path=str(p))

    _summary(logger, {
        "final_mAP50": last.get("metrics/mAP50(B)"),
        "final_mAP50-95": last.get("metrics/mAP50-95(B)"),
        "final_precision": last.get("metrics/precision(B)"),
        "final_recall": last.get("metrics/recall(B)"),
        "epochs": len(rows),
    })

    model = OutputModel(task=task, name="yolov8n-football-run1")
    model.update_weights(weights_filename=str(run_dir / "weights" / "best.pt"),
                         auto_delete_file=False)
    try:  # 2.x moved/renamed the design kwarg; args.yaml already in user properties
        model.update_design(config=args_text)
    except TypeError:
        try:
            model.update_design(args_text)
        except Exception:
            pass

    task.mark_completed()
    task.close()
    return task


def register_resnet():
    import torch  # venv torch
    task = Task.init(
        project_name="Football Commentator",
        task_name="ResNet18 event classifier — checkpoint registration",
        task_type=TASK_TRAIN,
        reuse_last_task_id=False,
    )
    ck_path = ROOT / "models" / "resnet18_events.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    classes = list(ck.get("classes") or [])
    val_acc = float(ck.get("val_acc") or 0.0)

    logger = task.get_logger()
    _summary(logger, {"final_val_accuracy": val_acc})
    if classes:
        task.set_model_label_enumeration({c: i for i, c in enumerate(classes)})
    task.set_user_properties(
        classes=json.dumps(classes),
        note="per-epoch history was not persisted at training time; "
             "final val_acc + checkpoint registered for provenance",
    )

    model = OutputModel(task=task, name="resnet18-events")
    model.update_weights(weights_filename=str(ck_path), auto_delete_file=False)
    try:  # 2.x: labels live on the task, not the model
        model.set_labels({c: i for i, c in enumerate(classes)})
    except AttributeError:
        pass

    task.mark_completed()
    task.close()
    return task


def register_lora():
    task = Task.init(
        project_name="Football Commentator",
        task_name="GPT-2 LoRA commentary adapter — registration",
        task_type=TASK_TRAIN,
        reuse_last_task_id=False,
    )
    lora_dir = ROOT / "models" / "gpt2_lora"
    cfg = json.loads((lora_dir / "adapter_config.json").read_text())
    props = {}
    for k in ("r", "lora_alpha", "lora_dropout", "target_modules",
              "base_model_name_or_path", "bias", "task_type"):
        if k in cfg:
            props[f"peft/{k}"] = str(cfg[k])
    props["note"] = ("training script/curves not persisted; adapter registered "
                     "for provenance and lineage (base: gpt2)")
    task.set_user_properties(**props)

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "gpt2_lora_adapter"
        zpath = shutil.make_archive(str(base), "zip", lora_dir)
        model = OutputModel(task=task, name="gpt2-lora-commentary")
        model.update_weights(weights_filename=zpath, auto_delete_file=False)

    task.mark_completed()
    task.close()
    return task


def audit_task():
    task = Task.init(
        project_name="Football Commentator",
        task_name="WC-final beIN footage — v4/v5 validation audit",
        task_type=TaskTypes.testing,
        reuse_last_task_id=False,
    )
    logger = task.get_logger()
    logger.report_scalar(title="audit", series="goal_recall (of 4 goals)", value=0.0, iteration=0)
    logger.report_scalar(title="audit", series="segments_with_repeats", value=7, iteration=0)
    logger.report_scalar(title="audit", series="high_excitement_segments", value=2, iteration=0)
    logger.report_scalar(title="runtime", series="realtime_ratio", value=0.269, iteration=0)
    logger.report_scalar(title="runtime", series="speech_seconds", value=192.6, iteration=0)
    task.set_user_properties(
        goals_in_ground_truth="4",
        goals_called="0",
        verdict="machinery stable; moment-detection accuracy poor on carousel-bug broadcasts",
    )
    for name in ["V5_FINDINGS.md", "ACCURACY_ROADMAP.md"]:
        p = ROOT / "output" / name
        if p.exists():
            task.upload_artifact(name.replace(".md", ""), artifact_object=str(p))
    task.mark_completed()
    task.close()
    return task


def live_design_task():
    task = Task.init(
        project_name="Football Commentator",
        task_name="Live-mode architecture (v6 plan)",
        task_type=TASK_DEV,
        reuse_last_task_id=False,
    )
    task.set_user_properties(
        constraint="no dependency on broadcast human commentary (ASR); live causal processing",
        signals="roar-spike detector; whistle anchors; page-classified bug clock/score; "
                "YOLO+ResNet visual events; replay-burst heuristic",
        latency_budget_s="8-12",
    )
    p = ROOT / "output" / "ACCURACY_ROADMAP.md"
    if p.exists():
        task.upload_artifact("accuracy_roadmap", artifact_object=str(p))
    task.mark_completed()
    task.close()
    return task


if __name__ == "__main__":
    modname = Path(__file__).stem  # 'upload_clearml'
    for fn in (replay_yolo, register_resnet, register_lora, audit_task, live_design_task):
        code = (
            "import sys; sys.path.insert(0, r'%s'); "
            "import importlib; m = importlib.import_module('%s'); m.%s()"
            % (str(ROOT / "scripts"), modname, fn.__name__)
        )
        # one task per process: ClearML allows a single live task per interpreter
        r = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(),
                           capture_output=True, text=True, cwd=str(ROOT))
        status = "ok" if r.returncode == 0 else "FAIL"
        print(f"[{status}] {fn.__name__}")
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()
            print("   ", tail[-1] if tail else "unknown error")
