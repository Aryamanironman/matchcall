"""Train Aryaman's YOLOv5lu football detector (FootballVision architecture).

What the repo actually improves
-------------------------------
FootballVision (mserra0/FootballVision) detects players/referees/ball with the
**large YOLOv5 family**: ``main.py`` loads ``models/yolov5lu/best.pt`` and
``training/train_yolov5.ipynb`` trains ``yolov5x.pt`` for 100 epochs. It also
fine-tunes **YOLOv8x** for pitch keypoints/homography. The repo does **not**
use YOLO26; the shipped ``yolov5su/best.pt`` is the small fallback checkpoint.

This script fine-tunes the same YOLOv5lu architecture on the full Roboflow
football-players V1 export already extracted at ``data/V1``:
  8,677 train / 2,996 val images, 3 classes (Player, Ball, Referee).

Everything is tracked in ClearML (project "Football Commentator"):
hyperparameters, dataset artifacts, live metrics/plots, final model registry.

Output:
  models/aryaman_yolo/aryaman_yolov5lu_football/weights/best.pt
  models/aryaman_yolo/aryaman_yolov5lu_football/results.png (+ curves)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from app.config import settings

PROJECT = "Football Commentator"
TASK_NAME = "Aryaman YOLOv5lu football detector — V1 fine-tune"
MODEL_NAME = "aryaman-yolov5lu-football"


def write_data_yaml(data_root: Path) -> Path:
    """V1 has no valid split; use the 2,996-image test split for validation."""
    yaml_path = data_root / "data_aryaman.yaml"
    yaml_path.write_text(
        f"path: {data_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/test\n"
        "nc: 3\n"
        "names: [Player, Ball, Referee]\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune Aryaman's YOLOv5lu")
    ap.add_argument("--data", default=str(settings.data_dir / "V1"))
    ap.add_argument("--base", default="yolov5lu.pt",
                    help="ultralytics YOLOv5 large checkpoint (the repo's "
                         "main.py architecture)")
    ap.add_argument("--out", default=str(
        settings.models_dir / "aryaman_yolo" / "aryaman_yolov5lu_football"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    from ultralytics import YOLO

    data_root = Path(args.data)
    out = Path(args.out)
    yaml_path = write_data_yaml(data_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_zip = data_root.parent / "V1.zip"

    # ---------------------------------------------------------------- ClearML
    task = None
    try:
        from clearml import OutputModel, Task, TaskTypes
        task_train = getattr(TaskTypes, "training", None) or TaskTypes.custom
        task = Task.init(
            project_name=PROJECT, task_name=TASK_NAME, task_type=task_train,
            output_uri=True, reuse_last_task_id=False,
            tags=["aryaman", "yolov5lu", "football", "v1"],
        )
        task.connect(vars(args))
        task.set_user_properties(
            repo="mserra0/FootballVision",
            architecture="YOLOv5lu (repo main.py; notebook trains yolov5x)",
            detector_note="repo improved detector is YOLOv5lu/xu; "
                          "YOLOv8x is used for pitch keypoints, no YOLO26",
            dataset="Roboflow football-players V1: 8,677 train / 2,996 val, "
                    "classes Player/Ball/Referee",
        )
        task.upload_artifact("data_yaml", artifact_object=str(yaml_path))
        if data_zip.exists():
            print(f"ClearML: uploading dataset artifact {data_zip.name} "
                  f"({data_zip.stat().st_size / 1e9:.2f} GB)...", flush=True)
            task.upload_artifact("dataset_v1_zip",
                                 artifact_object=str(data_zip))
        print("ClearML task:", task.get_output_log_web_page(), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"ClearML setup warning: {type(e).__name__}: {e}", flush=True)
        task = None

    print("Aryaman YOLOv5lu fine-tune", flush=True)
    print(f"  base: {args.base}", flush=True)
    print(f"  data: {yaml_path}", flush=True)
    print(f"  out : {out}", flush=True)
    print(f"  cuda: {torch.cuda.is_available()}", flush=True)

    def run_train(batch: int):
        model = YOLO(str(args.base))
        return model.train(
            data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
            batch=batch, project=str(out.parent), name=out.name,
            exist_ok=True,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=args.workers, patience=args.patience, seed=42,
            cache=False, plots=True, save=True, val=True, verbose=True,
        )

    try:
        run_train(args.batch)
    except RuntimeError as e:  # noqa: PERF203
        if "out of memory" in str(e).lower() and args.batch > 2:
            print("CUDA OOM — retrying with batch=2", flush=True)
            torch.cuda.empty_cache()
            run_train(2)
        else:
            raise

    best = out / "weights" / "best.pt"
    print(f"done -> {best} exists={best.exists()}", flush=True)

    # ------------------------------------------------------- ClearML finalize
    if task is not None:
        try:
            for name in ("results.csv", "results.png", "confusion_matrix.png",
                         "confusion_matrix_normalized.png", "BoxPR_curve.png",
                         "BoxF1_curve.png", "labels.jpg", "args.yaml"):
                p = out / name
                if p.exists():
                    task.upload_artifact(name, artifact_object=str(p))
            om = OutputModel(task=task, name=MODEL_NAME)
            om.update_weights(weights_filename=str(best),
                              auto_delete_file=False)
            try:
                om.update_design(config=(out / "args.yaml").read_text())
            except Exception:  # noqa: BLE001
                pass
            try:
                task.set_model_label_enumeration(
                    {"Player": 0, "Ball": 1, "Referee": 2})
            except Exception:  # noqa: BLE001
                pass
            task.mark_completed()
            print("ClearML task:", task.get_output_log_web_page(), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ClearML finalize warning: {type(e).__name__}: {e}",
                  flush=True)
        finally:
            task.close()


if __name__ == "__main__":
    main()
