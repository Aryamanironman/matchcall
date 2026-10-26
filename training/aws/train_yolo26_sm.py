"""SageMaker entry point: fine-tune YOLO26 on the V1 football dataset.

SageMaker downloads the ``train`` channel to /opt/ml/input/data/train and
packages everything written under /opt/ml/model into model.tar.gz. ClearML is
configured through environment variables set by the notebook.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT = "Football Commentator"
MODEL_NAME = "aryaman-yolo26-sagemaker"


def _find_dataset_root(base: Path) -> Path:
    """V1.zip may unzip to V1/ or directly to images/ + labels/."""
    for candidate in (base, base / "V1", base / "football-players"):
        if (candidate / "images" / "train").exists():
            return candidate
    for p in base.rglob("images"):
        if (p / "train").exists():
            return p.parent
    raise SystemExit(f"could not find images/train under {base}")


def _prepare_data(channel: Path) -> Path:
    root = Path("/tmp/V1")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    zips = list(channel.glob("*.zip")) + list(channel.rglob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(root)
        print(f"extracted {zips[0]} -> {root}", flush=True)
        data_root = _find_dataset_root(root)
    else:
        data_root = _find_dataset_root(channel)
    yaml_path = data_root / "data_aryaman.yaml"
    yaml_path.write_text(
        f"path: {data_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/test\n"
        "nc: 3\n"
        "names: [Player, Ball, Referee]\n",
        encoding="utf-8")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--base", default="yolo26s.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clearml", type=int, default=1)
    ap.add_argument("--task-name",
                    default="Aryaman YOLO26s football detector — V1 SageMaker")
    args, _unknown = ap.parse_known_args()

    import torch
    from ultralytics import YOLO

    channel = Path(os.environ.get("SM_CHANNEL_TRAIN",
                                  "/opt/ml/input/data/train"))
    yaml_path = _prepare_data(channel)
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"data: {yaml_path}\nmodel dir: {model_dir}", flush=True)

    task = None
    logger = None
    if args.clearml:
        try:
            from clearml import Task, TaskTypes
            task_train = getattr(TaskTypes, "training", None) or TaskTypes.custom
            task = Task.init(
                project_name=PROJECT, task_name=args.task_name,
                task_type=task_train, output_uri=True,
                reuse_last_task_id=False,
                tags=["aryaman", "yolo26", "sagemaker", "football", "v1"],
            )
            task.connect(vars(args))
            task.set_user_properties(
                architecture=f"YOLO26 ({args.base}), Ultralytics 8.4",
                dataset="Roboflow football-players V1: 8,677 / 2,996",
                platform="AWS SageMaker",
            )
            task.upload_artifact("data_yaml", artifact_object=str(yaml_path))
            logger = task.get_logger()
            print("ClearML task:", task.get_output_log_web_page(), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ClearML setup warning: {type(e).__name__}: {e}", flush=True)
            task = None

    model = YOLO(args.base)

    def on_fit_epoch_end(trainer):
        if logger is None:
            return
        epoch = int(trainer.epoch) + 1
        try:
            for k, v in (trainer.metrics or {}).items():
                if isinstance(v, (int, float)):
                    logger.report_scalar("metrics", k, float(v), epoch)
            try:
                losses = trainer.label_loss_items(trainer.tloss, prefix="train")
                for k, v in losses.items():
                    logger.report_scalar("loss", k, float(v), epoch)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    try:
        model.train(
            data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, project=str(model_dir), name="run",
            exist_ok=True, device=0 if torch.cuda.is_available() else "cpu",
            workers=args.workers, patience=args.patience, seed=args.seed,
            cache=False, amp=True, plots=True, save=True, val=True,
            verbose=True,
        )
    except Exception:
        if task is not None:
            try:
                task.mark_failed()
            except Exception:  # noqa: BLE001
                pass
        raise

    run_dir = model_dir / "run"
    best = run_dir / "weights" / "best.pt"
    print(f"done -> {best} exists={best.exists()}", flush=True)

    if task is not None:
        try:
            from clearml import OutputModel
            for name in ("results.csv", "results.png", "confusion_matrix.png",
                         "confusion_matrix_normalized.png", "BoxPR_curve.png",
                         "BoxF1_curve.png", "labels.jpg", "args.yaml"):
                p = run_dir / name
                if p.exists():
                    task.upload_artifact(name, artifact_object=str(p))
            if best.exists():
                om = OutputModel(task=task, name=MODEL_NAME)
                om.update_weights(weights_filename=str(best),
                                  auto_delete_file=False)
            task.mark_completed()
            (run_dir / "clearml_task.json").write_text(json.dumps({
                "id": task.id,
                "url": task.get_output_log_web_page(),
            }, indent=2), encoding="utf-8")
            print("ClearML task:", task.get_output_log_web_page(), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ClearML finalize warning: {type(e).__name__}: {e}",
                  flush=True)
        finally:
            task.close()


if __name__ == "__main__":
    main()
