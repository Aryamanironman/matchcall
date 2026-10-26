"""Fine-tune YOLO26 on the V1 football dataset, fully tracked in ClearML.

YOLO26 is Ultralytics' newest detector family (NMS-free end-to-end head). This
script mirrors the Aryaman YOLOv5lu run so the two can be compared directly:
same data YAML, same 3 classes, same validation split, ClearML scalars for every
epoch, plots/CSV as artifacts and the final weights in the model registry.

    python scripts/train_yolo26.py --epochs 50 --imgsz 512 --batch 1 --clearml

Output:
  models/yolo26/<name>/weights/best.pt
  models/yolo26/<name>/results.csv (+ curves)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]  # C:\Users\asus\football-commentator

PROJECT = "Football Commentator"
TASK_NAME = "Aryaman YOLO26n football detector — V1 fine-tune"
MODEL_NAME = "aryaman-yolo26n-football"


def _dataset_manifest(data_root: Path) -> dict:
    def count(split: str) -> int:
        p = data_root / "images" / split
        return sum(1 for _ in p.iterdir()) if p.exists() else 0

    return {
        "root": str(data_root),
        "train_images": count("train"),
        "val_images": count("test"),
        "classes": ["Player", "Ball", "Referee"],
        "yaml": "data_aryaman.yaml",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO26 on V1")
    ap.add_argument("--data", default=str(REPO / "data" / "V1"))
    ap.add_argument("--base", default=str(REPO / "weights" / "yolo26n.pt"))
    ap.add_argument("--out", default=str(REPO / "models" / "yolo26"))
    ap.add_argument("--name", default="yolo26n_v1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of the training split (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clearml", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    data_root = Path(args.data)
    yaml_path = data_root / "data_aryaman.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"missing dataset YAML: {yaml_path}")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = _dataset_manifest(data_root)

    # ---------------------------------------------------------------- ClearML
    task = None
    logger = None
    if args.clearml:
        try:
            from clearml import Task, TaskTypes
            task_train = getattr(TaskTypes, "training", None) or TaskTypes.custom
            task = Task.init(
                project_name=PROJECT, task_name=TASK_NAME, task_type=task_train,
                output_uri=True, reuse_last_task_id=False,
                tags=["aryaman", "yolo26", "football", "v1"],
            )
            try:
                task.connect(vars(args))
                task.connect(manifest, name="dataset")
                task.set_user_properties(
                    architecture="YOLO26n (Ultralytics 8.4, NMS-free end-to-end)",
                    dataset="Roboflow football-players V1: 8,677 / 2,996, "
                            "classes Player/Ball/Referee",
                    comparison="same data/YAML/splits as Aryaman YOLOv5lu",
                )
                task.upload_artifact("data_yaml", artifact_object=str(yaml_path))
                man_path = out_root / f"{args.name}_dataset.json"
                man_path.write_text(json.dumps(manifest, indent=2),
                                    encoding="utf-8")
                task.upload_artifact("dataset_manifest",
                                     artifact_object=str(man_path))
            except Exception as e:  # noqa: BLE001
                print(f"ClearML metadata warning: {type(e).__name__}: {e}",
                      flush=True)
            logger = task.get_logger()
            try:
                logger.report_text(
                    f"YOLO26n | train {manifest['train_images']} / "
                    f"val {manifest['val_images']} | imgsz {args.imgsz} | "
                    f"batch {args.batch} | epochs {args.epochs}")
            except Exception:  # noqa: BLE001
                pass
            print("ClearML task:", task.get_output_log_web_page(), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ClearML setup warning: {type(e).__name__}: {e}", flush=True)
            task = None

    model = YOLO(str(args.base))
    print(f"YOLO26 params: "
          f"{sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M",
          flush=True)

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
            for k, v in (getattr(trainer, "lr", None) or {}).items():
                logger.report_scalar("lr", k, float(v), epoch)
        except Exception:  # noqa: BLE001
            pass

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    def run(batch: int):
        if args.resume:
            last = out_root / args.name / "weights" / "last.pt"
            if not last.exists():
                raise SystemExit(f"cannot resume: {last} missing")
            return YOLO(str(last)).train(resume=True)
        return model.train(
            data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
            batch=batch, project=str(out_root), name=args.name,
            exist_ok=True, device=0 if torch.cuda.is_available() else "cpu",
            workers=args.workers, patience=args.patience, seed=args.seed,
            fraction=args.fraction, cache=False, amp=True, plots=True,
            save=True, val=True, verbose=True,
        )

    try:
        run(args.batch)
    except RuntimeError as e:  # noqa: PERF203
        if "out of memory" in str(e).lower() and args.batch > 1:
            print("CUDA OOM — retrying with batch=1", flush=True)
            torch.cuda.empty_cache()
            run(1)
        else:
            raise
    except Exception as e:  # noqa: BLE001
        if task is not None:
            try:
                task.mark_failed()
            except Exception:  # noqa: BLE001
                pass
        raise

    run_dir = out_root / args.name
    best = run_dir / "weights" / "best.pt"
    print(f"done -> {best} exists={best.exists()}", flush=True)

    if task is not None:
        try:
            from clearml import OutputModel
            for name in ("results.csv", "results.png", "confusion_matrix.png",
                         "confusion_matrix_normalized.png", "BoxPR_curve.png",
                         "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
                         "labels.jpg", "labels_correlogram.jpg", "args.yaml"):
                p = run_dir / name
                if p.exists():
                    task.upload_artifact(name, artifact_object=str(p))
            if best.exists():
                om = OutputModel(task=task, name=MODEL_NAME)
                om.update_weights(weights_filename=str(best),
                                  auto_delete_file=False)
                try:
                    om.update_design(config=(run_dir / "args.yaml").read_text())
                except Exception:  # noqa: BLE001
                    pass
            try:
                task.set_model_label_enumeration(
                    {"Player": 0, "Ball": 1, "Referee": 2})
            except Exception:  # noqa: BLE001
                pass
            task.mark_completed()
            (run_dir / "clearml_task.json").write_text(json.dumps({
                "id": task.id,
                "url": task.get_output_log_web_page(),
                "name": TASK_NAME,
            }, indent=2), encoding="utf-8")
            print("ClearML task:", task.get_output_log_web_page(), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ClearML finalize warning: {type(e).__name__}: {e}",
                  flush=True)
        finally:
            task.close()


if __name__ == "__main__":
    main()
