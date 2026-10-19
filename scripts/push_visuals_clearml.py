"""Push visuals to ClearML: training plots, detection samples, highlights reel."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clearml import Task
from clearml.logger import Logger

HL = ROOT / "output" / "highlights"
RUN1 = ROOT / "models" / "yolov8_football" / "run1"

# --- get the active YOLO task (newest non-archived) ---
s = Task._get_default_session()
r = s.send_request(
    service="tasks", action="get_all_ex",
    json={"project": ["ef18bd575abc408591532be2ac8f39b8"], "page_size": 30,
          "only_fields": ["id", "name", "system_tags"], "order_by": ["-created"]})
tasks = r.json().get("data", {}).get("tasks", [])
yolo = next(t for t in tasks if "YOLO" in t["name"]
            and "__ARCHIVED__" not in (t.get("system_tags") or []))
task = Task.get_task(task_id=yolo["id"])
logger = task.get_logger()
print("target task:", task.name, task.id)

# --- 1. training plots as debug samples (render in UI gallery) ---
for png in ["results.png", "BoxPR_curve.png", "BoxF1_curve.png",
            "confusion_matrix_normalized.png"]:
    p = RUN1 / png
    if p.exists():
        logger.report_image(title="training artifacts", series=p.stem,
                            iteration=0, local_path=str(p))
        print("logged:", png)

# --- 2. detection samples: run detector on highlight frames ---
try:
    from ultralytics import YOLO
    weights = ROOT / "models" / "yolov8_football" / "yolov8n-football" / "weights" / "best.pt"
    if not weights.exists():
        cands = list((ROOT / "models" / "yolov8_football").rglob("best.pt"))
        weights = cands[0]
    model = YOLO(str(weights))
    frames = sorted(HL.glob("frame_*.png"))[:4]
    for f in frames:
        res = model.predict(source=str(f), conf=0.35, verbose=False)[0]
        out_img = HL / f"det_{f.name}"
        res.save(filename=str(out_img))
        n = len(res.boxes)
        logger.report_image(title="detection samples", series=f.stem,
                            iteration=0, local_path=str(out_img))
        print(f"detected {n} objects in {f.name} -> logged")
except Exception as e:
    print("detection skipped:", type(e).__name__, str(e)[:120])

# --- 3. highlights reel into the UI as media ---
try:
    logger.report_media(title="highlights", series="reel",
                        iteration=0, local_path=str(HL / "highlights_arg_fra.mp4"))
    print("reel uploaded to ClearML")
except Exception as e:
    print("media upload failed:", type(e).__name__, str(e)[:120])

logger.flush()
print("done — check Debug Samples tab of", task.id)
