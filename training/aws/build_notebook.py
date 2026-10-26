import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text})

md("""# YOLO26 full vs YOLOv5 large — SageMaker

Fair generation test on the same V1 data, at the same 640 px and 100 epochs:

| Run | Base | Params | Batch |
|---|---|---|---|
| YOLO26 full | `yolo26x.pt` | 59.0 M | 8 |
| YOLOv5 large | `yolov5lu.pt` | 53 M | 16 |

Both log every epoch to ClearML and register their weights. `yolo26l.pt`
(26.3 M) is a cheaper option if you want the large tier rather than the full
model.

**Prerequisites:** open this notebook in SageMaker Studio inside `training/aws`
(upload the folder or clone the repo), a Studio execution role with S3 access,
and your ClearML keys.""")

code("""import os, boto3, sagemaker
from sagemaker import get_execution_role

BUCKET = "YOUR-BUCKET-NAME"          # <-- change
PREFIX = "football-commentator"
REGION = boto3.session.Session().region_name or "ap-southeast-2"
ROLE = get_execution_role()

INSTANCE = "ml.g5.2xlarge"           # 1x A10G 24 GB, 32 GB host RAM
EPOCHS = 100
IMGSZ = 640
PATIENCE = 20

RUNS = [
    {"base": "yolo26x.pt", "batch": 8,
     "task": "Aryaman YOLO26x football detector — V1 SageMaker"},
    {"base": "yolov5lu.pt", "batch": 16,
     "task": "Aryaman YOLOv5lu baseline — V1 SageMaker"},
]
print("region", REGION, "role", ROLE)""")

code("""s3 = boto3.client("s3")
try:
    s3.head_bucket(Bucket=BUCKET)
except Exception:
    if REGION == "us-east-1":
        s3.create_bucket(Bucket=BUCKET)
    else:
        s3.create_bucket(Bucket=BUCKET,
                         CreateBucketConfiguration={"LocationConstraint": REGION})

KEY = f"{PREFIX}/V1/V1.zip"
try:
    s3.head_object(Bucket=BUCKET, Key=KEY)
    print("dataset already at", f"s3://{BUCKET}/{KEY}")
except Exception:
    url = s3.generate_presigned_url(
        "put_object", Params={"Bucket": BUCKET, "Key": KEY}, ExpiresIn=3600)
    print("Upload data/V1.zip from the laptop (one command), then re-run this cell:")
    print()
    print(f'curl -X PUT --upload-file "data/V1.zip" "{url}"')
    print()
    raise SystemExit("waiting for the dataset upload")""")

md("""If the V1 zip is already on the Studio filesystem:

```python
sagemaker.session.Session().upload_data(path="V1.zip", bucket=BUCKET, key_prefix=f"{PREFIX}/V1")
```""")

code("""from getpass import getpass
for k in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"):
    if not os.environ.get(k):
        os.environ[k] = getpass(f"{k}: ")
os.environ.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
os.environ.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
os.environ.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")

CLEARML_ENV = {k: os.environ[k] for k in (
    "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY",
    "CLEARML_API_HOST", "CLEARML_WEB_HOST", "CLEARML_FILES_HOST")}
print("ClearML credentials set (not printed)")""")

code("""from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

estimators = {}
for cfg in RUNS:
    print(f"=== {cfg['base']} (batch {cfg['batch']}) ===")
    estimator = PyTorch(
        entry_point="train_yolo26_sm.py",
        source_dir=".",
        role=ROLE,
        framework_version="2.5.1",
        py_version="py311",
        instance_type=INSTANCE,
        instance_count=1,
        volume_size=100,
        max_run=6 * 3600,
        base_job_name="yolo26-" + cfg["base"].replace(".pt", ""),
        output_path=f"s3://{BUCKET}/{PREFIX}/yolo26/output",
        hyperparameters={
            "base": cfg["base"], "epochs": EPOCHS, "imgsz": IMGSZ,
            "batch": cfg["batch"], "workers": 8, "patience": PATIENCE,
            "clearml": 1, "task-name": cfg["task"],
        },
        environment=CLEARML_ENV,
        disable_profiler=True,
    )
    estimator.fit(
        {"train": TrainingInput(s3_data=f"s3://{BUCKET}/{PREFIX}/V1/",
                                input_mode="File")},
        wait=True, logs=True,
    )
    estimators[cfg["base"]] = estimator
    print("model artifact:", estimator.model_data)""")

code("""import tarfile
from urllib.parse import urlparse

for base, estimator in estimators.items():
    uri = estimator.model_data
    parsed = urlparse(uri)
    local_tar = f"model_{base.replace('.pt','')}.tar.gz"
    s3.download_file(parsed.netloc, parsed.path.lstrip("/"), local_tar)
    out_dir = f"model_{base.replace('.pt','')}"
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(local_tar) as t:
        t.extractall(out_dir)
    best = os.path.join(out_dir, "run", "weights", "best.pt")
    print(base, "best.pt:", best, os.path.exists(best))
    print(base, "results:", os.path.join(out_dir, "run", "results.csv"))""")

md("""## What to compare

1. **Accuracy:** mAP50, mAP50-95, precision, recall, and per-class Player/Ball/Referee AP.
2. **Efficiency:** params, GFLOPs, inference ms/image, batch throughput.
3. **Deployability:** does YOLO26x's NMS-free head behave with ByteTrack and the
   possession logic? Copy `best.pt` into `models/yolo26/` and rerun the pipeline with
   `--tracker-weights .../best.pt`.

**Cost:** `ml.g5.2xlarge` is about $1.21/hour. YOLO26x at 640/100 epochs on an A10G
is roughly 2–4 hours; the YOLOv5lu baseline roughly 2–3 hours. If the x run OOMs,
set its batch to 4 or use a larger instance. Keep the job under 6 hours to fit the
`max_run` setting.

**Tuning:** YOLO26x is a 59 M-param model on 8.7k images — 100 epochs is a real
fine-tune. If you want a quicker signal, use `yolo26l.pt` (26.3 M) at batch 16 for
the first comparison.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).with_name("yolo26_sagemaker.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, out.stat().st_size)
