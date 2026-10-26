# YOLO26 full vs YOLOv5lu on SageMaker

Fair generation test on the same V1 football dataset, same 640 px, same 100 epochs:

| Run | Base | Params | Batch |
|---|---|---|---|
| YOLO26 full | `yolo26x.pt` | 59.0 M | 8 |
| YOLOv5 large (baseline) | `yolov5lu.pt` | 53 M | 16 |

Both runs log every epoch to **ClearML** (losses, precision/recall, mAP50,
mAP50-95, learning rates), upload plots/CSV/args as artifacts, and register the
final weights. The notebook runs them sequentially on `ml.g5.2xlarge`
(1× A10G 24 GB, 32 GB host RAM).

## Files

- `yolo26_sagemaker.ipynb` — open in SageMaker Studio and run top to bottom.
- `train_yolo26_sm.py` — SageMaker entry point (extracts `V1.zip`, writes the
  data YAML, trains, logs to ClearML, writes `/opt/ml/model`).
- `requirements.txt` — installed by the SageMaker PyTorch container.
- `build_notebook.py` — regenerates the notebook if you edit the cells.

## Steps

1. **Put this folder in Studio** (upload `training/aws` or clone the repo).
2. Open `yolo26_sagemaker.ipynb`; set `BUCKET` to an S3 bucket you own.
3. Run the dataset cell. If `V1.zip` is not in S3 yet, it prints a presigned
   `curl -X PUT` command. Run that on the machine that has
   `data/V1.zip` (this laptop), then re-run the cell.
   - Alternative: upload `V1.zip` to `s3://<bucket>/football-commentator/V1/`
     with the S3 console or AWS CLI.
4. Enter your ClearML access/secret keys when prompted (they are not written to
   the notebook). The job container receives them as environment variables —
   SageMaker job definitions can show env values, so use an IAM-restricted
   bucket/role and rotate the keys if that matters.
5. Run the training cell. It launches YOLO26x first, then YOLOv5lu, waiting for
   each and streaming logs.
6. Run the download cell; it saves `model_yolo26x/run/weights/best.pt` and
   `model_yolov5lu/run/weights/best.pt` plus the result CSVs.

## Expected time and cost

- `ml.g5.2xlarge` ≈ **$1.21/hour** (check current pricing in your region).
- YOLO26x at 640, batch 8, 100 epochs: roughly **2–4 hours**.
- YOLOv5lu at 640, batch 16, 100 epochs: roughly **2–3 hours**.
- Total ≈ **$5–9**. If it is too slow, run `yolo26l.pt` (26.3 M) at batch 16
  first; it is a cheaper large-tier comparison.

## After training

Copy each `best.pt` into `models/yolo26/` locally, then rerun the pipeline with:

```powershell
python scripts/make_commented_highlights.py --tracker-weights "...\best.pt"
```

or just the overlay:

```powershell
python scripts/annotate_tracking.py --weights "...\best.pt" --out ...\overlay_yolo26x.mp4
```

Compare per-class Player/Ball/Referee AP, ball detection/interpolation rate,
possession, and the visual overlay against the YOLOv5lu run. The same
`--tracker-weights` flag works for either checkpoint.

## Why "newer" is not automatically better

The local pilot compared YOLO26n (2.5 M, 512 px, 5 epochs) with YOLOv5lu
(53 M, 640 px, 33 epochs); that is three disadvantages at once. This SageMaker
notebook removes them: both models are large/full, both at 640 px, both for
100 epochs on the same split. If YOLO26x still loses, that is a real result for
this small-ball broadcast dataset; if it wins, the generation claim holds here
too.

## Security

Do not commit ClearML keys, `clearml.conf`, or S3 credentials. The notebook
prompts for keys at runtime; `.env` and credentials are gitignored.
