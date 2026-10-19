# Football Commentary Dubber

AI play-by-play commentary over real football footage. Drop in a broadcast MP4,
get back the same clip with an energetic commentator dubbed on top — or a
10-second-per-moment commented highlights reel.

```
video ─▶ action spotting (score-bug OCR + YOLO/ResNet)
      ─▶ 10 s key-action clips
      ─▶ FootballVision tracking (yolov5su + ByteTrack + team possession)
      ─▶ grounded action description
      ─▶ GPT-2 LoRA commentary (validated, template fallback)
      ─▶ Fish Audio s2.1-pro-free / edge-tts
      ─▶ commented highlights reel
```

Events detected: **goal, shot, tackle, pass, corner, dead_time**.

## Commented highlights (the 5-stage pipeline)

```powershell
# ARG-FRA 2022 second half; score is 2-0 at t=0 and the broadcast clock is 34:38
.venv\Scripts\python scripts\make_commented_highlights.py `
  --video "C:\Users\asus\Downloads\3 - 3 ARG vs FRA - Second-Half - Bein.mp4" `
  --home Argentina --away France `
  --start-score-home 2 --start-score-away 0 --minute-offset 35.33
```

Outputs in `output/commented_highlights/`:

| file | what |
|---|---|
| `commented_highlights.mp4` | the reel (10 s per selected moment) |
| `annotated_goal.mp4` | tracking overlay (players, teams, ball, IDs) |
| `highlights.json` | moments, score anchors, descriptions, commentary, timings |
| `transcript.txt` | minute-by-minute commentary lines |
| `score_anchors.json` | score-bug goal evidence |

Stages map to the code:

1. **Action spotting** — `app/vision/score_flip.py` (score-bug digit-shape
   transitions with a verified per-feed cache) and `app/vision/analyze.py`
   (project YOLO + ResNet window scan).
2. **Tracking / visual enrichment** — `app/vision/fv_tracker.py` uses the
   FootballVision checkpoint (`models/footballvision/yolov5su_best.pt`) with
   ultralytics ByteTrack, KMeans jersey-colour teams and ball possession.
3. **Action description** — grounded facts (team, minute, score, possession,
   ball zone, tracked players) built in `scripts/make_commented_highlights.py`.
4. **Textual commentary** — `app/commentary_gpt2.py` (LoRA) with a validator
   that rejects invented teams, scores, cards or player names; templates are
   the per-line fallback.
5. **Audio commentary** — `app/tts.py` (Fish Audio `s2.1-pro-free` when
   `FISH_API_KEY` is real, edge-tts otherwise).

### Score-bug calibration

The reader is calibrated for the 1920x1080 beIN score bug. Boxes live in
`app/vision/score_flip.py`; they scale with the frame. If you change
broadcaster, update `BEIN_1080_BOXES` and `BEIN_1080_CLOCK`. The score already
on the board at `t0` must be passed with `--start-score-home/away`; the
official match-minute at `t0` with `--minute-offset` (ARG-FRA: 35.33, i.e. the
80' goal lands at video 2687 s).

For the ARG-FRA demo, `output/commented_highlights/score_anchors.json` is a
**verified cache** (2687 s = 80', 2784 s = 81') and the script reuses it. The
automatic reader is best-effort on this font; the cache is the calibrated
answer for this feed.

### GPT-2 LoRA

`models/gpt2_lora_v3/` is trained on a grounded corpus
(`scripts/build_commentary_corpus.py`, 6k examples) with completion-only loss,
so the adapter learns the `[EVENT] -> [COMMENTARY]` mapping rather than the
deterministic history tokens. Retrain with:

```powershell
.venv\Scripts\python scripts\build_commentary_corpus.py --limit 6000
.venv\Scripts\python scripts\train_commentator_lora.py `
  --data data\commentator_train_v2.jsonl --out models\gpt2_lora_v3 --epochs 3
```

### Fish Audio

Put a real key in `.env`:

```
FISH_API_KEY=sk-fish-...
FISH_MODEL=s2.1-pro-free
FISH_VOICE_ID=40f53434437e46a5bcbb5abb39f5079c
```

`***` is treated as missing; the pipeline falls back to edge-tts and says so.
As soon as the key is set, re-run the reel to re-voice every line with
`s2.1-pro-free` — no other change.

## Aryaman YOLOv5lu detector (product model)

The FootballVision repo's improved detection is **YOLOv5lu** (large;
`main.py` loads `models/yolov5lu/best.pt`) plus **YOLOv8x** for pitch
keypoints — it does **not** use YOLO26. Train Aryaman's detector on the full
V1 export (8,677 train / 2,996 val, 3 classes):

```powershell
.venv\Scripts\python scripts\train_aryaman_yolov5.py
```

- Base: `yolov5lu.pt` (the repo's detection architecture)
- Output: `models/aryaman_yolo/aryaman_yolov5lu_football/weights/best.pt`
- ClearML: project **Football Commentator**, task **Aryaman YOLOv5lu football
  detector — V1 fine-tune**, dataset artifact + live metrics/plots + registered
  output model `aryaman-yolov5lu-football`
- `app/vision/fv_tracker.py` prefers the Aryaman checkpoint once it exists

## Full-match dub

```powershell
.venv\Scripts\python -m app.cli --video "match.mp4" --home Argentina --away France `
  --score-home 2 --score-away 0 --minute-offset 35.33
```

The whole-match action-spotting pass uses the project YOLOv8 model by default
(fast). Set `FC_DETECTOR=footballvision` to use the FootballVision checkpoint
for the full scan too; the highlights script always uses FootballVision for
the 10-second clip tracking.


## Setup (Windows, CUDA GPU)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r app\requirements.txt
copy .env.example .env        # optional: add GEMINI_API_KEY for LLM lines
```

Requires `ffmpeg`/`ffprobe` on PATH, and `yolov8n.pt` in `models\`
(download once: https://github.com/ultralytics/assets/releases — or it is
fetched automatically on first run).

## Train the detector (one-time, ~2.5 h on RTX 3050)

Dataset: football-players detection set (Roboflow CC-BY mirror), 8,677 train /
2,996 test images with Player / Ball / Referee boxes. The zip is expected at
`data/V1.zip` (download link in `download_dataset.py`) and extracts to `data/V1/`.

```powershell
.venv\Scripts\python -m app.train_yolo data\V1 models\yolov8_football 50
```

Weights land at `models\yolov8_football\run1\weights\best.pt`; the pipeline
picks them up automatically (falls back to COCO `yolov8n.pt` with
person→player, sports-ball→ball mapping if absent).

## Commentary writers

`app/commentary.py` auto-selects, in order (each falls back to the next per line):

1. **Gemini 1.5 Flash** — if `GEMINI_API_KEY` is set.
2. **Fine-tuned GPT-2 (LoRA)** — `models/gpt2_lora/`: gpt2 + r=16 LoRA on `c_attn`,
   trained on markup-formatted match data
   (`[HISTORY] … [EVENT] … [INTENSITY] … [COMMENTARY] …`). Prompted in its native
   format it writes Opta-style lines (*"Header from the centre of the box to the
   top right corner. Assisted by…"*) and predicts its own intensity, which drives
   the excitement level. See `app/commentary_gpt2.py`.
3. **Energetic templates** — always available; also writes all filler lines.

The adapter was saved with PEFT ≥0.19 but this repo pins 0.13.2; the config's
newer keys were stripped from `models/gpt2_lora/adapter_config.json` so it loads.

## Web UI

```powershell
.venv\Scripts\python -m uvicorn app.webapp:app --port 8000
# open http://127.0.0.1:8000 — drag & drop, live progress bar, streaming
# transcript, inline player; jobs run in a background worker (POST /api/upload,
# GET /api/job/{id}, POST /api/job/{id}/cancel)
```

## Run

```powershell
# second half of the ARG-FRA match, from minute 15, 2-minute clip
.venv\Scripts\python -m app.cli --video "match.mp4" --home Argentina --away France --t0 900 --duration 120
```

Output directory contains:

| file | what |
|---|---|
| `final_commentary.mp4` | the dubbed clip |
| `transcript.txt` / `.json` | every line with timestamp + event tag |
| `analysis.json` | per-window vision features, labels, ResNet probs |
| `tts/` | per-segment voice lines (for QA) |

On GPU the whole 2-minute clip processes in well under a minute; even CPU-only
runs ~0.6× realtime.

## Design notes

- **Motion-gated sampling** (`app/vision/sampling.py`): one fast ffmpeg pass
  extracts a 480px/4fps proxy; frame-diff motion decides which frames the
  detector sees (dead time sampled sparsely). This is what makes the pipeline
  faster than realtime.
- **Weak supervision** (`app/vision/auto_label.py`): windows are auto-labeled
  by documented heuristics (ball in goalmouth + sustained presence → goal,
  speed spike → shot, motion spike + speed dip + crowd → tackle). ResNet18 is
  trained on those labels, then *refines* them: windows where the net is
  confident (≥0.55) override the heuristic.
- **No dead air** (`app/filler.py`): a filler engine inserts rotating ambient
  lines whenever narration gaps would exceed ~20s, so the commentary track is
  continuous.
- **Audio bed**: original broadcast audio is kept at low volume (0.12) under
  the commentator for atmosphere — disable with `--no-orig-audio`.
- **Three-tier writers**: Gemini → fine-tuned GPT-2 → templates. Every event
  falls back per line if a writer errors, so a dead API key or missing adapter
  degrades quality, never the run.
