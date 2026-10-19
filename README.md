# Football Commentary Dubber

AI play-by-play commentary over real football footage. Drop in a broadcast MP4,
get back the same clip with an energetic commentator dubbed on top.

```
video ─▶ motion-gated sampling ─▶ YOLOv8 (player/ball/referee)
      ─▶ auto-label (weak supervision) ─▶ ResNet18 event classifier
      ─▶ commentary (fine-tuned GPT-2 LoRA / Gemini / templates) ─▶ filler engine
      ─▶ edge-tts voice ─▶ ffmpeg mix (ducked crowd audio bed)
```

Events detected: **goal, shot, tackle, pass, dead_time**.

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
