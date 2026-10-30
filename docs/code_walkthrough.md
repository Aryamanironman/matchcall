# Code walkthrough — what to open, what to say

Do not read the whole repository. Show five files, in this order. Each stop is one idea.

## 1. The orchestration

**Open:** `app/pipeline.py:38` — `run_pipeline()`

Say:

> “This is the whole pipeline in one function. It probes the video, profiles the crowd, reads the score bug, runs the motion-gated analysis, writes the facts, generates commentary, synthesises voice and renders the final clip. Each stage reports progress to the web UI.”

Point at:
- `detector = make_detector()` — the detector is selected by config, not hardcoded.
- `detect_score_flips(...)` — score bug is the gold goal anchor.
- `analyze_video(...)` — motion-gated scan.
- `duo.event_pair(...)` — facts become commentary.
- `tts_mod.synthesize(...)` — voice.
- `render_mod.render(...)` — ffmpeg mix.

Don’t show the whole function; scroll between those calls.

## 2. Motion-gated scan

**Open:** `app/vision/sampling.py:98` — `detection_schedule()`

Say:

> “We extract proxy frames at 4 fps. For each frame pair I compute a mean absolute frame difference on a 128×72 grayscale image. If motion is above the gate, YOLO runs every half-second; if the frame is dead, it runs four times sparser, but never skips it completely.”

Show the core:

```python
step_active = max(1, int(round(settings.det_interval * settings.analysis_fps)))
step_dead = step_active * 4
gate = settings.motion_dead_gate
```

Then point at `app/vision/analyze.py:37` `analyze_video()` and say:

> “Those frames are pooled into 2.5-second windows: motion, ball speed, ball visibility, goal-zone time, corner time, player count and possession.”

If asked the honest number: 64.5-minute ARG–FRA → 15,484 proxy frames, 15,244 detector frames. The gate saves little on continuous play; the 4 fps sampling is the bigger saving.

## 3. Weak labels

**Open:** `app/vision/auto_label.py:68` — `label_windows()`

Say:

> “The labels are weak supervision, not a public event dataset. A score-bug flip is gold for a goal. Otherwise the rules combine goalmouth zone, ball speed, crowd rise, motion and player clustering. Low motion becomes dead time; everything else is a pass.”

Point at:
- `flip is not None` → `goal`
- `zone >= 0.20 ...` → `goal`
- `ball_speed >= BALL_SPEED_SHOT ...` → `shot`
- `corner_time >= 0.30 ...` → `corner`
- `motion < gate` → `dead_time`
- `enforce_caps()` — limits goals/shots/corners per match.

Say the limitation:

> “These are heuristics, so they over-fire. The caps and the crowd evidence are there to keep the event classes honest.”

## 4. Commentary and validation

**Open:** `app/commentary_gpt2.py:404` — `event_line()`

Say:

> “The model gets a structured prompt: history, event, team, scorer, intensity. In slot mode the facts are placeholders, and the real values are substituted after generation.”

Point at the slot prompt:

```python
history = "<HOME> vs <AWAY>, score <SCORE>, <MINUTE> minute."
event = f"{word} by <TEAM>"
if player:
    event += ", scorer <PLAYER>"
```

Then open `app/commentary_gpt2.py:339` — `_validate()`

Say:

> “This is the guard. It checks the team appears, blocked names are absent, digits and both colon and hyphen scores match the fact sheet, and a goal line contains a real goal action and no contradiction like ‘corner’, ‘saved’ or ‘out of play’.”

Point at:
- `_GOAL_CONTRADICTIONS`
- score regex
- `_phrase_in(body.lower(), _GOAL_ACTIONS)`

Say:

> “The validator is deliberately conservative. It checks words, not truth; it is a guard, not a proof.”

## 5. Voice

**Open:** `app/tts.py:74` — `_one()`

Say:

> “Fish Audio is the primary voice. The code only falls back to edge-tts if the Fish key is missing or the request fails. Then ffmpeg mixes the voice under the crowd.”

Point at:
- `if key and fish_voice:` → Fish
- `except ... — edge-tts fallback`
- `synthesize_all()` runs segments concurrently.

## Optional: the web wrapper

**Open:** `app/webapp.py:161` — `upload()`

Say:

> “The web UI is a thin wrapper: it saves the upload, creates a job, and puts it on a single-worker queue. The worker runs the same `run_pipeline` function. That is why uploads are serial and a second upload waits.”

## Optional: training

**Open:** `scripts/train_yolo26.py:150` — `model.train(...)`

Say:

> “Training is Ultralytics YOLO26 on the V1 dataset, with the same run logged to ClearML. The training card reads the completed run’s `results.csv`, `val_metrics.json` and plots.”

## If they ask “show me the model”

Don’t claim ResNet is a benchmark. Say:

> “The detector is YOLO26n, fine-tuned on V1: 8,677 train / 2,996 validation, mAP50 0.805, mAP50–95 0.481. ResNet-18 is a separate weak-supervised event classifier, not the detector.”

## If they ask “where is the LLM?”

Open `app/commentary_gpt2.py:404`. Say:

> “GPT-2 + LoRA. The adapter is loaded once, the prompt is built from the fact sheet, and generation is constrained for goals. The validator decides whether the line is allowed through.”

## If they ask “where is the API?”

Open `scripts/api_feed_current_example.py`. Say:

> “It fetches `/fixtures/events` for fixture 1551707, freezes the event into team/player/minute/score, and calls the same commentary and TTS stack. The key is read from the environment, never from the browser.”

## The one-sentence summary

> “The code separates measurement from language: `analyze_video` produces facts, `event_line` turns facts into a candidate, `_validate` decides whether it is allowed, `tts` speaks it, and `render` mixes it.”
