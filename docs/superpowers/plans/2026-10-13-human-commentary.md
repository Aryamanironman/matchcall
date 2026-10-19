# Human-Like Commentary Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Goal-catching fused detectors + selective VLM descriptions + a fine-tuned commentator LLM so the pipeline knows what's on screen (events, replays, gaps) and narrates it like a human.

**Architecture:** Cheap causal detectors (crowd-rise [exists], whistle, replay-burst, score-bug reader) vote in a precision-first fusion layer → pretrained Qwen2.5-VL 3B (4-bit, selective calls: event clips + sparse gap samples) describes moments → Gemma-3-4B QLoRA (trained on AWS from SN-Echoes pairs) writes event/replay/gap lines → validator blocks invented facts → edge-tts → existing ffmpeg render.

**Tech Stack:** Python 3.12 (.venv), numpy/ffmpeg signal processing, imagehash/pHash, transformers+bitsandbytes (VLM), peft+bitsandbytes (training, AWS T4), llama-cpp-python (local GGUF inference), pytest, existing FastAPI app untouched.

**Spec:** `docs/superpowers/specs/2026-10-13-human-commentary-design.md`

## Global Constraints

- Precision-first fusion: a false GOAL is the worst outcome; missing one is second worst (spec Anti-goals).
- Local realtime ratio must stay < 1.0 (current: 0.27). VLM calls are selective only — never per-frame.
- Local model budget: LLM GGUF Q4 ~2.5 GB; VLM 4-bit ~2 GB; RTX 2050 4 GB + system RAM.
- Single voice output; `COMMENTATOR_B` default 0.
- No quantile/relative thresholds for event classes; absolute floors + honest caps (goal ≤ 2 per clip-side, shot ≤ 24).
- Advisory classifiers promote-only; evidence gates always re-run after.
- Windows: output slug ≤ 80 chars computed once; ffmpeg filtergraph labels built in the same code path as their `-i` inputs; smoke-render both N=1 and N=40 branches.
- Match/test footage: `output/uploads/1791775787_3 - 3 ARG vs FRA - Second-Half - Bein.mp4`; score truth 2-0@882s, 2-1@2250s, 2-2@2450s (file-relative).
- Repo is not yet git — Task 0 fixes that before any commit step.

## Review Focus

1. **Loud crowd ≠ goal** (chants, normalized feeds): fusion must not fire a goal on crowd noise alone. → pinned by Task 5 chant test.
2. **Replay re-fire** (second GOAL during a replay): replay segments never emit event calls. → pinned by Task 5 exclusion test.
3. **Hallucinated state** (score/minute/player in LLM output): validator rejects or strips. → pinned by Task 9 adversarial tests.
4. **VLM absent/too slow on 4 GB**: stub backend keeps pipeline alive and fast. → pinned by Task 6 stub tests + Task 10 realtime assertion.
5. **Render blowup with new input counts** (event+replay+gap lines): filtergraph/labels built in same path, small-N and large-N both exercised. → pinned by Task 10 smoke renders.

---

### Task 0: Repo bootstrap (git + test runner)

**Files:**
- Create: `.gitignore`, `tests/__init__.py` (empty)

**Interfaces:**
- Produces: a committed repo every later task can commit to; `pytest` importable.

- [ ] **Step 1: Write `.gitignore`** — ignore: `.venv/`, `output/`, `models/`, `__pycache__/`, `*.pyc`, `.clearml/`, `work/`, `*.gguf`, `runs/`.
- [ ] **Step 2: Init and commit**: `git init && git add -A && git commit -m "chore: bootstrap repo"`; verify `output/` is NOT tracked (`git ls-files output` empty).
- [ ] **Step 3: Ensure pytest**: `.venv/Scripts/python.exe -m pytest --version`; if missing, `.venv/Scripts/pip.exe install pytest` and amend the commit.

### Task 1: Eval harness core + ground truth

**Files:**
- Create: `app/eval.py`, `tests/test_eval.py`, `data/truth/1791775787_arg_fra_2h.json`

**Interfaces:**
- Produces: `match_events(pred: list[dict], truth: list[dict], tol_s: float = 5.0) -> dict` where each dict is `{"t": float, "label": str}`; returns `{"precision": float, "recall": float, "f1": float, "matches": list[tuple[int,int]]}` (greedy nearest matching, each side used once).
- Produces: `repetition_rate(texts: list[str]) -> float` — 1 - unique/total over normalized sentences.
- Produces: `minute_error(pred_minutes: list[float], truth_minutes: list[float]) -> float` — mean absolute error.
- Produces: truth file schema `{"video": str, "events": [{"t": 2250.0, "label": "goal"}, ...], "notes": str}` (goals at 2250, 2450; HT at ~47min per V5 findings).

- [ ] **Step 1: Failing tests** — precision/recall on a hand-built case (2 predicted, 1 true within tol, 1 far): assert `precision == 0.5, recall == 1.0`; repetition_rate of `["x y.", "x y.", "z."]` == 1/3; minute_error([61],[63]) == 2.0.
- [ ] **Step 2: Run** `python -m pytest tests/test_eval.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** the three pure functions in `app/eval.py` (numpy allowed, no other deps).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Write truth JSON** for the test footage; commit.

### Task 2: Whistle detector

**Files:**
- Create: `app/vision/whistle.py`, `tests/test_whistle.py`
- Modify: `app/vision/audio.py` — add `extract_pcm(video: Path, sr: int = 8000, t0: float = 0.0, duration: float | None = None) -> np.ndarray` (refactor of the PCM part of `audio_rms_profile`; `audio_rms_profile` reimplemented on top of it, behavior unchanged).

**Interfaces:**
- Consumes: `extract_pcm` output.
- Produces: `detect_whistles(pcm: np.ndarray, sr: int = 8000) -> list[dict]` — `{"t": float, "conf": float, "dur": float}`; a whistle = ≥0.4s run where a 2.2–3.8 kHz band peak dominates its ±150 Hz neighborhood (spectral flatness in-band low) and band power ≥ 8 dB over the rolling 60s median. Merge detections < 2s apart.

- [ ] **Step 1: Failing tests** (synthetic PCM, no fixture files): (a) 3.0 kHz sine @ amplitude 0.5 over 0.8s bed of white noise 0.01 → one detection near its time, conf > 0.5; (b) pure white noise → zero detections; (c) chant-like 400 Hz sawtooth + harmonics → zero detections (band dominance rule); (d) two whistles 1.0s apart merge into one, 5s apart stay two.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — per-1s-window rFFT, band power via bin sums, dominance = band peak bin power / max(power of ±150 Hz side bins); rolling median floor per constraint "absolute floors + relative bed" hybrid.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit.**

### Task 3: Replay-burst detector

**Files:**
- Create: `app/vision/replay.py`, `tests/test_replay.py`

**Interfaces:**
- Consumes: proxy frames already on disk (`work/proxy_frames/*.jpg`, mapping via existing `app.vision.sampling.frame_time`).
- Produces: `frame_hash(img_bgr: np.ndarray) -> bytes` (8-byte pHash of 32x32 grayscale DCT low bits; use `pywt`-free manual DCT or cv2.dct).
- Produces: `detect_replays(hashes: list[bytes], times: list[float], min_gap_s: float = 8.0, max_lag_s: float = 120.0, min_matches: int = 3) -> list[dict]` — `{"t0": float, "t1": float, "orig_t": float, "conf": float}`; a replay = ≥`min_matches` consecutive-ish frames (allow 1 miss) whose hamming distance ≤ 6 to a hash seen `min_gap_s..max_lag_s` earlier.

- [ ] **Step 1: Failing tests**: (a) `frame_hash` of an image vs its slightly-noised copy → hamming ≤ 6; vs a different image → > 12. (b) synthetic stream: 10 hashes of image A at t=0..9s, then 10 of B (t=100..109), then A again (t=140..149) → one replay seg with `orig_t≈4`, `t0≈140`; (c) A repeated at t=4 after t=0..2 (gap < min_gap_s) → NOT a replay (live stutter).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** (numpy hamming via `np.unpackbits`; linear scan over a deque of last `max_lag_s` of hashes — buffer stays small at 1 fps).
- [ ] **Step 4: Run** → PASS. **Step 5: Commit.**

### Task 4: Score-bug page classifier + clock model

**Files:**
- Modify: `app/vision/score_bug.py` (add functions; keep existing `load_calibration`/`detect_goal_anchors` untouched)
- Create: `tests/test_score_pages.py`

**Interfaces:**
- Consumes: cached score-page templates (`output/calib/fr.npy` @ t=882/2250/2450, non-score @ t=913 — copy into `models/score_templates/` so tests don't depend on `output/`).
- Produces: `page_scores(frame_gray: np.ndarray, templates: dict[str, np.ndarray]) -> dict[str, float]` — normalized cross-correlation per template key ("score", "other").
- Produces: `read_score_timeline(video: Path, templates: dict[str, np.ndarray], box: tuple[int,int,int,int], t0: float, duration: float, sample_every_s: float = 1.0) -> dict` — `{"changes": [{"t": float, "home": int, "away": int}], "minutes": MinuteModel, "page_cov": float}`.
- Produces: `MinuteModel` — `.minute_at(t) -> int`, fit by robust linear regression (median-of-slopes fallback) over OCR'd clock readings; rejects non-monotonic outliers (> 90s jump) before fitting; `.ok` False when < 5 clean readings (callers fall back to `t/60`).

- [ ] **Step 1: Failing tests**: (a) page_scores of a template against itself (plus noise σ=0.02) → "score" NCC > "other" NCC; (b) a synthetic 3-digit box render (PIL-drawn "2-1" on flat bg) → read_score_timeline on a 10s synthetic video (ffmpeg color source + drawtext, written in the test's tmp dir) reports change to (2,1); (c) MinuteModel with readings t=0→17min, t=600→27min, t=1200→37min plus one corrupt (t=601→95min) → `.minute_at(600)==27` and the corrupt reading rejected.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (NCC via cv2.matchTemplate TM_CCOEFF_NORMED on the calibration box crop; digits: nearest-template glyph match per slot — slot boxes from calibration JSON).
- [ ] **Step 4: Run** → PASS. **Step 5: Commit.**

### Task 5: Fusion layer (precision-first) + replay exclusion

**Files:**
- Create: `app/vision/fusion.py`, `tests/test_fusion.py`
- Modify: `app/pipeline.py` — replace the ad-hoc `SCORE_ANCHORS`/ResNet promotion block (lines ~130–146) with a call into fusion; feed whistles + replays + score timeline in.

**Interfaces:**
- Consumes: `roar_events()` output (exists), `detect_whistles`, `detect_replays`, `read_score_timeline.changes`, windows from `analyze_video`.
- Produces: `fuse(roar: list[dict], whistles: list[dict], replays: list[dict], score_changes: list[dict], windows: list[Window], *, goal_second_signal: bool = True) -> list[dict]` — `{"t", "label": "goal"|"shot"|"whistle"|"replay"|"ambient", "tier": "confirmed"|"probable", "side": str|None, "replay_of": float|None}`.
- Fusion rules (each pinned by a test): GOAL requires roar(spike ≥2.0 rise within ±8s) AND ≥1 of {score-flip ±15s, replay-burst within 30s after, whistle 10–25s after}; no second signal → "probable" tier GOAL CANDIDATE demoted to "shot" if ball_speed floor met, else "ambient". REPLAY segments emit `label="replay"`, `replay_of=<orig_t>`; windows inside a replay segment cannot be promoted to goal/shot by anything (exclusion). Whistle-only → "whistle" anchor. Ties → precision-first (goal_second_signal=False flips to recall-first; config knob `FUSION_RECALL_FIRST`).

- [ ] **Step 1: Failing tests**: (a) roar@100 + score-flip@102 → confirmed goal, side from flip; (b) roar@100 alone → NOT a goal (ambient/probable per ball_speed); (c) roar@100 + whistle@115 → confirmed goal; (d) replay seg [300,320] covering a window the ResNet/promoter wants as goal → stays "replay", `replay_of` set, no goal emitted; (e) chant (rise 1.4 sustained, no second signal) → no goal.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Rewire `pipeline.py`** to call fusion (gated: only when audio_usable or score templates exist; falls back to current behavior otherwise); run `python -m pytest tests/ -v` → all PASS; commit.

### Task 6: Selective VLM describer (stub-first)

**Files:**
- Create: `app/vision/describe.py`, `tests/test_describe.py`

**Interfaces:**
- Produces: `describe_event_clip(video: Path, t_abs: float) -> str` (2 frames at t−1.0s, t+0.6s) and `describe_gap_sample(video: Path, t_abs: float) -> str` (1 frame), both cached in `work/descriptions.json` keyed by `(kind, round(t_abs,1))`.
- Produces: backend selection via env `DESC_BACKEND` ∈ {`stub`, `qwen`}; `stub` returns deterministic template text ("wide angle, players in mid third, ball with team in red" style, derived from frame brightness/motion so tests can vary input); `qwen` loads Qwen2.5-VL-3B-Instruct 4-bit (transformers+bitsandbytes), falls back to `stub` with a logged warning when load fails or model dir missing.
- Model dir contract: `models/qwen2.5-vl-3b/` (downloaded manually / on AWS; absent locally on first run → stub).

- [ ] **Step 1: Failing tests**: (a) `DESC_BACKEND=stub` → both functions return non-empty str; second call with same key does not re-extract (cache hit counter); (b) qwen with `DESC_QWEN_DIR` pointing nowhere → returns stub text AND logs fallback (caplog); (c) cache file round-trips (delete model, rerun → cached text served).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (ffmpeg single-frame extraction reusing sampling helpers; lazy model holder singleton; generation kwargs: max_new_tokens=60, prompt pins "one sentence, what is happening, no player names unless on the scoreboard").
- [ ] **Step 4: Run** → PASS. **Step 5: Commit.**

### Task 7: Training-pair builder (AWS-bound, schema-tested locally)

**Files:**
- Create: `scripts/build_training_pairs.py`, `tests/test_pairs.py`, `docs/aws-training.md`

**Interfaces:**
- Produces: JSONL records `{"prompt": str, "completion": str, "kind": "event"|"replay"|"gap", "meta": {...}}` — prompt embeds VLM description slot + facts; completion is the human line.
- Consumes: SN-Echoes arrow (HF, per skill: `SoccerNet/SN-echoes`, `whisper_v2_en`) + SoccerNet labels-v2.json per game + VLM descriptions generated on AWS over game frames at event timestamps.
- CLI: `python scripts/build_training_pairs.py --game <soccernet_slug> --out data/pairs/train.jsonl [--limit N] [--describe qwen|none]`.

- [ ] **Step 1: Failing tests**: (a) `build_pair(desc="strker shoots", facts={"label":"goal","minute":63,"score":"2-1"}, line="OH IT'S IN! 2-1!")` → prompt contains description + minute + score, completion == line, kind == "event"; (b) lines within ±2s of two events don't cross-pair (alignment window ±10s, first-match-wins); (c) gap pairs: label lines > 20s from any event as kind="gap" with state-only prompt; (d) replay pairs: lines 15–90s after a labeled event whose text matches replay-cue regex (`look at|replay|again|watch`) → kind="replay" referencing the event.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (arrow loading guarded by import; description generation via `app.vision.describe` with `--describe qwen`; `--describe none` writes prompts with empty description slot so training can still run on AWS without VLM).
- [ ] **Step 4: Run** → PASS. **Step 5: Write `docs/aws-training.md`** — exact g4dn.xlarge runbook (spot launch, `pip install -e .[train]`, HF login, download SN-echoes, run builder for 5–10 games, run Task 8 script, copy GGUF back to `models/`); commit.

### Task 8: QLoRA training script (AWS T4)

**Files:**
- Create: `scripts/train_commentator.py`, `tests/test_train_script.py`, `app/requirements-train.txt`

**Interfaces:**
- Produces: `load_pairs(path: Path) -> list[dict]` (validation: kind ∈ {event,replay,gap}, completion non-empty ≤ 400 chars) and `format_example(pair: dict, tokenizer) -> dict` (chat template with system prompt: "You are a football commentator. Use ONLY the facts given.").
- CLI: `--pairs data/pairs/*.jsonl --base google/gemma-3-4b-it --out models/commentator-lora --steps 1500 --merge --gguf q4_k_m` (merge + GGUF conversion via llama.cpp `convert_hf_to_gguf.py` + `quantize`, both subprocess-guarded).

- [ ] **Step 1: Failing tests** (CPU-safe): (a) load_pairs rejects bad records (kind typo, 500-char completion) and keeps good ones; (b) format_example output tokenizes and labels mask the prompt tokens (assert label == -100 for prompt span on one sample); (c) `--steps 2 --dry-run` executes the training loop body once with a randomly-initialized tiny causal LM (env `TRAIN_TINY_MODEL=1`) proving step/optimizer/saving wiring without downloading Gemma.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (peft LoRA r=16 α=32 on q,k,v,o + gate/up/down; bnb 4-bit NF4; cosine LR 2e-4; batch 1 + grad accum 16 — T4 16GB-safe per skill).
- [ ] **Step 4: Run** → PASS. **Step 5: Commit.**

### Task 9: Local brain (GGUF inference) + validator

**Files:**
- Create: `app/commentary_llm.py`, `app/validator.py`, `tests/test_brain.py`

**Interfaces:**
- Consumes: `models/commentator-q4.gguf` (from Task 8 runbook) via llama-cpp-python; absent → `Brain.available` False.
- Produces: `class Brain`: `.available: bool`, `.event_line(desc: str, facts: dict) -> str`, `.replay_line(desc: str, facts: dict, orig_t: float) -> str`, `.gap_line(state: dict, descs: list[str]) -> str`; all return validator-passed text, else raise `BrainUnavailable`/return None (caller falls back to templates).
- Produces: `validate_line(text: str, facts: dict) -> tuple[bool, str]` — facts provides allowed values `{"score", "minute", "teams": [..]}`; rules: any `\d+\s*[-–]\s*\d+` must equal facts["score"]; any `\b\d{1,3}(st|nd|rd|th)\b` minute must be within ±1 of facts["minute"]; capitalized multi-char tokens must ∈ facts["teams"] ∪ {"Referee","VAR","GOAL","KICKOFF","HALFTIME","FULLTIME"}; violation → strip offending sentence; if first sentence violated → `(False, "")`.

- [ ] **Step 1: Failing tests**: (a) `validate_line("It's 3-1 now!", {"score": None, "teams": ["Argentina"]})` → (False, ""); (b) with facts score "2-1", "2-1" passes, "2-2" fails; (c) minute guard: "in the 63rd minute" vs minute 63 passes, vs 61 fails; (d) team guard: "Messi" stripped when not in teams, sentence survives if it's sentence 2; (e) Brain with `BRAIN_GGUF=stub` (env) → returns canned but validator-filtered lines (deterministic test path without model).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (llama.cpp n_ctx=2048, temp 0.7, stop at newline; prompt templates per kind mirroring Task 7 pair format).
- [ ] **Step 4: Run** → PASS. **Step 5: Commit.**

### Task 10: Pipeline wiring — events, replays, gaps through the brain

**Files:**
- Modify: `app/pipeline.py` (stages: after fusion → describe event clips → brain event/replay lines → planner slots → brain gap lines with template fallback → tts → render); `app/filler.py` (accept a `gap_writer: Callable[[dict], str|None]`; template fallback when it returns None); `app/duo.py` (route through Brain when available — A lines only)
- Create: `tests/test_pipeline_wiring.py` (stub backend end-to-end on a 20s synthetic clip: ffmpeg color+noise source, assert transcript contains ≥1 event line + ≥1 gap line + no validator violations; realtime metric recorded)

**Interfaces:**
- Consumes: everything above; `Segment` unchanged (kind gains `"replay"`).
- Produces: `summary.json` gains `"goals_confirmed"`, `"replays_narrated"`, `"gap_lines"`, `"brain_backend"`, `"desc_backend"`; transcript.json unchanged schema (event_label `"replay"` for replay lines).

- [ ] **Step 1: Failing test** — stub end-to-end: transcript contains `("goal" in line) == False` when fusion emits none on a quiet synthetic clip; replay seg in fixture → exactly one replay line, zero second goal lines (Review Focus #2 at pipeline level).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement wiring** (all ffmpeg inputs+labels for render built in ONE code path — Review Focus #5; keep `orig_audio` mix behavior).
- [ ] **Step 4: Run pytest** → PASS. **Step 5: Smoke renders**: N=1 segment and N=40 segments through `render_mod.render` on the synthetic clip → both exit 0. **Step 6: Commit.**

### Task 11: Full-file validation + tuning + README

**Files:**
- Modify: `README.md` (new architecture section, model download table, AWS runbook link)
- Create: `output/` eval report is gitignored — write report to `docs/eval-2026-10-13.md` instead

- [ ] **Step 1: Run pipeline** on the WC final file end-to-end (stub or downloaded models), `time`-box wall clock; record realtime ratio.
- [ ] **Step 2: Eval report**: run `app/eval.py` functions against Task 1 truth (goals 2 expected) + repetition rate + minute error (when score templates available); write `docs/eval-2026-10-13.md` with before/after v4 numbers.
- [ ] **Step 3: Acceptance check**: goals recalled 2/2 with ≥1 signal pair each; zero false goals; realtime < 1.0; every line passes validator; gap lines present in gaps ≥ 75s.
- [ ] **Step 4:** Fix thresholds ONLY via config values if acceptance fails (no magic numbers in code); re-run Step 1–2. **Step 5: Commit + update README.**

## Self-Review

- Spec coverage: detectors (T2–4) → fusion (T5) → VLM selective (T6) → training data (T7) → training (T8) → brain+validator (T9) → wiring incl. gap/replay commentary (T10) → measurement (T1, T11). Anti-goals pinned by tests (precision-first T5, validator T9, realtime T10/T11). Model budgets honored in T6/T8/T9. Duo-off respected (single voice, A only).
- Step scan: every step names exact files/signatures/assertions or exact commands; no TBDs; algorithm bodies only where tests don't determine them (whistle dominance, pHash, NCC, minute fit).
- Type consistency: `detect_whistles`/`detect_replays`/`read_score_timeline`/`fuse`/`Brain.*`/`validate_line` names match across T5/T9/T10 interfaces.
- Review Focus: all five failure modes have owning-task tests (T5 a/b/e, T5 d + T10, T9, T6 b + T10, T10 step 5).
- Proportion: ~10 tasks, each with a failing-test → implement → pass → commit cycle; plan length ≈ spec length, code appears only as signatures/assertions.
