# Human-Like Football Commentary Pipeline — Design

Date: 2026-10-13
Status: draft for approval
Project: `football-commentator` (Windows laptop, RTX 2050 4 GB; AWS g4dn.xlarge for training)

## Intent and why

The current pipeline produces commentary that is structurally correct but flat, and
it misses the moments that matter: on the test file (ARG–FRA 2022 WC final, second
half, beIN feed) it called **0 of the 2 goals**. The owner wants commentary that
behaves like a human broadcaster: it **knows what is on screen** — including replays
and unimportant moments — and **reacts the way humans react**: an explosion when the
ball hits the net, calming analysis in the next breath, mood narration in quiet
stretches. Accuracy and human-likeness are the same project: a commentator can only
react like a human if it knows what happened, and can only be trusted if it never
invents facts.

## Architecture (decided)

**Cheap detectors find WHEN to look → pretrained VLM says WHAT → fine-tuned LLM
says it LIKE A HUMAN.**

```
video/audio ─► detector layer (causal, cheap, fused)
                ├─ crowd-rise over rolling bed      (exists: app/vision/audio.py)
                ├─ whistle detector (2–4 kHz tone)  (new)
                ├─ gasp/"ooh" transient             (new, optional phase 2)
                ├─ replay-burst (perceptual-hash)   (new)
                └─ score-bug page classifier + digit/clock OCR (new, per-broadcaster calib)
                       │
                       ▼
        event moments + confidence tiers
                       │
                       ▼
        VLM (pretrained Qwen2.5-VL 3B-class, 4-bit) — SELECTIVE calls:
          • 2 frames around each detected event → clip description
          • sparse 1 description / 20–30 s in long quiet gaps → filler feed
                       │
                       ▼
        LLM (Gemma-3-4B QLoRA, fine-tuned on SN-Echoes pairs) — the "brain":
          event:  (VLM description + fused facts) → commentator line
          replay: (replay flag + ref to original event + VLM description
                   of the replay angle) → analytical replay commentary
          gap:    (running state + sparse VLM descriptions) → color narration
                       │
                       ▼
        validator (no invented scores/players/minutes → else grounded template)
                       │
                       ▼
        edge-tts (single voice; B disabled) → ffmpeg mix (exists)
```

- The VLM is **not trained**. It is the description engine and, during dataset
  preparation, the **training-pair generator** on AWS.
- The LLM is the only model we train (QLoRA on a T4), producing a small artifact
  (~2–3 GB at 4-bit) that runs locally.
- Commentary output is **single-voice** (commentator B is switched off via
  `COMMENTATOR_B=0` default; the duo code stays, flag re-enables it).

### Gap and replay commentary (decided — both are trained-LLM jobs)

**Gaps (quiet stretches).** The existing filler *planner* keeps its scheduling
job — it finds gaps, respects margins around event calls, and enforces cadence
so narration never overlaps a goal call or turns into spam. What changes is the
content source: instead of canned templates, the planner hands each gap slot to
the fine-tuned LLM with (running match state, recent event list, the sparse VLM
descriptions sampled inside that gap) and the LLM writes 1–3 sentences of color
narration in human commentator style. SN-Echoes contains exactly this material
(humans talk continuously through build-ups), so gap-narration training pairs
come free from the same dataset. Fallback if the LLM is slow or unavailable:
existing template fillers (they stay in the code path).

**Replays.** The replay-burst detector emits replay *segments*
(t_start, t_end, reference to the original event ~15–90s earlier), not just a
flag. Each replay segment becomes a first-class prompt to the LLM:
"this is a replay of {event tier} at minute {m}" + the VLM's description of the
replay angle → the LLM writes *analytical* replay commentary ("look at that —
the keeper never moves"), the way humans narrate replays. Hard rule enforced in
fusion: **a replay segment can never itself emit an event call** — no second
"GOAL" when the replayed goal is shown again (dedupe cooldown + explicit
replay-kind exclusion). Replay pairs for training are built the same way as gap
pairs: VLM-style description of the replay window paired with the human line
spoken during that window in SN-Echoes.

## Goals

1. **Catch the goals**: 4/4 on test footage via fusion (roar + score-bug flip or
   replay burst). Goal requires ≥2 independent signals.
2. **Know what's on screen**: every VLM-visible moment class is describable —
   replays, build-ups, unimportant spells — not just goals/cards. Replay segments
   are recognized as replays, not mistaken for live events.
3. **Sound human**: emotional arc matched to event tier (explosion → analysis →
   mood filler), learned from real human commentary (SN-Echoes), not hand templates.
4. **Never lie**: score/minute/side asserted only from fused state; validator
   rejects hallucinated digits/names; confidence-tiered wording ("IT'S IN!" only
   when confirmed, hedged phrasing when probable).
5. **Stay faster than real-time locally** (currently 0.27×; hard ceiling 1.0×).
6. **Measure everything**: eval harness with ground-truth timelines; no pipeline
   change ships without an eval comparison.

## Non-goals (for now)

- Live stream ingest (architecture stays causal/live-shaped, but the deliverable
  is VOD → narrated VOD).
- Two-voice duo output (off; flag exists).
- Training or fine-tuning the VLM.
- Player-name recognition / face identification.
- Web UI redesign — the existing FastAPI app remains the front door.
- Broadcaster auto-generalization beyond the score-bug calibration-profile design.

## Anti-goals (failure = worse than not shipping)

- A claimed goal that did not happen (false explosion) — fusion is tuned
  precision-first. *(Owner was asked missing-vs-false and did not answer; this
  default is the designer's call, tunable via one config value.)*
- Hallucinated state in any line (scorelines, minutes, player names).
- Regressing local runtime above real-time.

## Constraints

- **Local runtime**: RTX 2050 4 GB, Windows, existing `.venv`. All inference local.
  Model budget: ~2.5 GB weights for the LLM, ~2 GB for the VLM, both 4-bit.
  VLM calls are selective (event clips + sparse gap sampling), never continuous.
- **Training**: AWS g4dn.xlarge (T4 16 GB), free access but spin up per job and
  tear down after (credit limits still exist). Training data: `SoccerNet/SN-echoes`
  (HF access already approved; see sports-video-commentary skill for the exact
  files/arrow conversion). Pair format: (VLM-style description of the moment) →
  (human commentary line at that timestamp).
- **Existing pipeline**: extend `app/pipeline.py` stages; keep `summary.json`
  metrics comparable; keep the existing web app contract.
- **Known pitfalls to honor** (from project history + skills): feed audio is
  loudness-normalized (use crowd RISE, never absolute loudness); no
  quantile-threshold event classes (absolute floors + honest caps); advisory
  classifiers may only promote, never override evidence gates; Windows path
  slugs ≤80 chars computed once; ffmpeg filtergraph labels built in the same
  code path as inputs; exercise both small-N and large-N render branches.
- **Test footage**: `output/uploads/1791775787_3 - 3 ARG vs FRA - Second-Half - Bein.mp4`
  (64 min). Verified score reads: 2-0 @ t≈882, 2-1 @ t≈2250, 2-2 @ t≈2450 — i.e.
  two score flips inside the clip (goals to confirm against the known 2022 final
  timeline). v4/v5 run artifacts live under `output/`.

## Builder's latitude (decide without asking)

- Local serving approach for Qwen2.5-VL-3B and Gemma-3-4B at 4-bit on 4 GB
  (llama.cpp vs transformers+bitsandbytes vs onnx; CPU fallback acceptable if
  selective calls keep VLM cost small — measure and pick).
- Quantization format, prompt templates (description prompt, pair-generation
  prompt, event/filler system prompts).
- SN-Echoes cleaning/filtering specifics; pair-generation run structure on AWS.
- Detector implementation details (FFmpeg/numpy FFT for whistle; pHash variant and
  recurrence window for replay bursts); initial thresholds from
  `ACCURACY_ROADMAP.md` + skill constants, calibrated later via the eval harness.
- Fusion window sizes and confidence tiers (precision-first default).
- Eval harness mechanics (ground-truth capture, metrics: event P/R @±5s, score
  accuracy, minute error, repetition rate, hallucination count).
- How sparse-gap VLM descriptions feed the existing filler planner.

## Explicitly left to later phases

Gasp transient detector, chant-vs-roar discriminator, celebration/slow-mo camera
grammar, VLM fine-tuning (only if descriptions prove too generic), live ingest,
duo re-enable + distinct B voice, per-broadcaster profile sharing.
