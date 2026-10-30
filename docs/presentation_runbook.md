# Matchcall — presentation runbook

A 7–8 minute walkthrough. Speak from the bullets; don’t read the screen.

## The shape of the talk

| Time | Beat | What they see |
|---:|---|---|
| 0:00–0:45 | The problem | Title / one-line claim |
| 0:45–2:00 | The system | Pipeline + pitch map |
| 2:00–2:45 | The design rule | Facts vs language |
| 2:45–4:00 | ARG–FRA | Scoreboard, reel, rundown |
| 4:00–5:00 | API-Football | Real fixture, events, GPT-2 line |
| 5:00–5:45 | Your match | Upload + match plan |
| 5:45–6:30 | Evidence | Training / ClearML |
| 6:30–7:30 | Limits + close | Honest failure analysis |

## Opening (say this)

“Most football matches are never commentated. The ones that are need a commentator, a producer and often an analyst. I built Matchcall to test a narrower question: can a small, modular system produce usable spoken commentary from a broadcast on a consumer GPU, with facts that are auditable?”

## Slide 1 — The problem

- Commentary is expensive and does not scale.
- Existing systems are either template-based and repetitive, or large end-to-end models that are costly, opaque and can hallucinate.
- My target is the middle: a modular, small-GPU pipeline where the facts are measured before the language model sees them.

## Slide 2 — The pipeline (show the pitch map)

“Five stages, one pass.”

1. **Action spotting:** score-bug OCR for goals; motion-gated window scan; ResNet-18 as an advisory event classifier.
2. **Tracking:** YOLO26n/YOLOv5lu + ByteTrack; jersey clustering for teams; possession from the tracked player nearest the ball.
3. **Fact sheet:** team, minute, score, ball zone, possession.
4. **Commentary:** GPT-2 + LoRA, then a grounding validator.
5. **Voice:** Fish Audio, mixed under the crowd by ffmpeg.

“Video in at the back, commentary out up front.”

## Slide 3 — The design rule

“The language model never decides the score, minute, team or event. Those are measured. The model only realises facts it was given.”

- This is what makes the system auditable.
- The validator is a guard, not a truth machine.
- If the candidate fails, a grounded template is the safety net.

## Slide 4 — ARG–FRA

- 64.5-minute second half.
- 43 fully voiced lines in 19.4 minutes on one consumer GPU.
- Both goals found: precision 1.00, recall 1.00.
- Overall event recall against ten curated moments: **0.40** at ±1 minute.
- First line after 16.6 minutes: it is pre-recorded, not live.

“The goal path is the strong part. The general event layer is not good enough for broadcast.”

## Slide 5 — API-Football

- Real fixture **1551707**, 26 October 2026.
- **LD Alajuelense 1–0 Perez Zeledon**, 85:00–90:00.
- API returned two yellow cards and a 90′ goal by A. Hernandez.
- We froze the event into facts: team, player, minute, score.
- GPT-2 produced the grounded line; Fish voiced it.
- API gives event truth; video gives shots, corners, possession and positions.

“The API is the clean event source. The video is the richer but noisier source. A production system uses both.”

## Slide 6 — Your match

- Upload any football video.
- Upload a **match plan**: teams, players, venue, positions.
- The UI parses JSON and fills the match facts; the plan is attached to the run.
- Current boundary: the backend parser for players/positions is the next step.
- The form still supports video-only runs with safe defaults.

## Slide 7 — Training / ClearML

- YOLO26n: mAP50 **0.806**, mAP50–95 **0.480**.
- YOLOv5lu: mAP50 0.795 / mAP50–95 0.459 — **not equal budget** (50 vs 33 epochs, 2.39 M vs 53 M parameters).
- GPT-2 v4: 2,481 steps, loss 3.877 → 0.779.
- GPT-2 v5 API-slot: 38,815 train / 1,723 validation, 4,852 steps, loss 0.903.
- Every run is a ClearML task with parameters, metrics, artifacts and a registered model.

## Slide 8 — Limits and close

“What works: score-bug goal anchoring, the fact layer, auditable generation, low cost. What does not: general event spotting, team attribution on some cuts, and live streaming. The system is pre-recorded. The contribution is an honest, reproducible full pipeline, not a new state of the art.”

Close with:

“The most useful result is not that the model can write a sentence. It is that we can see exactly which facts it was allowed to use, and where the pipeline fails.”

## Live demo sequence

Open **http://127.0.0.1:8100**.

1. **Pipeline:** point at the formation map; say “five stages, one pass.”
2. **ARG–FRA board:** let them read the 81′ call; play the reel; click one rundown moment; switch to the tracking tab.
3. **API section:** point at fixture, events, and the GPT-2 line. Say “same language stack, different input.”
4. **Upload:** drop a match plan; show the parsed teams/venue/player count. Don’t start a long run unless the queue is empty.
5. **Training:** point at the ClearML links, the YOLO curves and the GPT-2 loss. Say “everything is logged.”

If the queue is busy, skip the upload run and show the plan preview only.

## Numbers to memorise

- 64.5 min → 43 lines → 19.4 min; 0.30× real time.
- Goals 2/2; event recall 0.40.
- ROUGE-L 0.185 vs 0.093; METEOR 0.207 vs 0.027; BLEU-4 ≈ 0.
- YOLO26n 0.806 / 0.480.
- GPT-2 v5: 38,815 / 1,723; 4,852 steps; loss 0.903.
- API: 1–0, 90′ A. Hernandez; 76.1 s cold + 2.1 s TTS.
- ResNet-18: five classes; stored val 0.972, but not a benchmark.

## If asked about ResNet-18

“It is the event classifier in stage three. I didn’t train it on a public event dataset. The pipeline trains it with weak supervision on the video being processed: 2.5-second windows, 4 fps proxy frames, heuristic labels from score-bug flips, goalmouth zone, ball speed, crowd rise, motion and player clustering. It has five classes: goal, shot, pass, tackle, dead time. At inference it averages the class probabilities over a window and gives an advisory label. It can promote a quiet pass to shot or goal only when confidence and other evidence agree; it never overrides the score-bug anchor. The checkpoint says 0.972 validation accuracy, but the split is random frames rather than time-separated, so I don’t present that as a benchmark.”

## Likely questions

**Why is event recall only 0.40?**  
The score-bug path is strong for goals, but the general event layer is heuristic. Shots and tackles are hard; the reference is ten curated moments. I report it rather than hide it.

**Why is BLEU-4 so low?**  
BLEU-4 demands long exact word-sequence matches. Commentary is open-ended; many valid lines share no 4-gram. ROUGE-L and METEOR are more informative, but none measures factual truth.

**Is the GPT-2 actually writing, or are these templates?**  
GPT-2 is the primary generator. In the reel, 10/10 lines were GPT-2. With v5 goal decoding, the API goal and both ARG–FRA goals are GPT-2. The grounded template remains a safety net.

**How do you stop hallucination?**  
Three layers: measured facts, a validator, and a grounded template. The validator checks team, score, digits and goal language. It is a guard, not a truth machine.

**Is it live?**  
No. It is pre-recorded. The full match is faster than real time in aggregate, but the first line comes after 16.6 minutes. Live needs streaming perception and incremental speech.

**Why not a large multimodal model?**  
Cost, reproducibility and auditability. A 7B model is hard on a consumer GPU and decides facts and language together. I wanted the fact sheet inspectable before generation.

**Does the match plan work?**  
The UI parses JSON and fills teams, venue, score and minute; the file is attached to the run. The backend parser for players and positions is the next step, so I don’t claim it is already grounding the prompt.

**What would you do next?**  
Streaming architecture, a proper annotated event set, a match-plan parser wired into the fact layer, and a blind human evaluation of commentary quality.

## Do not say

- “It is fully live.” It is not.
- “It detects all events.” It does not.
- “ResNet is a benchmarked football event model.” It is weakly supervised.
- “The match plan already drives the grounded description.” The UI parses it; the backend is pending.
- “Use the old `:8000` page.” The new UI is on `:8100`.

## Before you present

- Start the server on `:8100`.
- Check no long upload is already in the queue.
- Open `http://127.0.0.1:8100` and leave it on the pipeline section.
- Have the reel ready at `C:\Users\asus\Downloads\Matchcall_ARG-FRA_reel.mp4`.
- Have a small match-plan JSON ready if you want to demo the parser.
