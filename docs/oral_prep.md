# Oral Prep — Matchcall Football Commentary Project

Use this as a speaking sheet, not a script to read. The key is to sound like you know the trade-offs, including the negative results.

## 1. The 60-second pitch

“Broadcast football commentary is expensive and does not scale: most matches are never covered, and the ones that are covered need a commentator, a producer and often an analyst. My project asks whether a small, modular system can produce usable spoken commentary from a broadcast on a consumer GPU.

The pipeline has five stages: action spotting from the score bug and a window classifier; tracking with YOLO26n, ByteTrack, jersey clustering and possession; a deterministic fact sheet; GPT-2 with LoRA for commentary under a grounding validator and template fallback; and Fish Audio for speech.

The important design choice is that the language model never decides the score, minute, team or event. It only realises facts that were already measured. That makes the system auditable and lets me measure exactly where it fails.

On a 64.5-minute ARG–FRA match it produced 43 voiced lines in 19.4 minutes. Goals are reliable — 2/2, precision and recall 1.0 — but general event recall is 0.40, and the current system is pre-recorded, not live. The contribution is an auditable, low-cost full pipeline with honest failure analysis, not a new state of the art.”

## 2. The pipeline in five stages

1. **Action spotting:** score-bug reader for goals; motion-gated window scan; ResNet-18 classifier for goal/shot/pass/tackle/dead time; crowd audio and ball speed refine decisions.
2. **Tracking:** YOLO26n detects players, ball and referee; ByteTrack keeps identities; KMeans jersey colours assign teams; possession goes to the side of the tracked player nearest the ball; bounded ball interpolation fills short gaps.
3. **Facts:** deterministic sheet with score, minute, ball zone, possession, tracked-player count and event team.
4. **Language:** GPT-2 v4 LoRA generates a line from the facts; a validator checks team, score, digits and event consistency; a grounded template is the fallback.
5. **Speech:** Fish Audio s2.1-pro-free; event intensity controls pacing; ffmpeg mixes voice under the crowd.

The design principle: **facts are measured, language is generated, speech is produced.** The model cannot invent a score because the score is not its decision.

## 3. Key numbers to remember

| Item | Number |
|---|---:|
| Full match | 64.5 min |
| Output | 43 lines, 43/43 voiced |
| Wall time | 19.4 min |
| Commentary comparison | 40 held-out prompts |
| GPT-2 accepted | 38/40 (95%) |
| Stricter independent acceptance | 15/40 (37.5%) |
| ROUGE-L | GPT-2 0.185 vs template 0.093 |
| METEOR | GPT-2 0.207 vs template 0.027 |
| BLEU-4 | 0.000304 vs 0.001038 |
| Average words | GPT-2 40.5; human 31.9; template 7.4 |
| Event recall ±1 min | 0.40 overall |
| Goals | 2/2; precision 1.00; recall 1.00 |
| Shots | 2/6 |
| Tackles | 0/2 |
| Timing MAE | 36.2 s |
| YOLO26n | mAP50 0.805; mAP50–95 0.481 |
| YOLOv5lu paused | mAP50 0.795; mAP50–95 0.459 |
| YOLO params / epochs | 2.39 M / 50 vs 53 M / 33 |
| Full-match first line | 16.6 min |
| 5-min chunk first line | 4.8 min |
| 18 s clip first line | 0.5 min |
| Local inference | $0.010/match |
| Cloud inference | $0.392/match |
| Training | 3.64 h; $0.109 local / $4.40 cloud |
| Human scenario | $300 live / $400 pre-recorded |
| TTS | 43 calls, 412.2 s speech |

## 4. Likely questions and answers

**Why did you build this?**
Broadcast commentary is manual, expensive and does not scale to every match or language. Existing AI systems are either template-based and repetitive, or large end-to-end models that are expensive, opaque and prone to hallucination. I wanted to test a middle point: a modular, small-GPU pipeline with an auditable fact layer.

**What is your main contribution?**
Not a new state of the art. It is an auditable, reproducible instantiation of the full broadcast-to-speech pipeline: deterministic facts separated from language, score-bug goal anchors, a grounding validator with template fallback, and an honest evaluation of where the system fails.

**Why not use a large multimodal model end-to-end?**
Cost, reproducibility and auditability. A 7B MLLM is hard to run on a 4 GB GPU, and it decides facts and language together, so a hallucinated score is hard to separate from a language error. My design lets me inspect the fact sheet before the language model sees it.

**Why GPT-2 and LoRA?**
GPT-2 is small enough to train and run on a consumer GPU. LoRA freezes the base weights and trains low-rank adapters, which reduces trainable parameters and memory. I am not claiming GPT-2 is better than a large model; I am testing whether a small model plus grounding is useful.

**Is GPT-2 actually being used, or is it templates?**
GPT-2 is the primary generator. The template is a fallback only when the validator rejects the line. In the v4 reel, 10/10 lines were GPT-2 and 0 were templates. In the corrected full-match run, the two goal lines fell back to templates because the goal validator rejected GPT-2’s goal attempts. The rest were mostly GPT-2.

**How do you prevent hallucination?**
Three layers: the fact sheet is measured before generation; the validator checks the generated line against team, event, score, digits and a goal-specific phrase check; and the template fallback is used if the candidate fails. The validator is a guard against invented facts, not a measure of fluency. It accepted 38/40 on one set, but only 15/40 under a stricter path, so I do not claim it solves hallucination.

**Why is the human reference column empty in the comparison?**
Because the human line is the reference, not a candidate. BLEU/ROUGE/METEOR compare a candidate to a reference. If I scored the human reference against itself, every metric would be 1.0 by definition. That would be a meaningless ceiling. The 31.9 average-word value is filled in because that is a descriptive property of the reference, not an overlap score.

**Why is BLEU-4 so low?**
BLEU-4 requires long exact n-gram matches, which is harsh for free-form commentary where many different sentences are valid. ROUGE-L and METEOR are more informative here, but none of these metrics measures factual correctness. That is why the paper also reports validator acceptance and qualitative error analysis.

**Why is general event recall only 0.40?**
The event classifier over-produces pass and dead-time lines and misses rare classes. Against the ten curated moments it finds both goals, two of six shots and no tackles. The goal path is strong because the score bug gives a precise anchor; shots and tackles have no equivalent anchor. The reference is also curated, not an exhaustive annotation, so this is a recall/consistency check, not a full event-spotting benchmark.

**Why are goals reliable but other events not?**
The score bug is a persistent broadcast graphic that flips exactly on a goal and names the side. That is much stronger evidence than a ResNet classifier looking at a short window. For shots and tackles, the classifier relies on visual and audio cues that are noisy under broadcast cuts.

**Is the system live?**
No. It is pre-recorded. The full match is faster than real time in aggregate — 19.4 minutes for 64.5 minutes — but the first line appears after 16.6 minutes because the pipeline scores, writes and only then voices. A 5-minute causal chunk gets the first line out in 4.8 minutes, but it mislabels both goals as Argentina and invents an earlier goal. So shorter inputs alone do not make it live; it needs streaming perception, incremental TTS and side propagation.

**Why did the chunk get the goal side wrong?**
The score-bug anchor knows the side — the anchors were away 44:33 and away 45:42 — but the side is not carried into the realised line in the chunk path, and the short context makes the event classifier over-fire. That is a concrete architectural failure, not just a language error.

**Why YOLO26n and not YOLOv5lu?**
I had a 4 GB GPU and a four-hour window. YOLO26x/l could not finish a meaningful run in that time; YOLO26n could. It reached mAP50 0.805 / mAP50–95 0.481 in 2.70 hours. The comparison with YOLOv5lu is not equal-budget: 50 vs 33 epochs and 2.39 M vs 53 M parameters. I do not claim YOLO26 is proven better; the fair 640 px / 100-epoch SageMaker run is still pending.

**Why use the score-bug reader?**
It gives high-precision goal anchors, names the scoring side, and survives replays and celebration cuts. It also infers the score already on the board, so a clip starting at 2–1 is not treated as 0–0. The limitation is that it is tuned to this broadcaster’s overlay; another feed would need recalibration.

**How does possession work?**
Jersey colours are clustered with KMeans into two teams. For each frame, possession is assigned to the side of the tracked player nearest the ball. It is a heuristic, not a possession model; broadcast cuts and close-ups break continuity, so the ball path is interpolated only across bounded gaps and teleports are rejected.

**Why Fish Audio?**
It is a production TTS with emotion/tone control and low latency, and the free tier made the demo possible. There is no peer-reviewed sports-commentary TTS benchmark, so I do not claim a sports-specific prosody evaluation. The paper labels Fish as a vendor source.

**What is ClearML doing?**
Every YOLO and GPT-2 run is a task with parameters, metrics, artifacts and a registered model. That gives reproducibility and an audit trail. The report links the tasks and shows the curves and logs.

**What is “grounded description”?**
It is the deterministic fact sheet: score, minute, ball zone, possession, tracked-player count and event team. The language model receives these facts and must realise them. It is not allowed to add a new score, team or player. The validator compares the generated line back to this sheet.

**What is novel here?**
Not a new detector or a new LLM. The novelty is the system-level combination: a deterministic fact layer, score-bug goal anchoring, a validator with template fallback, a full broadcast-to-speech run on a consumer GPU, and a paper that reports the negative results — live latency, event recall, team attribution — instead of hiding them.

**What are the biggest weaknesses?**
General event recall, team-side propagation, live latency, and the absence of a human evaluation. The reference moments are curated, not exhaustive; the human cost is illustrative; the YOLO comparison is unequal-budget; the generative model samples, so lines vary.

**What would you do next?**
Annotate a proper event test set; build a streaming mode; carry the score-bug side explicitly into the line; improve the ball detector; run the fair YOLO26x vs YOLOv5lu comparison; and run a blind human evaluation of AI vs human commentary.

**Is the comparison fair?**
For commentary, both GPT-2 and the template are scored against the same human references, so that comparison is controlled. For YOLO, it is not equal-budget, and I say so. For cost, the human rate is illustrative, not a quote. For event spotting, the reference is curated, not exhaustive.

**What about ethics?**
The system could affect commentary jobs, has pronunciation and bias risks, and needs rights-cleared footage. I did not study those; the paper’s claim is technical. A production system would need human review, especially for goals and team attribution.

**What would you change if you started again?**
I would build the streaming path and the event annotation set first, because they determine whether the system is useful. I would also collect a human evaluation early, not at the end.

## 5. Trap questions

**“So the human commentary is better?”**
“The experiment does not measure that. It measures similarity to human references; the human column is the reference, not a competitor. A proper answer needs a blind human preference or MOS study, which I have not run.”

**“Your GPT-2 is just templates.”**
“No — the reel is 10/10 GPT-2. The full-match goals fell back to templates because the goal validator rejected GPT-2’s goal attempts. The fallback is deliberate and auditable.”

**“YOLO26 is newer, so it must be better.”**
“Not necessarily at unequal size, resolution and budget. My YOLO26n run is 2.39 M parameters and 50 epochs; the YOLOv5lu run is 53 M and 33 epochs. The fair full-size 640 px / 100-epoch comparison is still pending.”

**“Your event recall is only 0.40, so the system does not work.”**
“The goal path works — 2/2 with precision and recall 1.0. The general event classifier does not. That is exactly the finding: score-bug anchors are strong; the window classifier is weak. The paper reports both.”

**“Why not just use a 7B model?”**
“It would likely improve fluency, but it would not solve grounding by itself, and it would not fit the 4 GB target. My contribution is the fact layer and the auditability, not the largest possible language model.”

## 6. 60-second demo script

“This is the Matchcall demo. The reel is 10 moments, voiced by Fish; the tracking tab shows players, ball, teams and possession; the YOLO26n tab is the newer detector; and the report has the ClearML tasks and curves.

If I upload a clip, the pipeline spots the score-bug goal anchor, tracks the players, writes a grounded description, GPT-2 realises it, the validator checks it, and Fish voices it. On the 81' France goal, the score bug flips 2–1 to 2–2, the reader infers the start score, and the goal is anchored. In the full match, both goals are found; the weakness is non-goal events and live latency.”

## 7. Final three sentences to remember

1. “The language model never decides the facts; it only realises them.”
2. “Goals are reliable because the score bug is a high-precision anchor; general event spotting is not.”
3. “The system is a low-cost, auditable pre-recorded pipeline, not a live broadcast replacement.”
