# Experiment Results

Generated from measured artifacts. Cost assumptions are explicit and are not vendor quotes.

**Research paper:** [`docs/Matchcall_Paper_v5.docx`](Matchcall_Paper_v5.docx)  
**Visual summary:** [`docs/matchcall_experiments.html`](matchcall_experiments.html) · **PDF:** [`docs/Matchcall_Experiments.pdf`](Matchcall_Experiments.pdf)

## 1. AI vs human commentary

Held-out SoccerReplay prompts (n=40). Human commentary is the reference; the template baseline uses the same facts.

| Metric | GPT-2 v4 | Template | Human reference |
|---|---:|---:|---:|
| BLEU-1 | 0.0300 | 0.0103 | — |
| BLEU-2 | 0.0014 | 0.0020 | — |
| BLEU-4 | 0.000304 | 0.001038 | — |
| ROUGE-L | 0.1845 | 0.0931 | — |
| METEOR | 0.2066 | 0.0269 | — |
| Avg words | 40.5 | 7.4 | 31.9 |
| Grounding accepted | 38/40 (95%) | — | — |

The human reference is the target, so its self-similarity is 1 by definition and is not shown as a system score. BLEU-4 is near zero for both systems; ROUGE-L and METEOR favour GPT-2. Overlap metrics do not measure factual correctness.

## 2. Event spotting

Reference: 10 curated ARG–FRA moments (6 shots, 2 tackles, 2 goals). Detected: 38 event lines from the corrected full-match run. Tolerance ±1 min.

| Kind | Reference | Matched | Recall |
|---|---:|---:|---:|
| goal | 2 | 2 | 1.000 |
| shot | 6 | 2 | 0.333 |
| tackle | 2 | 0 | 0.000 |
| **overall** | **10** | **4** | **0.400** |

Goal precision/recall against the known 80' and 81' goals: **1.00 / 1.00** (detected at [79.63, 80.78]). Timing MAE for matched events: 36.2 s.

Separate object-detection result (not event spotting): YOLO26n mAP50 **0.805**, mAP50–95 **0.481**, precision 0.864, recall 0.763.

Limitation: the reference moments are curated highlight moments, not an exhaustive independent annotation, so this is a consistency check and does not measure event precision.

## 3. Live vs pre-recorded

| Mode | Input | Wall time | Ratio to real time | Time to first line |
|---|---:|---:|---:|---:|
| 18 s clip | 0.3 min | 0.7 min | 2.21× | 0.5 min |
| 64.5 min full match | 64.5 min | 19.4 min | 0.30× | 16.6 min |

The full-match run is faster than real time in aggregate (0.30×), but it is a batch job: the first line appears only after scoring, classification and generation. The current system is **pre-recorded only**; a live mode needs a streaming pipeline with bounded latency.

## 4. Cost comparison

| Item | Local GPU | Cloud ml.g5.2xlarge | Human team |
|---|---:|---:|---:|
| Inference per 64.5-min match | $0.010 | $0.392 | $300 live / $400 pre-recorded |
| Training (YOLO26n + GPT-2 v4) | $0.109 | $4.40 | — |
| Training hours | 3.64 h | 3.64 h (same run) | — |
| TTS | 43 calls, 412.2 s speech | same | — |

Assumptions: cloud $1.21/h; local electricity at $0.30/kWh; human team of 2 at $50/h for 3 h live / 4 h pre-recorded. Human rates are illustrative; the range is $25–$100/h. Fish: s2.1-pro-free tier; no API charge measured. Cloud storage/egress and paid TTS are excluded.

Across the human rate range, compute is roughly two to three orders of magnitude cheaper than the illustrative human team; the trade-off is latency and commentary quality, not raw cost.

## 5. What the experiments support

- **AI vs human:** GPT-2 v4 is more similar to human commentary than templates on ROUGE-L/METEOR and is much longer, but BLEU-4 is near zero and factual quality still needs human evaluation.

- **Event spotting:** goals are reliable (2/2 via score bug); shots and tackles are weak; object detection mAP is a separate metric and should not be reported as spotting accuracy.

- **Live vs pre-recorded:** the current system is a batch pre-recorded pipeline; it is not live.

- **Cost:** compute is very cheap relative to human commentary; the open costs are latency, quality and TTS pricing.
