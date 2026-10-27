"""Compile the four paper comparisons into one JSON + Markdown result sheet.

Inputs are measured artifacts:
  output/comparison/commentary_comparison.json
  output/experiments/event_spotting.json
  output/experiments/full_match_job.json
  output/experiments/chunk_job.json

Cost assumptions are explicit in ASSUMPTIONS and are not measured prices.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
OUT = REPO / "output" / "experiments"
DOC = ROOT / "docs" / "experiments_results.md"
FIG = ROOT / "docs" / "figures"

ASSUMPTIONS = {
    "cloud_instance": "ml.g5.2xlarge",
    "cloud_usd_per_hour": 1.21,
    "local_gpu": "Local GPU",
    "local_power_kw": 0.10,
    "electricity_usd_per_kwh": 0.30,
    "human_team_size": 2,
    "human_live_hours": 3.0,
    "human_prerecorded_hours": 4.0,
    "human_usd_per_hour": 50.0,
    "human_usd_per_hour_range": [25.0, 50.0, 100.0],
    "fish_pricing": "s2.1-pro-free tier; no API charge measured",
}

# Measured from job 7c23a8a91829 (full match) and feebe1ed5d60 (5-min chunk).
LIVE = {
    "full_match": {
        "video_s": 3871.0, "wall_s": 1165.0, "ratio": 1165.0 / 3871.0,
        "time_to_first_line_s": 993.0, "segments": 43, "voiced": 43,
        "speech_s": 412.2,
    },
    "chunk_5min": {
        "video_s": 300.0, "wall_s": 311.8, "ratio": 311.8 / 300.0,
        "time_to_first_line_s": 286.0, "segments": 27, "voiced": 27,
        "speech_s": None,
    },
    "clip_18s": {
        "video_s": 18.0, "wall_s": 39.8, "ratio": 39.8 / 18.0,
        "time_to_first_line_s": 31.0, "segments": 2, "voiced": 2,
        "speech_s": 14.8,
    },
    "training": {"yolo26n_hours": 2.704, "gpt2_v4_hours": 0.935},
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    comp = load(REPO / "output" / "comparison" / "commentary_comparison.json")
    ev = load(OUT / "event_spotting.json")
    full = load(OUT / "full_match_job.json")
    chunk = load(OUT / "chunk_job.json")

    metrics = comp["metrics"]
    records = comp["records"]
    ref_words = [len(r["reference"].split()) for r in records]
    gpt_words = [r["words"] for r in records]

    a = ASSUMPTIONS
    wall_h = LIVE["full_match"]["wall_s"] / 3600.0
    train_h = (LIVE["training"]["yolo26n_hours"]
               + LIVE["training"]["gpt2_v4_hours"])
    local_kwh = wall_h * a["local_power_kw"]
    cost = {
        "inference_local_usd": round(
            local_kwh * a["electricity_usd_per_kwh"], 4),
        "inference_cloud_usd": round(wall_h * a["cloud_usd_per_hour"], 3),
        "inference_cloud_usd_per_match_minute": round(
            wall_h * a["cloud_usd_per_hour"] / (3871.0 / 60.0), 5),
        "training_local_usd": round(
            train_h * a["local_power_kw"] * a["electricity_usd_per_kwh"], 3),
        "training_cloud_usd": round(train_h * a["cloud_usd_per_hour"], 2),
        "training_hours": round(train_h, 3),
        "human_live_usd": round(
            a["human_team_size"] * a["human_live_hours"]
            * a["human_usd_per_hour"], 2),
        "human_prerecorded_usd": round(
            a["human_team_size"] * a["human_prerecorded_hours"]
            * a["human_usd_per_hour"], 2),
        "human_range_usd": {
            str(rate): {
                "live": round(a["human_team_size"] * a["human_live_hours"] * rate, 2),
                "prerecorded": round(
                    a["human_team_size"] * a["human_prerecorded_hours"] * rate, 2),
            } for rate in a["human_usd_per_hour_range"]
        },
        "fish_calls": LIVE["full_match"]["segments"],
        "fish_speech_s": LIVE["full_match"]["speech_s"],
        "fish_pricing": a["fish_pricing"],
    }

    out = {
        "assumptions": a,
        "ai_vs_human": {
            "n": metrics["n"],
            "gpt2_accepted": metrics["gpt2_accepted"],
            "accept_rate": metrics["accept_rate"],
            "avg_words_gpt2": metrics["avg_words_v4"],
            "avg_words_template": metrics["avg_words_template"],
            "avg_words_human_reference": round(statistics.mean(ref_words), 1),
            "gpt2": metrics["v4"],
            "template": metrics["template"],
            "human_reference": "SoccerReplay-1988 held-out human commentary",
        },
        "event_spotting": ev,
        "live_vs_prerecorded": LIVE,
        "cost": cost,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "experiment_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    # ---- Markdown result sheet ---------------------------------------------
    md = []
    md.append("# Experiment Results\n")
    md.append("Generated from measured artifacts. Cost assumptions are explicit "
              "and are not vendor quotes.\n")

    md.append("## 1. AI vs human commentary\n")
    md.append(f"Held-out SoccerReplay prompts (n={metrics['n']}). Human "
              "commentary is the reference; the template baseline uses the same "
              "facts.\n")
    md.append("| Metric | GPT-2 v4 | Template | Human reference |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| BLEU-1 | {metrics['v4']['bleu1']:.4f} | "
              f"{metrics['template']['bleu1']:.4f} | — |")
    md.append(f"| BLEU-2 | {metrics['v4']['bleu2']:.4f} | "
              f"{metrics['template']['bleu2']:.4f} | — |")
    md.append(f"| BLEU-4 | {metrics['v4']['bleu4']:.6f} | "
              f"{metrics['template']['bleu4']:.6f} | — |")
    md.append(f"| ROUGE-L | {metrics['v4']['rougeL']:.4f} | "
              f"{metrics['template']['rougeL']:.4f} | — |")
    md.append(f"| METEOR | {metrics['v4']['meteor']:.4f} | "
              f"{metrics['template']['meteor']:.4f} | — |")
    md.append(f"| Avg words | {metrics['avg_words_v4']:.1f} | "
              f"{metrics['avg_words_template']:.1f} | "
              f"{statistics.mean(ref_words):.1f} |")
    md.append(f"| Grounding accepted | {metrics['gpt2_accepted']}/"
              f"{metrics['n']} ({metrics['accept_rate']*100:.0f}%) | — | — |")
    md.append("\nThe human reference is the target, so its self-similarity is "
              "1 by definition and is not shown as a system score. BLEU-4 is "
              "near zero for both systems; ROUGE-L and METEOR favour GPT-2. "
              "Overlap metrics do not measure factual correctness.\n")

    md.append("## 2. Event spotting\n")
    md.append(f"Reference: {ev['reference_moments']} curated ARG–FRA moments "
              f"(6 shots, 2 tackles, 2 goals). Detected: "
              f"{ev['detected_event_lines']} event lines from the corrected "
              f"full-match run. Tolerance ±{ev['tolerance_minutes']:.0f} min.\n")
    md.append("| Kind | Reference | Matched | Recall |")
    md.append("|---|---:|---:|---:|")
    for kind in ("goal", "shot", "tackle"):
        ref_n = ev["reference_by_kind"].get(kind, 0)
        matched = ev["matched_by_kind"].get(kind, 0)
        rec = ev["recall_by_kind"].get(kind, 0.0)
        md.append(f"| {kind} | {ref_n} | {matched} | {rec:.3f} |")
    md.append(f"| **overall** | **{ev['reference_moments']}** | "
              f"**{sum(ev['matched_by_kind'].values())}** | "
              f"**{ev['recall_overall']:.3f}** |")
    md.append(f"\nGoal precision/recall against the known 80' and 81' goals: "
              f"**{ev['goal_spotting']['precision']:.2f} / "
              f"{ev['goal_spotting']['recall']:.2f}** "
              f"(detected at {ev['goal_spotting']['detected_minutes']}). "
              f"Timing MAE for matched events: {ev['timing_mae_s']} s.\n")
    y = ev["object_detection_yolo26n"]
    md.append(f"Separate object-detection result (not event spotting): "
              f"YOLO26n mAP50 **{y['mAP50']:.3f}**, mAP50–95 "
              f"**{y['mAP50_95']:.3f}**, precision {y['precision']:.3f}, "
              f"recall {y['recall']:.3f}.\n")
    md.append("Limitation: the reference moments are curated highlight "
              "moments, not an exhaustive independent annotation, so this is "
              "a consistency check and does not measure event precision.\n")

    md.append("## 3. Live vs pre-recorded\n")
    md.append("| Mode | Input | Wall time | Ratio to real time | "
              "Time to first line |")
    md.append("|---|---:|---:|---:|---:|")
    for name, label in (("clip_18s", "18 s clip"),
                        ("chunk_5min", "5 min chunk"),
                        ("full_match", "64.5 min full match")):
        x = LIVE[name]
        md.append(f"| {label} | {x['video_s']/60:.1f} min | "
                  f"{x['wall_s']/60:.1f} min | {x['ratio']:.2f}× | "
                  f"{x['time_to_first_line_s']/60:.1f} min |")
    md.append("\nThe full-match run is faster than real time in aggregate "
              "(0.30×), but it is a batch job: the first line appears only "
              "after scoring, classification and generation. The 5-minute "
              "chunk took 5.2 minutes and still mislabelled the goal side, so "
              "the current system is **pre-recorded only**; a live mode needs "
              "a streaming pipeline with bounded latency.\n")

    md.append("## 4. Cost comparison\n")
    md.append("| Item | Local GPU | Cloud ml.g5.2xlarge | Human team |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Inference per 64.5-min match | ${cost['inference_local_usd']:.3f} | "
              f"${cost['inference_cloud_usd']:.3f} | "
              f"${cost['human_live_usd']:.0f} live / "
              f"${cost['human_prerecorded_usd']:.0f} pre-recorded |")
    md.append(f"| Training (YOLO26n + GPT-2 v4) | "
              f"${cost['training_local_usd']:.3f} | "
              f"${cost['training_cloud_usd']:.2f} | — |")
    md.append(f"| Training hours | {cost['training_hours']:.2f} h | "
              f"{cost['training_hours']:.2f} h (same run) | — |")
    md.append(f"| TTS | {cost['fish_calls']} calls, "
              f"{cost['fish_speech_s']:.1f} s speech | same | — |")
    md.append(f"\nAssumptions: cloud ${a['cloud_usd_per_hour']}/h; local "
              f"{a['local_power_kw']*1000:.0f} W at "
              f"${a['electricity_usd_per_kwh']:.2f}/kWh; human team of "
              f"{a['human_team_size']} at ${a['human_usd_per_hour']:.0f}/h "
              f"for {a['human_live_hours']:.0f} h live / "
              f"{a['human_prerecorded_hours']:.0f} h pre-recorded. "
              f"Human rates are illustrative; the range is "
              f"${a['human_usd_per_hour_range'][0]:.0f}–"
              f"${a['human_usd_per_hour_range'][-1]:.0f}/h. "
              f"Fish: {cost['fish_pricing']}. Cloud storage/egress and paid "
              "TTS are excluded.\n")
    md.append("Across the human rate range, compute is roughly two to three "
              "orders of magnitude cheaper than the illustrative human team; "
              "the trade-off is latency and commentary quality, not raw cost.\n")

    md.append("## 5. What the experiments support\n")
    md.append("- **AI vs human:** GPT-2 v4 is more similar to human commentary "
              "than templates on ROUGE-L/METEOR and is much longer, but BLEU-4 "
              "is near zero and factual quality still needs human evaluation.\n")
    md.append("- **Event spotting:** goals are reliable (2/2 via score bug); "
              "shots and tackles are weak; object detection mAP is a separate "
              "metric and should not be reported as spotting accuracy.\n")
    md.append("- **Live vs pre-recorded:** the current system is a batch "
              "pre-recorded pipeline; it is not live.\n")
    md.append("- **Cost:** compute is very cheap relative to human commentary; "
              "the open costs are latency, quality and TTS pricing.\n")

    DOC.write_text("\n".join(md), encoding="utf-8")

    # ---- event recall chart -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ["goal", "shot", "tackle", "overall"]
    vals = [ev["recall_by_kind"].get("goal", 0),
            ev["recall_by_kind"].get("shot", 0),
            ev["recall_by_kind"].get("tackle", 0),
            ev["recall_overall"]]
    colors = ["#C9FF4F", "#4FE0D0", "#FF6B3D", "#8FA99B"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(kinds, vals, color=colors, width=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall at ±1 min")
    ax.set_title("Event spotting against 10 curated ARG–FRA moments",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "event_spotting.png", dpi=180)
    plt.close(fig)

    print("wrote", OUT / "experiment_results.json")
    print("wrote", DOC)
    print("wrote", FIG / "event_spotting.png")
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
