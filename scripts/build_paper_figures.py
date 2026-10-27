"""Build publication-style figures for the Word paper."""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
OUT = ROOT / "docs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

GREEN = "#1f5c3a"
TEAL = "#0b6b73"
RED = "#a33a1f"
GOLD = "#a36600"
PURPLE = "#6b4bb5"
GRAY = "#666b66"
LIGHT = "#e9ece8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.unicode_minus": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#d9ded8",
    "grid.linewidth": 0.6,
})


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def pipeline():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    stages = [
        ("1", "Action spotting", "score-bug anchors · motion windows · ResNet-18"),
        ("2", "Tracking", "YOLO26n · ByteTrack · jersey teams · possession"),
        ("3", "Facts", "score · minute · ball zone · possession · event team"),
        ("4", "Language", "GPT-2 LoRA · grounding validator · template fallback"),
        ("5", "Speech", "Fish Audio · event pacing · ffmpeg mix"),
    ]
    ys = [8.45, 6.55, 4.65, 2.75, 0.85]
    for i, (num, title, sub) in enumerate(stages):
        y = ys[i]
        box = FancyBboxPatch((0.45, y), 9.1, 1.30, boxstyle="round,pad=0.02,rounding_size=0.05",
                             linewidth=0.9, edgecolor="#9aa69b", facecolor="#f4f7f3")
        ax.add_patch(box)
        ax.text(0.85, y + 0.86, num, fontsize=10, color=GREEN, fontweight="bold")
        ax.text(1.35, y + 0.86, title, fontsize=11.5, color="#111111", fontweight="bold")
        ax.text(1.35, y + 0.38, sub, fontsize=8.4, color=GRAY, va="center")
        if i < len(stages) - 1:
            ax.annotate("", xy=(5.0, y - 0.60), xytext=(5.0, y - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="#666b66", lw=1.2))
    save(fig, "fig1_pipeline.png")


def commentary():
    comp = load(REPO / "output" / "comparison" / "commentary_comparison.json")
    m = comp["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), constrained_layout=True)
    ax = axes[0]
    labels = ["ROUGE-L", "METEOR"]
    gpt = [m["v4"]["rougeL"], m["v4"]["meteor"]]
    tpl = [m["template"]["rougeL"], m["template"]["meteor"]]
    x = np.arange(len(labels)); width = 0.34
    b1 = ax.bar(x - width/2, gpt, width, color=GREEN, label="GPT-2 v4")
    b2 = ax.bar(x + width/2, tpl, width, color=TEAL, label="Template")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.25); ax.set_ylabel("Score")
    ax.set_title("(a) Overlap with human reference", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.7)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax = axes[1]
    labels = ["GPT-2 v4", "Human", "Template"]
    vals = [m["avg_words_v4"], 31.9, m["avg_words_template"]]
    bars = ax.bar(labels, vals, color=[GREEN, TEAL, RED], width=0.55)
    ax.set_ylim(0, 48); ax.set_ylabel("Words per line")
    ax.set_title("(b) Average line length", loc="left")
    ax.grid(axis="y", alpha=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    save(fig, "fig2_commentary.png")


def latency():
    fig, ax = plt.subplots(figsize=(7.2, 2.3), constrained_layout=True)
    labels = ["18 s clip", "Full match"]
    vals = [0.5, 16.6]
    colors = [GREEN, TEAL]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.55)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 18); ax.set_xlabel("Time to first commentary line (min)")
    ax.grid(axis="x", alpha=0.7)
    for yy, v in zip(y, vals):
        ax.text(v+0.3, yy, f"{v:.1f}", va="center", fontsize=8)
    save(fig, "fig3_latency.png")


LINE = re.compile(r"^\[(\d+):(\d+)\]\s+\(([^|]+)\|")


def timeline():
    job = load(REPO / "output" / "experiments" / "full_match_job.json")
    ev = load(REPO / "output" / "experiments" / "event_spotting.json")
    off = 35.33
    detected = []
    for line in job.get("transcript", []):
        m = LINE.match(line)
        if not m:
            continue
        kind = m.group(3).strip()
        if kind == "ambience":
            continue
        detected.append((off + int(m.group(1)) + int(m.group(2))/60.0, kind))
    ref = [(38, "shot"), (40, "tackle"), (49, "shot"), (54, "tackle"),
           (65, "shot"), (76, "shot"), (80, "goal"), (81, "goal"),
           (85, "shot"), (99, "shot")]
    matched = {(m["reference_minute"], m["reference_kind"]): m["detected_minute"]
               for m in ev["matches"]}
    colors = {"goal": GREEN, "shot": TEAL, "tackle": GOLD, "corner": PURPLE,
              "pass": "#b5bdb4", "dead_time": "#d5dad2"}
    fig, ax = plt.subplots(figsize=(9, 3.0), constrained_layout=True)
    ax.set_xlim(35, 100); ax.set_ylim(-1.55, 1.55)
    for x in range(35, 101, 5):
        ax.axvline(x, color="#e1e5df", lw=0.6, zorder=0)
    ax.axhline(0, color="#c8cec6", lw=0.8)
    for minute, kind in ref:
        key = (float(minute), kind)
        is_match = key in matched
        ax.scatter(minute, 0.78, s=70, color=colors[kind], edgecolor=GREEN if is_match else RED,
                   linewidth=1.4, zorder=3)
        if not is_match:
            ax.scatter(minute, 0.78, s=70, color="white", marker="x", linewidth=1.8, zorder=4)
    for minute, kind in detected:
        strong = kind in ("goal", "shot", "tackle", "corner")
        ax.vlines(minute, -0.88, -0.55, color=colors[kind], lw=2.4 if strong else 0.9,
                  alpha=0.95 if strong else 0.45, zorder=2)
    for (rmin, kind), dmin in matched.items():
        ax.plot([rmin, dmin], [0.74, -0.52], color=GREEN, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.text(35.4, 1.22, "Reference", color=TEAL, fontsize=9, fontweight="bold")
    ax.text(35.4, -1.18, "Detected", color=GREEN, fontsize=9, fontweight="bold")
    ax.text(80.5, 1.02, "goals 80'/81'", color=GREEN, fontsize=8.5, ha="center")
    ax.set_yticks([]); ax.set_xticks(range(35, 101, 5))
    ax.set_xlabel("Match minute")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig4_timeline.png")


def recall():
    ev = load(REPO / "output" / "experiments" / "event_spotting.json")
    labels = ["Goals", "Shots", "Tackles", "Overall"]
    vals = [ev["recall_by_kind"].get("goal", 0), ev["recall_by_kind"].get("shot", 0),
            ev["recall_by_kind"].get("tackle", 0), ev["recall_overall"]]
    fig, ax = plt.subplots(figsize=(6.2, 2.6), constrained_layout=True)
    bars = ax.bar(labels, vals, color=[GREEN, TEAL, RED, GRAY], width=0.55)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Recall at ±1 min")
    ax.grid(axis="y", alpha=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    save(fig, "fig5_recall.png")


def yolo_curves():
    y = load(REPO / "output" / "experiments" / "yolo_training.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(y["epochs"], y["box"], color=GREEN, lw=1.6, label="box")
    ax.plot(y["epochs"], y["cls"], color=TEAL, lw=1.6, label="cls")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
    ax.set_title("(a) Training loss", loc="left"); ax.legend(frameon=False); ax.grid(alpha=0.7)
    ax = axes[1]
    ax.plot(y["epochs"], y["map50"], color=GREEN, lw=1.6, label="mAP50")
    ax.plot(y["epochs"], y["map5095"], color=TEAL, lw=1.6, label="mAP50–95")
    ax.set_ylim(0, 1); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation mAP")
    ax.set_title("(b) Validation mAP", loc="left"); ax.legend(frameon=False); ax.grid(alpha=0.7)
    save(fig, "fig6_yolo.png")


def gpt2_curve():
    h = load(REPO / "output" / "experiments" / "llm_training.json")
    steps = [p["step"] for p in h]; loss = [p["loss"] for p in h]
    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    ax.plot(steps, loss, color=GREEN, lw=1.5)
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Training loss")
    ax.set_ylim(0, 4.1); ax.grid(alpha=0.7)
    ax.annotate(f"{loss[-1]:.3f}", xy=(steps[-1], loss[-1]), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color=GREEN)
    save(fig, "fig7_gpt2.png")


def cost():
    fig, ax = plt.subplots(figsize=(7.2, 2.2), constrained_layout=True)
    labels = ["Local", "Cloud", "Human"]
    vals = [0.0002, 0.006, 4.65]
    colors = [GREEN, TEAL, RED]
    y = np.arange(len(labels))[::-1]
    ax.set_xscale("log")
    ax.set_xlim(0.0001, 10)
    ax.scatter(vals, y, s=70, color=colors, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v, yy+0.18, f"${v:g}", ha="center", va="bottom", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Cost per match minute (USD, log scale)")
    ax.grid(axis="x", which="major", alpha=0.7)
    ax.set_ylim(-0.55, 2.55)
    save(fig, "fig8_cost.png")


if __name__ == "__main__":
    pipeline()
    commentary()
    latency()
    timeline()
    recall()
    yolo_curves()
    gpt2_curve()
    cost()
