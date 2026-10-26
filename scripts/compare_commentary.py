"""Compare GPT-2 LoRA v4 against the template baseline and human references.

For held-out SoccerReplay prompts:
  * run the v4 announcer (grounded line, template fallback)
  * run the template announcer on the same facts
  * score both against the human reference commentary with BLEU-1/2/4,
    ROUGE-L and METEOR
  * report grounding acceptance, line length and lexical diversity

Outputs:
  output/comparison/commentary_comparison.json
  output/comparison/commentary_comparison.csv
  output/comparison/commentary_metrics.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
os.environ.setdefault("GPT2_LORA_DIR", str(REPO / "models" / "gpt2_lora_v4"))
sys.path.insert(0, str(ROOT))

from app.commentary import TemplateAnnouncer  # noqa: E402
from app import commentary_gpt2 as cg  # noqa: E402
from scripts.eval_commentator_lora import parse  # noqa: E402

OUT = REPO / "output" / "comparison"
FIG = ROOT / "docs" / "figures"


def _tok(text: str) -> list[str]:
    return [w for w in text.lower().replace("—", " ").split() if any(c.isalnum() for c in w)]


def _bleu(refs, hyps, n=4) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    weights = tuple([1.0 / n] * n)
    try:
        return float(corpus_bleu(refs, hyps, weights=weights,
                                 smoothing_function=SmoothingFunction().method1))
    except Exception:  # noqa: BLE001
        return 0.0


def _rouge_l(refs, hyps) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    return float(statistics.mean(scores)) if scores else 0.0


def _meteor(refs, hyps) -> float:
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:  # noqa: BLE001
            pass
    from nltk.translate.meteor_score import meteor_score
    scores = []
    for r, h in zip(refs, hyps):
        try:
            scores.append(float(meteor_score([_tok(r)], _tok(h))))
        except Exception:  # noqa: BLE001
            pass
    return float(statistics.mean(scores)) if scores else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default=str(ROOT / "data" / "commentator_valid_v4.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["GPT2_LORA_DIR"] = str(REPO / "models" / "gpt2_lora_v4")
    dev = args.device
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(REPO / "models" / "gpt2_lora_v4"))
    model.eval()
    if dev != "cpu":
        model = model.to(dev)
    cg._MODEL = (tok, model)

    rows = [json.loads(l)["text"] for l in Path(args.valid).read_text(encoding="utf-8").splitlines()
            if l.strip()][:args.n]
    records = []
    for text in rows:
        p = parse(text)
        if not p:
            continue
        kind, team, teams, feats = p
        ref = text.split("[COMMENTARY]", 1)[1].strip()
        tpl, tpl_exc = TemplateAnnouncer(teams).event_line(kind, feats)
        src = "gpt2"
        try:
            line, exc = cg.event_line(kind, feats, teams)
        except Exception:  # noqa: BLE001
            line, exc = tpl, tpl_exc
            src = "template"
        records.append({
            "kind": kind, "team": team, "reference": ref,
            "gpt2_or_fallback": line, "template": tpl,
            "source": src, "excitement": exc,
            "words": len(line.split()),
            "template_words": len(tpl.split()),
        })
        print(f"[{src:8s}] {kind}/{team}: {line[:100]}", flush=True)

    refs = [_tok(r["reference"]) for r in records]
    hyp_v4 = [_tok(r["gpt2_or_fallback"]) for r in records]
    hyp_tpl = [_tok(r["template"]) for r in records]
    n = len(records)
    metrics = {
        "n": n,
        "gpt2_accepted": sum(1 for r in records if r["source"] == "gpt2"),
        "accept_rate": round(sum(1 for r in records if r["source"] == "gpt2") / max(1, n), 3),
        "avg_words_v4": round(statistics.mean(r["words"] for r in records), 1),
        "avg_words_template": round(statistics.mean(r["template_words"] for r in records), 1),
        "v4": {
            "bleu1": _bleu(refs, hyp_v4, 1),
            "bleu2": _bleu(refs, hyp_v4, 2),
            "bleu4": _bleu(refs, hyp_v4, 4),
            "rougeL": _rouge_l([r["reference"] for r in records],
                               [r["gpt2_or_fallback"] for r in records]),
            "meteor": _meteor([r["reference"] for r in records],
                              [r["gpt2_or_fallback"] for r in records]),
        },
        "template": {
            "bleu1": _bleu(refs, hyp_tpl, 1),
            "bleu2": _bleu(refs, hyp_tpl, 2),
            "bleu4": _bleu(refs, hyp_tpl, 4),
            "rougeL": _rouge_l([r["reference"] for r in records],
                               [r["template"] for r in records]),
            "meteor": _meteor([r["reference"] for r in records],
                              [r["template"] for r in records]),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / "commentary_comparison.json").write_text(
        json.dumps({"metrics": metrics, "records": records}, indent=2),
        encoding="utf-8")
    with (OUT / "commentary_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    # ---- chart -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["BLEU-1", "BLEU-2", "BLEU-4", "ROUGE-L", "METEOR"]
    keys = ["bleu1", "bleu2", "bleu4", "rougeL", "meteor"]
    v4 = [metrics["v4"][k] for k in keys]
    tp = [metrics["template"][k] for k in keys]
    x = range(len(labels))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    width = 0.38
    b1 = ax.bar([i - width / 2 for i in x], v4, width, label="GPT-2 v4 (grounded)",
                color="#3987e5")
    b2 = ax.bar([i + width / 2 for i in x], tp, width, label="Template baseline",
                color="#d95926")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score vs human reference")
    ax.set_title("Commentary quality against held-out human commentary", loc="left",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax2.bar(["accept"], [metrics["accept_rate"]], color="#199e70", width=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_title("Grounding\nacceptance", fontsize=9)
    ax2.text(0, metrics["accept_rate"] + 0.03, f"{metrics['accept_rate']*100:.0f}%",
             ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "commentary_metrics.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print("wrote", OUT / "commentary_comparison.json")
    print("wrote", FIG / "commentary_metrics.png")


if __name__ == "__main__":
    main()
