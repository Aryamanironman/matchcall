"""Bootstrap confidence intervals for the 40-prompt commentary comparison.

Reads output/comparison/commentary_comparison.json (main checkout) and writes
commentary_bootstrap.json next to it. No model inference; this is a statistics
pass over the recorded lines.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve()
_CANDIDATES = [
    HERE.parents[4] / "output" / "comparison",  # worktree -> main checkout
    HERE.parents[2] / "output" / "comparison",  # main checkout
]
OUT = next((p for p in _CANDIDATES
            if (p / "commentary_comparison.json").exists()), _CANDIDATES[0])
SRC = OUT / "commentary_comparison.json"


def tok(text: str) -> list[str]:
    return [w for w in text.lower().replace("—", " ").split()
            if any(c.isalnum() for c in w)]


def bleu(refs, hyps, n: int) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    weights = tuple([1.0 / n] * n)
    try:
        return float(corpus_bleu(refs, hyps, weights=weights,
                                 smoothing_function=SmoothingFunction().method1))
    except Exception:  # noqa: BLE001
        return 0.0


def rouge_l(refs: list[str], hyps: list[str]) -> list[float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]


def meteor(refs: list[str], hyps: list[str]) -> list[float]:
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:  # noqa: BLE001
            pass
    from nltk.translate.meteor_score import meteor_score
    out = []
    for r, h in zip(refs, hyps):
        try:
            out.append(float(meteor_score([tok(r)], tok(h))))
        except Exception:  # noqa: BLE001
            out.append(0.0)
    return out


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    recs = payload["records"]
    refs = [r["reference"] for r in recs]
    v4 = [r["gpt2_or_fallback"] for r in recs]
    tpl = [r["template"] for r in recs]
    ref_tok = [tok(x) for x in refs]
    v4_tok = [tok(x) for x in v4]
    tpl_tok = [tok(x) for x in tpl]
    n = len(recs)

    # per-record components (resample the means without recomputing each time)
    rl_v4 = rouge_l(refs, v4)
    rl_tpl = rouge_l(refs, tpl)
    mt_v4 = meteor(refs, v4)
    mt_tpl = meteor(refs, tpl)

    def corpus(metric: str, idx: list[int], side: str):
        rs = [ref_tok[i] for i in idx]
        hs = [v4_tok[i] if side == "v4" else tpl_tok[i] for i in idx]
        if metric.startswith("bleu"):
            return bleu(rs, hs, int(metric[-1]))
        arr = rl_v4 if side == "v4" else rl_tpl
        if metric == "rougeL":
            return statistics.mean(arr[i] for i in idx)
        arr = mt_v4 if side == "v4" else mt_tpl
        return statistics.mean(arr[i] for i in idx)

    metrics = ["bleu1", "bleu2", "bleu4", "rougeL", "meteor"]
    all_idx = list(range(n))
    point = {m: {"v4": corpus(m, all_idx, "v4"),
                 "template": corpus(m, all_idx, "template")} for m in metrics}
    rng = random.Random(17)
    B = 1000
    draws = {m: {"v4": [], "template": [], "diff": []} for m in metrics}
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        for m in metrics:
            a = corpus(m, idx, "v4")
            b = corpus(m, idx, "template")
            draws[m]["v4"].append(a)
            draws[m]["template"].append(b)
            draws[m]["diff"].append(a - b)

    rows = []
    for m in metrics:
        d = draws[m]["diff"]
        lo, hi = pct(d, .025), pct(d, .975)
        # two-sided bootstrap p: how often the difference crosses zero
        p = 2 * min(sum(1 for x in d if x <= 0), sum(1 for x in d if x >= 0)) / B
        rows.append({
            "metric": m,
            "gpt2": round(point[m]["v4"], 5),
            "template": round(point[m]["template"], 5),
            "difference": round(point[m]["v4"] - point[m]["template"], 5),
            "ci95_low": round(lo, 5),
            "ci95_high": round(hi, 5),
            "p_bootstrap": round(min(1.0, p), 4),
        })

    out = {
        "n": n,
        "resamples": B,
        "seed": 17,
        "note": "Percentile bootstrap over the 40 recorded prompts; paired by prompt. "
                "p is a two-sided bootstrap crossing probability, not a parametric test.",
        "rows": rows,
    }
    (OUT / "commentary_bootstrap.json").write_text(json.dumps(out, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
