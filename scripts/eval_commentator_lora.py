"""Measure a commentary adapter on held-out SoccerReplay prompts.

Reports the two numbers that matter for the demo: how often the grounding
validator accepts a generated line, and how long the accepted lines are.
It uses the runtime cleaner and validator, so the result is what the pipeline
would actually speak.

    python scripts/eval_commentator_lora.py --model-dir models/gpt2_lora_v4 --n 40
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.commentary_gpt2 import _clean, _validate  # noqa: E402

EVENT_RE = re.compile(r"\[EVENT\]\s*(\w+)\s+by\s+(.+?)\s+\[INTENSITY\]")
HISTORY_RE = re.compile(r"\[HISTORY\]\s*(.*?)\s*\[EVENT\]")
SCORE_RE = re.compile(r"(.+?)\s+(\d+)-(\d+)\s+(.+?)(?:,|$)")
AT_RE = re.compile(r"at\s+(.+?)\s+v\s+(.+?)(?:[;.]|$)")
MINUTE_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\s+minute|(\d+)\s+minutes?\s+gone")


def parse(text: str):
    ev = EVENT_RE.search(text)
    hist = HISTORY_RE.search(text)
    if not ev or not hist:
        return None
    kind, team = ev.group(1), ev.group(2).strip()
    h = hist.group(1)
    h = re.sub(r"^\d+'\s*[—–-]\s*", "", h)  # "79' — Spain 0-1 Italy"
    m = SCORE_RE.search(h)
    if m:
        home, hs, as_, away = (m.group(1).strip(), int(m.group(2)),
                               int(m.group(3)), m.group(4).strip())
    else:
        at = AT_RE.search(h)
        if not at:
            return None
        home, away = at.group(1).strip(), at.group(2).strip()
        hs = as_ = 0
    mm = MINUTE_RE.search(h)
    minute = int(mm.group(1) or mm.group(2)) if mm else 0
    return kind, team, {"home": home, "away": away}, {
        "score_home": hs, "score_away": as_, "minute": minute}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "gpt2_lora_v4"))
    ap.add_argument("--data", default=str(ROOT / "data" / "commentator_valid_v4.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--write-eval", default=None,
                    help="write metrics JSON (e.g. output/llm_eval.json)")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base, args.model_dir)
    model.eval()
    if dev != "cpu":
        model = model.half().to(dev)

    rows = [json.loads(l)["text"] for l in
            Path(args.data).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = rows[:args.n]
    accepted = 0
    lengths = []
    accepted_lengths = []
    sent2 = 0
    shown = 0
    for text in rows:
        parsed = parse(text)
        if not parsed:
            continue
        kind, team, teams, feats = parsed
        prompt = text.split("[COMMENTARY]", 1)[0] + "[COMMENTARY]"
        with torch.no_grad():
            ids = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **ids, max_new_tokens=96, min_new_tokens=14, do_sample=True,
                temperature=0.85, top_p=0.9, repetition_penalty=1.25,
                no_repeat_ngram_size=4, pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        if "[COMMENTARY]" in gen:
            gen = gen.split("[COMMENTARY]", 1)[1]
        line = _clean(gen)
        # mirror runtime/commentary_gpt2.event_line: a grounded sentence that
        # omits the team gets a broadcast-style attribution instead of a reject
        candidates = [line]
        if team and team.lower() not in line.lower():
            candidates.append(f"{team} — {line}")
        chosen = next((c for c in candidates
                       if c and _validate(c, feats, teams, kind, team)), None)
        ok = chosen is not None
        if chosen:
            line = chosen
        accepted += ok
        n_words = len(line.split())
        lengths.append(n_words)
        if ok:
            accepted_lengths.append(n_words)
        if sum(line.count(c) for c in ".!?") >= 2:
            sent2 += 1
        if shown < args.show:
            print(f"[{'OK ' if ok else 'REJ'}] ({kind}/{team}, {n_words}w) {line}")
            shown += 1

    n = len(lengths)
    mean_words = statistics.mean(lengths) if lengths else 0.0
    accepted_mean = (statistics.mean(accepted_lengths) if accepted_lengths
                     else mean_words)
    print(f"\nmodel {args.model_dir}")
    print(f"accepted {accepted}/{n} ({100 * accepted / max(1, n):.0f}%)")
    print(f"mean words {mean_words:.1f}; "
          f"median {statistics.median(lengths) if lengths else 0:.0f}; "
          f"max {max(lengths) if lengths else 0}")
    print(f"accepted mean words {accepted_mean:.1f}")
    print(f"2+ sentence lines {sent2}/{n}")
    if args.write_eval:
        payload = {
            "eval": "SoccerReplay-1988 held-out (40 matches)",
            "total": n,
            "gpt2_accepted": accepted,
            "template_fallback": n - accepted,
            "validator_rejections": n - accepted,
            "accept_rate": round(accepted / max(1, n), 3),
            "avg_words": round(accepted_mean, 1),
            "median_words": statistics.median(accepted_lengths)
            if accepted_lengths else 0,
            "two_sentence_share": round(sent2 / max(1, n), 3),
            "model_dir": args.model_dir,
            "note": ("Held-out SoccerReplay prompts; the runtime cleaner and "
                     "grounding validator decide acceptance."),
        }
        Path(args.write_eval).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_eval).write_text(json.dumps(payload, indent=2),
                                         encoding="utf-8")
        print("wrote", args.write_eval)


if __name__ == "__main__":
    main()
