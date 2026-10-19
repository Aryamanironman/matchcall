"""GPT-2 + LoRA announcer: a locally fine-tuned commentary model.

The adapter was trained on markup-formatted match data:
    [HISTORY] <context>. [EVENT] <event> by <team> [INTENSITY] <level> [COMMENTARY] <line>
so we prompt in that format and parse out both the commentary line and the
model's own intensity prediction. Generation is constrained (short,
repetition-penalized, sentence-trimmed) because small models loop easily.
"""
from __future__ import annotations

import re
import threading

from app.config import settings

_LOCK = threading.Lock()
_MODEL = None  # lazy singleton


def available() -> bool:
    return settings.gpt2_lora_dir.exists()


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base, str(settings.gpt2_lora_dir))
    model.eval()
    if settings.device == "cuda":
        model = model.half().to("cuda")
    _MODEL = (tok, model)
    return _MODEL


_LABEL_WORD = {"goal": "goal", "shot": "shot", "tackle": "tackle",
               "pass": "pass", "dead_time": "foul"}

_DEFAULT_EXCITEMENT = {"goal": "huge", "shot": "high", "tackle": "medium",
                       "pass": "low", "dead_time": "calm"}


def _clean(text: str) -> str:
    text = text.split("\n")[0]
    text = text.split("[")[0]  # never leak follow-on markup
    # cut at the first sentence end; fall back to the whole trimmed text
    m = re.search(r"^(.+[.!?])(\s|$)", text)
    if m:
        text = m.group(1)
    else:
        text = text.rstrip(",;: ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", "ignore").decode()  # drop tokenizer mojibake
    # drop leading dangling conjunctions from mid-sentence starts
    text = re.sub(r"^(and|but|so|which|who)\s+", "", text, flags=re.I)
    if text:
        text = text[0].upper() + text[1:]
    return text


def event_line(label: str, features: dict, teams: dict,
               excitement_hint: str = "medium") -> tuple[str, str]:
    """One commentary line from the fine-tuned model. Returns (line, excitement)."""
    excitement = _DEFAULT_EXCITEMENT.get(label, excitement_hint)
    side = "home" if (features.get("ball_x_frac", 0.5) or 0.5) >= 0.5 else "away"
    team = teams.get(side, "the home side")
    other = teams.get("away" if side == "home" else "home", "the visitors")
    minute = int(features.get("minute", 0) or 0)
    word = _LABEL_WORD.get(label, label)
    prompt = (f"[HISTORY] {team} v {other}. {minute}' {word} by {team}. "
              f"[EVENT] {word} by {team} ")
    try:
        import torch
        tok, model = _load()
        with _LOCK, torch.no_grad():
            ids = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **ids, max_new_tokens=42, min_new_tokens=6, do_sample=True,
                temperature=0.85, top_p=0.9, repetition_penalty=1.25,
                no_repeat_ngram_size=4, pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True)

        # excitement: prefer the model's own [INTENSITY] prediction
        m = re.search(r"\[INTENSITY\]\s*(\w+)", gen, flags=re.I)
        if m and m.group(1).lower() in ("huge", "high", "medium", "low", "calm"):
            excitement = m.group(1).lower()

        # the line itself: after [COMMENTARY] when present, else the raw tail
        if "[COMMENTARY]" in gen:
            gen = gen.split("[COMMENTARY]", 1)[1]
        line = _clean(gen)
        n_words = sum(1 for w in line.split() if any(c.isalpha() for c in w))
        if n_words >= 4:
            return line, excitement
    except Exception as e:  # noqa: BLE001
        print(f"  [gpt2-lora] generation failed ({type(e).__name__}: {e})")
    raise RuntimeError("gpt2-lora produced no usable line")


def opener(teams: dict) -> str:
    return (f"We're underway — {teams.get('home', 'the home side')} against "
            f"{teams.get('away', 'the visitors')}, live commentary powered by a "
            f"fine-tuned GPT-2.")
