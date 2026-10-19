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
    if settings.device not in ("cpu", ""):
        dev = (f"cuda:{settings.device}" if settings.device.isdigit()
               else settings.device)
        model = model.half().to(dev)
    _MODEL = (tok, model)
    return _MODEL


_LABEL_WORD = {"goal": "goal", "shot": "shot", "tackle": "tackle",
               "pass": "pass", "dead_time": "foul", "corner": "corner",
               "save": "save"}

_DEFAULT_EXCITEMENT = {"goal": "huge", "shot": "high", "tackle": "medium",
                       "pass": "low", "dead_time": "calm",
                       "corner": "medium", "save": "high"}


def _clean(text: str) -> str:
    text = text.split("\n")[0]
    text = re.sub(r"[\[\]]", " ", text)  # never leak markup
    # Do NOT cut at the first period/exclamation: lines legitimately start
    # with "GOAL!" or "It's in!" followed by the actual sentence. Only trim a
    # genuinely rambling generation at a sentence boundary near the end.
    if len(text.split()) > 30:
        m = re.search(r"^(.{40,300}?[.!?])(\s|$)", text)
        if m:
            text = m.group(1)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", "ignore").decode()  # drop tokenizer mojibake
    text = re.sub(r"\s+", " ", text).strip()
    # Keep the spoken line inside the ~10 s clip (and inside one breath).
    words = text.split()
    if len(words) > 20:
        text = " ".join(words[:20]).rstrip(" ,;:")
        m = re.search(r"^(.*[.!?])(\s|$)", text)
        if m:
            text = m.group(1)
    # drop leading dangling conjunctions from mid-sentence starts
    text = re.sub(r"^(and|but|so|which|who)\s+", "", text, flags=re.I)
    if text:
        text = text[0].upper() + text[1:]
    return text


# Names the runtime vision pipeline cannot verify. A grounded line must not
# contain them; the templates are the fallback when the model strays.
_BLOCK_NAMES = {
    "real madrid", "barcelona", "psg", "inter", "milan", "napoli",
    "atletico", "chelsea", "arsenal", "liverpool", "manchester",
    "tottenham", "bayern", "juventus", "roma", "brazil", "germany",
    "spain", "italy", "england", "portugal", "netherlands", "belgium",
    "croatia", "morocco", "japan", "korea", "uruguay", "colombia",
    "chile", "peru", "ecuador", "senegal", "ghana", "nigeria",
    "cameroon", "egypt", "algeria", "tunisia", "australia", "canada",
    "qatar", "saudi", "iran", "poland", "denmark", "sweden", "norway",
    "switzerland", "austria", "serbia", "wales", "scotland", "ireland",
    "turkey", "greece", "russia", "ukraine", "romania", "hungary",
    "argentine", "french", "paris", "london", "madrid", "munich",
    "turin", "rome", "berlin", "doha",
    "messi", "mbappe", "mbappé", "di maria",
}
_BLOCK_CLAIMS = {"penalty", "red card", "yellow card", "offside", "var",
                 "substitution", "injury", "disallowed"}

# A line must actually be about the event it is attached to. Without this a
# small model happily calls a goal "a brilliant stop".
_EVENT_WORDS = {
    "goal": {"goal", "score", "scored", "scores", "net", "buried", "equal",
             "levelled", "level", "lead", "ahead", "breakthrough", "erupts",
             "punish", "finish", "converted", "ripples", "celebrate", "have it",
             "make it", "made it", "back in", "break the deadlock", "drama",
             "sensational", "magic", "spectacular", "opened the scoring"},
    "shot": {"shot", "strike", "struck", "effort", "chance", "keeper",
             "save", "saved", "wide", "post", "blocked", "deflect", "volley",
             "header", "tries", "try", "shoot", "drive", "sight of goal"},
    "tackle": {"tackle", "challenge", "win", "won", "defend", "defending",
               "clear", "clearance", "recover", "muscle", "snuff", "shut",
               "dig", "turnover", "ball back"},
    "corner": {"corner", "delivery", "swing", "set-piece", "box", "whipped",
               "send the big men"},
}


def _validate(line: str, features: dict, teams: dict, label: str,
              team: str = "") -> bool:
    """Reject lines that claim facts the pipeline did not observe."""
    low = line.lower()
    home = (teams.get("home") or "").lower()
    away = (teams.get("away") or "").lower()
    # Every line must name the team the event belongs to; this is the single
    # strongest guard against "A moment of magic from Paris" style drift.
    if team and team.lower() not in low:
        return False
    if "]" in line or "[" in line:
        return False
    if line.count("!") > 2:
        return False
    if any(p in low for p in _BLOCK_CLAIMS):
        return False
    for name in _BLOCK_NAMES:
        if name in low and name not in {home, away}:
            return False
    words = _EVENT_WORDS.get(label)
    if words is not None and not any(w in low for w in words):
        return False
    hs = features.get("score_home")
    as_ = features.get("score_away")
    minute = int(features.get("minute", 0) or 0)
    for m in re.finditer(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})", line):
        if hs is not None and (int(m.group(1)), int(m.group(2))) != (int(hs), int(as_)):
            return False
    allowed = {str(minute)}
    if hs is not None:
        allowed |= {str(hs), str(as_)}
    for d in re.findall(r"\d+", line):
        if d not in allowed:
            return False
    return True


def event_line(label: str, features: dict, teams: dict,
               excitement_hint: str = "medium") -> tuple[str, str]:
    """One grounded commentary line from the fine-tuned model.

    Prompt shape matches the training corpus:
      [HISTORY] <scoreline, minute, possession> [EVENT] <event> by <team>
    The model then predicts [INTENSITY] and [COMMENTARY].
    """
    excitement = _DEFAULT_EXCITEMENT.get(label, excitement_hint)
    home = teams.get("home", "the home side")
    away = teams.get("away", "the visitors")
    side = features.get("goal_side")
    if side not in ("home", "away"):
        poss = features.get("possession")
        side = poss if poss in ("home", "away") else (
            "home" if (features.get("ball_x_frac", 0.5) or 0.5) >= 0.5
            else "away")
    team = teams.get(side, home)
    minute = int(features.get("minute", 0) or 0)
    hs = features.get("score_home", 0)
    as_ = features.get("score_away", 0)
    scoreline = features.get("scoreline") or f"{home} {hs}-{as_} {away}"
    poss = features.get("possession", "even")
    poss_team = teams.get(poss, "both sides") if poss in ("home", "away") \
        else "both sides"
    word = _LABEL_WORD.get(label, label)
    history = (f"{scoreline}, {minute}th minute, "
               f"{poss_team} in possession.")
    # Prefill the control tokens so the model only has to write the line.
    # This is the completion-only distribution the adapter was trained on.
    prompt = (f"[HISTORY] {history} [EVENT] {word} by {team} "
              f"[INTENSITY] {excitement} [COMMENTARY]")
    import torch
    tok, model = _load()
    last_line = ""
    for attempt in range(3):
        try:
            with _LOCK, torch.no_grad():
                ids = tok(prompt, return_tensors="pt").to(model.device)
                out = model.generate(
                    **ids, max_new_tokens=42, min_new_tokens=8, do_sample=True,
                    temperature=max(0.55, 0.90 - 0.15 * attempt),
                    top_p=0.9, repetition_penalty=1.25,
                    no_repeat_ngram_size=4, pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True)

            m = re.search(r"\[INTENSITY\]\s*(\w+)", gen, flags=re.I)
            if m and m.group(1).lower() in ("huge", "high", "medium", "low", "calm"):
                excitement = m.group(1).lower()
            if "[COMMENTARY]" in gen:
                gen = gen.split("[COMMENTARY]", 1)[1]
            line = _clean(gen)
            last_line = line
            n_words = sum(1 for w in line.split() if any(c.isalpha() for c in w))
            if n_words >= 4 and _validate(line, features, teams, label, team):
                return line, excitement
        except Exception as e:  # noqa: BLE001
            print(f"  [gpt2-lora] generation failed ({type(e).__name__}: {e})")
    if last_line:
        print(f"  [gpt2-lora] rejected: {last_line}")
    raise RuntimeError("gpt2-lora produced no usable grounded line")


def opener(teams: dict) -> str:
    return (f"We're underway — {teams.get('home', 'the home side')} against "
            f"{teams.get('away', 'the visitors')}, live commentary powered by a "
            f"fine-tuned GPT-2.")
