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
        # the YOLO run can leave only a couple of hundred MiB free; generating
        # a few hundred tokens needs far less on CPU and must not OOM the demo
        if dev.startswith("cuda"):
            free, _total = torch.cuda.mem_get_info()
            if free < 1_200_000_000:
                dev = "cpu"
        model = model.half().to(dev)
    _MODEL = (tok, model)
    return _MODEL


_LABEL_WORD = {"goal": "goal", "shot": "shot", "tackle": "tackle",
               "pass": "pass", "dead_time": "foul", "corner": "corner",
               "save": "save"}

_DEFAULT_EXCITEMENT = {"goal": "huge", "shot": "high", "tackle": "medium",
                       "pass": "low", "dead_time": "calm",
                       "corner": "medium", "save": "high"}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _bad_word_ids(tok, home: str, away: str) -> list[list[int]]:
    """Token sequences for teams the model is not allowed to name.

    Suppressing them during generation is stronger than rejecting afterwards:
    the adapter otherwise drifts to the national teams in its pretraining.
    """
    allowed = [t for t in (home.lower(), away.lower()) if t]
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for name in sorted(_BLOCK_NAMES):
        if any(name in t for t in allowed):
            continue
        # Case variants matter: "lyon" was blocked but the model generated
        # "Lyon" and slipped past the token-level guard.
        forms = {name, name.capitalize(), name.title(), name.upper()}
        for form in forms:
            for prefix in (" ", ""):
                seq = tok.encode(prefix + form, add_special_tokens=False)
                if not seq:
                    continue
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    out.append(seq)
    return out


def _repair_team_names(line: str, home: str, away: str, team: str) -> str:
    """Replace an invented team with the grounded event team."""
    allowed = {t.lower() for t in (home, away)}
    for name in sorted(_BLOCK_NAMES, key=len, reverse=True):
        if name in allowed or name in team.lower():
            continue
        line = re.sub(r"\b%s\b" % re.escape(name), team, line, flags=re.I)
    return line


_MAX_WORDS = 50  # two to three spoken sentences (~14-16 s of speech)


def _clean(text: str) -> str:
    text = text.split("\n")[0]
    text = re.sub(r"[\[\]]", " ", text)  # never leak markup
    text = re.sub(r"\s+", " ", text).strip()
    # Keep every complete sentence that fits inside the word cap. Trimming at
    # the *first* sentence (the old behaviour) threw away the second half of
    # the longer lines the v4 adapter is trained to write.
    words = text.split()
    if len(words) > _MAX_WORDS:
        cut = " ".join(words[:_MAX_WORDS])
        bounds = list(re.finditer(r"[.!?](?=\s|$)", cut))
        text = cut[:bounds[-1].end()] if bounds else cut.rstrip(" ,;:")
    text = text.encode("ascii", "ignore").decode()  # drop tokenizer mojibake
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # fromFrance -> from France
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
    # national sides the model drifts to
    "mexico", "usa", "united states", "czech", "slovakia", "slovenia",
    "bosnia", "albania", "montenegro", "bulgaria", "finland", "iceland",
    "israel", "georgia", "armenia", "azerbaijan", "kazakhstan", "uzbekistan",
    "china", "india", "new zealand", "south africa", "ivory coast", "mali",
    "burkina faso", "dr congo", "zambia", "angola", "cape verde", "paraguay",
    "bolivia", "venezuela", "costa rica", "honduras", "panama", "jamaica",
    # clubs the model has invented on held-out prompts
    "wigan", "sunderland", "leeds", "blackburn", "bolton", "northampton",
    "sheffield", "tottenham", "everton", "fulham", "brighton", "wolves",
    "watford", "burnley", "southampton", "leicester", "newcastle",
    "aston villa", "crystal palace", "nottingham", "bournemouth",
    "brentford", "atletico", "sevilla", "valencia", "villarreal",
    "real sociedad", "athletic", "betis", "lazio", "fiorentina",
    "atalanta", "dortmund", "leipzig", "leverkusen", "frankfurt",
    "wolfsburg", "monchengladbach", "stuttgart", "hamburg", "schalke",
    "lyon", "marseille", "monaco", "lille", "nice", "rennes", "ajax",
    "psv", "feyenoord", "porto", "benfica", "sporting", "celtic",
    "rangers", "galatasaray", "fenerbahce", "besiktas", "olympiacos",
    "panathinaikos",
    "messi", "mbappe", "mbappé", "di maria",
}
_BLOCK_CLAIMS = {"penalty", "red card", "yellow card", "offside", "var",
                 "substitution", "injury", "disallowed"}

# proper-noun pairs that are safe even though they are not the two teams
_GENERIC_PROPER = {
    "premier league", "champions league", "europa league", "fa cup",
    "league cup", "carabao cup", "community shield", "world cup",
    "european championship", "nations league", "copa america",
    "african cup", "serie a", "la liga", "bundesliga", "ligue 1",
    "boxing day", "new year", "copa del rey", "dfb pokal", "coppa italia",
    "coupe de france", "efl cup", "conference league",
}
_PROPER_PAIR = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def _unknown_proper_noun(line: str, home: str, away: str) -> bool:
    """True when the line names a club/competition we were not told about."""
    allowed = [t for t in (home, away) if t]
    for m in _PROPER_PAIR.finditer(line):
        low = m.group(1).lower()
        if low in _GENERIC_PROPER or any(t in low for t in allowed):
            continue
        return True
    return False

# A line must actually be about the event it is attached to. Without this a
# small model happily calls a goal "a brilliant stop".
_EVENT_WORDS = {
    "goal": {"goal", "goals", "score", "scored", "scores", "scoring", "net",
             "buried", "equal", "equaliser", "equalizer", "levelled", "level",
             "lead", "ahead", "breakthrough", "erupts", "punish", "finish",
             "finished", "converted", "ripples", "celebrate", "have it",
             "make it", "made it", "back in", "break the deadlock", "drama",
             "sensational", "magic", "spectacular", "opened the scoring",
             "opener", "winner", "consolation", "header", "strike", "strikes",
             "shot", "double", "drew level", "pull one back"},
    "shot": {"shot", "shots", "strike", "struck", "effort", "chance",
             "keeper", "save", "saved", "wide", "post", "bar", "blocked",
             "block", "deflect", "volley", "header", "tries", "try", "shoot",
             "drive", "sight of goal", "attempt", "curler", "dipping",
             "rebound", "distance", "free kick", "penalty area", "crossbar",
             "goalmouth", "on target", "off target"},
    "tackle": {"tackle", "challenge", "win", "won", "defend", "defending",
               "defender", "clear", "cleared", "clearance", "recover",
               "muscle", "snuff", "shut", "dig", "turnover", "ball back",
               "intercept", "intercepted", "avert", "dispossess", "press",
               "pressure", "block", "blocked", "slide tackle"},
    "corner": {"corner", "corners", "delivery", "swing", "swung",
               "set-piece", "box", "whipped", "cross", "crossed", "header",
               "flag", "send the big men", "short corner", "in-swinger"},
}


_GOAL_STRONG = (
    "goal", "it's in", "it is in", "buried", "net", "scores", "scored",
    "scoring", "equaliser", "equalizer", "equalise", "equalised", "equalize",
    "equalized", "make it", "made it", "levelled",
    "levels", "erupts", "winner", "converted", "opened the scoring",
    "back of the net",
)
_GOAL_CONTRADICTIONS = (
    "denied", "deny", "denies", "wide", "off target", "saved", "save",
    "saves", "block", "blocks", "blocked", "clear", "clears", "cleared",
    "clearance", "disallowed", "ruled out", "no goal", "goal kick",
    "missed", "over the bar", "hit the post", "hit the bar", "past the post",
    "corner", "corner kick", "out of play", "free kick", "throw-in",
    "awarded a corner",
)

_GOAL_ACTIONS = (
    "score", "scored", "scores", "scoring", "finish", "finished", "net",
    "buried", "make it", "made it", "equaliser", "equalizer", "equalise",
    "equalised", "converted", "header", "strike", "strikes", "shot", "goal",
    "winner", "levels", "levelled", "ripples", "celebrate",
)


def _phrase_in(text: str, phrases) -> bool:
    """Substring for multi-word phrases; word boundaries for single words."""
    for p in phrases:
        if " " in p or "'" in p:
            if p in text:
                return True
        elif re.search(rf"\b{re.escape(p)}\b", text):
            return True
    return False


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
        # allow a blocked substring when it is part of the prompted team:
        # "manchester" must not reject a line about "Manchester City".
        if name in low and not any(name in t for t in (home, away)):
            return False
    if _unknown_proper_noun(line, home, away):
        return False
    words = _EVENT_WORDS.get(label)
    if words is not None and not any(w in low for w in words):
        return False
    if label == "goal":
        if _phrase_in(low, _GOAL_CONTRADICTIONS):
            return False
        # The goal call must lead the line. "...another goal" at the end of a
        # foul/corner description is not a goal call. The forced prefix
        # "GOAL! <team>" is stripped before checking the body, otherwise it
        # would satisfy the goal-word test by itself.
        first = re.split(r"(?<=[.!?])\s+", line.strip(), maxsplit=1)[0].lower()
        if not _phrase_in(first, _GOAL_STRONG):
            return False
        body = re.sub(r"^\s*GOAL!\s*", "", line, count=1, flags=re.I)
        if team:
            body = re.sub(rf"^\s*{re.escape(team)}\s*", "", body,
                          count=1, flags=re.I)
        if not _phrase_in(body.lower(), _GOAL_ACTIONS):
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
    history = (f"{scoreline}, {_ordinal(minute)} minute, "
               f"{poss_team} in possession.")
    # Prefill the control tokens so the model only has to write the line.
    # This is the completion-only distribution the adapter was trained on.
    prompt = (f"[HISTORY] {history} [EVENT] {word} by {team} "
              f"[INTENSITY] {excitement} [COMMENTARY]")
    import torch
    tok, model = _load()
    bad = _bad_word_ids(tok, home, away)
    last_line = ""
    for attempt in range(3):
        try:
            # Retries anchor the line on the event team; goals also anchor the
            # first sentence so the model cannot open with a corner/clearance.
            if label == "goal":
                prefill = f"GOAL! {team} "
            else:
                prefill = "" if attempt == 0 else f"{team} "
            prompt_in = prompt if not prefill else f"{prompt} {prefill.strip()}"
            with _LOCK, torch.no_grad():
                ids = tok(prompt_in, return_tensors="pt").to(model.device)
                out = model.generate(
                    **ids, max_new_tokens=96, min_new_tokens=14, do_sample=True,
                    temperature=max(0.55, 0.90 - 0.15 * attempt),
                    top_p=0.9, repetition_penalty=1.25,
                    no_repeat_ngram_size=4, pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    bad_words_ids=bad or None,
                )
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            if prefill:
                gen = prefill + gen.lstrip()

            m = re.search(r"\[INTENSITY\]\s*(\w+)", gen, flags=re.I)
            if m and m.group(1).lower() in ("huge", "high", "medium", "low", "calm"):
                excitement = m.group(1).lower()
            if "[COMMENTARY]" in gen:
                gen = gen.split("[COMMENTARY]", 1)[1]
            line = _clean(gen)
            last_line = line
            # A line can be grounded in substance but omit the team name. The
            # validator requires the team, so give the model's sentence a
            # broadcast-style attribution rather than dropping it to a template.
            candidates = [line]
            if team and team.lower() not in line.lower():
                candidates.append(f"{team} — {line}")
            repaired = _repair_team_names(line, home, away, team)
            if repaired != line:
                candidates.append(repaired)
            for cand in candidates:
                n_words = sum(1 for w in cand.split()
                              if any(c.isalpha() for c in w))
                if n_words >= 4 and _validate(cand, features, teams, label,
                                              team):
                    return cand, excitement
        except Exception as e:  # noqa: BLE001
            print(f"  [gpt2-lora] generation failed ({type(e).__name__}: {e})")
    if last_line:
        print(f"  [gpt2-lora] rejected: {last_line}")
    raise RuntimeError("gpt2-lora produced no usable grounded line")


def opener(teams: dict) -> str:
    return (f"We're underway — {teams.get('home', 'the home side')} against "
            f"{teams.get('away', 'the visitors')}, live commentary powered by a "
            f"fine-tuned GPT-2.")
