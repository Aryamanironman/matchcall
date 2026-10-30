"""Build a Word (.docx) research paper with native styles, tables and figures."""
from __future__ import annotations

import html
import json
import os
import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX,
                            WD_LINE_SPACING)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
OUT = Path(os.getenv("PAPER_OUT", str(ROOT / "docs" / "Matchcall_Paper_v5.docx")))
FIG = ROOT / "docs" / "paper_figures"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


AI = load(REPO / "output" / "comparison" / "commentary_comparison.json")
EV = load(REPO / "output" / "experiments" / "event_spotting.json")
M = AI["metrics"]

doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.27)
section.page_height = Inches(11.69)
section.top_margin = Inches(1.0)
section.bottom_margin = Inches(1.0)
section.left_margin = Inches(1.0)
section.right_margin = Inches(1.0)
section.different_first_page_header_footer = True

# ---------------------------------------------------------------- styles
styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
normal.paragraph_format.line_spacing = 1.15
normal.paragraph_format.space_after = Pt(6)

for name, size, before, after in (
        ("Heading 1", 14, 14, 6), ("Heading 2", 12, 10, 4)):
    st = styles[name]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.keep_with_next = True

cap = styles.add_style("Paper Caption", WD_STYLE_TYPE.PARAGRAPH)
cap.font.name = "Times New Roman"
cap.font.size = Pt(9)
cap.font.color.rgb = RGBColor(70, 70, 70)
cap.paragraph_format.space_before = Pt(2)
cap.paragraph_format.space_after = Pt(6)
cap.paragraph_format.line_spacing = 1.0

abstract_style = styles.add_style("Paper Abstract", WD_STYLE_TYPE.PARAGRAPH)
abstract_style.base_style = normal
abstract_style.font.size = Pt(10)
abstract_style.paragraph_format.left_indent = Inches(0.25)
abstract_style.paragraph_format.right_indent = Inches(0.25)
abstract_style.paragraph_format.space_after = Pt(6)

ref_style = styles.add_style("Paper Reference", WD_STYLE_TYPE.PARAGRAPH)
ref_style.font.name = "Times New Roman"
ref_style.font.size = Pt(9)
ref_style.paragraph_format.left_indent = Inches(0.32)
ref_style.paragraph_format.first_line_indent = Inches(-0.32)
ref_style.paragraph_format.space_after = Pt(2)
ref_style.paragraph_format.line_spacing = 1.0
ref_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

# ---------------------------------------------------------------- helpers
def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def cell_margins(table, top=40, bottom=40, left=70, right=70):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", top), ("left", left), ("bottom", bottom),
                      ("right", right)):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), str(val))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def table_borders(table, color="BFBFBF"):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), color)
        borders.append(e)
    tblPr.append(borders)


def no_split(row):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:cantSplit")
    trPr.append(e)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:tblHeader")
    trPr.append(e)


def add_runs(par, segments, size=8.5, bold=False):
    if isinstance(segments, str):
        segments = [(segments, None)]
    for text, kind in segments:
        parts = str(text).split("\n")
        for k, part in enumerate(parts):
            r = par.add_run(part)
            r.font.name = "Times New Roman"
            r.font.size = Pt(size)
            r.bold = bold
            if k < len(parts) - 1:
                r.add_break()
            if kind == "bad":
                r.font.highlight_color = WD_COLOR_INDEX.PINK
            elif kind == "good":
                r.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
            elif kind == "warn":
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def cell_para(cell, segments, size=8.5, bold=False):
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_runs(p, segments, size=size, bold=bold)


def table_caption(n, text):
    p = doc.add_paragraph(style="Paper Caption")
    r = p.add_run(f"Table {n}. ")
    r.bold = True
    p.add_run(text)
    p.paragraph_format.keep_with_next = True


def add_table(headers, rows, widths, font_size=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table_borders(table)
    cell_margins(table)
    for i, h in enumerate(headers):
        c = table.rows[0].cells[i]
        c.width = Inches(widths[i])
        shade(c, "E3E9E2")
        cell_para(c, h, size=font_size, bold=True)
        c.vertical_alignment = WD_ALIGN_VERTICAL.BOTTOM
    repeat_header(table.rows[0])
    no_split(table.rows[0])
    for row in rows:
        if row[0] == "GROUP":
            tr = table.add_row()
            merged = tr.cells[0].merge(tr.cells[-1])
            shade(merged, "F1F3EF")
            cell_para(merged, row[1], size=font_size, bold=True)
            no_split(tr)
            continue
        tr = table.add_row()
        no_split(tr)
        for i, val in enumerate(row):
            c = tr.cells[i]
            c.width = Inches(widths[i])
            cell_para(c, val, size=font_size)
            c.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    return table


def add_figure(path, number, caption, width=6.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(style="Paper Caption")
    r = cap.add_run(f"Figure {number}. ")
    r.bold = True
    cap.add_run(caption)


def P(text, style=None):
    p = doc.add_paragraph(style=style)
    p.add_run(text)
    return p


def left_para(text, size=9.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    return p


def BP(lead, text):
    p = doc.add_paragraph()
    r = p.add_run(lead)
    r.bold = True
    p.add_run(" " + text)
    return p


def add_field(par, instr):
    run = par.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(it)
    run._r.append(fld2)


def horizontal_rule(p, color="A6B0A5", size="6"):
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), color)
    pBdr.append(bottom)
    pPr.append(pBdr)


def code_block(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.0
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F2F4F1")
    pPr.append(shd)
    for i, line in enumerate(text.splitlines()):
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(9)
        if i < len(text.splitlines()) - 1:
            r.add_break()


# ---------------------------------------------------------------- header/footer
header = section.header.paragraphs[0]
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
hr = header.add_run("Matchcall: Fact-Grounded Football Commentary")
hr.font.name = "Times New Roman"
hr.font.size = Pt(8.5)
hr.font.color.rgb = RGBColor(90, 90, 90)
horizontal_rule(header, color="D0D6CF", size="4")

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = footer.add_run("Page ")
fr.font.name = "Times New Roman"
fr.font.size = Pt(9)
add_field(footer, "PAGE")
fr2 = footer.add_run(" of ")
fr2.font.name = "Times New Roman"
fr2.font.size = Pt(9)
add_field(footer, "NUMPAGES")

# ---------------------------------------------------------------- title page
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(28)
p.paragraph_format.space_after = Pt(8)
r = p.add_run("Matchcall: A Fact-Grounded, Low-Cost Pipeline for Spoken Football Commentary")
r.bold = True
r.font.name = "Times New Roman"
r.font.size = Pt(18)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("Aryaman")
r.bold = True
r.font.size = Pt(12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("Supervisor: Nabin Sir")
r.font.size = Pt(11)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("Department of Computer Science, [University]")
r.font.size = Pt(9.5)
r.font.color.rgb = RGBColor(80, 80, 80)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(16)
r = p.add_run("27 October 2026")
r.font.size = Pt(9.5)
r.font.color.rgb = RGBColor(80, 80, 80)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.keep_with_next = True
r = p.add_run("ABSTRACT")
r.bold = True
r.font.size = Pt(9)
r.font.color.rgb = RGBColor(31, 92, 58)

abstract = (
    "Automatic football commentary must detect events, identify players and teams, decide what is "
    "worth saying, generate grounded language, and voice it in time. Recent systems apply large "
    "multimodal models, but they are expensive, difficult to reproduce on consumer hardware, and "
    "can hallucinate facts. We present Matchcall, a modular pipeline that keeps deterministic facts "
    "separate from surface realisation. A score-bug reader supplies goal anchors; a motion-gated "
    "window scan and a ResNet-18 classifier propose other events; YOLO26n with ByteTrack, jersey "
    "clustering and possession estimation supply player-level facts; GPT-2 with LoRA generates "
    "commentary under a grounding validator with a grounded template as a safety net; Fish Audio synthesises speech. "
    "On a 64.5-minute ARG–FRA broadcast, the pipeline produces 43 fully voiced lines in 19.4 minutes "
    "on a single consumer GPU. Both goals are recovered with precision and recall 1.00, but overall event "
    "recall against ten curated moments is 0.40 at ±1 minute. On 40 held-out human references, GPT-2 "
    "v4 improves ROUGE-L from 0.093 to 0.185 and METEOR from 0.027 to 0.207 over templates, while "
    "BLEU-4 remains near zero. The system is pre-recorded only; a deployed live service would need "
    "streaming perception and incremental speech. Cloud inference costs $0.39 per match versus an "
    "illustrative $300 human team. We conclude that low-cost, auditable pre-recorded commentary is "
    "feasible; live commentary and general event spotting remain open."
)
P(abstract, style="Paper Abstract")
p = doc.add_paragraph(style="Paper Abstract")
r = p.add_run("Keywords: ")
r.bold = True
p.add_run("football commentary, action spotting, multi-object tracking, LoRA, grounding, "
          "text-to-speech, reproducibility")
horizontal_rule(p, color="B9C2B7", size="6")

# ---------------------------------------------------------------- 1 introduction
doc.add_heading("1. Introduction", level=1)
P("A football broadcast is expensive to describe. A live commentator, a producer and often a second "
  "analyst must watch every match, identify the important moments, remember the score and the "
  "narrative, and speak continuously. Most matches are never covered at all; the ones that are "
  "covered are described in one language for one audience. Automatic commentary could widen that "
  "coverage and lower its cost, but only if it is accurate enough to trust and cheap enough to run.")
P("Three families of systems have been proposed. Rule-based and template systems such as MIKE "
  "separate fact selection from surface realisation and remain interpretable, but their language is "
  "repetitive [6]. Neural captioning systems, including SoccerNet-Caption and MatchTime, learn "
  "from broadcast commentary and produce more varied text, but depend on large aligned corpora and "
  "report overlap or alignment metrics rather than factuality [7], [8]. More recent multimodal "
  "LLMs such as TimeSoccer and SoccerComment consume full matches or retrieve external knowledge, "
  "but they are heavy and do not expose an auditable fact layer [9], [10]. Each family trades "
  "one desirable property for another.")
P("This paper asks whether a deliberately small, modular system can occupy a different point in that "
  "trade-off. Matchcall keeps five stages separate: event spotting, tracking and team/possession "
  "facts, deterministic description, language generation, and speech. The language model never "
  "decides the score, minute, team or possession; it only realises facts that were already measured. "
  "A validator checks every candidate line against those facts, and a grounded template is used when "
  "the candidate fails. The design target is a full match on a single consumer GPU, with every "
  "training run and artifact logged.")
P("We report three comparisons on a real broadcast: (i) GPT-2 commentary against 40 held-out human "
  "references and a template baseline; (ii) event spotting against ten curated reference moments; and "
  "(iii) compute, TTS and human cost. The results are deliberately mixed. Goals are reliable because "
  "the broadcast score bug is a high-precision anchor. General event spotting and live latency are not "
  "solved. The contribution is not a new state of the art on SoccerNet; it is an auditable, "
  "reproducible instantiation of the full broadcast-to-speech pipeline, with its failure modes "
  "measured rather than hidden.")
BP("Contributions.", "(1) A modular five-stage pipeline that separates deterministic facts from "
   "language and speech. (2) A score-bug goal anchor that infers the on-screen starting score and "
   "survives short clips. (3) A full 64.5-minute end-to-end run with 43 voiced lines on a single consumer GPU. "
   "(4) Three empirical comparisons, including negative results for general event spotting and API "
   "event coverage. (5) ClearML-logged training runs and a reproducible evaluation script.")

# ---------------------------------------------------------------- 2 related work
doc.add_heading("2. Related Work", level=1)
doc.add_heading("2.1 Action spotting and event understanding", level=2)
P("SoccerNet defined action spotting as predicting the temporal location and class of sparse events "
  "in long broadcasts, with Average-mAP at multiple tolerances [1]. SoccerNet-v2 expanded the "
  "taxonomy to seventeen classes and added camera-shot and replay annotations [2]. Later work "
  "improves temporal context and timestamp precision, but the common lesson is that temporal "
  "precision and a large, well-annotated corpus matter. Our system instead uses a broadcast-specific "
  "shortcut for goals — the score bug — and a lightweight window classifier for other events. The "
  "trade-off is lower coverage on rare classes, which we measure rather than assume away.")
doc.add_heading("2.2 Detection, tracking, team identification and possession", level=2)
P("YOLO established single-pass real-time detection [21]. ByteTrack associates low-score boxes to "
  "recover occluded objects and is the tracker used here [3]. Team identity and possession are "
  "semantic quantities that track IDs alone do not provide. DeepPlayer-Track uses jersey-colour "
  "masks for team recognition [4], and Guo et al. combine YOLOv8, DBSCAN jersey clustering and "
  "Norfair tracking for possession estimation on full matches [5]. Our pipeline follows the "
  "label-free jersey-clustering approach and adds a bounded ball-interpolation rule; the ball "
  "remains a weak point because its object-detection AP is low.")
doc.add_heading("2.3 From templates to neural and LLM commentary", level=2)
P("MIKE used six analysers, importance-scored proposition selection and template generation [6]. "
  "SoccerNet-Caption introduced dense captioning with almost 37,000 timestamped commentaries across "
  "715.9 hours [7]. MatchTime showed that alignment quality is a first-order determinant of "
  "generation quality and released an aligned dataset and the MatchVoice model [8]. TimeSoccer "
  "consumes full 45-minute matches with frame compression [9], while SoccerComment combines "
  "LanguageBind, Vicuna-7B and retrieval [10]. MatchVision/SoccerReplay-1988 provides a large "
  "multi-modal corpus [23]. These systems are more fluent than templates but heavier; our work "
  "tests a small GPT-2 with LoRA [11], [12] under a deterministic validator.")
doc.add_heading("2.4 Grounding, hallucination and evaluation", level=2)
P("Hallucination in natural-language generation is surveyed by Ji et al. [14]. FActScore "
  "decomposes long-form text into atomic facts and checks each against a knowledge source [13]. "
  "Football commentary has no dedicated faithfulness benchmark. SoccerNet-Caption, MatchTime, "
  "TimeSoccer and SoccerComment report captioning overlap or alignment; BLEU, ROUGE and METEOR "
  "measure similarity to references, not truth [18]–[20]. Our validator is closer to atomic "
  "checking than to retrieval-only grounding: it checks the generated line against the measured "
  "score, minute, team and event before acceptance.")
doc.add_heading("2.5 Speech synthesis and prosody", level=2)
P("Tacotron 2 established neural TTS naturalness [15]; FastSpeech 2 added explicit duration, pitch "
  "and energy control [16]. Fish Audio's S1 family offers emotion and tone markers and reports low "
  "real-time factors on high-end GPUs [17]. Sports-commentary prosody is not covered by these "
  "benchmarks. We use a production TTS and map event intensity to pacing, but we do not claim a "
  "sports-specific TTS evaluation.")
doc.add_heading("2.6 Positioning", level=2)
P("The literature is rich in perception benchmarks and increasingly rich in generative commentary, "
  "but sparse in systems that expose a deterministic fact layer, run the full broadcast-to-speech "
  "path on consumer hardware, and report their negative results. Matchcall is positioned at that "
  "intersection. It borrows the score-bug idea for goals, the jersey-clustering idea for teams, the "
  "LoRA idea for small-GPU language, and the FActScore idea for validation; it combines them into "
  "one auditable pipeline and measures where the combination fails.")

# ---------------------------------------------------------------- 3 system
doc.add_heading("3. System", level=1)
doc.add_heading("3.1 Overview", level=2)
P("Matchcall has five stages, shown in Figure 1. The first two produce observations; the third turns "
  "them into facts; the fourth turns facts into language; the fifth turns language into speech and "
  "mixes it with the match. The separation is deliberate: an error in the language model cannot "
  "change the score, and an error in perception is visible in the fact sheet before it reaches the "
  "viewer.")
add_figure(FIG / "fig1_pipeline.png", 1,
           "The five-stage pipeline. Stages 1–2 observe; stage 3 fixes the facts; stage 4 realises "
           "them; stage 5 voices and renders. The language model never writes the score or the event "
           "team.", width=6.2)
doc.add_heading("3.2 Event layer", level=2)
P("The event layer combines three signals. A score-bug reader samples the two score digits in the "
  "broadcast overlay, classifies each digit by its binary shape, and detects a persistent transition "
  "to the expected next score. It infers the score already on the board from the first three samples, "
  "so a clip that starts at 2–1 is not treated as 0–0. A motion-gated window scan proposes candidate "
  "events, and a ResNet-18 classifier assigns one of five labels (goal, shot, pass, tackle, dead "
  "time); crowd-audio rise and ball speed refine the goal/shot decision. The score-bug anchor is "
  "treated as gold for goals: it is placed a few seconds before the digit flip because the broadcast "
  "updates after the ball crosses the line.")
doc.add_heading("3.3 Tracking, teams and facts", level=2)
P("YOLO26n detects players, the ball and the referee in each scheduled frame; ByteTrack maintains "
  "identities across frames [3], [21]. Jersey colours are clustered with KMeans into two teams, and "
  "possession is assigned to the side of the tracked player nearest the ball. Because broadcast cuts "
  "and close-ups break the ball path, raw detections are interpolated only across bounded gaps "
  "(16 frames), single-frame teleports are rejected, and predicted frames are marked. The output is "
  "a deterministic fact sheet: score, minute, ball zone, possession, tracked-player count and event "
  "team. No language model is allowed to add to this sheet.")
doc.add_heading("3.4 Commentary and validation", level=2)
P("GPT-2 v4 is a LoRA adapter [11] over GPT-2 [12], trained on team-grounded SoccerReplay "
  "commentary [23]. The prompt supplies the event label, team, scoreline, minute, phase and measured "
  "vision features. A v5 adapter uses an API-slot prompt (`<HOME>`, `<AWAY>`, `<TEAM>`, `<PLAYER>`, "
  "`<SCORE>`, `<MINUTE>`) and substitutes the facts after generation, so the same model can consume "
  "an API-Football event feed. Candidate lines are cleaned and checked: the event team must appear, "
  "blocked team and player names are rejected, digits and both colon and hyphen scores must match the "
  "fact sheet, and a goal line must contain a real goal phrase in its opening sentence and no "
  "contradiction such as “denied,” “wide” or “saved.” Goal candidates are prefilled with “Goal!” plus "
  "the `<PLAYER>` or `<TEAM>` slot and decoded with beam search, with contradiction and off-score "
  "tokens suppressed; the validator strips the prefix and requires a goal action in the body. Blocked "
  "team names are suppressed in lowercase, capitalised and uppercase forms during decoding. If every "
  "candidate fails, a grounded template is the last resort [6]. The validator is intentionally "
  "conservative; it is a guard against invented facts, not a measure of fluency.")
doc.add_heading("3.5 Speech, render and MLOps", level=2)
P("Accepted lines are synthesised with Fish Audio s2.1-pro-free [17]. Event intensity controls "
  "pacing, the voice is mixed under the original crowd at a low level, and ffmpeg renders the full "
  "match. Every YOLO and GPT-2 run is a ClearML task with parameters, per-epoch or per-step scalars, "
  "plots, artifacts and a registered model [22]. The evaluation artifacts in this paper were "
  "produced from the same logged runs.")

# ---------------------------------------------------------------- 4 setup
doc.add_heading("4. Experimental Setup", level=1)
doc.add_heading("4.1 Data and tasks", level=2)
P("All end-to-end results use a 64.5-minute, 1920×1080 recording of the Argentina–France second "
  "half. The recording contains two goals, at broadcast minutes 80 and 81; the score bug flips from "
  "2–1 to 2–2. The event reference is the ten curated moments used by the earlier highlight reel: "
  "six shots, two tackles and two goals. This is a reference for recall/consistency, not an "
  "exhaustive independent annotation, so event precision is not measured. The commentary comparison "
  "uses 40 held-out SoccerReplay prompts with human reference lines [23]. The object detector is "
  "trained on the Roboflow V1 football dataset (8,677 train / 2,996 "
  "validation images, three classes: Player, Ball, Referee) [24]. These are the only datasets used "
  "for training in this paper; SoccerNet and other benchmarks appear only as related work.")
doc.add_heading("4.2 Models and training", level=2)
P("Table 1 summarises the trained components. YOLO26n was fine-tuned at 640 px for 50 epochs with "
  "batch 1 on a consumer GPU; the best checkpoint was the final epoch, with mAP50 0.806 and mAP50–95 "
  "0.480. GPT-2 v4 used LoRA rank 32, 160-token sequences, completion-only loss, three epochs and "
  "2,481 optimizer steps; the loss fell from 3.877 to 0.779. A second adapter, v5, used the same "
  "base and LoRA rank but the API-slot prompt and a 38,815-example training split (1,723 validation); "
  "it ran for 2 epochs and 4,852 steps (1.09 h) and reached a final training loss of 0.903 (ClearML "
  "task 6acdac3186d640cc8a398898752dc90a). Fish Audio was used through its free tier, so no paid TTS "
  "price is included.")
table_caption(1, "Trained components and schedules. The detector and language models were trained "
                 "locally on one consumer GPU; ClearML logged all runs.")
add_table(
    ["Component", "Model / data", "Params", "Schedule", "Result"],
    [
        ["Detector", "YOLO26n on V1 (8,677 / 2,996)", "2.39 M",
         "640 px, 50 epochs, batch 1, 2.70 h", "mAP50 0.806; mAP50–95 0.480"],
        ["Commentary v4", "GPT-2 + LoRA r=32 on SoccerReplay-1988", "124 M base",
         "3 epochs, 2,481 steps, 160 tokens", "loss 3.877 → 0.779"],
        ["Commentary v5", "GPT-2 + LoRA r=32, API-slot template", "124 M base",
         "2 epochs, 4,852 steps, 38,815 rows", "loss → 0.903"],
        ["Speech", "Fish Audio s2.1-pro-free", "—", "43 calls, 412.2 s speech", "43/43 lines voiced"],
    ],
    [1.0, 2.15, 0.65, 1.3, 1.05], font_size=8.2)
doc.add_heading("4.3 Metrics", level=2)
P("Commentary is scored against human references with BLEU-1/2/4, ROUGE-L and METEOR [18]–[20], and "
  "with validator acceptance and line length. Event spotting is scored as recall against the ten "
  "reference moments at ±1 minute, with goal precision/recall checked against the known 80' and 81' "
  "goals. Live operation is measured by wall time, wall-time ratio and time to first commentary "
  "line. Cost is reported for local electricity, a cloud ml.g5.2xlarge at $1.21/h, and an "
  "illustrative two-person human team; the human rates are assumptions, not quotes.")
doc.add_heading("4.4 Hardware and reproducibility", level=2)
P("Inference and training used a single consumer GPU. The full-match run, ClearML "
  "tasks, model checkpoints, comparison JSON and evaluation scripts are retained in the project "
  "repository. The key commands are listed in Appendix A.")

# ---------------------------------------------------------------- 5 results
doc.add_heading("5. Results", level=1)
doc.add_heading("5.1 AI versus human commentary", level=2)
P("Table 2 reports the 40-prompt comparison. GPT-2 v4 more than doubles ROUGE-L over the template "
  f"baseline ({M['v4']['rougeL']:.3f} vs {M['template']['rougeL']:.3f}) and raises METEOR from "
  f"{M['template']['meteor']:.3f} to {M['v4']['meteor']:.3f}, while writing lines of human length "
  f"({M['avg_words_v4']:.1f} words vs 31.9 for humans). BLEU-4 is near zero for both systems, which "
  "is expected for free-form commentary with many valid paraphrases. The validator accepts 38 of 40 "
  "lines; an independent held-out evaluation with a stricter acceptance path accepted 15 of 40, so "
  "the 95% figure should be read as a property of this prompt set, not a general truth rate.")
table_caption(2, "Commentary against 40 held-out human references. The human column is the target; "
                 "self-similarity is 1 and is not a system score.")
add_table(
    ["Metric", "GPT-2 v4", "Template", "Human reference"],
    [
        ["BLEU-1", f"{M['v4']['bleu1']:.4f}", f"{M['template']['bleu1']:.4f}", "—"],
        ["BLEU-2", f"{M['v4']['bleu2']:.4f}", f"{M['template']['bleu2']:.4f}", "—"],
        ["BLEU-4", f"{M['v4']['bleu4']:.6f}", f"{M['template']['bleu4']:.6f}", "—"],
        ["ROUGE-L", f"{M['v4']['rougeL']:.4f}", f"{M['template']['rougeL']:.4f}", "—"],
        ["METEOR", f"{M['v4']['meteor']:.4f}", f"{M['template']['meteor']:.4f}", "—"],
        ["Average words", f"{M['avg_words_v4']:.1f}", f"{M['avg_words_template']:.1f}", "31.9"],
        ["Grounding accepted", f"{M['gpt2_accepted']}/{M['n']}", "—", "—"],
    ],
    [1.9, 1.3, 1.3, 1.6], font_size=8.8)
add_figure(FIG / "fig2_commentary.png", 2,
           "GPT-2 v4 is closer to human overlap and length than the template baseline. BLEU-4 is "
           "omitted because it is near zero for both systems; exact values are in Table 2.", width=6.2)
P("Table 3 quotes three held-out records to show what the aggregate scores hide. In the first, "
  "GPT-2 names the correct team and produces a human-length line, but invents a shot that the "
  "reference does not contain. In the second, both GPT-2 and the template attribute the corner to "
  "Napoli while the event belongs to Eintracht Frankfurt; this is a shared feature-layer failure, "
  "not only a language failure. In the third, GPT-2 misses a goal entirely and describes a stop and "
  "a corner; the template at least calls a goal, but names the wrong team. These correspond to the "
  "three failure modes: invented detail, wrong team, missed event.")
table_caption(3, "Held-out commentary records. Pink highlights mark wrong or unsupported content; "
                 "green highlights mark a correct event call.")
add_table(
    ["Event", "Source", "Line", "Difference"],
    [
        ["GROUP", "Record A · pass / West Ham · human and GPT-2 both describe the same passage"],
        ["pass", "Human",
         "Wonderful movement by West Ham. The players are holding possession, mostly exchanging "
         "one-touch passes. They control the pace of the game now. The midfielder attempts to feed "
         "the defender with a chip pass, but the goalkeeper reacts well and comes off his line to "
         "intercept.", "Reference line"],
        ["pass", "GPT-2 v4",
         [("West Ham are showcasing great team work by exchanging some short passes and one-two "
           "passing balls with one another to create new attacking opportunities. The midfielder is "
           "unlucky enough not to shoot from the edge of the box as ", None),
          ("his effort flies just wide of the right post", "bad"), (".", None)],
         [("invented shot", "bad"), ("; team correct", None)]],
        ["pass", "Template", "Patient stuff from West Ham, switching the play smartly.",
         "correct but short"],
        ["GROUP", "Record B · corner / Eintracht Frankfurt · both systems name the wrong team"],
        ["corner", "Human",
         "The forward strikes the ball towards goal, but it fails to reach the target after being "
         "well blocked by one of the defenders. Corner kick. Eintracht Frankfurt will have an "
         "opportunity.", "Reference line"],
        ["corner", "GPT-2 v4",
         [("Napoli", "bad"),
          (" failed to take advantage of the corner as their opponents were alert and averted the "
           "threat at the last moment! The ball goes out of play. Neat result from the forward's "
           "attempted effort with a cross into the box resulting in a dangerous clearance.", None)],
         [("wrong team", "bad"), ("; topic correct", None)]],
        ["corner", "Template",
         [("The delivery comes in for ", None), ("Napoli", "bad"), ("!", None)],
         [("wrong team", "bad")]],
        ["GROUP", "Record C · goal / B. Monchengladbach · GPT-2 misses the event"],
        ["goal", "Human",
         "The goal stands! That took a long time for the referee to decide. The relieved B. "
         "Monchengladbach players can finally celebrate.", "Reference line"],
        ["goal", "GPT-2 v4",
         [("The forward races towards goal but the defender gets back well to make a challenge. One "
           "of the defending players reacts superbly to get his teammates down and intercepts one. ",
           None),
          ("A promising stop for Inter! Corner kick from long range.", "bad")],
         [("missed goal", "bad"), ("; wrong team", None)]],
        ["goal", "Template",
         [("GOAL!", "good"), (" ", None), ("Inter", "bad"),
          (" make it count — sheer precision at a vital time!", None)],
         [("calls goal", "good"), ("; wrong team", "bad")]],
    ],
    [0.62, 0.55, 3.9, 1.18], font_size=8.2)
BP("Result.", "GPT-2 is the better surface realiser: longer, more varied, closer to human overlap. "
   "It is not the better fact layer. The validator raises precision but accepts 95% of lines, and "
   "in one record both the model and the template inherit a wrong team from the shared feature layer.")

doc.add_heading("5.2 Pre-recorded full context versus an API-fed goal", level=2)
P("The full-context run is a batch job: it reads the whole match, detects events, writes lines and "
  "only then synthesises speech. It calls both goals correctly and attributes them to France. Table 4 "
  "shows v5 GPT-2 goal lines generated from the same fact sheet, alongside a real API-Football event "
  "feed. The API path removes perception entirely and can therefore be used to test the language and "
  "speech stages in isolation.")
table_caption(4, "Pre-recorded full context and a real API-Football event-feed example. The full-context "
                 "goal lines are v5 GPT-2 generations from the same fact sheet; the API row is a "
                 "different match (Alajuelense–Perez Zeledon), so compare system behavior, not "
                 "wording. Green marks the correct side.")
add_table(
    ["Mode", "First line", "Goal commentary", "Outcome"],
    [
        ["Pre-recorded, full context", "16.6 min",
         [("[80'] Goal! ", None), ("France", "good"),
          (" earn a first-time opportunity as they open up a promising attacking opportunity. The "
           "score is 2:1.\n[81'] Goal! ", None), ("France", "good"),
          (" earn a first-time opportunity as they open up the defence to hit the ball from long "
           "range. The score is 2:2.", None)],
         "2/2 goals; correct side; slower"],
        ["API-fed example (different match)", "76.1 s cold + 2.1 s TTS",
         [("Goal! A. Hernandez (LD Alajuelense) latches on to a precise pass inside the box and "
           "unleashes a shot from the edge of the box into the left side of the goal. The score is "
           "1:0.", None)],
         [("1/1 goal", "good"), ("; correct team, player and score; 2 cards skipped", None)]],
    ],
    [1.35, 0.7, 3.25, 1.1], font_size=8.2)
P("The API-fed row is a real API-Football event feed for LD Alajuelense 1–0 Perez Zeledon "
  "(fixture 1551707, 26 October 2026). The match is finished; this is a live-style event feed, not a "
  "live broadcast. In the 85:00–90:00 window the feed contains two yellow cards and a 90' goal by "
  "A. Hernandez. The goal was fed as structured facts — team, player, minute and score 1–0 — and the "
  "v5 slot adapter produced the line shown, with the correct team, player and score; Fish voiced it "
  "in 2.1 s. The 76.1 s generation figure is a cold start that includes loading GPT-2; a deployed "
  "service would keep the model warm. The two cards are skipped because the current commentary label "
  "set has no card class. This is a different match from ARG–FRA, so compare the system behavior, not "
  "the wording: the API removes perception and gets the team and score right, but it supplies only "
  "major events; the video pipeline supplies richer events.")
add_figure(FIG / "fig3_latency.png", 3,
           "Latency to the first commentary line for the 18 s clip and the full match. The full-match "
           "run is faster than real time in aggregate (0.30×), but the first line appears only after "
           "the whole match is scored and written.", width=5.6)

doc.add_heading("5.3 Event spotting", level=2)
P(f"Figure 4 compares the ten reference moments with the 38 event lines from the full run. Figure 5 "
  "summarises recall. Goals are the strong case: both are found within 13–22 seconds, with precision "
  "and recall 1.00. Shots are weaker: two of six are matched, with a timing error of 51–59 seconds. "
  "Both tackles are missed. Overall recall is 0.40 at ±1 minute. The pipeline's label distribution "
  "explains part of the gap: it emits 19 pass and 7 dead-time lines, but only 5 shot, 5 corner and "
  "2 goal lines.")
add_figure(FIG / "fig4_timeline.png", 4,
           "Reference moments (top) and detected event lines (bottom). Faint ticks are passes and "
           "dead time; strong ticks are goals, shots, corners and tackles. Dashed lines mark matches "
           "within ±1 minute; red-ringed markers mark missed reference events.", width=6.2)
add_figure(FIG / "fig5_recall.png", 5,
           "Event recall against the ten curated moments. Goal precision and recall are both 1.00; "
           "timing MAE for matched events is 36.2 s.", width=4.7)
add_figure(ROOT / "docs" / "goal_clip_contact.jpg", 6,
           "The 81' goal clip. The broadcast score flips from 2–1 to 2–2; the score-bug reader "
           "infers the starting score and anchors the away goal at 5 s. Frames at 0, 3, 6, 9, 12 "
           "and 15 s.", width=6.2)
table_caption(5, "Reference moments versus detected events at ±1 minute. A dash means no same-kind "
                 "detection in the window.")
add_table(
    ["Minute", "Reference", "Detected", "Result"],
    [
        ["38", "shot", "38.85 shot", "matched (+50.8 s)"],
        ["40", "tackle", "—", "missed"],
        ["49", "shot", "49.98 shot", "matched (+58.8 s)"],
        ["54", "tackle", "—", "missed"],
        ["65", "shot", "65.56 corner", "missed"],
        ["76", "shot", "—", "missed"],
        ["80", "goal", "79.63 goal", "matched (−22.2 s)"],
        ["81", "goal", "80.78 goal", "matched (−13.2 s)"],
        ["85", "shot", "—", "missed"],
        ["99", "shot", "—", "missed"],
    ],
    [0.75, 0.95, 2.2, 2.4], font_size=8.5)
P("Object detection is a separate measurement. Table 6 compares YOLO26n with the paused YOLOv5lu "
  "run. YOLO26n is fractionally ahead in aggregate mAP, but the budgets are not equal: 50 vs 33 "
  "epochs and 2.39 M vs 53 M parameters. Per-class YOLO26n mAP50–95 is 0.621 for Player, 0.291 for "
  "Ball and 0.530 for Referee. The low Ball score is why the ball path is interpolated rather than "
  "trusted, and it is a plausible contributor to the missed shot/tackle events.")
table_caption(6, "Object detection on the V1 validation split. This is not event-spotting accuracy.")
add_table(
    ["Model", "Params", "Epochs", "mAP50", "mAP50–95", "Precision", "Recall"],
    [
        ["YOLO26n", "2.39 M", "50", "0.805", "0.481", "0.864", "0.763"],
        ["YOLOv5lu", "53 M", "33", "0.795", "0.459", "0.854", "0.752"],
    ],
    [0.9, 0.65, 0.65, 0.85, 0.95, 0.95, 0.9], font_size=8.2)
BP("Result.", "The score-bug goal path is reliable enough to build on. The general event classifier "
   "is not: it finds one third of reference shots, no tackles, and emits more pass/dead-time lines "
   "than event lines. Detection mAP must not be reported as event-spotting accuracy.")

doc.add_heading("5.4 Training dynamics", level=2)
P("Figure 7 shows the YOLO26n training and validation curves. The box and classification losses fall "
  "steadily, and mAP50 rises from 0.538 at epoch 1 to 0.806 at epoch 50; mAP50–95 follows from 0.269 "
  "to 0.480. The curve is still improving at the final epoch, so 50 epochs is a budget choice rather "
  "than a converged optimum. Figure 8 shows the GPT-2 v4 LoRA loss: a rapid fall from 3.877 to about "
  "1.0 in the first epoch, then a slow, noisy decline to 0.779 at step 2,481. The v5 API-slot "
  "adapter ran for 4,852 steps and ended at 0.903 training loss; its lower final loss did not remove "
  "the need for constrained goal decoding, because the target corpus contains goal passages with "
  "corner and clearance language. The final held-out acceptance is not a monotone function of "
  "training loss; language quality and factual grounding are different objectives.")
add_figure(FIG / "fig6_yolo.png", 7,
           "YOLO26n training curves, 640 px and 50 epochs. mAP50 reaches 0.806 and mAP50–95 0.480 at "
           "the final epoch; the model is still improving, so the schedule is a compute constraint, "
           "not a converged optimum.", width=6.2)
add_figure(FIG / "fig7_gpt2.png", 8,
           "GPT-2 v4 LoRA training loss over 2,481 steps. The rapid first-epoch drop is followed by a "
           "slow, noisy decline; the final loss is 0.779.", width=5.7)

doc.add_heading("5.5 Cost", level=2)
P("Table 7 reports the measured compute, the API-Football cost and an illustrative human baseline. A "
  "full match costs about one cent of local electricity or $0.39 on a cloud A10G. Training both "
  "models for 3.64 hours costs $0.109 locally or $4.40 at the cloud rate. The human scenarios are "
  "illustrative: a two-person team at $50/h for three live hours or four pre-recorded hours costs "
  "$300 or $400. Across the $25–$100/h range, cloud compute is roughly 400–1,500× cheaper. The API "
  "example used the free API-Football tier, so the API cost was $0 for the requests made; paid plans "
  "are excluded. The caveat is that compute is not the only cost: latency, quality review and paid "
  "TTS are excluded.")
table_caption(7, "Cost per 64.5-minute match. Local electricity and cloud rates are measured/assumed; "
                 "API-Football was used on the free tier; human rates are illustrative, not quotes.")
add_table(
    ["Item", "Local GPU", "Cloud g5.2xlarge", "API-Football", "Human team"],
    [
        ["Inference", "$0.010", "$0.392", "—", "$300 live / $400 pre-recorded"],
        ["Training (both models)", "$0.109", "$4.40", "—", "—"],
        ["Training hours", "3.64 h", "3.64 h", "—", "—"],
        ["TTS calls / speech", "43 / 412.2 s", "43 / 412.2 s", "—", "—"],
        ["API feed", "—", "—", "$0 free tier; paid plan excluded", "—"],
    ],
    [1.15, 0.95, 0.95, 1.25, 1.45], font_size=8.2)
add_figure(FIG / "fig8_cost.png", 9,
           "Cost per match minute on a log scale. The local and cloud points are two to three orders "
           "of magnitude below the human scenario; the barrier to deployment is quality and latency, "
           "not compute. The API-Football free tier used here is $0 and cannot be shown on a log "
           "axis.", width=6.0)

doc.add_heading("5.6 Tracking, possession and ball path", level=2)
P("Figure 10 shows the tracking overlay across six moments of the ARG–FRA clip. Boxes are tracked "
  "players, the green dot and trail are the ball path, and the bar along the bottom is the live "
  "possession split. Figure 11 isolates the same overlay around the 80′–81′ sequence, where the "
  "possession bar and the ball trail are easiest to read.")
add_figure(ROOT / "docs" / "tracking_contact.jpg", 10,
           "Tracking overlay across six moments. Player boxes are ByteTrack identities; the green "
           "dot/trail is the ball path; the bottom bar is the possession split (Argentina white, "
           "France blue). The tracker, not the language model, supplies these facts.", width=6.2)
add_figure(ROOT / "docs" / "possession_contact.jpg", 11,
           "Possession and ball path around the 80′–81′ sequence. The bottom bar reports the "
           "team possession split; the green trail shows the smoothed ball path. Both are aggregated "
           "into the fact sheet used by the commentary stage.", width=6.2)
P("The overlay is generated in three steps. YOLO26n detects players, the referee and the ball in "
  "each scheduled frame; ByteTrack keeps identities across frames. Jersey colours are clustered into "
  "two teams with KMeans, and possession is assigned to the team of the tracked player nearest the "
  "smoothed ball. Short ball gaps are interpolated only across bounded gaps, and single-frame "
  "teleports are rejected. The per-frame possession indicator is then averaged into the 2.5-second "
  "window facts: possession_home, possession_away, ball visibility, ball zone and tracked-player "
  "count.")
BP("Why the overlay matters.", "The same numbers in the overlay become the fact sheet. When the "
   "commentary says a side is in control, it is reading the possession feature, not inventing a "
   "narrative. The goal-side anchor comes from the score bug; the tracker supplies the surrounding "
   "play.")
BP("What the overlay does not solve.", "Broadcast cuts, close-ups and crowd shots break the ball "
   "path, and similar kits can confuse the jersey clusters. Possession is therefore an estimate, not "
   "a tracking ground truth. The pipeline marks interpolated gaps and never treats a missing ball as "
   "a fact.")

doc.add_heading("5.7 Commercialisation and cost advantage", level=2)
P("The measured cost is the commercial opening. A 64.5-minute match costs about $0.392 on a cloud "
  "A10G, or about $0.010 of local electricity. Against an illustrative two-person human team at "
  "$50/h, that is 765× cheaper than a $300 live team and 1,020× cheaper than a $400 pre-recorded "
  "team on cloud compute. Local electricity is 30,000–40,000× cheaper, but that figure excludes the "
  "GPU hardware, so the cloud comparison is the honest one. Across a $25–$100/h human range, the "
  "cloud advantage is roughly 380–1,530×.")
table_caption(8, "Cost advantage per 64.5-minute match. Human rates are illustrative; compute is "
                 "measured. The local column is electricity only and excludes GPU hardware. Paid TTS, "
                 "paid API plans, storage, rights and human review are excluded.")
add_table(
    ["Baseline", "Cost / match", "Matchcall cloud", "Matchcall local", "Advantage"],
    [
        ["Human live team (2 × $50/h × 3 h)", "$300", "$0.392", "$0.010",
         "765× cloud; 30,000× local"],
        ["Human pre-recorded team (2 × $50/h × 4 h)", "$400", "$0.392", "$0.010",
         "1,020× cloud; 40,000× local"],
        ["Human range, $25–$100/h", "$150–$600", "$0.392", "—",
         "380–1,530× cloud"],
        ["API-Football, free tier", "$0 requests", "—", "—",
         "Paid plan excluded; not a compute comparison"],
    ],
    [2.4, 1.0, 1.0, 1.0, 1.55], font_size=8.0)
P("These are compute-only comparisons. They exclude the costs that decide a real product: paid TTS "
  "above the free tier, paid API-Football, storage and egress, rights clearance, human quality "
  "review and support. They also exclude the cost of the GPU if the service is deployed on-premise. "
  "At an illustrative $10–$50 per match, cloud compute is 0.8–3.9% of revenue before those costs; "
  "that is the margin the product has to protect.")
BP("Per-match SaaS for rights-holders.", "Sell processed commentary or highlight packages to "
   "broadcasters, clubs and lower leagues that cannot justify a full commentary team. The customer "
   "pays per match or per minute of output; the provider runs the GPU pipeline.")
BP("On-premise or white-label.", "Rights-holders often cannot send video to a third party. A "
   "containerised on-premise version keeps the video local and licenses the pipeline and models. "
   "This is the natural route for federations and large clubs.")
BP("Highlights and social clips.", "The same facts and voice can produce short vertical clips, "
   "multilingual descriptions and automated match reports. The marginal cost is another generation "
   "and TTS pass, not another camera crew.")
BP("API-fed second screen.", "Where a live event feed exists, the API path can anchor goals, cards "
   "and substitutions without reading the video. The video path then adds shots, corners, "
   "possession and player context. The two feeds are complementary, not competing.")
P("The commercial moat is not the language model. It is the auditable fact layer, the score-bug "
  "anchor and the ability to run the whole chain on one consumer GPU. The risks are equally clear: "
  "general event recall is still 0.40, the system is pre-recorded rather than live, the commentary "
  "needs human review, and rights and league data licences are commercial prerequisites. A credible "
  "first market is lower-league and highlights automation, where the alternative is no commentary at "
  "all; live top-flight commentary is a later, harder market.")

# ---------------------------------------------------------------- 6 discussion
doc.add_heading("6. Discussion", level=1)
P("The experiments support a narrower claim than “automatic commentary.” The pipeline is a cheap, "
  "auditable, pre-recorded system whose goal path is reliable and whose general event layer is not "
  "yet good enough for broadcast output. Three findings organise the rest.")
BP("Score-bug anchoring is the strongest component.", "A persistent digit flip is a high-precision "
   "signal that survives replays, close-ups and celebration cuts, and it names the scoring side. The "
   "full-match run finds both goals with precision and recall 1.00. The main engineering requirement "
   "is to carry the anchored side into the realised line.")
BP("Language quality and factuality are separate objectives.", "GPT-2 v4 more than doubles ROUGE-L "
   "over templates and writes human-length lines, but it invents shots, misnames teams and can miss "
   "a goal. The validator improves precision but accepts 95% of lines on this prompt set; it checks "
   "words, not truth. A grounded template is safer but not sufficient when the feature layer itself "
   "carries the wrong team.")
BP("Latency is an architectural problem, not a speed knob.", "The full run is faster than real "
   "time in aggregate, but the first line appears after 16.6 minutes. A live system needs streaming "
   "perception, incremental speech and explicit side propagation; the current batch architecture "
   "should be described as pre-recorded.")
P("For the paper's framing, the strongest contribution is the separation of deterministic facts from "
  "language and the score-bug goal path. The weakest links are general event spotting, team-side "
  "propagation and live latency. Reporting those weaknesses is more useful than claiming another "
  "state-of-the-art commentary model.")

# ---------------------------------------------------------------- 7 threats
doc.add_heading("7. Threats to Validity", level=1)
BP("Reference validity.", "The ten event moments are curated from the earlier highlight reel, not an "
   "exhaustive independent annotation. The 0.40 recall is a consistency check against those moments; "
   "event precision for shots, tackles and corners is not measured. The 2/2 goal result is exact for "
   "this recording but has only two samples.")
BP("External validity.", "All end-to-end results come from one 64.5-minute broadcast, one competition "
   "and one production style. The score-bug boxes are tuned to this overlay; another broadcaster "
   "would need recalibration. The detector was trained on a small Roboflow export.")
BP("Construct validity.", "BLEU, ROUGE and METEOR measure overlap, not truth or entertainment. The "
   "validator acceptance rate is not a human preference score. No human evaluation of commentary "
   "quality was run; the paper reports the need for one.")
BP("Budget validity.", "YOLO26n and YOLOv5lu did not receive equal epochs or parameter budgets, so the "
   "mAP comparison is descriptive, not a controlled model comparison. The human cost is illustrative; "
   "paid TTS, storage and egress are excluded.")
BP("Reproducibility.", "The pipeline is deterministic given the same video and model weights, but the "
   "generative language model samples; lines vary across runs. The reported commentary examples are "
   "single samples.")

# ---------------------------------------------------------------- 8 conclusion
doc.add_heading("8. Conclusion", level=1)
P("Matchcall shows that a small, modular, fact-grounded pipeline can turn a full broadcast into "
  "voiced commentary on a consumer GPU at a cost far below a human team. The goal path works because "
  "the score bug is an explicit, high-precision signal; the language path works well enough to "
  "produce human-length lines; the cost is negligible. The system does not yet work as live "
  "commentary, does not find most non-goal events, and can attribute a goal to the wrong team. Those "
  "are the next problems: streaming architecture, side propagation, better event annotation and a "
  "human evaluation of commentary quality. The broader lesson is that separating facts from language "
  "makes both the successes and the failures measurable.")

# ---------------------------------------------------------------- references
doc.add_page_break()
doc.add_heading("References", level=1)
paper_html = (ROOT / "docs" / "matchcall_paper.html").read_text(encoding="utf-8")
refs = re.findall(r"<li>(.*?)</li>", paper_html, re.S)
refs = [html.unescape(re.sub(r"<[^>]+>", "", r)).strip() for r in refs]
# Keep only the sources this paper actually relies on, in citation order.
REF_ORDER = [1, 2, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             21, 22, 25, 27, 28, 29, 32, 33, 34, 37]
refs = [refs[i - 1] for i in REF_ORDER]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph(style="Paper Reference")
    p.add_run(f"[{i}] ").font.name = "Times New Roman"
    p.add_run(r)

# ---------------------------------------------------------------- appendix
doc.add_page_break()
doc.add_heading("Appendix A. Reproducibility", level=1)
left_para("The main artifacts are produced by the following commands. The full-match job is "
          "7c23a8a91829; the corrected goal clip is 8cdba651f954.", size=9.5)
code_block("python scripts/compare_commentary.py --n 40 --device cuda\n"
           "python scripts/evaluate_event_spotting.py\n"
           "python scripts/compile_experiments.py\n"
           "python scripts/api_feed_current_example.py")
left_para("Data and outputs: output/comparison/commentary_comparison.json; output/experiments/"
          "event_spotting.json; output/experiments/experiment_results.json; output/experiments/"
          "full_match_job.json; output/experiments/api_feed_current_example.json. Models: "
          "models/yolo26/yolo26n_v1_640; models/gpt2_lora_v4; models/gpt2_lora_v5. Commit 7f1a826.",
          size=9.5)

# ---------------------------------------------------------------- properties
doc.core_properties.title = "Matchcall: A Fact-Grounded, Low-Cost Pipeline for Spoken Football Commentary"
doc.core_properties.author = "Aryaman"
doc.core_properties.subject = "Football commentary, action spotting, tracking, grounded generation, TTS"
doc.core_properties.comments = "Experiment paper; supervisor: Nabin Sir."

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("wrote", OUT, OUT.stat().st_size)
print("references", len(refs))
