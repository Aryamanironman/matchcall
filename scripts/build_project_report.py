"""Build the Matchcall capstone report from the UTS project-report template.

Usage:
  python scripts/build_project_report.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
DOCS = ROOT / "docs"
FIG = DOCS / "paper_figures"
TEMPLATE = Path(r"C:\Users\asus\Downloads\project report template.docx")
OUT = DOCS / "Matchcall_Project_Report.docx"

AI = json.loads((REPO / "output" / "comparison" / "commentary_comparison.json")
                .read_text(encoding="utf-8"))
M = AI["metrics"]
EV = json.loads((REPO / "output" / "experiments" / "event_spotting.json")
                .read_text(encoding="utf-8"))
_boot_path = REPO / "output" / "comparison" / "commentary_bootstrap.json"
BOOT = json.loads(_boot_path.read_text(encoding="utf-8")) if _boot_path.exists() else None

doc = Document(str(TEMPLATE))
section = doc.sections[0]

# ---------------------------------------------------------------- styles
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
normal.paragraph_format.line_spacing = 1.15
normal.paragraph_format.space_after = Pt(6)

for name, size, before, after in (
        ("Heading 1", 16, 16, 8), ("Heading 2", 13, 12, 5),
        ("Heading 3", 11.5, 8, 3)):
    st = doc.styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.keep_with_next = True

cap = doc.styles.add_style("Report Caption", WD_STYLE_TYPE.PARAGRAPH)
cap.font.name = "Calibri"
cap.font.size = Pt(9)
cap.font.color.rgb = RGBColor(80, 80, 80)
cap.paragraph_format.space_before = Pt(2)
cap.paragraph_format.space_after = Pt(8)
cap.paragraph_format.line_spacing = 1.0

ref_style = doc.styles.add_style("Report Reference", WD_STYLE_TYPE.PARAGRAPH)
ref_style.font.name = "Calibri"
ref_style.font.size = Pt(9.5)
ref_style.paragraph_format.left_indent = Inches(0.32)
ref_style.paragraph_format.first_line_indent = Inches(-0.32)
ref_style.paragraph_format.space_after = Pt(3)
ref_style.paragraph_format.line_spacing = 1.0
ref_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

# ---------------------------------------------------------------- title page
title_p = doc.paragraphs[7]
title_p.clear()
title_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
for label, value, brk in (
        ("Project Title: ", "Matchcall — A Fact-Grounded, Low-Cost Pipeline for Spoken Football Commentary", True),
        ("Student Name: ", "Aryaman Sharma", True),
        ("Student ID: ", "25158763", True),
        ("Supervisor Name: ", "Prof. Nabin Sharma", False)):
    r = title_p.add_run(label)
    r.bold = True
    title_p.add_run(value)
    if brk:
        title_p.add_run().add_break()

date_p = doc.paragraphs[12]
date_p.clear()
date_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = date_p.add_run("Date of Submission: ")
r.bold = True
date_p.add_run("30 October 2026")

# The template already carries a right-aligned page number in the footer.

# ---------------------------------------------------------------- helpers
def page():
    doc.add_page_break()


def H1(text):
    doc.add_heading(text, level=1)


def H2(text):
    doc.add_heading(text, level=2)


def H3(text):
    doc.add_heading(text, level=3)


def P(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def BP(lead, body):
    p = doc.add_paragraph()
    r = p.add_run(lead + " ")
    r.bold = True
    p.add_run(body)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def bullet(text):
    p = doc.add_paragraph("•  " + text)
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.space_after = Pt(3)
    return p


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


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


def caption(text):
    p = doc.add_paragraph(text, style="Report Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def add_table(headers, rows, widths=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = 1
    table_borders(table)
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(str(h))
        r.bold = True
        r.font.size = Pt(font_size)
        r.font.name = "Calibri"
        shade(cell, "E7E6E6")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(value))
            r.font.size = Pt(font_size)
            r.font.name = "Calibri"
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def figure(path, number, cap_text, width=6.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width))
    caption(f"Figure {number}. {cap_text}")


def load_refs():
    text = (DOCS / "matchcall_paper.html").read_text(encoding="utf-8")
    raw = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
           for x in re.findall(r"<li>(.*?)</li>", text, re.S)]
    # Report citation order: the numbers used in the text, not the source HTML order.
    order = [1, 12, 13, 14, 15, 32, 6, 11, 16, 17, 19, 18,
             27, 28, 29, 21, 22, 25, 37, 33]
    refs = [raw[i - 1] for i in order]
    refs.append("SoccerReplay-1988 dataset, Hugging Face. "
                "https://huggingface.co/datasets/Homie0609/SoccerReplay-1988")
    return refs


# remove the template's instruction body, keep title page and contents
for p in list(doc.paragraphs[28:]):
    p._element.getparent().remove(p._element)

page()

# ---------------------------------------------------------------- 1 executive summary
H1("1. Executive Summary")
P("Football commentary is expensive to produce and does not scale to every match. A live broadcast "
  "needs a commentator, a producer and often an analyst; most matches, especially outside the top "
  "leagues, are never covered at all. This project asks whether a small, modular system can produce "
  "usable spoken commentary from a broadcast on a single consumer GPU, with facts that can be "
  "audited before the language model writes a line.")
P("Matchcall is a five-stage pipeline. A score-bug reader finds goal anchors; a motion-gated window "
  "scan proposes other events; YOLO26n and ByteTrack track players, the referee and the ball; jersey "
  "clustering assigns teams and possession; a deterministic fact sheet is written; GPT-2 with a LoRA "
  "adapter generates the commentary under a validator; and Fish Audio voices the accepted lines. A "
  "web interface exposes the pipeline, the ARG–FRA demo, an API-Football example and a match-plan "
  "upload.")
P(f"On a 64.5-minute Argentina–France broadcast the pipeline produced 43 fully voiced lines in 19.4 "
  f"minutes. Both goals were recovered with precision and recall 1.00, but overall event recall "
  f"against ten curated moments was {EV['recall_overall']:.2f} at ±1 minute. On 40 held-out human "
  f"references, GPT-2 v4 reached ROUGE-L {M['v4']['rougeL']:.3f} versus {M['template']['rougeL']:.3f} "
  f"for a template baseline, and METEOR {M['v4']['meteor']:.3f} versus {M['template']['meteor']:.3f}. "
  "A paired bootstrap puts the ROUGE-L and METEOR advantages above zero; BLEU-2 and BLEU-4 are not "
  "distinguishable from the template.")
P("The system is deliberately reported as a pre-recorded system, not a live one. Its contribution "
  "is an auditable fact layer, a reliable score-bug goal path, a grounding validator and a complete "
  "broadcast-to-speech pipeline at a measured compute cost. General event spotting, live latency, "
  "team-side propagation and human evaluation remain open. The report also sets out the "
  "commercialisation path: cloud inference is roughly 765× cheaper than a $300 human live team and "
  "1,020× cheaper than a $400 pre-recorded team, before rights, paid TTS, API plans and human review.")

# ---------------------------------------------------------------- 2 introduction
H1("2. Introduction")
H2("2.1 Background")
P("A football broadcast is a continuous, time-sensitive description task. The commentator must know "
  "the score, the minute, the teams, the players and the narrative, and must say something useful "
  "within seconds of an event. Producing that coverage is labour-intensive, which is why most "
  "matches are not covered and why lower leagues, women's competitions and smaller federations are "
  "under-represented.")
P("Automatic commentary is difficult for two different reasons. The perception problem is to find "
  "events and identify the actors; the language problem is to say something accurate, varied and "
  "appropriate. End-to-end multimodal models can do both, but they are expensive, opaque and prone "
  "to hallucination. Template systems are auditable but repetitive. This project tests the middle: a "
  "small modular pipeline with a deterministic fact layer between perception and language.")
H2("2.2 Project Context")
P("The project was built as an AI/data-analytics capstone on a consumer GPU. It uses a real "
  "Argentina–France broadcast, the SoccerReplay-1988 commentary corpus and the Roboflow V1 football "
  "detection dataset. The system is a research prototype, not a deployed broadcast product. The "
  "demo, training runs, metrics and figures are retained in the repository and in ClearML so the "
  "claims can be inspected.")
H2("2.3 Objectives")
P("The project has five objectives:")
for t in (
    "Build a complete video-to-voice pipeline that runs on one consumer GPU.",
    "Separate measured facts from generated language so the facts can be audited.",
    "Make the goal path reliable using the broadcast score bug as a high-precision anchor.",
    "Evaluate the commentary against held-out human references and a template baseline.",
    "Report the failure modes honestly: event recall, team attribution, latency and cost.",
):
    bullet(t)
H2("2.4 Report Structure")
P("Section 3 reviews related work. Section 4 defines the objectives and scope. Section 5 describes "
  "the methodology. Section 6 documents the data sources and analysis. Section 7 describes model "
  "development and implementation. Section 8 reports results and discusses them. Section 9 concludes, "
  "Section 10 gives recommendations, and Section 11 lists references. The appendices contain "
  "reproducibility commands, the match-plan schema and cost assumptions.")

# ---------------------------------------------------------------- 3 literature
H1("3. Literature Review")
H2("3.1 Action Spotting and Event Understanding")
P("SoccerNet formalised action spotting as predicting the temporal location and class of sparse "
  "events in long broadcasts, evaluated with Average-mAP at multiple tolerances [1]. Later work "
  "extended the task to dense captioning and commentary generation. SoccerNet-Caption introduced "
  "almost 37,000 timestamped commentaries across 715.9 hours [2]. MatchTime showed that alignment "
  "quality is a first-order determinant of generation quality and released an aligned dataset and "
  "the MatchVoice model [3]. TimeSoccer consumes full 45-minute matches with frame compression [4], "
  "while SoccerComment combines LanguageBind, Vicuna-7B and retrieval [5].")
P("These systems establish the benchmark, but they optimise event localisation or caption overlap. "
  "They do not expose a deterministic fact layer that a broadcaster can inspect before a line is "
  "spoken. That separation is the design target here.")
H2("3.2 Detection, Tracking, Team Identification and Possession")
P("YOLO established single-pass real-time detection [6]. ByteTrack associates low-score boxes to "
  "recover occluded objects and is the tracker used here [7]. Team identification from jersey colours "
  "and possession estimation from the player nearest the ball are common broadcast-analysis "
  "components; they are approximate under broadcast cuts, close-ups and similar kits. The project "
  "uses YOLO26n for detection, ByteTrack for identity and KMeans for team assignment, and treats "
  "possession as an estimate rather than ground truth.")
H2("3.3 Neural and LLM Commentary")
P("MIKE used six analysers, importance-scored proposition selection and template generation [8]. "
  "Neural captioning systems produce more varied text but depend on large aligned corpora and report "
  "overlap or alignment metrics rather than factuality [2], [3]. Recent multimodal LLMs consume full "
  "matches or retrieve external knowledge, but they are heavy and do not expose an auditable fact "
  "layer [4], [5]. The project tests a small GPT-2 with LoRA [9], [10] under a deterministic "
  "validator, not a large end-to-end model.")
H2("3.4 Grounding, Hallucination and Evaluation")
P("Hallucination in natural-language generation is surveyed by Ji et al. [11]. FActScore decomposes "
  "long-form text into atomic facts and checks each against a knowledge source [12]. The present "
  "system cannot check open-world facts; it checks a closed fact sheet: team, score, digits, event "
  "and goal language. BLEU, ROUGE and METEOR [13]–[15] measure overlap with a reference, not truth. "
  "This report therefore adds bootstrap intervals and states the need for a human study rather than "
  "treating overlap as a factuality score.")
H2("3.5 Speech Synthesis and Prosody")
P("Tacotron 2 established neural TTS naturalness [16]; FastSpeech 2 added explicit duration, pitch "
  "and energy control [17]. Fish Audio's S1 family offers emotion and tone markers and low-latency "
  "synthesis [18]. The project uses Fish Audio s2.1-pro-free, with edge-tts as a fallback when the "
  "hosted key or request fails. Event intensity controls pacing, and ffmpeg mixes the voice under the "
  "original crowd.")
H2("3.6 Research Gap")
P("The literature is rich in perception benchmarks and increasingly rich in generative commentary, "
  "but sparse in systems that (i) expose a deterministic fact layer, (ii) use a score bug as a gold "
  "goal anchor, (iii) run the whole broadcast-to-speech chain on a consumer GPU, and (iv) report "
  "negative results alongside the positive ones. Matchcall is positioned as a system paper addressing "
  "that gap, not as a new state of the art on SoccerNet.")

# ---------------------------------------------------------------- 4 objectives/scope
H1("4. Project Objectives and Scope")
H2("4.1 Project Aim")
P("To build and evaluate an auditable, low-cost pipeline that turns a football broadcast into "
  "grounded spoken commentary on a single consumer GPU.")
H2("4.2 Specific Objectives")
for t in (
    "Implement score-bug goal anchoring and a motion-gated event scan.",
    "Track players, referee and ball; estimate teams and possession.",
    "Produce a deterministic fact sheet for every event.",
    "Generate commentary with GPT-2 + LoRA under a grounding validator.",
    "Synthesise speech with Fish Audio and render a complete clip.",
    "Evaluate commentary, event spotting, training behaviour and cost.",
    "Expose the system through a reproducible web demo and ClearML runs.",
):
    bullet(t)
H2("4.3 In Scope")
P("One 64.5-minute ARG–FRA broadcast; a 40-prompt held-out SoccerReplay comparison; YOLO26n and "
  "GPT-2 v4/v5 training; a real API-Football event-feed example; a local web UI; a match-plan upload "
  "that is parsed in the browser; cost accounting for local and cloud inference.")
H2("4.4 Out of Scope")
P("Live streaming and sub-second latency; a full independent event annotation across many matches; "
  "a human preference study; broadcast rights clearance; paid TTS and paid API plans; multilingual "
  "commentary; a production multi-tenant service. These are stated as limitations and future work, "
  "not claimed as completed.")
H2("4.5 Success Criteria")
P("The project succeeds if it produces a complete voiced clip, recovers both goals with correct "
  "sides, generates commentary that passes a deterministic validator, improves overlap over a "
  "template baseline, and documents its failures with reproducible artifacts.")

# ---------------------------------------------------------------- 5 methodology
H1("5. Methodology")
H2("5.1 Overview")
P("Matchcall has five stages: action spotting, tracking and team/possession facts, deterministic "
  "description, language generation, and speech. The language model never decides the score, minute, "
  "team or event; it only realises facts measured by the earlier stages. Figure 1 shows the flow.")
figure(FIG / "fig1_pipeline.png", 1,
       "The five-stage pipeline. Stages 1–2 observe; stage 3 fixes the facts; stage 4 realises them; "
       "stage 5 voices and renders. The language model never writes the score or the event team.")
H2("5.2 Action Spotting")
P("The broadcast score bug is sampled until its digits flip. The reader infers the score already on "
  "the board from the first samples, so a clip starting at 2–1 is not treated as 0–0. A motion-gated "
  "window scan proposes other events: proxy frames are extracted at 4 fps, a mean absolute "
  "frame-difference score is computed on 128×72 grayscale images, and YOLO runs every half-second "
  "when motion is above the dead-time gate and four times sparser when it is not. The frame features "
  "are pooled into 2.5-second windows.")
P("A weak-supervision labeler turns window features into candidate events: score-bug flips are gold "
  "for goals; goalmouth zone, ball speed, crowd rise, motion and player clustering label the rest; "
  "caps limit goals, shots and corners per match. A ResNet-18 classifier is an advisory second "
  "opinion, trained on auto-labelled frames. It can promote a quiet pass to shot or goal only when "
  "confidence and other evidence agree; it never overrides the score bug. Its validation accuracy "
  "is a weak-label split property, not an event-spotting benchmark.")
H2("5.3 Tracking, Teams and Possession")
P("YOLO26n detects players, the referee and the ball in each scheduled frame; ByteTrack maintains "
  "identities across frames. Jersey colours are clustered with KMeans into two teams, and possession "
  "is assigned to the team of the tracked player nearest the smoothed ball. Short ball gaps are "
  "interpolated only across bounded gaps, and single-frame teleports are rejected. The output is a "
  "deterministic fact sheet: score, minute, ball zone, possession, tracked-player count and event "
  "team.")
H2("5.4 Commentary and Validation")
P("GPT-2 v4 is a LoRA adapter over GPT-2, trained on team-grounded SoccerReplay commentary. The "
  "prompt supplies the event label, team, scoreline, minute, phase and measured vision features. A "
  "v5 adapter uses an API-slot prompt (`<HOME>`, `<AWAY>`, `<TEAM>`, `<PLAYER>`, `<SCORE>`, "
  "`<MINUTE>`) and substitutes the facts after generation, so the same stack can consume an "
  "API-Football event feed. Candidate lines are cleaned and checked: the event team must appear, "
  "blocked names are rejected, digits and both colon and hyphen scores must match the fact sheet, "
  "and a goal line must contain a real goal action and no contradiction such as ‘corner’, ‘saved’ or "
  "‘out of play’. A grounded template is the last resort.")
H2("5.5 Speech and Rendering")
P("Accepted lines are synthesised with Fish Audio s2.1-pro-free. Fish is primary; edge-tts is a "
  "fallback if the key is missing or the request fails. Event intensity controls pacing, the voice is "
  "mixed under the crowd, and ffmpeg renders the full clip at the original timestamps.")
H2("5.6 MLOps and Reproducibility")
P("Every YOLO and GPT-2 run is a ClearML task with parameters, dataset information, scalars, plots, "
  "artifacts and a registered model [20]. The web demo reads the same run artifacts, so the training "
  "curves in the report and the curves in the UI come from the same source.")

# ---------------------------------------------------------------- 6 data
H1("6. Data Collection and Analysis")
H2("6.1 Data Sources")
P("Four data sources are used. The end-to-end video is a 64.5-minute, 1920×1080 Argentina–France "
  "second-half broadcast. The commentary comparison uses 40 held-out SoccerReplay-1988 prompts with "
  "human reference lines [21]. The detector is trained on the Roboflow V1 football dataset (8,677 "
  "train / 2,996 validation images; Player, Ball, Referee) [19]. The API example uses the real "
  "API-Football fixture 1551707, LD Alajuelense 1–0 Perez Zeledon, 26 October 2026.")
H2("6.2 Data Characteristics")
P("The broadcast contains two goals at 80′ and 81′, with the score bug flipping from 2–1 to 2–2. The "
  "ten reference moments are six shots, two tackles and two goals; this is a recall/consistency "
  "reference, not an exhaustive annotation. The SoccerReplay comparison is held out from training. "
  "The detector dataset has three object classes and a long-tailed ball class.")
H2("6.3 Data Cleaning and Preprocessing")
P("Proxy frames are resized to 512×288 for analysis; detector frames are scheduled by motion. Ball "
  "detections are interpolated only across bounded gaps, single-frame teleports are rejected, and "
  "predicted frames are marked. Commentary text is cleaned of markup and ASCII mojibake. Scores and "
  "minutes are parsed into the fact sheet rather than copied from the model.")
H2("6.4 Exploratory Analysis")
P("The pipeline emits 38 event lines on the full broadcast: 19 pass, 7 dead time, 5 shot, 5 corner "
  "and 2 goal. The label distribution already shows the weakness: the general event layer produces "
  "far more pass/dead-time lines than shot/tackle lines. Goal detection is the strong path because "
  "the score bug is a persistent, high-precision signal.")
H2("6.5 Analysis Approach")
P("Commentary is scored against human references with BLEU-1/2/4, ROUGE-L and METEOR, plus validator "
  "acceptance and word count. Event spotting is scored as recall against the ten reference moments "
  "at ±1 minute, with goal precision/recall against the known 80′ and 81′ goals. Cost is measured "
  "for local electricity and a cloud A10G; the human team is an illustrative baseline.")

# ---------------------------------------------------------------- 7 implementation
H1("7. Model Development and Implementation")
H2("7.1 Detector")
P("YOLO26n was fine-tuned at 640 px for 50 epochs on the V1 dataset. The best checkpoint was the "
  "final epoch, with mAP50 0.806 and mAP50–95 0.480. It detects Player, Ball and Referee; ByteTrack "
  "and jersey clustering sit above it. YOLOv5lu is retained as a descriptive comparison but the "
  "budgets are not equal: 50 vs 33 epochs and 2.39 M vs 53 M parameters.")
P("Table 1 summarises the trained components. The detector and language models were trained locally "
  "on one consumer GPU; ClearML logged all runs [20].")
caption("Table 1. Trained components and schedules.")
add_table(["Component", "Model / data", "Params", "Schedule", "Result"], [
    ["Detector", "YOLO26n on V1 (8,677 / 2,996)", "2.39 M",
     "640 px, 50 epochs, 2.70 h", "mAP50 0.806; mAP50–95 0.480"],
    ["Commentary v4", "GPT-2 + LoRA r=32 on SoccerReplay-1988", "124 M base",
     "3 epochs, 2,481 steps", "loss 3.877 → 0.779"],
    ["Commentary v5", "GPT-2 + LoRA r=32, API-slot template", "124 M base",
     "2 epochs, 4,852 steps", "loss → 0.903"],
    ["Speech", "Fish Audio s2.1-pro-free", "—", "43 calls, 412.2 s", "43/43 voiced"],
], [1.0, 2.0, 0.75, 1.25, 1.2], font_size=8.2)
H2("7.2 Event Classifier")
P("The ResNet-18 classifier is trained by the pipeline with weak supervision on auto-labelled "
  "windows from the processed video. Every proxy frame inherits its window label; the model is "
  "fine-tuned from ImageNet weights with a five-class head (goal, shot, pass, tackle, dead time). It "
  "averages class probabilities over a window and returns an advisory label and confidence. It is "
  "not a separately benchmarked football event model, and its validation number should not be read "
  "as event-spotting accuracy.")
H2("7.3 Language Model")
P("GPT-2 v4 uses LoRA rank 32, 160-token sequences, completion-only loss, three epochs and 2,481 "
  "optimizer steps; the loss fell from 3.877 to 0.779. A v5 adapter uses the API-slot prompt and a "
  "38,815-example training split (1,723 validation), 2 epochs and 4,852 steps, reaching a final "
  "training loss of 0.903. The v5 goal path uses slot-token prefill, beam search, contradiction and "
  "off-score token suppression, and a validator that checks both colon and hyphen scores.")
H2("7.4 Speech, UI and MLOps")
P("Fish Audio voices the accepted lines; ffmpeg mixes them under the crowd. The web demo orders the "
  "system as pipeline, ARG–FRA board, API-Football example, upload and training. The upload accepts a "
  "video and a match plan; the plan is parsed in the browser for teams, venue, score, minute and "
  "player/position counts. The backend still needs the plan parser wired into the grounded prompt. "
  "ClearML records every training run and model.")
figure(FIG / "fig6_yolo.png", 2,
       "YOLO26n training curves, 640 px and 50 epochs. mAP50 reaches 0.806 and mAP50–95 0.480 at the "
       "final epoch; the model is still improving, so the schedule is a compute constraint.")
figure(FIG / "fig7_gpt2.png", 3,
       "GPT-2 v4 LoRA training loss over 2,481 steps. The rapid first-epoch drop is followed by a "
       "slow, noisy decline; the final loss is 0.779.")

# ---------------------------------------------------------------- 8 results
H1("8. Results and Discussion")
H2("8.1 Commentary Against Human References")
P(f"Table 2 reports the 40-prompt comparison. GPT-2 v4 more than doubles ROUGE-L over the template "
  f"baseline ({M['v4']['rougeL']:.3f} vs {M['template']['rougeL']:.3f}) and raises METEOR from "
  f"{M['template']['meteor']:.3f} to {M['v4']['meteor']:.3f}, while writing lines of human length "
  f"({M['avg_words_v4']:.1f} words vs 31.9 for humans). BLEU-4 is near zero for both systems, which "
  "is expected for free-form commentary with many valid paraphrases.")
caption("Table 2. Commentary against 40 held-out human references. The human column is the target; "
        "self-similarity is 1 and is not a system score.")
add_table(["Metric", "GPT-2 v4", "Template", "Human reference"], [
    ["BLEU-1", f"{M['v4']['bleu1']:.4f}", f"{M['template']['bleu1']:.4f}", "—"],
    ["BLEU-2", f"{M['v4']['bleu2']:.4f}", f"{M['template']['bleu2']:.4f}", "—"],
    ["BLEU-4", f"{M['v4']['bleu4']:.6f}", f"{M['template']['bleu4']:.6f}", "—"],
    ["ROUGE-L", f"{M['v4']['rougeL']:.4f}", f"{M['template']['rougeL']:.4f}", "—"],
    ["METEOR", f"{M['v4']['meteor']:.4f}", f"{M['template']['meteor']:.4f}", "—"],
    ["Average words", f"{M['avg_words_v4']:.1f}", f"{M['avg_words_template']:.1f}", "31.9"],
    ["Grounding accepted", f"{M['gpt2_accepted']}/{M['n']}", "—", "—"],
], [1.8, 1.3, 1.3, 1.6])
if BOOT:
    b = {r["metric"]: r for r in BOOT["rows"]}
    def ci(k):
        r = b[k]
        p = "p < 0.001" if r["p_bootstrap"] == 0 else f"p = {r['p_bootstrap']:.3f}"
        return f"{r['difference']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}], {p}"
    P(f"A paired percentile bootstrap over the same {BOOT['n']} prompts ({BOOT['resamples']} "
      f"resamples) puts confidence intervals around the differences. ROUGE-L improves by {ci('rougeL')}; "
      f"METEOR by {ci('meteor')}; BLEU-1 by {ci('bleu1')}. BLEU-2 is not distinguishable "
      f"({ci('bleu2')}) and BLEU-4 is not distinguishable ({ci('bleu4')}). The overlap advantage is "
      "therefore measured for ROUGE-L, METEOR and BLEU-1, but not for the longer n-gram metrics.")
figure(FIG / "fig2_commentary.png", 4,
       "GPT-2 v4 is closer to human overlap and length than the template baseline. BLEU-4 is omitted "
       "because it is near zero for both systems; exact values are in Table 2.")
H2("8.2 Error Analysis")
P("Table 3 quotes three held-out records. In the first, GPT-2 names the correct team and produces a "
  "human-length line but invents a shot. In the second, both systems attribute a corner to the wrong "
  "team; this is a shared feature-layer failure, not only a language failure. In the third, GPT-2 "
  "misses a goal and describes a stop and a corner; the template calls a goal but names the wrong "
  "team. The failure modes are invented detail, wrong team and missed event.")
caption("Table 3. Held-out commentary records. The examples show what aggregate overlap hides.")
add_table(["Event", "Source", "Line", "Difference"], [
    ["pass / West Ham", "Human", "Wonderful movement by West Ham. The players are holding possession, mostly exchanging one-touch passes.", "Reference"],
    ["pass / West Ham", "GPT-2", "West Ham are showcasing great team work by exchanging some short passes… his effort flies just wide of the right post.", "Invented shot; team correct"],
    ["corner / E. Frankfurt", "GPT-2", "Napoli failed to take advantage of the corner as their opponents were alert…", "Wrong team; topic correct"],
    ["goal / B. Monchengladbach", "GPT-2", "The forward races towards goal but the defender gets back well to make a challenge… A promising stop for Inter!", "Missed goal; wrong team"],
], [1.4, 0.8, 3.2, 1.2], font_size=8.2)
H2("8.3 Event Spotting")
P(f"Figure 5 compares the ten reference moments with the 38 event lines from the full run. Goals are "
  f"the strong case: both are found within 13–22 seconds, with precision and recall "
  f"{EV['goal_spotting']['precision']:.2f}. Shots are weaker: two of six are matched, with a timing "
  f"error of 51–59 seconds. Both tackles are missed. Overall recall is "
  f"{EV['recall_overall']:.2f} at ±1 minute. The pipeline emits more pass and dead-time lines than "
  "event lines, which explains part of the gap.")
caption("Table 4. Reference moments versus detected events at ±1 minute. A dash means no same-kind "
        "detection in the window.")
add_table(["Minute", "Reference", "Detected", "Result"], [
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
], [0.8, 1.1, 1.2, 1.7], font_size=8.6)
figure(FIG / "fig4_timeline.png", 5,
       "Reference moments and detected event lines. Faint ticks are passes and dead time; strong "
       "ticks are goals, shots, corners and tackles.")
figure(FIG / "fig5_recall.png", 6,
       "Event recall against the ten curated moments. Goal precision and recall are both 1.00; timing "
       "MAE for matched events is 36.2 s.")
figure(DOCS / "goal_clip_contact.jpg", 7,
       "The 81′ goal clip. The broadcast score flips from 2–1 to 2–2; the score-bug reader infers the "
       "starting score and anchors the away goal.", width=6.2)
H2("8.4 Possession and Ball Tracking")
P("Figure 8 shows the tracking overlay across six moments. Boxes are tracked players, the green dot "
  "and trail are the ball path, and the bottom bar is the live possession split. Figure 9 isolates "
  "the 80′–81′ sequence, where the possession bar and ball trail are easiest to read.")
figure(DOCS / "tracking_contact.jpg", 8,
       "Tracking overlay across six moments. Player boxes are ByteTrack identities; the green "
       "dot/trail is the ball path; the bottom bar is the possession split.", width=6.2)
figure(DOCS / "possession_contact.jpg", 9,
       "Possession and ball path around the 80′–81′ sequence. The bottom bar reports the team "
       "possession split; the green trail shows the smoothed ball path.", width=6.2)
P("The overlay is not a tracking benchmark. Broadcast cuts, close-ups and crowd shots break the "
  "ball path, and similar kits can confuse the jersey clusters. Possession is an estimate, not a "
  "tracking ground truth. The figures are qualitative; MOTA, IDF1 and possession accuracy would "
  "require frame-level ground truth and are left as future work.")
H2("8.5 API-Football Example")
P("The real API-Football fixture 1551707 returned two yellow cards and a 90′ goal by A. Hernandez. "
  "The goal was frozen into facts — team, player, minute, score 1–0 — and the same GPT-2/validator/TTS "
  "stack produced a correct line. The two cards are skipped because the current commentary label set "
  "has no card class. This is a different match from ARG–FRA, so it demonstrates the API path rather "
  "than a same-match comparison.")
caption("Table 5. API-Football example. The API row is a different match from the video run.")
add_table(["Mode", "Input", "Goal commentary", "Outcome"], [
    ["Video, full context", "ARG–FRA, 64.5 min", "France make it 2–2. The ground is up.", "2/2 goals; correct side"],
    ["API feed", "Alajuelense 1–0, 90′", "Goal! A. Hernandez (LD Alajuelense) … The score is 1:0.", "Correct team, player and score"],
], [1.2, 1.4, 3.2, 1.4], font_size=8.4)
H2("8.6 Object Detection")
P("YOLO26n is fractionally ahead of the paused YOLOv5lu run in aggregate mAP, but the budgets are not "
  "equal: 50 vs 33 epochs and 2.39 M vs 53 M parameters. The comparison is descriptive, not a "
  "controlled model comparison. Per-class YOLO26n mAP50–95 is 0.621 for Player, 0.291 for Ball and "
  "0.530 for Referee; the ball remains the weak class.")
caption("Table 6. Object detection on the V1 validation split. This is not event-spotting accuracy.")
add_table(["Model", "Params", "Epochs", "mAP50", "mAP50–95", "Precision", "Recall"], [
    ["YOLO26n", "2.39 M", "50", "0.805", "0.481", "0.864", "0.763"],
    ["YOLOv5lu", "53 M", "33", "0.795", "0.459", "0.854", "0.752"],
], [1.0, 0.8, 0.7, 0.7, 0.9, 0.9, 0.8], font_size=8.6)
H2("8.7 Cost and Commercial Potential")
P("A full match costs about one cent of local electricity or $0.392 on a cloud A10G. Training both "
  "models for 3.64 hours costs $0.109 locally or $4.40 at the cloud rate. Against an illustrative "
  "two-person team at $50/h, cloud inference is 765× cheaper than a $300 live team and 1,020× "
  "cheaper than a $400 pre-recorded team. Across a $25–$100/h human range, the cloud advantage is "
  "roughly 380–1,530×. Local electricity is 30,000–40,000× cheaper, but that excludes the GPU "
  "hardware, so the cloud comparison is the honest one.")
caption("Table 7. Cost per 64.5-minute match. Human rates are illustrative; compute is measured.")
add_table(["Item", "Local GPU", "Cloud g5.2xlarge", "API-Football", "Human team"], [
    ["Inference", "$0.010", "$0.392", "—", "$300 live / $400 pre-recorded"],
    ["Training (both models)", "$0.109", "$4.40", "—", "—"],
    ["Training hours", "3.64 h", "3.64 h", "—", "—"],
    ["TTS calls / speech", "43 / 412.2 s", "43 / 412.2 s", "—", "—"],
    ["API feed", "—", "—", "$0 free tier; paid plan excluded", "—"],
], [1.1, 0.9, 0.95, 1.25, 1.45], font_size=8.2)
caption("Table 8. Cost advantage per match. Compute-only; excludes paid TTS, API, storage, rights "
        "and human review.")
add_table(["Baseline", "Cost / match", "Matchcall cloud", "Matchcall local", "Advantage"], [
    ["Human live team, $50/h × 2 × 3 h", "$300", "$0.392", "$0.010", "765× cloud; 30,000× local"],
    ["Human pre-recorded team, $50/h × 2 × 4 h", "$400", "$0.392", "$0.010", "1,020× cloud; 40,000× local"],
    ["Human range, $25–$100/h", "$150–$600", "$0.392", "—", "380–1,530× cloud"],
    ["API-Football free tier", "$0 requests", "—", "—", "Paid plan excluded"],
], [2.2, 0.95, 1.0, 0.95, 1.55], font_size=8.0)
figure(FIG / "fig8_cost.png", 10,
       "Cost per match minute on a log scale. Local and cloud compute are two to three orders of "
       "magnitude below the human scenario; the barrier is quality and latency, not compute.")
H2("8.8 Discussion")
P("The score-bug goal path is the strongest component: a persistent digit flip survives replays, "
  "close-ups and celebration cuts, and it names the scoring side. The language path is a better "
  "surface realiser than the template, but it is not a better fact layer. It invents details, "
  "misnames teams and can miss a goal. The validator raises precision but checks words, not truth. "
  "The general event layer is not good enough for broadcast: it finds one third of reference shots, "
  "no tackles and emits more pass/dead-time lines than event lines.")
P("The system is pre-recorded. The full match is faster than real time in aggregate (0.30×), but the "
  "first line appears after 16.6 minutes. A live system needs streaming perception, incremental "
  "speech and explicit side propagation. Reporting that as a negative result is more useful than "
  "claiming a live system.")
H2("8.9 Evaluation Limitations and Human Study")
P("Automatic overlap metrics cannot establish whether commentary is accurate, natural or suitable "
  "for broadcast. A human study should use 60 generated lines stratified by event type, three blind "
  "raters and five-point scales for factual correctness, naturalness and broadcast suitability, "
  "with Krippendorff's alpha. That study was not run for this report; the protocol is stated so the "
  "absence of a human result is visible rather than hidden behind BLEU/ROUGE/METEOR.")
H2("8.10 Demo Interface")
P("The browser demo exposes the pipeline in the order the project runs: the five-stage pipeline, the "
  "ARG–FRA match board and reel, the API-Football example, the upload/match-plan form, and the "
  "training/ClearML evidence. Figures 11–13 show the live Stadium Board build. The screenshots are "
  "from the running application, not mock-ups.")
figure(DOCS / "demo_pipeline.png", 11,
       "Live pipeline screen. The formation map and five stage cards expose action spotting, tracking, "
       "the fact sheet, commentary and voice.", width=6.2)
figure(DOCS / "demo_match.png", 12,
       "Live ARG–FRA board and reel. The scoreboard, commentary bus, possession bar, player tracking "
       "and rundown are all driven by the same event/fact artifacts.", width=6.2)
figure(DOCS / "demo_api.png", 13,
       "Live API-Football section. The real fixture, returned events, grounded GPT-2 line and "
       "API-to-voice flow are shown next to the video path.", width=6.2)

# ---------------------------------------------------------------- 9 conclusion
H1("9. Conclusion")
P("Matchcall shows that a small, modular, fact-grounded pipeline can turn a full broadcast into "
  "voiced commentary on a consumer GPU at a cost far below a human team. The goal path works because "
  "the score bug is an explicit, high-precision signal; the language path produces human-length lines; "
  "the cost is negligible. The system does not yet work as live commentary, does not find most "
  "non-goal events, and can attribute a goal to the wrong team.")
P("The project met its build objectives: a complete video-to-voice pipeline, a deterministic fact "
  "sheet, a score-bug anchor, a validator and a reproducible demo. It did not meet the stronger "
  "research objective of a multi-match, human-evaluated, live system. The contribution is an "
  "auditable, reproducible instantiation of the full broadcast-to-speech pipeline with measured "
  "failure modes, not a new state of the art.")

# ---------------------------------------------------------------- 10 recommendations
H1("10. Recommendations")
H2("10.1 Practical Applications")
P("The first market is lower-league and highlights automation, where the alternative is no "
  "commentary at all. A per-match SaaS can sell processed commentary to rights-holders, clubs and "
  "broadcasters; an on-premise or white-label version keeps video local for federations and large "
  "clubs; the same facts can drive vertical social clips and multilingual descriptions; and the API "
  "path can power a second-screen text/audio feed where a live event feed exists.")
P("The commercial moat is not the language model. It is the auditable fact layer, the score-bug "
  "anchor and the ability to run the whole chain on one consumer GPU. At an illustrative $10–$50 per "
  "match, cloud compute is 0.8–3.9% of revenue before paid TTS, API, storage, rights and human "
  "review.")
H2("10.2 Future Work")
for t in (
    "Annotate 2–3 full matches independently and report event precision, recall and F1.",
    "Run a blind human evaluation with three raters and report agreement.",
    "Add a modern LLM baseline under the same fact sheet.",
    "Report tracking metrics (MOTA/IDF1) and possession accuracy against manual labels.",
    "Run an equal-budget YOLO26 vs YOLOv5 comparison.",
    "Compare API and video on the same match, then fuse them.",
    "Build a streaming mode with incremental perception and speech.",
    "Wire the uploaded match plan into the grounded prompt and validator.",
):
    bullet(t)

# ---------------------------------------------------------------- 11 references
H1("11. References")
refs = load_refs()
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph(f"[{i}] {ref}", style="Report Reference")
    p.paragraph_format.left_indent = Inches(0.32)

# ---------------------------------------------------------------- appendices
H1("12. Appendices")
H2("Appendix A — Reproducibility")
P("The commands below assume the project virtual environment and the ARG–FRA video. The full-match "
  "job is 7c23a8a91829 and the corrected goal clip is 8cdba651f954. The API example is "
  "scripts/api_feed_current_example.py. ClearML project: Football Commentator "
  "(ef18bd575abc408591532be2ac8f39b8).")
for line in (
    "python scripts/compare_commentary.py --n 40 --device cuda",
    "python scripts/bootstrap_commentary.py",
    "python scripts/evaluate_event_spotting.py",
    "python scripts/compile_experiments.py",
    "python scripts/api_feed_current_example.py",
    "python scripts/train_yolo26.py --epochs 50 --imgsz 640 --batch 16 --clearml",
    "python -m uvicorn app.webapp:app --host 127.0.0.1 --port 8100",
):
    p = doc.add_paragraph(line)
    p.paragraph_format.left_indent = Inches(0.2)
    for r in p.runs:
        r.font.name = "Consolas"
        r.font.size = Pt(9)
caption("Table 9. Reproducibility artifacts and where they live.")
add_table(["Artifact", "Where", "Purpose"], [
    ["Commentary comparison", "output/comparison/commentary_comparison.json", "40-prompt metrics and lines"],
    ["Bootstrap statistics", "output/comparison/commentary_bootstrap.json", "95% CIs and p-values"],
    ["Event spotting", "output/experiments/event_spotting.json", "Ten-moment recall and goal precision"],
    ["Full-match job", "7c23a8a91829", "43-line ARG–FRA run"],
    ["Corrected goal clip", "8cdba651f954", "Goal-anchor verification"],
    ["YOLO26n run", "models/yolo26/yolo26n_v1_640", "Weights, results.csv, curves"],
    ["GPT-2 v4 / v5", "models/gpt2_lora_v4, models/gpt2_lora_v5", "LoRA adapters"],
    ["ClearML tasks", "Football Commentator project", "Scalars, plots, models"],
], [1.6, 3.0, 2.0], font_size=8.4)

H2("Appendix B — Match-Plan Schema")
P("The upload accepts a JSON match plan. The browser parses the file and fills the match facts; the "
  "backend parser for players and positions is the next implementation step. The schema is deliberately "
  "small so a club analyst can produce it from a team sheet.")
for line in (
    '{ "home": "Argentina", "away": "France", "venue": "Lusail Stadium",',
    '  "score_home": 2, "score_away": 2, "minute": 81,',
    '  "players": [{"name": "player", "team": "ARG", "position": "GK"}],',
    '  "positions": {"ARG": [], "FRA": []} }',
):
    p = doc.add_paragraph(line)
    p.paragraph_format.left_indent = Inches(0.2)
    for r in p.runs:
        r.font.name = "Consolas"
        r.font.size = Pt(9)
caption("Table 10. Match-plan fields.")
add_table(["Field", "Type", "Use"], [
    ["home / away", "string", "Team names and commentary attribution"],
    ["venue", "string", "Grounded venue fact"],
    ["score_home / score_away", "integer", "Starting score for the fact sheet"],
    ["minute", "integer", "Broadcast minute at the start"],
    ["players", "array", "Name, team, shirt number, role"],
    ["positions", "object", "Formation/role or pitch coordinates"],
], [1.8, 0.9, 3.8], font_size=8.6)

H2("Appendix C — Cost Assumptions")
P("Cloud: $1.21/h for an ml.g5.2xlarge. Local electricity: $0.30/kWh at 100 W. Human team: two people "
  "at $50/h for three live hours or four pre-recorded hours; this is illustrative, not a quote. "
  "API-Football: free tier for the requests made; paid plans excluded. Fish Audio: free tier; paid "
  "TTS, storage and egress excluded. The local electricity figure excludes GPU hardware.")
caption("Table 11. Cost calculations used in Section 8.7.")
add_table(["Item", "Calculation", "Result"], [
    ["Cloud inference", "$1.21/h × 19.4/60 h", "$0.392"],
    ["Local electricity", "0.1 kW × 19.4/60 h × $0.30/kWh", "$0.010"],
    ["Cloud training", "$1.21/h × 3.64 h", "$4.40"],
    ["Local training", "0.1 kW × 3.64 h × $0.30/kWh", "$0.109"],
    ["Human live team", "2 people × $50/h × 3 h", "$300"],
    ["Human pre-recorded", "2 people × $50/h × 4 h", "$400"],
], [1.6, 3.4, 1.1], font_size=8.6)

H2("Appendix D — Demo Screenshots")
P("The upload and training screens complete the demo walkthrough. The upload accepts a video and a "
  "match plan; the training screen reads the YOLO26n run and the GPT-2 adapters from ClearML-backed "
  "artifacts. These are live application screenshots, not mock-ups.")
figure(DOCS / "demo_upload.png", 14,
       "Live upload screen. The video drop and match-plan drop are separate; JSON plans are parsed for "
       "teams, venue, score, minute and player/position counts.", width=6.2)
figure(DOCS / "demo_training.png", 15,
       "Live training screen. YOLO26n metrics and curves, GPT-2 v4/v5 loss, ClearML links and the "
       "model artifacts are shown in the same board.", width=6.2)

H2("Appendix E — Repository Map")
caption("Table 12. Main repository areas.")
add_table(["Path", "Contents"], [
    ["app/pipeline.py", "End-to-end orchestration"],
    ["app/vision/", "Score bug, sampling, labels, detector, tracking"],
    ["app/commentary_gpt2.py", "GPT-2 prompt, generation, validator"],
    ["app/tts.py", "Fish Audio primary, edge-tts fallback"],
    ["app/webapp.py", "Upload job queue and web API"],
    ["scripts/", "Training, evaluation, report builders"],
    ["docs/", "Report, figures, runbook, walkthrough"],
    ["output/", "Generated clips, transcripts, metrics (gitignored)"],
    ["models/", "Checkpoints and adapters (gitignored)"],
], [2.2, 4.3], font_size=8.6)

H2("Appendix F — Evaluation Checklist")
caption("Table 13. What is complete and what still needs external data or people.")
add_table(["Item", "Status", "Note"], [
    ["Commentary metrics, 40 prompts", "Complete", "BLEU/ROUGE/METEOR + bootstrap CIs"],
    ["Event spotting, 10 moments", "Complete", "Recall 0.40; goals 2/2"],
    ["Reproducibility / ClearML", "Complete", "Scripts, artifacts, task IDs"],
    ["Independent event annotation", "Pending", "Needs human labels on 2–3 matches"],
    ["Human commentary evaluation", "Pending", "Protocol in Section 8.9"],
    ["Tracking MOTA/IDF1", "Pending", "Needs frame-level ground truth"],
    ["Equal-budget YOLO comparison", "Pending", "Needs matched epochs/GPU time"],
    ["Same-match API vs video", "Pending", "Needs API access and the match video"],
], [2.0, 0.9, 3.6], font_size=8.6)

doc.core_properties.title = ("Matchcall: A Fact-Grounded, Low-Cost Pipeline for Spoken Football "
                             "Commentary")
doc.core_properties.author = "Aryaman Sharma"
doc.core_properties.subject = "AI/Data Analytics Capstone Project"
doc.core_properties.comments = "Built from the UTS project report template; supervisor: Prof. Nabin Sharma."

doc.save(str(OUT))
print("wrote", OUT, OUT.stat().st_size)
print("references", len(refs))
