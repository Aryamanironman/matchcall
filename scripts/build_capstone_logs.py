"""Generate nine weekly capstone progress logs as separate DOCX files."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "capstone_logs_final"
OUT.mkdir(parents=True, exist_ok=True)

STUDENT = "Aryaman Sharma"
STUDENT_ID = "25158763"
SUPERVISOR = "Prof. Nabin Sharma"
COURSE = "31243 AI/Data Analytics Capstone Project"

LOGS = [
    {
        "week": 1,
        "dates": "31 August – 4 September 2026",
        "summary": ("Project scoping and literature review. Defined the five-stage "
                    "video-to-voice pipeline and audited the available datasets and compute."),
        "activities": [
            "Reviewed SoccerNet, SoccerNet-Caption, MatchTime, TimeSoccer and SoccerComment as the main related work.",
            "Defined the project aim: a modular, fact-grounded commentary pipeline on one consumer GPU.",
            "Audited the Roboflow V1 detector dataset and the SoccerReplay-1988 commentary corpus.",
            "Set up the Python environment, CUDA/PyTorch, ffmpeg and the repository structure.",
            "Wrote the initial risk list: event annotation, broadcast cuts, LLM hallucination and TTS latency.",
        ],
        "progress": [
            ("Problem definition and related work", "Complete"),
            ("Dataset and hardware audit", "Complete"),
            ("Pipeline architecture", "Drafted"),
        ],
        "issues": [
            "No single public dataset covers detection, tracking, events and commentary end to end.",
            "The available GPU has limited memory, so model size and batch size must be constrained.",
        ],
        "next": [
            "Build the video/proxy-frame ingestion path.",
            "Prototype the score-bug reader and motion-gated detector schedule.",
        ],
        "hours": 10,
    },
    {
        "week": 2,
        "dates": "7 – 11 September 2026",
        "summary": ("Built the action-spotting front end and the first reproducible video scan."),
        "activities": [
            "Implemented 4 fps proxy-frame extraction and frame-difference motion scoring.",
            "Implemented the motion-gated detector schedule (dense on active play, sparse on dead time).",
            "Prototyped the score-bug digit reader and the goal-anchor concept.",
            "Created the 2.5-second window representation and the first heuristic labels.",
            "Ran the first baseline YOLO detections and inspected false positives on broadcast cuts.",
        ],
        "progress": [
            ("Video ingestion and proxy frames", "Complete"),
            ("Motion-gated window scan", "Complete"),
            ("Score-bug goal anchor", "Prototype"),
        ],
        "issues": [
            "Camera cuts and crowd shots trigger motion, so the gate opens on non-play frames.",
            "Fast events can fall between sparse samples.",
        ],
        "next": [
            "Train the detector on the V1 football export.",
            "Add ByteTrack and team/possession features.",
        ],
        "hours": 11,
    },
    {
        "week": 3,
        "dates": "14 – 18 September 2026",
        "summary": ("Added tracking, team assignment and possession, and produced the first annotated clip."),
        "activities": [
            "Fine-tuned the YOLOv5lu detector on the V1 dataset and logged the run in ClearML.",
            "Integrated ByteTrack for player and referee identities.",
            "Implemented KMeans jersey clustering for the two teams.",
            "Implemented ball tracking with bounded gap interpolation and nearest-player possession.",
            "Produced the first annotated goal clip and compared it with the raw broadcast.",
        ],
        "progress": [
            ("Detector training", "Complete"),
            ("Tracking and team assignment", "Complete"),
            ("Possession estimate", "Prototype"),
        ],
        "issues": [
            "Ball detections drop on close-ups and replays.",
            "Similar kits and referee colours can confuse the jersey clusters.",
        ],
        "next": [
            "Build the deterministic fact sheet from tracking and score-bug evidence.",
            "Start the GPT-2 commentary adapter.",
        ],
        "hours": 12,
    },
    {
        "week": 4,
        "dates": "21 – 25 September 2026",
        "summary": ("Built the first language and voice stages with a grounded validator."),
        "activities": [
            "Built the SoccerReplay commentary corpus and the completion-only LoRA training script.",
            "Trained the first GPT-2 LoRA adapter and generated initial event lines.",
            "Added a validator for team names, blocked names, digits and goal contradictions.",
            "Integrated Fish Audio TTS with an edge-tts fallback.",
            "Rendered the first voiced clip and checked timestamp alignment.",
        ],
        "progress": [
            ("GPT-2 LoRA training", "Complete"),
            ("Grounding validator", "Prototype"),
            ("Fish Audio integration", "Complete"),
        ],
        "issues": [
            "The model invents shots and sometimes names teams outside the prompt.",
            "Sampling produces different lines on repeated runs.",
        ],
        "next": [
            "Run the full ARG–FRA match.",
            "Measure line quality against human references.",
        ],
        "hours": 12,
    },
    {
        "week": 5,
        "dates": "28 September – 2 October 2026",
        "summary": ("Completed the first full-match run and measured the system honestly."),
        "activities": [
            "Ran the 64.5-minute ARG–FRA broadcast end to end.",
            "Produced 43 fully voiced lines in 19.4 minutes on the consumer GPU.",
            "Measured both goals at 80′ and 81′ with correct sides.",
            "Built the ten-moment event reference and measured recall at ±1 minute.",
            "Compared GPT-2 against a template baseline on 40 held-out prompts.",
        ],
        "progress": [
            ("Full-match run", "Complete"),
            ("Goal detection", "2/2; precision and recall 1.00"),
            ("Event recall", "0.40; weakness documented"),
        ],
        "issues": [
            "General event spotting finds only two of six reference shots and no tackles.",
            "The first line appears after the whole match is scored; the system is pre-recorded.",
        ],
        "next": [
            "Train YOLO26n and compare with YOLOv5lu.",
            "Start the paper draft and cost analysis.",
        ],
        "hours": 12,
    },
    {
        "week": 6,
        "dates": "5 – 9 October 2026",
        "summary": ("Trained YOLO26n, evaluated detection and drafted the paper structure."),
        "activities": [
            "Fine-tuned YOLO26n for 50 epochs at 640 px and logged the run in ClearML.",
            "Recorded mAP50 0.806 and mAP50–95 0.480 on the V1 validation split.",
            "Compared the detector with the paused YOLOv5lu run and documented the unequal budgets.",
            "Built the event timeline, recall figure and goal-clip contact sheet.",
            "Drafted the paper's related work, system and experimental setup sections.",
        ],
        "progress": [
            ("YOLO26n training", "Complete"),
            ("Detection evaluation", "Complete"),
            ("Paper draft", "In progress"),
        ],
        "issues": [
            "YOLO26n and YOLOv5lu did not receive equal epochs or parameter budgets.",
            "The ball remains the weakest detection class.",
        ],
        "next": [
            "Add the real API-Football event feed.",
            "Improve goal generation for out-of-distribution teams and players.",
        ],
        "hours": 14,
    },
    {
        "week": 7,
        "dates": "12 – 16 October 2026",
        "summary": ("Integrated API-Football and trained the v5 API-slot commentary adapter."),
        "activities": [
            "Connected the API-Football fixture 1551707 (LD Alajuelense 1–0 Perez Zeledon).",
            "Built the API-slot training corpus: 38,815 train / 1,723 validation examples.",
            "Trained GPT-2 v5 for 2 epochs and 4,852 steps; final training loss 0.903.",
            "Added slot-token prefill, beam search and contradiction/off-score suppression.",
            "Verified the API goal line and both ARG–FRA goal lines as GPT-2 outputs.",
        ],
        "progress": [
            ("API-Football integration", "Complete"),
            ("GPT-2 v5 adapter", "Complete"),
            ("Goal validator hardening", "Complete"),
        ],
        "issues": [
            "The 124M model still produces awkward clauses even when the facts are correct.",
            "API and video examples are from different matches.",
        ],
        "next": [
            "Build the Stadium Board UI.",
            "Add the match-plan upload and final report.",
        ],
        "hours": 14,
    },
    {
        "week": 8,
        "dates": "19 – 23 October 2026",
        "summary": ("Redesigned the demo UI, added the match-plan upload and strengthened the report."),
        "activities": [
            "Designed and implemented the Stadium Board UI: pipeline, ARG–FRA board, API, upload, training.",
            "Added the ARG–FRA reel, possession/ball-tracking visuals and API-Football section.",
            "Added a match-plan document drop and client-side JSON parsing for teams, venue, score, minute and positions.",
            "Added bootstrap confidence intervals for the 40-prompt commentary comparison.",
            "Moved the report into the UTS project-report template and added the commercialisation section.",
        ],
        "progress": [
            ("Stadium Board UI", "Complete"),
            ("Match-plan upload UI", "Complete"),
            ("Statistical analysis", "Complete"),
        ],
        "issues": [
            "The backend still needs to consume players/positions from the plan.",
            "The demo transcript/audio is the existing reel, not a freshly v5-voiced reel.",
        ],
        "next": [
            "Finalise the 22-page report and appendices.",
            "Prepare the oral presentation and submission package.",
        ],
        "hours": 15,
    },
    {
        "week": 9,
        "dates": "26 – 30 October 2026",
        "summary": ("Final integration, report completion, demo checks and submission preparation."),
        "activities": [
            "Completed the UTS-template report: 22 pages, 15 figures, 13 tables and 21 references.",
            "Added live demo screenshots and filled appendices A–F.",
            "Verified the web UI on port 8100: pipeline → match board → API → upload → training.",
            "Tested the match-plan JSON parser and checked the upload path returns a job ID.",
            "Wrote the presentation runbook, code walkthrough, and published the repository.",
        ],
        "progress": [
            ("Final report", "Complete"),
            ("Demo verification", "Complete"),
            ("Presentation preparation", "Complete"),
            ("Human evaluation", "Pending — external raters required"),
            ("Independent event annotation", "Pending — external annotation required"),
        ],
        "issues": [
            "The research checklist still needs human raters, independent event labels, tracking ground truth and an equal-budget YOLO run.",
            "Broadcast screenshots require permission before a public release.",
        ],
        "next": [
            "Submit the report and repository link.",
            "Run the human evaluation and independent annotation if the project continues.",
        ],
        "hours": 12,
    },
    {
        "week": 10,
        "dates": "2 – 6 November 2026",
        "summary": ("Final submission and verification week. Completed the consistency pass, "
                    "checked the public repository and prepared the final presentation package."),
        "activities": [
            "Proofread the final report: title page, references, figure/table numbering and appendices.",
            "Verified the ten weekly logs and the public repository links.",
            "Ran a final smoke test of the Stadium Board demo: pipeline, match board, API, upload and training.",
            "Finalised the presentation runbook and code walkthrough.",
            "Archived the report, logs, screenshots and submission links.",
        ],
        "progress": [
            ("Final report", "Complete"),
            ("Ten weekly logs", "Complete"),
            ("Public repository", "Complete"),
            ("Presentation package", "Complete"),
            ("Human evaluation / independent annotation", "Pending — external work"),
        ],
        "issues": [
            "The research checklist still needs human raters, independent event labels, tracking ground truth and an equal-budget YOLO run.",
            "The public repository contains broadcast screenshots; confirm copyright clearance for the submission context.",
        ],
        "next": [
            "Submit the report, logs and repository link.",
            "If the project continues, run the human evaluation and independent annotation.",
        ],
        "hours": 10,
    },
]


def shade(cell, fill):
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc.append(shd)


def page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    for kind, text in (("begin", None), ("instr", "PAGE"), ("separate", None),
                       ("text", "1"), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
            run._r.append(el)
        elif kind == "text":
            t = OxmlElement("w:t")
            t.text = text
            run._r.append(t)
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
            run._r.append(el)


def build(log):
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, attr, Inches(1.0))

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12

    for name, size, before, after in (("Heading 1", 16, 8, 4),
                                      ("Heading 2", 12.5, 10, 3)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(COURSE)
    r.bold = True
    r.font.size = Pt(12)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Weekly Progress Log")
    r.bold = True
    r.font.size = Pt(20)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Week {log['week']}  ·  {log['dates']}")
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(80, 80, 80)

    info = doc.add_table(rows=4, cols=2)
    info.style = "Table Grid"
    for row, (label, value) in enumerate((
            ("Student Name", STUDENT), ("Student ID", STUDENT_ID),
            ("Supervisor", SUPERVISOR), ("Week / Date", f"Week {log['week']} · {log['dates']}"))):
        info.rows[row].cells[0].text = label
        info.rows[row].cells[1].text = value
        info.rows[row].cells[0].paragraphs[0].runs[0].bold = True
        shade(info.rows[row].cells[0], "E7E6E6")
        for c in info.rows[row].cells:
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for row in info.rows:
        row.cells[0].width = Inches(1.55)
        row.cells[1].width = Inches(5.0)

    doc.add_paragraph()
    doc.add_heading("Weekly summary", level=2)
    doc.add_paragraph(log["summary"])

    doc.add_heading("Activities completed", level=2)
    for item in log["activities"]:
        p = doc.add_paragraph("•  " + item)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(3)

    doc.add_heading("Progress against objectives", level=2)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "Objective / work item"
    table.rows[0].cells[1].text = "Status"
    for c in table.rows[0].cells:
        c.paragraphs[0].runs[0].bold = True
        shade(c, "E7E6E6")
    for objective, status in log["progress"]:
        cells = table.add_row().cells
        cells[0].text = objective
        cells[1].text = status
    table.rows[0].cells[0].width = Inches(4.4)
    table.rows[0].cells[1].width = Inches(2.1)

    doc.add_heading("Issues and risks", level=2)
    for item in log["issues"]:
        p = doc.add_paragraph("•  " + item)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(3)

    doc.add_heading("Next week", level=2)
    for item in log["next"]:
        p = doc.add_paragraph("•  " + item)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(3)

    p = doc.add_paragraph()
    r = p.add_run("Estimated effort this week: ")
    r.bold = True
    p.add_run(f"{log['hours']} hours")

    p = doc.add_paragraph()
    r = p.add_run("Project links: ")
    r.bold = True
    p.add_run("ClearML — https://app.clear.ml/projects/ef18bd575abc408591532be2ac8f39b8  ·  "
              "GitHub — https://github.com/Aryamanironman/matchcall")

    page_number(section.footer.paragraphs[0])

    name = f"Week_{log['week']:02d}_Progress_Log.docx"
    doc.save(OUT / name)
    print("wrote", OUT / name)


if __name__ == "__main__":
    for log in LOGS:
        build(log)
