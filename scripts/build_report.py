"""Build the standalone training + pipeline report.

Reads the real YOLO/LLM/demo state (``app.training_status``) and the ClearML
snapshot, then writes a single self-contained HTML file to
``output/Aryaman_Football_AI_Report.html``. No network, no GPU.

    .venv\\Scripts\\python scripts\\build_report.py
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import training_status as ts
from app.config import settings

OUT = settings.out_root / "Aryaman_Football_AI_Report.html"

C = {
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500",
    "sky": "#4FE0D0", "ink": "#EAF7EE", "ink2": "#B9CDBF",
    "muted": "#8FA99B", "grid": "#1C3A2C", "axis": "#2A4C3A",
}
e = html.escape


def fmt(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


def yolo26_summary() -> dict | None:
    """Read the completed local YOLO26n run, if it exists."""
    run = settings.models_dir / "yolo26" / "yolo26n_v1_640"
    csv_path = run / "results.csv"
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows:
        return None

    def fl(row, key):
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return None

    best = max(rows, key=lambda r: fl(r, "metrics/mAP50-95(B)") or -1)
    metrics_path = run / "val_metrics.json"
    metrics = (json.loads(metrics_path.read_text(encoding="utf-8"))
               if metrics_path.exists() else {})
    return {
        "run": run,
        "epochs": len(rows),
        "hours": (fl(rows[-1], "time") or 0) / 3600,
        "precision": fl(best, "metrics/precision(B)"),
        "recall": fl(best, "metrics/recall(B)"),
        "mAP50": fl(best, "metrics/mAP50(B)"),
        "mAP50_95": fl(best, "metrics/mAP50-95(B)"),
        "metrics": metrics,
    }


def line_svg(series, *, width=760, height=230, y_max=None, y_fmt=".2f",
             x_label="epoch", title="") -> str:
    series = [s for s in series if s["points"]]
    if not series:
        return f'<div class="empty">Waiting for data — {e(title)}</div>'
    pad = {"l": 46, "r": 62, "t": 12, "b": 30}
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    x0, x1 = min(xs), max(xs)
    if x0 == x1:
        x0 -= .5
        x1 += .5
    top = y_max if y_max is not None else max(ys) * 1.1
    X = lambda v: pad["l"] + (v - x0) / (x1 - x0) * (width - pad["l"] - pad["r"])
    Y = lambda v: height - pad["b"] - (v / top) * (height - pad["t"] - pad["b"])
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{e(title)}" class="chart-svg">']
    for i in range(5):
        v = top * i / 4
        parts.append(f'<line class="grid" x1="{pad["l"]}" x2="{width-pad["r"]}" '
                     f'y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{pad["l"]-7}" y="{Y(v)+3:.1f}" '
                     f'text-anchor="end">{format(v, y_fmt)}</text>')
    for i in range(6):
        v = x0 + (x1 - x0) * i / 5
        parts.append(f'<text class="tick" x="{X(v):.1f}" y="{height-8}" '
                     f'text-anchor="middle">{v:.0f}</text>')
    parts.append(f'<line class="axis" x1="{pad["l"]}" x2="{width-pad["r"]}" '
                 f'y1="{height-pad["b"]}" y2="{height-pad["b"]}"/>')
    ends = []
    for s in series:
        d = " ".join(("M" if i == 0 else "L") + f"{X(p[0]):.1f} {Y(p[1]):.1f}"
                     for i, p in enumerate(s["points"]))
        parts.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" '
                     f'stroke-width="2" stroke-linejoin="round" '
                     f'stroke-linecap="round"/>')
        ends.append([s, s["points"][-1], Y(s["points"][-1][1])])
    ends.sort(key=lambda z: z[2])
    last = -1e9
    for s, p, yy in ends:
        yy = max(yy, last + 12)
        last = yy
        parts.append(f'<text class="endlabel" x="{width-pad["r"]+6}" '
                     f'y="{yy+3:.1f}">{format(p[1], y_fmt)}</text>')
    parts.append(f'<text class="tick" x="{width-pad["r"]}" y="{height-2}" '
                 f'text-anchor="end">{e(x_label)}</text>')
    parts.append("</svg>")
    legend = "".join(
        f'<span><i style="background:{s["color"]}"></i>{e(s["name"])}</span>'
        for s in series)
    return (f'<figure class="chart"><figcaption><span class="ctitle">{e(title)}'
            f'</span><span class="legend">{legend}</span></figcaption>'
            + "".join(parts) + "</figure>")


def img_b64(path: Path, alt: str, cap: str) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure class="asset"><img src="data:image/png;base64,{data}" '
            f'alt="{e(alt)}"><figcaption>{e(cap)}</figcaption></figure>')


def card(v, k, cls="") -> str:
    return f'<div class="metric"><div class="v {cls}">{v}</div><div class="k">{e(k)}</div></div>'


def main() -> None:
    y = ts.yolo_status()
    l = ts.llm_status()
    d = ts.demo_status()
    t = ts.tracking_status()
    snap = ts._json(ts.SNAPSHOT) or {"tasks": [], "models": [],
                                     "project_url": ts.PROJECT_URL}

    b = y.get("best_metrics") or y.get("last_metrics") or {}
    pz = y.get("paused") or {}
    y_paused_note = ""
    if pz and y.get("status") == "paused":
        y_paused_note = (
            '<div class="note"><b>Training paused by request.</b> '
            f'Fine-tuning stopped at epoch {pz.get("epoch_completed", "—")}/'
            f'{y.get("epochs_total", "—")}; best checkpoint epoch '
            f'{pz.get("best_epoch", "—")} with mAP50 '
            f'{pz.get("best_mAP50", "—")} and mAP50-95 '
            f'{pz.get("best_mAP50_95", "—")}. The weights are preserved and '
            'resumable with <span class="mono">--resume</span>.</div>')
    rows = y.get("rows") or []
    hist = l.get("loss_history") or []
    ev = l.get("eval") or {}
    v4 = "SoccerReplay" in (l.get("dataset") or "")
    moments = d.get("moments") or []
    goals = [m for m in moments if m.get("kind") == "goal"]
    now = dt.datetime.now().strftime("%d %b %Y, %H:%M")

    y26 = yolo26_summary()
    y26_section = ""
    if y26:
        pc50 = y26["metrics"].get("per_class_AP50", {})
        pc = y26["metrics"].get("per_class_mAP50_95", {})
        pp = y26["metrics"].get("per_class_precision", {})
        pr = y26["metrics"].get("per_class_recall", {})
        pc_rows = "".join(
            f"<tr><td>{e(name)}</td><td>{fmt(pc50.get(name))}</td>"
            f"<td>{fmt(pc.get(name))}</td><td>{fmt(pp.get(name))}</td>"
            f"<td>{fmt(pr.get(name))}</td></tr>"
            for name in ("Player", "Ball", "Referee"))
        y26_section = f"""
<section>
<h2>YOLO26n — local speed run.</h2>
<p class="sub">2.39 M parameters · 640 px · {y26['epochs']} epochs ·
{y26['hours']:.2f} h on a consumer GPU · ClearML task
<a href="https://app.clear.ml/projects/ef18bd575abc408591532be2ac8f39b8/tasks/63128dc3e2b64a69bd31cee4982c707a/output/log">63128dc3</a>.
This run was sized to finish inside a four-hour window, so it is a nano-model
result rather than the planned full-size comparison.</p>
<div class="grid g4">
{card(fmt(y26['mAP50']), 'mAP50', 'lime')}
{card(fmt(y26['mAP50_95']), 'mAP50–95', 'sky')}
{card(fmt(y26['precision']), 'precision')}
{card(fmt(y26['recall']), 'recall')}
</div>
<table style="margin-top:16px"><thead><tr><th>Class</th><th>AP50</th>
<th>AP50–95</th><th>Precision</th><th>Recall</th></tr></thead>
<tbody>{pc_rows}</tbody></table>
<div class="note"><b>How to read this.</b> YOLO26n has 2.39 M parameters and
trained for {y26['epochs']} epochs; the paused YOLOv5lu run has 53 M parameters
and 33 epochs. YOLO26n edges the aggregate mAP50 ({fmt(y26['mAP50'])} vs
{fmt(b.get('mAP50'))}) and mAP50–95 ({fmt(y26['mAP50_95'])} vs
{fmt(b.get('mAP50_95'))}), but its Ball AP50–95 is only
{fmt(pc.get('Ball'))}, and on the 10 s goal overlay it keeps the ball in
22/40 frames versus 38/40 for Aryaman YOLOv5lu. This is not evidence that
YOLO26 is superior overall; the controlled 640 px / 100-epoch SageMaker run
is still pending.</div>
<div class="links">
<a href="commented_highlights/yolo26n_comparison/annotated_goal_possession.mp4">YOLO26n tracking overlay</a>
<a href="../models/yolo26/yolo26n_v1_640/results.csv">results.csv (local)</a>
<a href="https://app.clear.ml/projects/ef18bd575abc408591532be2ac8f39b8/tasks/63128dc3e2b64a69bd31cee4982c707a/output/log">ClearML task</a>
</div>
</section>
"""

    loss_svg = line_svg([
        {"name": "box", "color": C["s1"],
         "points": [(r["epoch"], r["box_loss"]) for r in rows if r["box_loss"] is not None]},
        {"name": "cls", "color": C["s2"],
         "points": [(r["epoch"], r["cls_loss"]) for r in rows if r["cls_loss"] is not None]},
        {"name": "dfl", "color": C["s3"],
         "points": [(r["epoch"], r["dfl_loss"]) for r in rows if r["dfl_loss"] is not None]},
    ], title="YOLOv5lu — training losses")
    map_svg = line_svg([
        {"name": "precision", "color": C["s1"],
         "points": [(r["epoch"], r["precision"]) for r in rows if r["precision"] is not None]},
        {"name": "recall", "color": C["s2"],
         "points": [(r["epoch"], r["recall"]) for r in rows if r["recall"] is not None]},
        {"name": "mAP50", "color": C["s3"],
         "points": [(r["epoch"], r["mAP50"]) for r in rows if r["mAP50"] is not None]},
        {"name": "mAP50–95", "color": C["s4"],
         "points": [(r["epoch"], r["mAP50_95"]) for r in rows if r["mAP50_95"] is not None]},
    ], title="YOLOv5lu — validation metrics", y_max=1, y_fmt=".2f")
    llm_title = ("GPT-2 LoRA v4" if "SoccerReplay" in (l.get("dataset") or "")
                 else "GPT-2 LoRA v3")
    llm_svg = line_svg([
        {"name": "loss", "color": C["sky"],
         "points": [(h["step"], h["loss"]) for h in hist if h.get("loss") is not None]},
    ], title=f"{llm_title} — training loss", x_label="step", height=210)

    moment_rows = "".join(
        f"<tr><td>{m.get('minute')}′</td><td>{e(str(m.get('kind','')))}</td>"
        f"<td>{e(str(m.get('team','')))}</td>"
        f"<td>{m['score'][0]}–{m['score'][1]}</td>"
        f"<td>{e(m.get('commentary',''))}</td></tr>"
        for m in moments)

    tasks = snap.get("tasks") or []
    task_rows = "".join(
        f'<tr><td>{e(str(t_.get("name","")))}</td><td>{e(str(t_.get("type","")))}</td>'
        f'<td>{e(str(t_.get("status","")))}</td>'
        f'<td><a href="{e(str(t_.get("url","")))}">{e(str(t_.get("url","")))}</a></td></tr>'
        for t_ in tasks) or '<tr><td colspan="4">Snapshot not generated yet.</td></tr>'

    assets = "".join([
        img_b64(ts.YOLO_RUN / "results.png", "YOLO results",
                "Ultralytics results.png (all curves)"),
        img_b64(ts.YOLO_RUN / "confusion_matrix.png", "confusion matrix",
                "Confusion matrix"),
        img_b64(ts.YOLO_RUN / "labels.jpg", "dataset labels",
                "V1 label distribution"),
        img_b64(ts.YOLO_RUN / "train_batch0.jpg", "training batch",
                "A training batch (mosaic)"),
    ])

    status_line = (f"{y.get('status')} · epoch {y.get('epochs_done', 0)}/"
                   f"{y.get('epochs_total', 100)} · best mAP50-95 "
                   f"{fmt(b.get('mAP50_95'))}")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aryaman Football AI — training &amp; pipeline report</title>
<style>
:root{{color-scheme:dark;--bg:#05110C;--panel:#0A2118;--panel2:#0E2C20;
--line:#1C3A2C;--line2:#2A4C3A;--ink:#EAF7EE;--ink2:#B9CDBF;--muted:#8FA99B;
--lime:#C9FF4F;--coral:#FF6B3D;--sky:#4FE0D0;
--fd:"Barlow Condensed","Arial Narrow",sans-serif;
--fb:"Barlow","Segoe UI",system-ui,sans-serif;--fm:"IBM Plex Mono",monospace}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 var(--fb);-webkit-font-smoothing:antialiased}}
.wrap{{width:min(1120px,calc(100% - 48px));margin:auto;padding:40px 0 64px}}
h1,h2,h3,p,figure,table{{margin:0}}h1{{font:800 clamp(2.8rem,6vw,4.8rem)/.94 var(--fd)}}
h2{{font:700 clamp(1.7rem,3vw,2.4rem)/1.1 var(--fd);margin-bottom:6px}}
h3{{font:600 1.05rem/1.3 var(--fb)}}
a{{color:var(--lime)}}.muted{{color:var(--muted)}}.mono{{font-family:var(--fm)}}
header{{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:30px}}
.kicker{{color:var(--muted);font-size:13px;margin-bottom:8px}}
.lede{{color:var(--ink2);max-width:62ch;margin-top:12px}}
.meta-line{{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:14px}}
section{{border-top:1px solid var(--line);padding:34px 0}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:18px}}
.grid{{display:grid;gap:12px}}.g4{{grid-template-columns:repeat(4,minmax(0,1fr))}}
.g2{{grid-template-columns:repeat(2,minmax(0,1fr))}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px}}
.metric .v{{font:700 1.6rem/1.1 var(--fd);font-variant-numeric:tabular-nums}}
.metric .k{{color:var(--muted);font-size:11.5px;margin-top:3px}}
.metric .v.lime{{color:var(--lime)}}.metric .v.coral{{color:var(--coral)}}
.metric .v.sky{{color:var(--sky)}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}
.chart{{background:#07160F;border:1px solid var(--line);border-radius:10px;padding:12px;margin-top:14px}}
.chart figcaption{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:6px}}
.ctitle{{font-size:12px;color:var(--ink2);font-weight:600}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:10.5px;color:var(--muted)}}
.legend i{{display:inline-block;width:14px;height:2px;margin-right:5px;vertical-align:middle}}
.chart-svg{{width:100%;height:auto;overflow:visible}}
.grid{{stroke:var(--line);stroke-width:1}}.axis{{stroke:var(--line2)}}
.tick{{fill:var(--muted);font-family:var(--fb);font-size:9.5px}}
.endlabel{{fill:var(--ink2);font-family:var(--fm);font-size:9px}}
.empty{{color:var(--muted);text-align:center;padding:34px;font-size:12px}}
.asset-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:16px}}
.asset{{background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.asset img{{width:100%;height:auto;display:block}}
.asset figcaption{{padding:6px 9px;font-size:10.5px;color:var(--muted);background:#07160F}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:14px}}
th,td{{text-align:left;padding:8px 9px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--muted);font-weight:500;font-size:11px}}
pre{{background:#04100A;border:1px solid var(--line);border-radius:8px;padding:12px;
font:11px/1.5 var(--fm);color:#7fc7a4;max-height:280px;overflow:auto;white-space:pre-wrap}}
.note{{border-left:2px solid var(--coral);background:rgba(255,107,61,.06);
padding:10px 14px;border-radius:0 8px 8px 0;color:var(--ink2);font-size:12.5px;margin-top:12px}}
.note b{{color:var(--coral)}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
video{{width:100%;border-radius:8px;background:#000}}
.links{{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:12.5px}}
footer{{border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:11.5px}}
@media(max-width:780px){{.g4,.two,.asset-grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:520px){{.g4,.two,.asset-grid,.g2{{grid-template-columns:1fr}}}}
@media print{{body{{background:#fff;color:#111}}header,section{{border-color:#ddd}}
.panel,.metric,.chart,.asset,pre{{background:#fff;border-color:#ddd;color:#111}}
.tick,.endlabel,.muted,.sub,.asset figcaption{{color:#555}}a{{color:#0645ad}}
.chart-svg .line{{filter:none}}}}
</style></head><body><div class="wrap">

<header>
<div class="kicker">Aryaman Football AI · training &amp; pipeline report</div>
<h1>Every goal, called the moment it lands.</h1>
<p class="lede">Two models, one pipeline: an Aryaman YOLOv5lu detector fine-tuned on
the V1 football set, a GPT-2 LoRA commentator, and Fish Audio s2.1-pro-free in
front of a verified ARG–FRA highlights reel. Everything logged in ClearML.</p>
<div class="meta-line"><span>Generated {now}</span>
<span>ClearML: <a href="{e(snap.get('project_url') or ts.PROJECT_URL)}">Football Commentator</a></span>
<span>YOLO task: <a href="{e(y.get('task_url') or ts.TASK_URL)}">f020de2d</a></span>
<span>Status: {e(status_line)}</span></div>
</header>

<section>
<h2>At a glance.</h2>
<p class="sub">The shipped evidence, not the plan.</p>
<div class="grid g4">
{card(d.get('clips','—'), 'moments called', 'lime')}
{card(len(goals), 'goals anchored', 'coral')}
{card(f"{round(d.get('reel_seconds') or 0)} s", 'of voiced commentary')}
{card(fmt(b.get('mAP50')), 'YOLO mAP50', 'sky')}
</div>
<div class="grid g4" style="margin-top:12px">
{card(fmt(b.get('mAP50_95')), 'YOLO mAP50–95', 'sky')}
{card(f"{y.get('epochs_done',0)}/{y.get('epochs_total',100)}", 'epochs complete')}
{card(fmt(l.get('final_loss')), 'GPT-2 final loss')}
{card(f"{ev.get('gpt2_accepted', '—')}/{ev.get('total', '—')}", 'GPT-2 lines accepted')}
</div>
</section>

<section>
<h2>The pipeline.</h2>
<p class="sub">Action spotting → tracking → grounded description → commentary → voice.</p>
<div class="two">
<div class="panel"><h3>01 · Action spotting</h3>
<p class="sub" style="margin:6px 0 0">Broadcast score-bug reading plus a
motion-gated YOLO/ResNet window scan. The ARG–FRA goals were verified at
<b>2687 s (80′)</b> and <b>2784 s (81′)</b>.</p></div>
<div class="panel"><h3>02 · Tracking</h3>
<p class="sub" style="margin:6px 0 0">Aryaman YOLOv5lu + ByteTrack for players,
referee and ball, KMeans jersey teams, and a bounded ball interpolation.
Overlay: <a href="commented_highlights/annotated_goal_possession.mp4">annotated_goal_possession.mp4</a>.</p></div>
<div class="panel"><h3>03 · Grounded description</h3>
<p class="sub" style="margin:6px 0 0">Team, minute, score, possession, ball zone
and tracked-player count. No invented facts are allowed downstream.</p></div>
<div class="panel"><h3>04 · Commentary</h3>
<p class="sub" style="margin:6px 0 0">{llm_title} with a grounding validator;
rejected lines fall back to templates, so the reel always has a line.</p></div>
<div class="panel"><h3>05 · Voice</h3>
<p class="sub" style="margin:6px 0 0">Fish Audio s2.1-pro-free, one voice, every
line placed on its timestamp and mixed under the crowd.</p></div>
<div class="panel"><h3>The reel</h3>
<p class="sub" style="margin:6px 0 0">{d.get('clips','—')} moments × 10 s,
{d.get('fish_model','Fish')}. <a href="commented_highlights/{Path(str(d.get('video_url') or 'commented_highlights.mp4')).name}">Watch the reel</a>.</p></div>
</div>

<h3 style="margin-top:26px">Moment log</h3>
<table><thead><tr><th>Minute</th><th>Event</th><th>Team</th><th>Score</th>
<th>Commentary</th></tr></thead><tbody>{moment_rows}</tbody></table>

<h3 style="margin-top:26px">Ball tracking — what the repo actually does</h3>
<div class="note"><b>Honest note.</b> FootballVision's README says ByteTrack
tracks the ball. In the code, ball IDs are discarded: the ball is kept from raw
per-frame detections with a fixed ID and then linearly interpolated offline. We
adopt that interpolation, but bound it to 16-frame gaps and reject single-frame
teleports, and we mark predicted frames with a ring. The real accuracy gain is
the <b>Aryaman YOLOv5lu V1 fine-tune</b>, whose Ball class replaces the shipped
small checkpoint.</div>
<p class="sub" style="margin-top:10px">Overlay model:
<b>{e(t.get('model','—'))}</b> · interpolated frames:
<b>{t.get('meta',{}).get('interpolated_frames','—')}</b> /
{t.get('meta',{}).get('frames','—')} · method:
{e(t.get('meta',{}).get('ball_tracking', t.get('ball_tracking','—')))}.</p>
</section>

<section>
<h2>Aryaman YOLOv5lu.</h2>
<p class="sub">The FootballVision architecture — YOLOv5lu, not YOLO26 — fine-tuned
on the full V1 export. ClearML task
<a href="{e(y.get('task_url') or ts.TASK_URL)}">f020de2d</a> carries the dataset,
parameters, metrics, plots and the registered model.</p>
<div class="grid g4">
{card(y.get('model','yolov5lu.pt'), 'base checkpoint')}
{card(f"{y.get('dataset',{}).get('train','—')} / {y.get('dataset',{}).get('val','—')}", 'train / val images')}
{card(', '.join(y.get('dataset',{}).get('classes',[])) or '—', 'classes')}
{card(f"{y.get('args',{}).get('batch','—')} @ {y.get('args',{}).get('imgsz','—')}", 'batch / imgsz')}
</div>
{y_paused_note}
{loss_svg}
{map_svg}
<div class="asset-grid">{assets}</div>
<h3 style="margin-top:22px">Training log</h3>
<pre>{e((y.get('log_tail') or 'No log yet.').strip())}</pre>
<div class="links"><a href="{e(y.get('task_url') or ts.TASK_URL)}">ClearML task</a>
<a href="{e(y.get('project_url') or ts.PROJECT_URL)}">ClearML project</a>
<a href="../models/aryaman_yolo/aryaman_yolov5lu_football/results.csv">results.csv (local)</a></div>
</section>

{y26_section}
<section>
<h2>{llm_title}.</h2>
<p class="sub">{e(l.get('method') or 'LoRA, completion-only loss')} on
{e(l.get('dataset') or 'a grounded corpus')}, {l.get('epochs','—')} epochs.
The adapter is <span class="mono">{e(str(l.get('adapter',{}).get('path','')))}</span>.</p>
{f'<p class="sub">Data: <a href="https://huggingface.co/datasets/Homie0609/SoccerReplay-1988">SoccerReplay-1988</a> (1,000 matches); player and referee names are removed before training, and every target names the event team.</p>' if v4 else ''}
<div class="grid g4">
{card(fmt(l.get('final_loss')), 'final epoch loss', 'sky')}
{card(l.get('steps','—'), 'optimizer steps')}
{card(l.get('epochs','—'), 'epochs')}
{card(l.get('corpus',{}).get('examples','—'), 'corpus examples')}
</div>
{llm_svg}
<div class="note"><b>Evaluation.</b> Held-out SoccerReplay prompts:
<b>{ev.get('gpt2_accepted','—')}</b> of <b>{ev.get('total','—')}</b> generated lines
passed the grounding validator; accepted lines average
<b>{ev.get('avg_words','—')}</b> words. The ARG–FRA reel used
<b>{d.get('gpt2_lines','—')}</b> GPT-2 lines and
<b>{d.get('template_lines','—')}</b> template fallbacks.
{f"The adapter was trained on {e(l.get('dataset') or '')}." if v4 else "The v3 adapter's free-form lines drifted, so the validator refuses invented teams, scores or players by design."}</div>
<h3 style="margin-top:22px">Loss history</h3>
<pre>{e(json.dumps(l.get('logs') or [], indent=1)[:4000])}</pre>
<div class="links">{f'<a href="{e(l["task_url"])}">ClearML task</a>' if l.get('task_url') else '<span class="muted">ClearML task pending</span>'}
<a href="{e(str(l.get('trainer_state','')))}">trainer_state.json</a></div>
</section>

<section>
<h2>ClearML ledger.</h2>
<p class="sub">Every task and model in the project, with live links.</p>
<table><thead><tr><th>Task</th><th>Type</th><th>Status</th><th>Link</th></tr></thead>
<tbody>{task_rows}</tbody></table>
</section>

<section>
<h2>Next: YOLO26 full vs YOLOv5 large.</h2>
<p class="sub">A fair generation test is prepared in
<code>training/aws/yolo26_sagemaker.ipynb</code>. Both models are the large/full
tier, both train at 640 px for 100 epochs on the same V1 split, and both log
every epoch to ClearML on an <code>ml.g5.2xlarge</code> instance. The notebook
runs YOLO26x (59.0 M parameters) and YOLOv5lu (53 M) sequentially, registers
each checkpoint, and returns <code>best.pt</code> for both. The pipeline can
then be rerun with <code>--tracker-weights &lt;best.pt&gt;</code> to measure the
change in ball detection, possession and the rendered overlay.</p>
<p class="sub">The full-size comparison has <b>not started</b>: it needs an AWS
account with an S3 bucket and ClearML keys. The local YOLO26n speed run above
is complete, but it is not a like-for-like replacement for this 640 px /
100-epoch large/full test, and the YOLOv5lu section above remains the paused
33/100 checkpoint.</p>
</section>

<section>
<h2>Reproduce.</h2>
<pre>.venv\\Scripts\\python scripts\\make_commented_highlights.py --video "...ARG vs FRA..." --home Argentina --away France --start-score-home 2 --start-score-away 0 --minute-offset 35.33
.venv\\Scripts\\python scripts\\train_aryaman_yolov5.py
.venv\\Scripts\\python scripts\\annotate_tracking.py
.venv\\Scripts\\python scripts\\build_report.py</pre>
</section>

<footer>Matchcall — built by Aryaman · generated {now} · ClearML project
“Football Commentator” · FootballVision (mserra0) · Fish Audio s2.1-pro-free.</footer>
</div></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
