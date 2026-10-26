"""Web demo: upload a match video, pipeline runs in the background, watch progress live.

Endpoints:
  GET  /                 -> static UI
  POST /api/upload       -> multipart(file, home, away, t0?, duration?, orig_audio?)
  GET  /api/job/{id}     -> live status: stage, overall progress, log, transcript
  GET  /api/jobs         -> job history
  POST /api/job/{id}/cancel
  GET  /media/{path}     -> outputs (final videos), rooted at settings.out_root

One background worker thread processes jobs serially (GPU-friendly). Jobs are
in-memory; uploaded files and outputs persist on disk.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings

UPLOAD_DIR = settings.out_root / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Football Commentary Dubber")


@app.middleware("http")
async def _revalidate_shell(request, call_next):
    """Keep the HTML shell out of the browser cache after a reel rebuild."""
    response = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-store"
    return response


class Job:
    def __init__(self, filename: str, home: str, away: str, t0: float,
                 duration: float | None, orig_audio: bool,
                 score_home: int = 0, score_away: int = 0,
                 minute_offset: float = 0.0):
        self.id = uuid.uuid4().hex[:12]
        self.filename = filename
        self.home, self.away = home, away
        self.t0, self.duration = t0, duration
        self.orig_audio = orig_audio
        self.score_home, self.score_away = score_home, score_away
        self.minute_offset = minute_offset
        self.status = "queued"          # queued|running|done|error|cancelled
        self.stage = "waiting"
        self.progress = 0.0             # 0..1 overall
        self.message = ""
        self.log: list[str] = []
        self.transcript: list[str] = []
        self.segments_done: list[dict] = []   # live: voiced segments as they finish
        self.live_seq = 0                     # SSE clients resume from here
        self.result: dict | None = None
        self.error = ""
        self.created = time.time()
        self.cancel_flag = False


_JOBS: dict[str, Job] = {}
_QUEUE: "queue.Queue[Job]" = queue.Queue()


def _as_int(value: str, default: int = 0) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: str, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


class JobCancelled(Exception):
    pass


def _worker():
    while True:
        job = _QUEUE.get()
        if job.cancel_flag:
            job.status = "cancelled"
            continue
        job.status = "running"
        try:
            _run_job(job)
        except JobCancelled:
            job.status = "cancelled"
            job.message = "cancelled by user"
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.log.append(f"ERROR {job.error}")
            traceback.print_exc()


def _run_job(job: Job):
    from app import pipeline

    def progress(stage: str, frac: float, msg: str = ""):
        if job.cancel_flag:
            raise JobCancelled
        job.stage = stage
        job.progress = min(0.999, max(job.progress, frac))
        job.message = msg
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {stage}: {msg}" if msg else f"[{stamp}] {stage}"
        if not job.log or job.log[-1] != line:
            job.log.append(line)
            if len(job.log) > 400:
                del job.log[:200]

    def on_transcript(lines: list[str]):
        job.transcript = lines

    def on_segment(seg):
        job.segments_done.append({
            "t0": round(max(0.0, seg.t0), 2),
            "label": seg.event_label or "ambience",
            "excitement": seg.excitement,
            "text": seg.text,
            "audio_url": f"/api/tts/{job.id}/{Path(seg.audio_path).name}",
        })
        job.live_seq += 1

    print(f"[web] job {job.id} start: {job.filename}")
    summary = pipeline.run_pipeline(
        Path(job.filename), job.home, job.away, t0=job.t0,
        duration=job.duration, orig_audio=job.orig_audio,
        start_score={"home": job.score_home, "away": job.score_away},
        minute_offset=job.minute_offset,
        progress=progress, on_transcript=on_transcript, on_segment=on_segment,
        _cancel_check=lambda: job.cancel_flag,
    )
    job.progress = 1.0
    job.stage = "done"
    job.status = "done"
    job.message = "final video ready"
    job.result = summary


threading.Thread(target=_worker, daemon=True).start()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), home: str = Form("Home"),
                 away: str = Form("Away"), t0: str = Form("0"),
                 dur: str = Form(""), orig_audio: bool = Form(True),
                 score_home: str = Form("0"), score_away: str = Form("0"),
                 minute_offset: str = Form("0")):
    if not file.filename:
        raise HTTPException(400, "no file")
    dest = UPLOAD_DIR / f"{int(time.time())}_{Path(file.filename).name}"
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
            size += len(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")
    try:
        t0_f = max(0.0, float(t0 or 0))
    except ValueError:
        t0_f = 0.0
    try:
        dur_f = float(dur) if dur and float(dur) > 0 else None
    except ValueError:
        dur_f = None
    job = Job(str(dest), home.strip() or "Home", away.strip() or "Away",
              t0_f, dur_f, orig_audio, _as_int(score_home),
              _as_int(score_away), _as_float(minute_offset))
    _JOBS[job.id] = job
    _QUEUE.put(job)
    return {"job_id": job.id}


@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {
        "id": job.id, "status": job.status, "stage": job.stage,
        "progress": job.progress, "message": job.message,
        "log": job.log[-60:], "transcript": job.transcript,
        "result": job.result, "error": job.error,
        "video_url": _media_url(job) if job.status == "done" else None,
    }


@app.get("/api/job/{job_id}/live")
async def job_live(job_id: str, after: int = 0):
    """SSE: pushes each commentary line the moment its TTS finishes (live mode)."""
    import json as _json
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")

    async def gen():
        seq = after
        while True:
            while seq < len(job.segments_done):
                seg = job.segments_done[seq]
                seq += 1
                yield f"data: {_json.dumps(seg)}\n\n"
            if job.status in ("done", "error", "cancelled") and \
                    seq >= len(job.segments_done):
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/tts/{job_id}/{name}")
async def tts_file(job_id: str, name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400)
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    out_dir = Path(job.result["output"]).parent if job.result else \
        settings.out_root / f"{Path(job.filename).stem}_{int(job.t0)}s"
    path = out_dir / "tts" / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/jobs")
async def jobs():
    return [{"id": j.id, "file": Path(j.filename).name, "status": j.status,
             "progress": j.progress, "created": j.created}
            for j in sorted(_JOBS.values(), key=lambda j: -j.created)]


@app.post("/api/job/{job_id}/cancel")
async def cancel(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    job.cancel_flag = True
    return {"ok": True}


@app.post("/api/upload_existing")
async def upload_existing(home: str = Form("Home"), away: str = Form("Away"),
                          t0: str = Form("0"), dur: str = Form(""),
                          orig_audio: bool = Form(True),
                          score_home: str = Form("0"),
                          score_away: str = Form("0"),
                          minute_offset: str = Form("0"),
                          file: str = Form(...)):
    """Re-register an already-uploaded file (server-side path) as a new job —
    lets a big earlier upload be reprocessed without re-uploading."""
    p = (UPLOAD_DIR / file).resolve()
    if UPLOAD_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "no such upload")
    try:
        t0_f = float(t0)
    except ValueError:
        t0_f = 0.0
    try:
        dur_f = float(dur) if dur and float(dur) > 0 else None
    except ValueError:
        dur_f = None
    job = Job(str(p), home.strip() or "Home", away.strip() or "Away",
              t0_f, dur_f, orig_audio, _as_int(score_home),
              _as_int(score_away), _as_float(minute_offset))
    _JOBS[job.id] = job
    _QUEUE.put(job)
    return {"job_id": job.id}


def _media_url(job: Job) -> str | None:
    if not job.result:
        return None
    out = Path(job.result.get("output", ""))
    try:
        rel = out.relative_to(settings.out_root).as_posix()
    except ValueError:
        return None
    return f"/media/{rel}"


@app.get("/api/demo")
async def demo_state():
    """The precomputed ARG-FRA reel, moments and stats."""
    from app import training_status
    return training_status.demo_status()


@app.get("/api/training")
async def training_state():
    """Live YOLO/LLM training state, demo stats and the ClearML ledger."""
    from app import training_status
    return training_status.training_bundle()


@app.get("/api/training/asset/{name}")
async def training_asset(name: str):
    """Serve whitelisted plots/files from the YOLO run directory."""
    from app import training_status
    if name not in training_status.ASSETS:
        raise HTTPException(404, "unknown asset")
    target = training_status.YOLO_RUN / name
    if not target.exists():
        raise HTTPException(404)
    return FileResponse(target)


@app.get("/media/{path:path}")
async def media(path: str):
    root = settings.out_root.resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)) or not target.exists():
        raise HTTPException(404)
    return FileResponse(target)


app_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(app_dir), html=True), name="static")
