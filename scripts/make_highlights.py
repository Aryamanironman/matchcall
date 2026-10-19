"""Build a highlights reel + key frames from the ARG-FRA commentary run."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output" / "1791775787_3 - 3 ARG vs FRA - Second-Half - Bein_0s"
VIDEO = RUN / "final_commentary.mp4"
OUT = ROOT / "output" / "highlights"
OUT.mkdir(parents=True, exist_ok=True)

FFMPEG = "ffmpeg"

# 1. pick top high-excitement shot moments, spread out
t = json.loads((RUN / "transcript.json").read_text(encoding="utf-8"))
shots = [s for s in t if s.get("kind") == "event"
         and s.get("event_label") == "shot" and s.get("excitement") == "high"]
shots.sort(key=lambda s: s["t0"])

picked = []
for s in shots:
    if all(abs(s["t0"] - p) > 120 for p in picked):  # >=2 min apart
        picked.append(s["t0"])
    if len(picked) == 8:
        break
print("picked moments (s):", [round(p, 1) for p in picked])

# 2. cut 8s clips (2s before the call), chain-concat with fades, 720p25
CLIP_LEN, PRE = 8, 2
inputs, filters = [], []
for i, t0 in enumerate(picked):
    ss = max(0, t0 - PRE)
    inputs += ["-ss", f"{ss:.2f}", "-t", str(CLIP_LEN), "-i", str(VIDEO)]
    filters.append(
        f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,"
        f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=25,setsar=1,"
        f"fade=t=in:st=0:d=0.4,fade=t=out:st={CLIP_LEN-0.4:.2f}:d=0.4[v{i}];"
        f"[{i}:a]afade=t=in:st=0:d=0.3,afade=t=out:st={CLIP_LEN-0.3:.2f}:d=0.3[a{i}]"
    )
n = len(picked)
fc = ";".join(filters) + ";" + "".join(f"[v{i}][a{i}]" for i in range(n)) + \
     f"concat=n={n}:v=1:a=1[v][a]"
reel = OUT / "highlights_arg_fra.mp4"
cmd = [FFMPEG, "-y", *inputs, "-filter_complex", fc,
       "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast",
       "-crf", "23", "-c:a", "aac", "-b:a", "128k", str(reel)]
r = subprocess.run(cmd, capture_output=True, text=True)
print("reel:", reel.name, reel.stat().st_size // 1024, "KB" if r.returncode == 0
      else f"FAILED: {r.stderr[-400:]}")

# 3. key frames (full-res stills) at picked moments
frames = []
for i, t0 in enumerate(picked[:6]):
    m, sec = divmod(int(t0), 60)
    fp = OUT / f"frame_{i}_{m:02d}m{sec:02d}s.png"
    subprocess.run([FFMPEG, "-y", "-ss", f"{t0:.2f}", "-i", str(VIDEO),
                    "-frames:v", "1", str(fp)], capture_output=True)
    if fp.exists():
        frames.append(fp)
print("frames:", [f.name for f in frames])

# 4. 2x3 montage for one-glance view
try:
    from PIL import Image
    thumbs = [Image.open(f).resize((640, 360)) for f in frames]
    grid = Image.new("RGB", (1280, 1080), "black")
    for i, im in enumerate(thumbs):
        grid.paste(im, ((i % 2) * 640, (i // 2) * 360))
    montage = OUT / "key_moments_montage.png"
    grid.save(montage)
    print("montage:", montage.name, montage.stat().st_size // 1024, "KB")
except ImportError:
    print("PIL unavailable, skipping montage")

(OUT / "picked_moments.json").write_text(
    json.dumps([{"t0": round(p, 2), "text": next(
        s["text"] for s in shots if s["t0"] == p)} for p in picked],
        ensure_ascii=False, indent=1), encoding="utf-8")
print("done ->", OUT)
