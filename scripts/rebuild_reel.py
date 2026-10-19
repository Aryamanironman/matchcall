"""Rebuild the highlights reel: distinct A/B voices + commentator-free crowd bed.

Pipeline: pick high-excitement moments (v4 transcript) -> re-synthesize TTS
(Fish A / edge B, per-speaker) -> demucs two-stems separation of the source
audio (keep no_vocals = crowd) -> per-moment clips with mixed audio -> concat
with fades.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from app import tts as tts_mod  # noqa: E402
from app.events import Segment  # noqa: E402
from app.render import _voice_chain  # noqa: E402

RUN = ROOT / "output" / "1791775787_3 - 3 ARG vs FRA - Second-Half - Bein_0s_v4"
SRC = ROOT / "output" / "uploads" / "1791775787_3 - 3 ARG vs FRA - Second-Half - Bein.mp4"
OUT = ROOT / "output" / "highlights_reel_v2"
OUT.mkdir(parents=True, exist_ok=True)
FF = "ffmpeg"

CLIP_LEN, PRE = 8, 2

# ---- 1. pick moments -------------------------------------------------------
t = json.loads((RUN / "transcript.json").read_text(encoding="utf-8"))
events = [s for s in t if s.get("kind") == "event"]
xc = {"huge": 4, "high": 3, "medium": 2, "low": 1, "calm": 0}
events.sort(key=lambda s: (-xc.get(s.get("excitement"), 0), s["t0"]))
picked, chosen = [], []
for s in events:
    if all(abs(s["t0"] - p) > 120 for p in picked):
        picked.append(s["t0"])
        chosen.append(s)
    if len(picked) == 6:
        break
picked.sort()
chosen = sorted(chosen, key=lambda s: s["t0"])
print("picked moments:", [round(p, 1) for p in picked])

# ---- 2. fresh TTS per picked segment (+ paired B reactions) ----------------
tts_dir = OUT / "tts"
by_t0 = {round(s["t0"], 2): s for s in t}
chosen_extra = []
for s in chosen:
    nxt = by_t0.get(round(s["t0"] + 1.2, 2))
    if nxt and nxt.get("speaker") == "B":
        chosen_extra.append(nxt)
segs = [Segment(t0=s["t0"], kind=s.get("kind", "event"),
                event_label=s.get("event_label"),
                excitement=s.get("excitement", "medium"),
                text=s["text"], speaker=s.get("speaker", "A"))
        for s in chosen + chosen_extra]
done = tts_mod.synthesize(segs, tts_dir)
for s in done:
    assert s.audio_path, f"tts failed: {s.text[:40]}"
print("tts done:", len(done), "lines "
      f"(A={sum(1 for s in done if s.speaker == 'A')}, "
      f"B={sum(1 for s in done if s.speaker == 'B')})")

# ---- 3. demucs: crowd bed for just the picked windows ----------------------
WIN = CLIP_LEN + 0.5  # per-window slot in the concat wav
windows = OUT / "windows.wav"
cmd = [FF, "-y", "-v", "error"]
for t0 in picked:
    cmd += ["-ss", f"{max(0, t0 - PRE):.3f}", "-t", f"{WIN:.3f}", "-i", str(SRC)]
fc = [f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{i}]"
      for i in range(len(picked))]
fc.append("".join(f"[a{i}]" for i in range(len(picked))) +
          f"concat=n={len(picked)}:v=0:a=1[out]")
cmd += ["-filter_complex", ";".join(fc), "-map", "[out]", "-ac", "2",
        "-ar", "44100", str(windows)]
subprocess.run(cmd, check=True)
print(f"separating {len(picked)} windows ({len(picked) * WIN:.0f}s)...")
sep_dir = OUT / "separated"
crowd = sep_dir / "htdemucs" / windows.stem / "no_vocals.wav"
if crowd.exists():
    print("crowd bed reused from cache:", crowd.stat().st_size // 1024, "KB")
else:
    r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals",
                        "-n", "htdemucs", "--device", "cuda",
                        "-o", str(sep_dir), str(windows)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not crowd.exists():
        print("demucs failed:", r.stderr[-500:])
        sys.exit(1)
    print("crowd bed ready:", crowd.stat().st_size // 1024, "KB")

# ---- 4. per-moment clips (video copy + crowd + TTS mix) --------------------
parts = []
for i, t0 in enumerate(picked):
    clip0 = t0 - PRE
    rel = {id(s): max(0.0, s.t0 - clip0) for s in done}
    cmd = [FF, "-y", "-v", "error", "-ss", f"{clip0:.3f}", "-t", str(CLIP_LEN),
           "-i", str(SRC)]                              # 0: video+orig audio
    cmd += ["-ss", f"{i * WIN:.3f}", "-t", str(CLIP_LEN),
            "-i", str(crowd)]                           # 1: crowd bed slot
    voiced = [s for s in done if clip0 - 1 <= s.t0 < clip0 + CLIP_LEN]
    for s in voiced:
        cmd += ["-i", s.audio_path]                     # 2..n: TTS
    fc, labels = [], []
    for j, s in enumerate(voiced):
        fc.append(_voice_chain(j + 2, s) +
                  f",adelay={int(round(rel[id(s)] * 1000))}:all=1[t{j}]")
        labels.append(f"[t{j}]")
    if labels:
        fc.append("".join(labels) +
                  f"amix=inputs={len(labels)}:duration=longest:normalize=0[vx]")
    else:
        fc.append("anullsrc=r=44100:cl=stereo,atrim=0:0.05[vx]")
    fc.append("[1:a]volume=0.9[crowd]")
    fc.append("[vx][crowd]amix=inputs=2:duration=first:"
              "normalize=0,alimiter=limit=0.95[aout]")
    part = OUT / f"part_{i}.mp4"
    cmd += ["-filter_complex", ";".join(fc), "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", str(part)]
    subprocess.run(cmd, check=True)
    parts.append(part)
    print(f"  part {i}: t0={t0:.0f}s, {len(voiced)} lines")

# ---- 5. concat with fades ---------------------------------------------------
cmd = [FF, "-y", "-v", "error"]
for p in parts:
    cmd += ["-i", str(p)]
fc = []
for i in range(len(parts)):
    fc.append(f"[{i}:v]fps=25,scale=1280:720:force_original_aspect_ratio=decrease,"
              f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
              f"fade=t=in:st=0:d=0.4,fade=t=out:st={CLIP_LEN - 0.4}:d=0.4[fv{i}]")
    fc.append(f"[{i}:a]afade=t=in:st=0:d=0.3,"
              f"afade=t=out:st={CLIP_LEN - 0.35}:d=0.35[fa{i}]")
fc.append("".join(f"[fv{i}][fa{i}]" for i in range(len(parts))) +
          f"concat=n={len(parts)}:v=1:a=1[v][a]")
final = OUT / "highlights_reel_v2.mp4"
cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(final)]
subprocess.run(cmd, check=True)
print("FINAL:", final, final.stat().st_size // 1024, "KB")
