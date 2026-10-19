"""ffmpeg renderer: mix TTS segments at exact offsets over ducked original audio."""
from __future__ import annotations

import subprocess
from pathlib import Path

from app.config import settings
from app.events import Segment


def _voice_chain(i: int, s: Segment) -> str:
    """Per-speaker tone: B gets a ~0.7-semitone lower pitch (asetrate+atempo
    keeps duration), so A/B stay distinguishable even with similar voices."""
    fmt = (f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo")
    if s.speaker == "B":
        return (fmt + f",asetrate=44100*0.96,aresample=44100,"
                f"atempo={1 / 0.96:.5f},volume={settings.voice_boost}")
    return fmt + f",volume={settings.voice_boost}"


def render(video: Path, segments: list[Segment], out_path: Path,
           t0: float = 0.0, duration: float | None = None,
           orig_audio: bool = True) -> Path:
    """Segments' t0 are clip-relative seconds (0 = clip start).

    ffmpeg option order matters: `-ss` goes BEFORE the video input (fast seek),
    but `-t` and all output options go AFTER every input — otherwise they bind
    to the last input (a 3-second mp3) and the filter graph breaks.
    """
    voice_segments = [s for s in segments if s.audio_path and s.audio_dur > 0.1]

    if not voice_segments:
        # nothing synthesized: plain copy
        cmd = [settings.ffmpeg, "-y", "-v", "error"]
        if t0 > 0:
            cmd += ["-ss", f"{t0:.3f}"]
        cmd += ["-i", str(video)]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-c", "copy", str(out_path)]
        subprocess.run(cmd, check=True)
        return out_path

    work = out_path.parent / "work"
    work.mkdir(exist_ok=True)

    # Windows limits a process command line (~32K chars); hundreds of `-i` paths
    # blow past it. Pre-mix the voice track in chunks so no command is huge.
    _CHUNK = 80
    chunk_files: list[Path] = []
    if len(voice_segments) > _CHUNK:
        for c0 in range(0, len(voice_segments), _CHUNK):
            grp = voice_segments[c0:c0 + _CHUNK]
            wav = work / f"voicemix_{c0:04d}.wav"
            cmd = [settings.ffmpeg, "-y", "-v", "error"]
            for s in grp:
                cmd += ["-i", s.audio_path]
            fc_parts, labels = [], []
            for i, s in enumerate(grp):
                delay_ms = max(0, int(round(s.t0 * 1000)))
                fc_parts.append(
                    _voice_chain(i, s) +
                    f",adelay={delay_ms}:all=1[v{i}]")
                labels.append(f"[v{i}]")
            fc_parts.append("".join(labels) +
                            f"amix=inputs={len(labels)}:duration=longest:normalize=0[aout]")
            cmd += ["-filter_complex", ";".join(fc_parts), "-map", "[aout]", str(wav)]
            subprocess.run(cmd, check=True)
            chunk_files.append(wav)

    # ---- final pass inputs ----
    cmd = [settings.ffmpeg, "-y", "-v", "error"]
    if t0 > 0:
        cmd += ["-ss", f"{t0:.3f}"]
    cmd += ["-i", str(video)]
    for w in chunk_files:
        cmd += ["-i", str(w)]

    # ---- filter graph (segment t0 values are clip-relative seconds) ----
    fc_parts, labels = [], []
    if chunk_files:
        for i in range(len(chunk_files)):
            fc_parts.append(f"[{i + 1}:a]aformat=sample_rates=44100:channel_layouts=stereo[vm{i}]")
            labels.append(f"[vm{i}]")
    else:
        for i, s in enumerate(voice_segments):
            cmd += ["-i", s.audio_path]
            delay_ms = max(0, int(round(s.t0 * 1000)))
            # input 0 is the video; TTS inputs start at 1
            fc_parts.append(
                _voice_chain(i + 1, s) +
                f",adelay={delay_ms}:all=1[v{i}]")
            labels.append(f"[v{i}]")
    fc_parts.append("".join(labels) +
                    f"amix=inputs={len(labels)}:duration=longest:normalize=0[vmix]")
    if duration is not None:
        fc_parts.append(f"[vmix]atrim=0:{duration:.3f}[vmix]")
    if orig_audio:
        fc_parts.append(f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                        f"volume={settings.original_audio_volume}[orig]")
        fc_parts.append("[vmix][orig]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        fc_parts.append("[vmix]anull[aout]")

    # ---- output options (AFTER all inputs) ----
    cmd += ["-filter_complex", ";".join(fc_parts),
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def concat_tts(segments: list[Segment], out_path: Path) -> Path | None:
    """DEBUG helper: single continuous TTS track for QA listening."""
    import json
    voice = [s for s in segments if s.audio_path and s.audio_dur > 0.1]
    if not voice:
        return None
    cmd = [settings.ffmpeg, "-y", "-v", "error"]
    for s in voice:
        cmd += ["-i", s.audio_path]
    fc = "".join(f"[{i}:a]" for i in range(len(voice))) + \
        f"concat=n={len(voice)}:v=0:a=1[aout]"
    cmd += ["-filter_complex", fc, "-map", "[aout]", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path
