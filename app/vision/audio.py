"""Crowd-audio analysis: per-second loudness profile + roar (goal-spike) events.

Goal roars are the single most match-independent goal signal in a broadcast
feed: vision heuristics fire on every attack, but a genuine roar is a
sustained 6-20 dB jump over the ambient crowd bed. We extract 8 kHz mono PCM
and compute per-second RMS, then find sustained spikes ("roars").
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from app.config import settings

_SR = 8000  # 8 kHz mono is plenty for loudness and keeps the pipe small


def audio_rms_profile(video: Path, t0: float = 0.0,
                      duration: float | None = None,
                      skip_audio: bool = False) -> np.ndarray:
    """Per-second RMS (linear power) of the video's audio track.

    Returns an empty array when the video has no audio stream or ffmpeg fails —
    callers must treat that as "audio gating unavailable", never as "no roars".
    """
    if skip_audio:
        return np.zeros(0)
    cmd = [settings.ffmpeg, "-v", "error"]
    if t0 > 0:
        cmd += ["-ss", f"{t0:.3f}"]
    cmd += ["-i", str(video)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "a:0", "-ac", "1", "-ar", str(_SR),
            "-f", "s16le", "pipe:1"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    except (subprocess.TimeoutExpired, OSError):
        return np.zeros(0)
    if len(raw) < _SR * 2:  # < 1 second of PCM
        return np.zeros(0)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n_sec = len(pcm) // _SR
    if n_sec == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(pcm[:n_sec * _SR].reshape(n_sec, _SR) ** 2, axis=1))


def crowd_rise(rms: np.ndarray, smooth_s: int = 8) -> np.ndarray:
    """Per-second crowd energy relative to a slow local median (rise factor).

    Broadcast loudness normalization squashes absolute dynamics, so we compute
    energy rise over a 10-minute rolling median of a smoothed envelope. Values
    hover ~1.0 at the crowd bed and reach ~1.3-2.5 at genuine surges (goals,
    big chances). Returns ones() for unusable input (no audio / too short) so
    callers never have to special-case.
    """
    n = len(rms)
    if n < 30 or float(np.median(rms)) <= 0.0:
        return np.ones(max(n, 1))
    k = max(1, smooth_s)
    env = np.convolve(rms, np.ones(k) / k, mode="same")
    half = 300  # 10-minute rolling bed
    bed = np.array([np.median(env[max(0, i - half):i + half + 1])
                    for i in range(n)])
    return np.maximum(env / np.maximum(bed, 1e-6), 0.5)


def audio_evidence(rise: np.ndarray, t: float, window_s: float = 30.0) -> float:
    """Max crowd rise within +-window_s of t (1.0 = no audio / nothing there)."""
    if len(rise) < 2:
        return 1.0
    a = max(0, int(t - window_s))
    b = min(len(rise), int(t + window_s) + 1)
    return float(np.max(rise[a:b])) if b > a else 1.0


def roar_events(rms: np.ndarray, fps: float = 1.0,
                min_rise: float = 2.2, min_sustain_s: float = 3.0,
                cooldown_s: float = 25.0,
                min_baseline: float = 0.004) -> list[dict]:
    """Sustained loudness spikes over the ambient crowd bed.

    rms: per-second RMS. A roar = a run of >= min_sustain_s seconds whose
    power exceeds baseline * min_rise, where baseline is the running median
    (robust to long quiet passages). Events closer than cooldown_s are merged
    (replays / celebrations directly after a goal are one event). Returns
    [{"t": seconds_from_profile_start, "peak": rms, "rise": peak/baseline}].
    """
    if len(rms) < 30 or float(np.median(rms)) <= 0.0:
        return []
    events: list[dict] = []
    # running median over a 121 s window (crowd level drifts through a match)
    half = 60
    med = np.array([np.median(rms[max(0, i - half):i + half + 1])
                    for i in range(len(rms))])
    med = np.maximum(med, min_baseline)
    hot = rms > med * min_rise

    i = 0
    while i < len(hot):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < len(hot) and hot[j]:
            j += 1
        if (j - i) >= min_sustain_s:
            peak_i = i + int(np.argmax(rms[i:j]))
            events.append({"t": peak_i / fps, "peak": float(rms[peak_i]),
                           "rise": float(rms[peak_i] / med[peak_i]),
                           "start": i / fps, "end": j / fps})
        i = j
    # merge events inside cooldown: keep the highest-rise one
    merged: list[dict] = []
    for ev in sorted(events, key=lambda e: e["t"]):
        if merged and ev["t"] - merged[-1]["t"] < cooldown_s:
            if ev["rise"] > merged[-1]["rise"]:
                merged[-1] = ev
        else:
            merged.append(ev)
    return merged


def extract_pcm(video: Path, t0: float = 0.0, duration: float | None = None, sr: int = 8000) -> np.ndarray:
    """
    Extract mono PCM audio from a video file at the given sample rate.
    Returns a float32 numpy array in [-1, 1].
    If the video has no audio or ffmpeg fails, returns an empty array.
    """
    cmd = [settings.ffmpeg, "-v", "error"]
    if t0 > 0:
        cmd += ["-ss", f"{t0:.3f}"]
    cmd += ["-i", str(video)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "a:0", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    except (subprocess.TimeoutExpired, OSError):
        return np.zeros(0, dtype=np.float32)
    if len(raw) < 2:  # less than one sample
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm
