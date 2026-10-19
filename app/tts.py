"""TTS: Fish Audio (s2.1-pro) primary, edge-tts fallback — one mp3 per segment."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import edge_tts

from app.config import settings
from app.events import Segment

# --- Fish Audio (primary) ---
_FISH_URL = "https://api.fish.audio/v1/tts"
_FISH_MODEL = os.getenv("FISH_MODEL", "s2.1-pro")
_FISH_VOICE_A = os.getenv("FISH_VOICE_ID", "40f53434437e46a5bcbb5abb39f5079c")
# B (color) gets its own Fish voice ONLY if FISH_VOICE_B is set; otherwise B
# falls through to edge-tts so the two commentators never share one timbre.
_FISH_VOICE_B = os.getenv("FISH_VOICE_B", "").strip()
_FISH_TIMEOUT = float(os.getenv("FISH_TIMEOUT", "60"))
_FISH_SEM = asyncio.Semaphore(3)  # polite concurrency for the hosted API

# --- edge-tts (fallback) ---
_SEM = asyncio.Semaphore(4)
# pacing multipliers per excitement (edge-tts rate strings)
_RATE_FOR = {"huge": "+18%", "high": "+12%", "medium": "+8%", "low": "+4%", "calm": "+0%"}


def _fish_key() -> str:
    return os.getenv("FISH_API_KEY", "").strip()


def _fish_one(seg: Segment, path: Path, key: str, voice_id: str) -> None:
    """Blocking Fish Audio request for one segment (runs in a worker thread)."""
    import httpx

    # s2.1 delivers energy naturally; nudge pace slightly by excitement
    speed = {"huge": 1.15, "high": 1.10, "low": 0.98, "calm": 0.95}.get(seg.excitement, 1.05)
    payload = {
        "text": seg.text,
        "reference_id": voice_id,
        "format": "mp3",
        "normalize": True,
        "prosody": {"speed": speed},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "model": _FISH_MODEL,
    }
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=_FISH_TIMEOUT) as client:
                r = client.post(_FISH_URL, json=payload, headers=headers)
            if r.status_code == 200 and r.content:
                if len(r.content) > 500:  # sanity: real audio, not an error stub
                    path.write_bytes(r.content)
                    return
                last_err = RuntimeError(f"tiny response ({len(r.content)}B)")
            elif r.status_code in (401, 402, 403):
                raise PermissionError(f"fish audio rejected the key ({r.status_code})")
            else:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fish audio failed after retries: {last_err}")


async def _one(seg: Segment, out_dir: Path, idx: int, on_done=None) -> Segment:
    path = out_dir / f"seg_{idx:04d}.mp3"

    # --- Fish Audio first (A always; B only when a distinct voice is set) ---
    key = _fish_key()
    fish_voice = (_FISH_VOICE_B if seg.speaker == "B" and _FISH_VOICE_B
                  else _FISH_VOICE_A if seg.speaker == "A" else "")
    if key and fish_voice:
        try:
            async with _FISH_SEM:
                await asyncio.to_thread(_fish_one, seg, path, key, fish_voice)
            dur = await asyncio.to_thread(_duration, path)
            seg.audio_path = str(path)
            seg.audio_dur = dur
            if on_done:
                on_done(seg)
            return seg
        except Exception as e:  # noqa: BLE001
            print(f"  [tts] fish audio failed seg {idx} "
                  f"({type(e).__name__}: {e}) — edge-tts fallback")

    # --- edge-tts fallback ---
    rate = _RATE_FOR.get(seg.excitement, settings.tts_rate)
    voice = settings.tts_voice_b if seg.speaker == "B" else settings.tts_voice
    for attempt in range(3):
        try:
            async with _SEM:
                comm = edge_tts.Communicate(seg.text, voice, rate=rate)
                await comm.save(str(path))
            dur = await asyncio.to_thread(_duration, path)
            seg.audio_path = str(path)
            seg.audio_dur = dur
            if on_done:
                on_done(seg)
            return seg
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  [tts] FAILED seg {idx} ({type(e).__name__}: {e}) — skipped")
                if on_done:
                    on_done(seg)
                return seg
            await asyncio.sleep(1.5 * (attempt + 1))
    return seg


def _duration(path: Path) -> float:
    import json
    import subprocess
    from app.config import settings as s
    out = subprocess.run(
        [s.ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


async def synthesize_all(segments: list[Segment], out_dir: Path,
                         on_done=None) -> list[Segment]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [_one(seg, out_dir, i, on_done=on_done) for i, seg in enumerate(segments)]
    return list(await asyncio.gather(*tasks))


def synthesize(segments: list[Segment], out_dir: Path, on_line=None) -> list[Segment]:
    try:
        return asyncio.run(synthesize_all(segments, out_dir, on_done=on_line))
    except RuntimeError:  # event loop already running (notebook)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(synthesize_all(segments, out_dir, on_done=on_line))
