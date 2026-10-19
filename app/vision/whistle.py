"""Whistle detector: sustained 2.2-3.8 kHz tonality."""

from __future__ import annotations

import numpy as np
from typing import List, Dict


def detect_whistles(pcm: np.ndarray, sr: int = 8000) -> List[Dict]:
    """
    Detect whistle-like tones in the 2.2-3.8 kHz band.

    A whistle is detected if the energy in the 2.2-3.8 kHz band is significantly
    higher than the energy in the adjacent side bands (2.05-2.2 kHz and 3.8-3.95 kHz),
    indicating a tonal signal in that band.

    Returns a list with a single dict if a whistle is detected: {"t": float, "conf": float, "dur": float}
    where t is the time of the detection (middle of the signal), conf is the detection confidence (0-1),
    and dur is the duration of the signal in seconds.

    If no whistle is detected, returns an empty list.
    """
    # Minimum length: 0.1 seconds to avoid detecting clicks
    if len(pcm) < 0.1 * sr:
        return []

    # Compute FFT
    freqs = np.fft.rfftfreq(len(pcm), 1.0 / sr)
    fft_vals = np.fft.rfft(pcm)
    power = np.abs(fft_vals) ** 2

    # Define bands
    whistle_mask = (freqs >= 2200) & (freqs <= 3800)
    low_mask = (freqs >= 2050) & (freqs < 2200)
    high_mask = (freqs > 3800) & (freqs <= 3950)

    # Energy in each band
    whistle_energy = np.sum(power[whistle_mask])
    low_energy = np.sum(power[low_mask])
    high_energy = np.sum(power[high_mask])
    side_energy = low_energy + high_energy

    # Avoid division by zero
    if side_energy == 0:
        side_energy = 1e-10

    # Ratio of whistle energy to side energy
    ratio = whistle_energy / side_energy

    # Spectral flatness in whistle band: exp(mean(log)) / mean
    whistle_bands = power[whistle_mask]
    # Avoid zeros in log
    epsilon = 1e-10
    geom_mean = np.exp(np.mean(np.log(whistle_bands + epsilon)))
    arith_mean = np.mean(whistle_bands)
    flatness = geom_mean / (arith_mean + epsilon)
    # flatness in [0,1], lower is more tonal

    # Thresholds: ratio > 20.0 and flatness < 0.005
    if ratio > 20.0 and flatness < 0.005:
        # Confidence: combine ratio and flatness
        # Normalize ratio: at ratio=100, conf_ratio=1.0; at ratio=20, conf_ratio=0.0? We'll clip.
        conf_ratio = min((ratio - 20.0) / (100.0 - 20.0), 1.0) if ratio > 20.0 else 0.0
        # Normalize flatness: at flatness=0, conf_flat=1.0; at flatness=0.005, conf_flat=0.0
        conf_flat = max(0.0, (0.005 - flatness) / 0.005)
        conf = (conf_ratio + conf_flat) / 2.0
        # Time at the middle of the signal
        t = len(pcm) / (2.0 * sr)
        dur = len(pcm) / sr
        return [{"t": float(t), "conf": float(conf), "dur": float(dur)}]
    else:
        return []
