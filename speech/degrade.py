"""Controlled acoustic degradations (provided). Each takes a mono float waveform and returns one of the
same length, so the same test files can be evaluated under every condition.

    degrade(y, sr, "noise", snr_db=10, rng=rng)        white noise at a given SNR
    degrade(y, sr, "babble", snr_db=5, others=[...])   sum of other speakers (FSDD) at a given SNR
    degrade(y, sr, "reverb", rt60=0.6, rng=rng)        synthetic room impulse response (exponential decay)
    degrade(y, sr, "telephone")                        300-3400 Hz band + 8-bit mu-law codec
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt


def _mix_at_snr(y, noise, snr_db):
    p_s = np.mean(y ** 2) + 1e-12
    p_n = np.mean(noise ** 2) + 1e-12
    return y + noise * np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))


def synthetic_rir(sr: int, rt60: float, rng, direct: float = 1.0) -> np.ndarray:
    """Exponentially decaying noise tail (energy -60 dB at rt60 seconds) after a direct path."""
    n = int(sr * rt60)
    t = np.arange(n) / sr
    tail = rng.normal(0, 1, n) * np.exp(-6.908 * t / rt60)  # ln(1000) = 6.908 -> amplitude -60 dB
    h = 0.3 * tail
    h[0] += direct
    return h / np.sqrt(np.sum(h ** 2))


def mu_law(y, mu: int = 255, bits: int = 8):
    y = np.clip(y / (np.max(np.abs(y)) + 1e-9), -1, 1)
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round((c + 1) / 2 * (2 ** bits - 1)) / (2 ** bits - 1) * 2 - 1
    return np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu


def degrade(y: np.ndarray, sr: int, kind: str, rng=None, snr_db: float = 10.0, rt60: float = 0.5,
            others: list | None = None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng(0)
    if kind == "clean":
        return y
    if kind == "noise":
        return _mix_at_snr(y, rng.normal(0, 1, len(y)), snr_db)
    if kind == "babble":
        assert others, "babble needs other recordings"
        b = np.zeros(len(y))
        for o in rng.choice(len(others), size=min(4, len(others)), replace=False):
            x = others[o]
            b += np.resize(x, len(y)) if len(x) < len(y) else x[:len(y)]
        return _mix_at_snr(y, b, snr_db)
    if kind == "reverb":
        out = fftconvolve(y, synthetic_rir(sr, rt60, rng))[:len(y)]
        return out * np.sqrt(np.mean(y ** 2) / (np.mean(out ** 2) + 1e-12))
    if kind == "telephone":
        hi = min(3400, 0.45 * sr)
        sos = butter(4, [300, hi], btype="band", fs=sr, output="sos")
        out = mu_law(sosfiltfilt(sos, y))
        return out * np.sqrt(np.mean(y ** 2) / (np.mean(out ** 2) + 1e-12))
    raise ValueError(kind)
