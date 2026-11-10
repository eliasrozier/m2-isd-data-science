"""Metrics shared by the live server and the replay harness, so live and offline numbers are comparable.
  first_partial_ms : audio start -> first partial text (streaming feel)
  final_latency_ms : record button released -> final text (what the user waits for)
  rtf              : compute time / audio duration (must stay < 1 to keep up)
"""
from __future__ import annotations

import time

import numpy as np


def wer_or_none(reference: str, hyp: str):
    if not reference.strip():
        return None
    import jiwer
    from speech.data import normalise
    return float(jiwer.wer(normalise(reference), normalise(hyp) or "<empty>"))


def summarise(name, streaming, y, sr, compute_s, first_partial_ms, final_latency_ms, n_partials, text, reference, load_ms=0.0):
    dur = len(y) / sr if len(y) else 0.0
    return {"backend": name, "streaming": bool(streaming), "text": text, "audio_s": round(dur, 3),
            "first_partial_ms": None if first_partial_ms is None else round(first_partial_ms, 1),
            "final_latency_ms": round(final_latency_ms, 1), "rtf": round(compute_s / dur, 4) if dur else None,
            "n_partials": n_partials, "wer": wer_or_none(reference, text), "load_ms": round(load_ms, 1)}


def run_session_offline(backend, y, sr, reference="", realtime=False, chunk_s=0.1):
    """Feed a recording through a backend exactly like the browser does (100 ms chunks, then finish).
    realtime=True sleeps so that audio arrives at the speed of speech (true first-partial latency)."""
    from asr_demo.contract import resample
    y = resample(y, sr, backend.sample_rate)
    sr = backend.sample_rate
    s = backend.new_session()
    n = int(chunk_s * sr)
    compute, first, nparts = 0.0, None, 0
    t0 = time.perf_counter()
    for i in range(0, len(y), n):
        if realtime:
            time.sleep(max(0.0, t0 + i / sr - time.perf_counter()))
        c = time.perf_counter()
        txt = s.accept_chunk(y[i:i + n])
        compute += time.perf_counter() - c
        if txt:
            nparts += 1
            first = first or ((time.perf_counter() - t0) * 1000 if realtime else (i + n) / sr * 1000)
    c = time.perf_counter()
    final = s.finish()
    fin = time.perf_counter() - c
    return summarise(backend.name, backend.streaming, y, sr, compute + fin, first, fin * 1000, nparts, final, reference)
