"""The contract every recogniser plugs into (demo server, replay harness, benchmarks).

A Backend is loaded once (heavy: model weights) and opens one Session per recording (light).
  session.accept_chunk(audio) -> str | None   current partial hypothesis (None = nothing new)
  session.finish()            -> str          final transcript, after the record button is released
Audio: mono float32 in [-1, 1] at backend.sample_rate (16 kHz unless stated), chunks of ~100 ms.
Timing is measured OUTSIDE the backend (server / replay harness), identically for every backend.

Offline models (Whisper, wav2vec2, your GMM-HMM...) subclass OfflineBackend and only implement
transcribe(audio) -> str. Streaming models subclass Backend and return their own Session.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Session(ABC):
    @abstractmethod
    def accept_chunk(self, audio: np.ndarray) -> str | None: ...

    @abstractmethod
    def finish(self) -> str: ...


class Backend(ABC):
    name: str = "unnamed"
    description: str = ""
    sample_rate: int = 16000
    streaming: bool = True

    @abstractmethod
    def new_session(self) -> Session: ...


class _BufferSession(Session):
    def __init__(self, backend: "OfflineBackend"):
        self.b, self.buf, self.since = backend, [], 0

    def accept_chunk(self, audio):
        self.buf.append(np.asarray(audio, np.float32))
        self.since += len(audio)
        every = self.b.partial_every_s
        if every and self.since >= every * self.b.sample_rate:  # optional re-decoding for live partials
            self.since = 0
            return self.b.transcribe(np.concatenate(self.buf))
        return None

    def finish(self):
        return self.b.transcribe(np.concatenate(self.buf)) if self.buf else ""


class OfflineBackend(Backend):
    """Wraps a whole-utterance recogniser. partial_every_s=None: text appears only after release
    (lowest compute); a number: re-decode the growing buffer that often (live text, quadratic cost)."""
    streaming = False
    partial_every_s: float | None = None

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str: ...

    def new_session(self) -> Session:
        return _BufferSession(self)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)
