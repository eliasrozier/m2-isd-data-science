"""Day 3: your streaming policy, LIVE. Turns any offline recogniser into a streaming one that fits the
demo contract (asr_demo.contract): re-decode the growing buffer every `step_s` seconds of new audio and
commit words with LocalAgreement-n (as in speech/streaming_sim.py, but driven by real audio chunks).
Committed words are never retracted; the partial shown = committed + tentative tail.
With a record button there is no endpointing to do: release = finish().
"""
from __future__ import annotations

import numpy as np

from asr_demo.contract import Backend, OfflineBackend, Session


def common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


class LocalAgreementSession(Session):
    def __init__(self, inner: OfflineBackend, step_s: float, n_agree: int):
        self.inner, self.step, self.n = inner, step_s, n_agree
        self.buf: list[np.ndarray] = []
        self.new = 0
        self.committed: list[str] = []
        self.history: list[list[str]] = []

    def accept_chunk(self, audio):
        raise NotImplementedError("TODO(d3): buffer audio; every step_s of NEW audio re-transcribe the whole buffer, keep the last n hypotheses (words after the committed prefix), commit their common prefix, return committed + tentative text (None if no update)")

    def finish(self):
        raise NotImplementedError("TODO(d3): transcribe the full buffer once more; keep the committed words and append the remaining words of the final hypothesis")


class LocalAgreementBackend(Backend):
    def __init__(self, inner: OfflineBackend, step_s: float = 1.0, n_agree: int = 2):
        self.inner, self.step_s, self.n_agree = inner, step_s, n_agree
        self.sample_rate = inner.sample_rate
        self.description = f"LocalAgreement-{n_agree} every {step_s}s around {getattr(inner, 'name', 'model')}"

    def new_session(self):
        return LocalAgreementSession(self.inner, self.step_s, self.n_agree)
