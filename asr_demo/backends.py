"""Backend registry. The demo lists every registered backend; the browser picks one PER RECORDING.

Register your own (this is how you "hijack" the demo) in my_backends.py at the bundle root:

    from asr_demo.backends import register
    from asr_demo.contract import OfflineBackend

    @register("student/my-model", "what it is, in one line")
    class Mine(OfflineBackend):
        def __init__(self): ...                 # load once
        def transcribe(self, audio): return "..."

Factories are lazy: a model is loaded the first time someone selects it, then cached.
"""
from __future__ import annotations

import importlib
import os
import traceback
from pathlib import Path

import numpy as np

from asr_demo.contract import Backend, OfflineBackend, Session, resample

REGISTRY: dict[str, dict] = {}
_CACHE: dict[str, Backend] = {}
_FAILED: dict[str, str] = {}  # name -> one-line reason; a failed factory is not retried until the server restarts
ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("DSP_MODELS", ROOT / "models"))


class BackendUnavailable(RuntimeError):
    """Unknown backend, or a factory failed (missing package, missing model file, ...). str(e) says what to do."""


def explain(e: BaseException) -> str:
    if isinstance(e, ModuleNotFoundError):
        return f"missing Python module '{e.name}' (install it in the demo's environment)"
    return f"{type(e).__name__}: {e}"


def register(name: str, description: str = ""):
    def deco(factory):
        REGISTRY[name] = {"factory": factory, "description": description}
        return factory
    return deco


def get(name: str) -> Backend:
    if name in _CACHE:
        return _CACHE[name]
    if name not in REGISTRY:
        raise BackendUnavailable(f"unknown backend {name}")
    if name not in _FAILED:
        try:
            b = REGISTRY[name]["factory"]()
        except Exception as e:
            _FAILED[name] = explain(e)
            if not isinstance(e, (ModuleNotFoundError, FileNotFoundError)):  # those two are explained, not bugs
                traceback.print_exc()
        else:
            b.name = name
            _CACHE[name] = b
            return b
    raise BackendUnavailable(f"{name} is unavailable: {_FAILED[name]}. Fix it, then restart the server.")


def available() -> list[dict]:
    return [{"name": n, "description": r["description"], "loaded": n in _CACHE, "error": _FAILED.get(n)}
            for n, r in REGISTRY.items()]


# ----------------------------------------------------------------- reference: on-device streaming SOTA
class _SherpaSession(Session):
    def __init__(self, rec):
        self.rec, self.s, self.last = rec, rec.create_stream(), ""

    def _text(self):
        r = self.rec.get_result(self.s)
        return (r if isinstance(r, str) else r.text).strip().lower()

    def accept_chunk(self, audio):
        self.s.accept_waveform(16000, audio)
        while self.rec.is_ready(self.s):
            self.rec.decode_stream(self.s)
        t = self._text()
        if t != self.last:
            self.last = t
            return t
        return None

    def finish(self):
        self.s.accept_waveform(16000, np.zeros(int(0.5 * 16000), np.float32))  # tail padding flushes the encoder
        self.s.input_finished()
        while self.rec.is_ready(self.s):
            self.rec.decode_stream(self.s)
        return self._text()


class SherpaZipformer(Backend):
    """Streaming transducer (k2/icefall zipformer) exported to ONNX, run by sherpa-onnx on CPU."""
    def __init__(self, folder: str, int8: bool = True, threads: int = 2):
        import sherpa_onnx
        d = MODELS / folder
        q = ".int8" if int8 else ""
        f = lambda part: str(next(d.glob(f"{part}-*{q}.onnx")))  # noqa: E731
        if not int8:
            f = lambda part: str(next(p for p in d.glob(f"{part}-*.onnx") if ".int8" not in p.name))  # noqa: E731
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=f("encoder"), decoder=f("decoder"), joiner=f("joiner"),
            num_threads=threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")

    def new_session(self):
        return _SherpaSession(self.rec)


ZIPFORMER_20M = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
register("reference/zipformer-20M-int8", "streaming transducer, 20M params, int8, CPU (sherpa-onnx)")(
    lambda: SherpaZipformer(ZIPFORMER_20M, int8=True))
register("reference/zipformer-20M-fp32", "same model in fp32: what does quantisation cost or save?")(
    lambda: SherpaZipformer(ZIPFORMER_20M, int8=False))


# ----------------------------------------------------------------- Hugging Face offline models
class HFPipeline(OfflineBackend):
    def __init__(self, model_id: str, partial_every_s=None):
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model=model_id, device="cpu")
        self.partial_every_s = partial_every_s

    def transcribe(self, audio):
        return self.asr({"raw": audio, "sampling_rate": 16000})["text"].strip().lower()


register("hf/whisper-tiny.en", "Whisper tiny (39M), offline: text after release")(lambda: HFPipeline("openai/whisper-tiny.en"))
register("hf/whisper-base.en+partials", "Whisper base, re-decoded every 1 s for live text")(
    lambda: HFPipeline("openai/whisper-base.en", partial_every_s=1.0))
register("hf/wav2vec2-base-960h", "CTC, greedy decoding, no LM")(lambda: HFPipeline("facebook/wav2vec2-base-960h"))


# ----------------------------------------------------------------- student systems (use YOUR code)
class GMMDigits(OfflineBackend):
    """Day 1 system: your GMM-HMM word models + word-loop Viterbi + bigram LM, on spoken digits."""
    def __init__(self):
        import pickle
        from speech.gmm_hmm import WordLoopDecoder, mfcc_features
        path = Path(os.environ.get("DSP_GMM_MODEL", ROOT / "results/speech_classical/gmm_standard.pkl"))
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run speech/run_classical_baseline.py --part isolated --split standard first, or set DSP_GMM_MODEL")
        self.wm, self.norm = pickle.load(open(path, "rb"))
        self.dec = WordLoopDecoder(self.wm, [str(d) for d in range(10)] + ["sil"], None, word_penalty=-100.0)
        self.mfcc = mfcc_features

    def transcribe(self, audio):
        words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        y = resample(audio, 16000, 8000)
        y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.05
        return " ".join(words[int(d)] for d in self.dec.decode(self.norm(self.mfcc(y, 8000))))


register("student/gmm-hmm-digits", "your day-1 recogniser (digits only!)")(GMMDigits)


def _local_agreement(inner: str, step_s: float = 1.0):
    def factory():
        from speech.streaming_live import LocalAgreementBackend
        return LocalAgreementBackend(get(inner), step_s=step_s)
    return factory


register("student/local-agreement(whisper-tiny.en)", "your day-3 streaming policy around Whisper tiny")(
    _local_agreement("hf/whisper-tiny.en"))

# students' own backends
try:
    importlib.import_module("my_backends")
except ModuleNotFoundError as e:
    if e.name != "my_backends":
        raise
