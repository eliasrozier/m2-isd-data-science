"""Offline ASR Pareto front on LibriSpeech: WER vs real-time factor, latency, energy, size.

    python -m speech.asr_benchmark --librispeech /data/LibriSpeech --subset test-clean --n 200 \
        --models openai/whisper-tiny openai/whisper-small facebook/wav2vec2-base-960h --batch 1 16
    python -m speech.asr_benchmark --plot results/asr_bench.jsonl

batch=1 measures LATENCY (interactive use); batch=16 measures THROUGHPUT (offline transcription).
The two give different fronts -- one of the key messages of the project.
UNTESTED in the authoring sandbox (no model downloads possible there). Verify model ids on the Hub;
some families (e.g. Moonshine, Parakeet) may need a recent transformers or their own library.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common.bench import (BenchRecord, cuda_sync, energy_scope, log_record, peak_memory_mb, reset_peak_memory,
                          summarize_latencies)
from speech.data import load_librispeech, wer

DEFAULT_MODELS = [
    "openai/whisper-tiny.en", "openai/whisper-base.en", "openai/whisper-small.en", "openai/whisper-medium.en",
    "openai/whisper-large-v3-turbo", "distil-whisper/distil-large-v3",
    "facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h-lv60-self",
    "UsefulSensors/moonshine-tiny", "UsefulSensors/moonshine-base",
]


def bench_model(model_id, data, batch, device, fp16):
    raise NotImplementedError("TODO(d2): HF ASR pipeline at the requested batch size and dtype, warm-up, time every batch (sync GPU), return metrics (wer, rtf, latency percentiles, params, energy, memory) and hypotheses")


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.bench import load_records
    from common.pareto import plot_pareto
    df = load_records(path)
    df["label"] = df.model.str.split("/").str[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, b in zip(axes, sorted(df.cfg_batch.unique())[:2]):
        plot_pareto(df[df.cfg_batch == b].reset_index(drop=True), "rtf", "wer", "min", "min", label="label",
                    ax=ax, logx=True, title=f"WER vs real-time factor, batch={b}")
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", type=Path)
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/asr_bench.jsonl"))
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()
    if args.plot:
        return plot(args.plot)
    data = load_librispeech(args.librispeech, args.subset, args.n)
    print(f"{len(data)} utterances, {sum(len(d['audio']) / d['sr'] for d in data) / 60:.1f} min of audio")
    for mid in args.models:
        for b in args.batch:
            try:
                m, _ = bench_model(mid, data, b, args.device, args.fp16)
            except Exception as e:
                print(f"[skip] {mid} batch={b}: {e}")
                continue
            log_record(BenchRecord("asr", mid, dict(subset=args.subset, batch=b, fp16=args.fp16, n=len(data)), m), args.out)
            print(mid, b, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
