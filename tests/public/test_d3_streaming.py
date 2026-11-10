import numpy as np
import pytest

from speech.streaming_sim import _mock, commit_lags, simulate_streaming

pytestmark = pytest.mark.d3


def setup(seed=0):
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(50)]
    ends = np.cumsum(rng.uniform(0.25, 0.6, 50)) + 0.3
    audio = np.zeros(int((ends[-1] + 0.5) * 16000), np.float32)
    return rng, words, ends, audio


def test_fast_model_streams_with_bounded_lag():
    import jiwer
    rng, words, ends, audio = setup()
    r = simulate_streaming(audio, 16000, _mock(words, ends, 16000, rng), step_s=0.5, cost_model=lambda d: 0.05 * d + 0.02)
    assert jiwer.wer(" ".join(words), r["text"]) < 0.12
    assert np.median(commit_lags(r["words"], r["commit_times"], words, ends)) < 2.5
    assert not r["behind"]


def test_slow_model_falls_behind():
    rng, words, ends, audio = setup(1)
    r = simulate_streaming(audio, 16000, _mock(words, ends, 16000, rng), step_s=0.5, cost_model=lambda d: 0.5 * d + 0.1)
    assert r["behind"]
