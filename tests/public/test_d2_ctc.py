import numpy as np
import pytest

from speech.ctc_lm import greedy_ctc

pytestmark = pytest.mark.d2
LABELS = ["", " ", "a", "b", "c"]


def onehot(ids):
    lp = np.full((len(ids), len(LABELS)), -10.0)
    lp[np.arange(len(ids)), ids] = 0.0
    return lp


def test_greedy_collapses_repeats_and_blanks():
    assert greedy_ctc(onehot([2, 2, 0, 2, 3, 3, 1, 4]), LABELS, blank=0) == "aab c"
    assert greedy_ctc(onehot([0, 0, 0]), LABELS, blank=0) == ""


def test_sweep_rows():
    pytest.importorskip("pyctcdecode")
    from speech.ctc_lm import sweep
    lps = [onehot([2, 0, 3, 1, 4])]
    rows = sweep(lps, ["ab c"], LABELS, 0, None, alphas=[0.5], betas=[0.0], beams=[1, 8])
    names = [r["decoder"] for r in rows]
    assert names.count("beam, no LM") == 2 and rows[0]["wer"] == 0
    assert all({"decoder", "alpha", "beta", "beam", "wer", "ms_per_utt"} <= set(r) for r in rows)
