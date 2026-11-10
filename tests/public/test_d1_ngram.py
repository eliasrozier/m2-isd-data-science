import numpy as np
import pytest

from speech.ngram import NGramLM, sample_second_order, second_order_chain

pytestmark = pytest.mark.d1
V = ["a", "b", "c"]
CORPUS = [["a", "b", "c"], ["a", "c"], ["b", "b", "a"]]


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("sm", ["add-k", "interpolated"])
def test_distributions_sum_to_one(order, sm):
    lm = NGramLM(order, sm).fit(CORPUS, V)
    for hist in ([], ["a"], ["b", "b"], ["c", "a"]):
        assert sum(lm.prob(w, hist) for w in V + ["</s>"]) == pytest.approx(1.0, abs=1e-9)


def test_perplexity_known_value():
    lm = NGramLM(1, "add-k", k=0.0).fit([["a", "b"]] * 10, ["a", "b"])  # P(a)=P(b)=P(</s>)=1/3
    assert lm.perplexity([["a", "b"]]) == pytest.approx(3.0)


def test_trigram_wins_on_a_second_order_source():
    rng = np.random.default_rng(0)
    P = second_order_chain(rng)
    train, test = sample_second_order(rng, P, 5000), sample_second_order(rng, P, 500)
    pp = {n: NGramLM(n, "add-k").fit(train, [str(i) for i in range(10)]).perplexity(test) for n in (1, 2, 3)}
    assert pp[3] < pp[2] < pp[1]
