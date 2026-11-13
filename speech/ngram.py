"""N-gram language models measured ON THEIR OWN (before any decoding): perplexity vs order N,
training-set size and smoothing. Day 1 morning.

    lm = NGramLM(order=3, smoothing="interpolated").fit(train_sentences, vocab)
    lm.perplexity(test_sentences)

The lesson: higher N captures longer dependencies but sees each history less often (sparsity), so the
best N depends on the amount of text and on smoothing. The decoding graph also grows with N (lecture).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

BOS, EOS = "<s>", "</s>"


class NGramLM:
    def __init__(self, order: int = 2, smoothing: str = "add-k", k: float = 0.1, lambdas: list | None = None):
        assert order >= 1 and smoothing in ("add-k", "interpolated")
        self.order, self.smoothing, self.k = order, smoothing, k
        # interpolation weights for orders N, N-1, ..., 1 (sum to 1)
        self.lambdas = lambdas or [0.6, 0.3, 0.1, 0.05, 0.05][:order]
        self.lambdas = [w / sum(self.lambdas) for w in self.lambdas]

    
    def fit(self, sentences: list[list[str]], vocab: list[str]):
        """Count n-grams of every order 1..N."""

        self.vocab = list(vocab) + [EOS]

        # Initialise counters for each order.
        self.counts = {
            n: Counter() for n in range(1, self.order + 1)
        }
        self.context = {
            n: Counter() for n in range(1, self.order + 1)
        }

        for sentence in sentences:
            # Add beginning and end tokens.
            padded = [BOS] * (self.order - 1) + list(sentence) + [EOS]

            # Iterate only over predicted tokens (words + EOS).
            for i in range(self.order - 1, len(padded)):
                for n in range(1, self.order + 1):

                    # Current n-gram ending at position i.
                    ngram = tuple(padded[i - n + 1:i + 1])

                    # History of the n-gram.
                    history = ngram[:-1]

                    self.counts[n][ngram] += 1
                    self.context[n][history] += 1

        return self


    
    def prob(self, word: str, history: list[str]) -> float:
        """P(word | last N-1 tokens of history)."""

        V = len(self.vocab)

        # Keep the last N-1 tokens and pad missing history with BOS.
        hist = list(history[-(self.order - 1):]) if self.order > 1 else []
        hist = [BOS] * (self.order - 1 - len(hist)) + hist

        def add_k_prob(n):
            # Use the last n-1 history tokens.
            context = tuple(hist[-(n - 1):]) if n > 1 else ()

            ngram = context + (word,)

            count_ngram = self.counts[n][ngram]
            count_context = self.context[n][context]

            return (
                (count_ngram + self.k)
                / (count_context + self.k * V)
            )

        if self.smoothing == "add-k":
            return add_k_prob(self.order)

        # Jelinek-Mercer interpolation: N, N-1, ..., 1
        return sum(
            weight * add_k_prob(n)
            for n, weight in zip(
                range(self.order, 0, -1),
                self.lambdas
            )
        )

    def sentence_logprob(self, sentence: list[str]) -> float:
        hist, lp = [], 0.0
        for w in list(sentence) + [EOS]:
            lp += math.log(self.prob(w, hist))
            hist.append(w)
        return lp


    def perplexity(self, sentences: list[list[str]]) -> float:
        """exp(-(sum of log P) / number of predicted tokens), EOS included."""

        total_logprob = 0.0
        total_tokens = 0

        for sentence in sentences:
            total_logprob += self.sentence_logprob(sentence)

            # Each sentence predicts its words and one EOS token.
            total_tokens += len(sentence) + 1

        if total_tokens == 0:
            return float("inf")

        return math.exp(-total_logprob / total_tokens)

    @property
    def n_entries(self) -> int:
        return sum(len(c) for c in self.counts.values())

    def as_bigram(self, vocab: list[str]) -> dict:
        """Bigram table {u: {v: P(v|u)}} for the word-loop decoder (uses this model's own smoothing)."""
        rows = {}
        for u in [BOS] + list(vocab):
            p = {v: self.prob(v, [u]) for v in vocab}
            z = sum(p.values())
            rows[u] = {v: x / z for v, x in p.items()}
        return rows


def second_order_chain(rng, n_symbols: int = 10, concentration: float = 0.3):
    """A digit 'grammar' where the next digit depends on the previous TWO: a trigram has something to
    learn that a bigram cannot. Returns P[a, b, c] = P(c | a, b), with index n_symbols = start."""
    P = rng.dirichlet(np.full(n_symbols, concentration), size=(n_symbols + 1, n_symbols + 1))
    return P


def sample_second_order(rng, P, n: int, lmin: int = 4, lmax: int = 9) -> list[list[str]]:
    S = P.shape[-1]
    out = []
    for _ in range(n):
        a = b = S  # start symbols
        s = []
        for _ in range(int(rng.integers(lmin, lmax + 1))):
            c = int(rng.choice(S, p=P[a, b]))
            s.append(str(c))
            a, b = b, c
        out.append(s)
    return out
