"""Classical acoustic-only baseline: MFCC features + one left-to-right GMM-HMM per word,
and a word-loop Viterbi decoder into which a bigram language model can be plugged.

Why per-WORD models rather than per-PHONE models?  A phone-level recogniser needs
(i) frame-level phone alignments or embedded training, (ii) a pronunciation lexicon and
(iii) a decoding graph (phone loop -> words). That is the Kaldi/HTK stack and does not fit
in a 25 h project.  Word models on a small vocabulary keep every concept (MFCC, GMM
emissions, HMM topology, Viterbi, LM integration, LM weight, insertion penalty) visible
in ~200 lines of numpy, which is the pedagogical point of the baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import librosa
import numpy as np
from hmmlearn.hmm import GMMHMM


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13, win_ms: float = 25, hop_ms: float = 10,
                  deltas: bool = True, cmvn: bool = False) -> np.ndarray:
    """(T, D) MFCC (+delta, +delta-delta). cmvn=True -> per-utterance normalisation.
    For connected-word decoding prefer a GLOBAL normaliser (see Normaliser) so that isolated
    training words and long test utterances share the same feature statistics."""
    n_fft = int(round(sr * win_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop,
                             win_length=n_fft, n_mels=40, fmax=sr / 2)
    feats = [m]
    if deltas and m.shape[1] >= 9:
        feats += [librosa.feature.delta(m, order=1), librosa.feature.delta(m, order=2)]
    elif deltas:  # very short segment: pad deltas with zeros
        feats += [np.zeros_like(m), np.zeros_like(m)]
    f = np.vstack(feats).T
    if cmvn:
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    return f.astype(np.float64)


class Normaliser:
    """Global mean/variance normalisation estimated on the training set."""

    def fit(self, feats: list[np.ndarray]):
        X = np.concatenate(feats)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, f: np.ndarray) -> np.ndarray:
        return (f - self.mu) / self.sd


def add_noise(y: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return y
    p_sig = np.mean(y ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    return y + rng.normal(0, np.sqrt(p_noise), size=y.shape)


# --------------------------------------------------------------------------- #
# Word models
# --------------------------------------------------------------------------- #
def left_to_right_hmm(n_states: int, n_mix: int, n_iter: int = 15, seed: int = 0) -> GMMHMM:
    """Bakis topology: each state loops or moves to the next one. EM keeps zeros at zero,
    so the topology is preserved during training."""
    m = GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="diag", n_iter=n_iter,
               random_state=seed, init_params="mcw", params="stmcw", min_covar=1e-3)
    m.startprob_ = np.eye(n_states)[0]
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i == n_states - 1:
            A[i, i] = 1.0
        else:
            A[i, i], A[i, i + 1] = 0.6, 0.4
    m.transmat_ = A
    return m


@dataclass
class WordModels:
    models: dict  # word -> GMMHMM
    n_states: int
    n_mix: int
    train_time_s: float

    @property
    def n_params(self) -> int:
        n = 0
        for m in self.models.values():
            S, M, D = m.n_components, m.n_mix, m.n_features
            n += S * M * (2 * D + 1) + S * S  # means, diag covars, weights, transitions
        return n


def uniform_segmentation_init(m: GMMHMM, seqs: list[np.ndarray], seed: int = 0):
    """HTK-style flat start: cut every training sequence into n_states equal chunks, fit one
    GMM per state on its chunk frames. Much more robust than k-means on pooled frames
    (which often leaves a state with an empty mixture component -> NaNs)."""
    from sklearn.mixture import GaussianMixture
    S, M = m.n_components, m.n_mix
    D = seqs[0].shape[1]
    per_state = [[] for _ in range(S)]
    for q in seqs:
        for s, chunk in enumerate(np.array_split(q, S)):
            if len(chunk):
                per_state[s].append(chunk)
    means = np.zeros((S, M, D))
    covars = np.ones((S, M, D))
    weights = np.full((S, M), 1.0 / M)
    for s in range(S):
        X = np.concatenate(per_state[s])
        k = min(M, max(1, len(X) // 10))
        g = GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, random_state=seed, max_iter=50).fit(X)
        means[s, :k], covars[s, :k], weights[s, :k] = g.means_, g.covariances_, g.weights_
        if k < M:  # pad unused components with perturbed copies
            for j in range(k, M):
                means[s, j] = g.means_[0] + 0.1 * np.random.default_rng(seed + j).standard_normal(D)
                covars[s, j] = g.covariances_[0]
            weights[s] = 1.0 / M
    m.means_, m.covars_, m.weights_ = means, covars, weights / weights.sum(1, keepdims=True)
    m.n_features = D


def train_word_models(train: dict[str, list[np.ndarray]], n_states: int = 5, n_mix: int = 2,
                      n_iter: int = 8, seed: int = 0) -> WordModels:
    t0 = time.perf_counter()
    models = {}
    for w, seqs in train.items():
        # silence is acoustically homogeneous: many mixtures collapse onto it (degenerate covariances)
        s, k = (n_states, n_mix) if w != "sil" else (3, min(n_mix, 2))
        m = left_to_right_hmm(s, k, n_iter, seed)
        m.init_params = ""
        uniform_segmentation_init(m, seqs, seed)
        X = np.concatenate(seqs)
        m.fit(X, lengths=[len(q) for q in seqs])
        if not np.all(np.isfinite(m.means_)) or not np.isfinite(m.transmat_).all():
            raise RuntimeError(f"training diverged for word {w}")
        models[w] = m
    return WordModels(models, n_states, n_mix, time.perf_counter() - t0)



def recognise_isolated(wm: WordModels, feats: np.ndarray, vocab: list[str]) -> str:
    """Return the word with the highest acoustic log-likelihood."""

    best_word = None
    best_score = -np.inf

    for word in vocab:
        model = wm.models[word]
        score = model.score(feats)

        if score > best_score:
            best_score = score
            best_word = word

    return best_word


# --------------------------------------------------------------------------- #
# Connected-word decoding: one big HMM = loop over all word HMMs.
# Transition (end of word u) -> (start of word v) carries  lm_weight * log P(v|u) + insertion penalty.
# Because the source state identifies u, a bigram needs no state expansion.
# --------------------------------------------------------------------------- #
class WordLoopDecoder:
    def __init__(self, wm: WordModels, vocab: list[str], bigram: dict | None = None,
                 lm_weight: float = 1.0, word_penalty: float = 0.0, exit_logp: float = np.log(0.5)):
        self.vocab = vocab
        self.wm = wm
        self.offsets, self.sizes = {}, {}
        o = 0
        for w in vocab:
            self.offsets[w], self.sizes[w] = o, wm.models[w].n_components
            o += self.sizes[w]
        self.S = o
        self.word_of_state = np.empty(self.S, dtype=object)
        for w in vocab:
            self.word_of_state[self.offsets[w]:self.offsets[w] + self.sizes[w]] = w
        V = len(vocab)
        if bigram is None:  # acoustic-only: uniform "LM"
            bigram = {u: {v: 1.0 / V for v in vocab} for u in ["<s>"] + vocab}
        self.bigram = bigram
        self.lm_weight, self.word_penalty, self.exit_logp = lm_weight, word_penalty, exit_logp
        self._build()

    def _lm(self, u, v):
        return self.lm_weight * np.log(max(self.bigram[u].get(v, 1e-12), 1e-12)) + self.word_penalty

    
    def _build(self):
        """Build the composite HMM transition matrix and start scores."""

        S = self.S

        # Log transition matrix: impossible transitions are -inf.
        self.logA = np.full((S, S), -np.inf)

        # Initial probabilities for the composite HMM.
        self.logpi = np.full(S, -np.inf)

        # Identify transitions that start a new word.
        self.entry = np.zeros((S, S), dtype=bool)

        # 1. Copy the internal transitions of each word HMM.
        for word in self.vocab:
            model = self.wm.models[word]
            offset = self.offsets[word]
            size = self.sizes[word]

            with np.errstate(divide="ignore"):
                log_trans = np.log(model.transmat_)

            self.logA[
                offset:offset + size,
                offset:offset + size
            ] = log_trans

            # Initial probability: P(word | <s>) and HMM startprob.
            with np.errstate(divide="ignore"):
                log_start = np.log(model.startprob_)

            self.logpi[offset:offset + size] = (
                self._lm("<s>", word) + log_start
            )

        # 2. Connect final states to the first state of each word.
        for prev_word in self.vocab:
            prev_end = (
                self.offsets[prev_word]
                + self.sizes[prev_word] - 1
            )

            for next_word in self.vocab:
                next_start = self.offsets[next_word]

                transition_score = (
                    self.exit_logp
                    + self._lm(prev_word, next_word)
                )

                # Keep the best transition if two edges share a pair.
                self.logA[prev_end, next_start] = max(
                    self.logA[prev_end, next_start],
                    transition_score
                )

                self.entry[prev_end, next_start] = True

    def emissions(self, feats: np.ndarray) -> np.ndarray:
        """(T, S) log b_s(o_t) concatenated over word models."""
        cols = [self.wm.models[w]._compute_log_likelihood(feats) for w in self.vocab]
        return np.hstack(cols)

    def decode(self, feats: np.ndarray, drop: tuple = ("sil",)) -> list[str]:
        return self.decode_emissions(self.emissions(feats), drop)

    
    def decode_emissions(self, B: np.ndarray,
                        drop: tuple = ("sil",)) -> list[str]:
        """Decode emissions using log-domain Viterbi."""

        T, S = B.shape

        if T == 0:
            return []

        # Viterbi scores and backpointers.
        delta = np.full((T, S), -np.inf)
        psi = np.zeros((T, S), dtype=int)

        # 1. Initialisation.
        delta[0] = self.logpi + B[0]

        # 2. Recursion.
        for t in range(1, T):
            scores = delta[t - 1][:, None] + self.logA

            psi[t] = np.argmax(scores, axis=0)
            delta[t] = (
                scores[psi[t], np.arange(S)] + B[t]
            )

        # 3. Terminate in a word-final state.
        final_states = [
            self.offsets[word] + self.sizes[word] - 1
            for word in self.vocab
        ]

        last_state = max(
            final_states,
            key=lambda s: delta[-1, s]
        )

        # 4. Backtracking.
        path = np.zeros(T, dtype=int)
        path[-1] = last_state

        for t in range(T - 1, 0, -1):
            path[t - 1] = psi[t, path[t]]

        # 5. Recover words from word-entry transitions.
        words = [self.word_of_state[path[0]]]

        for t in range(1, T):
            prev_state = path[t - 1]
            curr_state = path[t]

            if self.entry[prev_state, curr_state]:
                words.append(self.word_of_state[curr_state])

        # 6. Remove silence tokens.
        return [word for word in words if word not in drop]


# --------------------------------------------------------------------------- #
# Bigram LM with add-k smoothing
# --------------------------------------------------------------------------- #

def train_bigram(sentences: list[list[str]], vocab: list[str], k: float = 0.1,
                 sil_token: str | None = "sil", p_sil: float = 0.15) -> dict:
    """Train an add-k bigram LM with optional silence transitions."""

    from collections import Counter

    words = [w for w in dict.fromkeys(vocab) if w != sil_token]
    counts = {}

    # Count transitions, including the beginning of each sentence.
    for sentence in sentences:
        prev = "<s>"

        for word in sentence:
            if word not in words:
                continue

            if prev not in counts:
                counts[prev] = Counter()

            counts[prev][word] += 1
            prev = word

    # Build smoothed probability distributions.
    bigram = {}
    V = len(words)

    for prev in ["<s>"] + words:
        row_counts = counts.get(prev, Counter())
        total = sum(row_counts.values())

        row = {
            word: (row_counts[word] + k) / (total + k * V)
            for word in words
        }

        if sil_token is not None:
            row = {word: prob * (1.0 - p_sil)
                   for word, prob in row.items()}
            row[sil_token] = p_sil

        bigram[prev] = row

    # After silence, resume with a smoothed start distribution.
    if sil_token is not None:
        bigram[sil_token] = {
            word: (1.0 - p_sil) * (
                counts.get("<s>", Counter())[word] + k
            ) / (sum(counts.get("<s>", Counter()).values()) + k * V)
            for word in words
        }
        bigram[sil_token][sil_token] = p_sil

    return bigram
