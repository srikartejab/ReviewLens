from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import Config

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class RelevancyModel:
    def __init__(self) -> None:
        cfg = Config()
        self.model_name = cfg.models.get("relevancy_cross_encoder", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.model = None
        self.backend = "ce"
        self._warned = False

    def load(self) -> None:
        if CrossEncoder is None:
            self._fallback("sentence-transformers not available")
            return
        try:
            self.model = CrossEncoder(self.model_name)
            self.backend = "ce"
        except Exception as exc:
            self._fallback(f"failed to load cross-encoder: {exc}")

    def _fallback(self, reason: str) -> None:
        self.model = None
        self.backend = "tfidf"
        if not self._warned:
            print(f"Warning: relevancy model fallback to TF-IDF ({reason}).")
            self._warned = True

    def score_pairs(self, pairs: Iterable[Tuple[str, str]]) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.array([], dtype=float)
        if self.model is None and self.backend == "ce":
            self.load()
        if self.model is not None and self.backend == "ce":
            scores = np.asarray(self.model.predict(pairs), dtype=float)
            if scores.min() < 0.0 or scores.max() > 1.0:
                scores = 1.0 / (1.0 + np.exp(-scores))
            return scores
        return self._score_tfidf(pairs)

    def _score_tfidf(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        texts = [a or "" for a, _ in pairs]
        descs = [b or "" for _, b in pairs]
        corpus = texts + descs
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        mat = vec.fit_transform(corpus)
        n = len(pairs)
        a = mat[:n]
        b = mat[n:]

        # Cosine similarity for each aligned pair.
        num = (a.multiply(b)).sum(axis=1).A1
        a_norm = np.sqrt(a.multiply(a).sum(axis=1)).A1
        b_norm = np.sqrt(b.multiply(b).sum(axis=1)).A1
        denom = a_norm * b_norm
        sims = np.zeros_like(num, dtype=float)
        mask = denom > 0
        sims[mask] = num[mask] / denom[mask]
        sims = np.clip(sims, 0.0, 1.0)
        return sims
