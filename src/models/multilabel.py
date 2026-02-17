from __future__ import annotations

from typing import Iterable, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from ..config import LABELS


class MultiLabelSklearn:
    def __init__(
        self,
        labels: Optional[List[str]] = None,
        max_features: int = 20000,
        ngram_range: tuple = (1, 2),
    ) -> None:
        self.labels = labels or LABELS
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.binarizer = MultiLabelBinarizer(classes=self.labels)
        base = LogisticRegression(max_iter=1000, solver="liblinear")
        self.clf = OneVsRestClassifier(base)

    def fit(self, texts: Iterable[str], labels_list: Iterable[List[str]]) -> "MultiLabelSklearn":
        texts = list(texts)
        labels_list = list(labels_list)
        X = self.vectorizer.fit_transform(texts)
        y = self.binarizer.fit_transform(labels_list)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        X = self.vectorizer.transform(list(texts))
        probs = self.clf.predict_proba(X)
        return np.asarray(probs, dtype=float)

    def predict(self, texts: Iterable[str], threshold: float = 0.5) -> List[List[str]]:
        probs = self.predict_proba(texts)
        out: List[List[str]] = []
        for row in probs:
            labs = [self.labels[i] for i, p in enumerate(row) if p >= threshold]
            out.append(labs)
        return out

    def save(self, path: str) -> None:
        payload = {
            "labels": self.labels,
            "vectorizer": self.vectorizer,
            "binarizer": self.binarizer,
            "clf": self.clf,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "MultiLabelSklearn":
        payload = joblib.load(path)
        obj = cls(labels=payload.get("labels") or LABELS)
        obj.vectorizer = payload["vectorizer"]
        obj.binarizer = payload["binarizer"]
        obj.clf = payload["clf"]
        return obj
