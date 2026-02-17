
import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
from dataclasses import dataclass
from typing import Tuple, List
from .utils import normalize_text
from .config import Config

DetectorFactory.seed = 0

@dataclass
class SplitConfig:
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42

def load_samples() -> Tuple[pd.DataFrame, pd.DataFrame]:
    reviews = pd.read_csv("data/sample_reviews.csv")
    places = pd.read_csv("data/sample_places.csv")
    return reviews, places

def _maybe_translate(texts: List[str]) -> List[str]:
    cfg = Config()
    if not cfg.translate_non_en:
        return texts
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "Helsinki-NLP/opus-mt-mul-en"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        outs = []
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                outs.append(t); continue
            try:
                lang = detect(t)
            except Exception:
                lang = "unknown"
            if lang == "en" or lang == "unknown":
                outs.append(t)
            else:
                inpt = tok(t, return_tensors="pt", truncation=True, max_length=512)
                gen = model.generate(**inpt, max_length=512)
                outs.append(tok.decode(gen[0], skip_special_tokens=True))
        return outs
    except Exception:
        return texts

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(normalize_text)
    langs = []
    for t in df["text"].tolist():
        try:
            langs.append(detect(t) if isinstance(t, str) and t.strip() else "unknown")
        except Exception:
            langs.append("unknown")
    df["lang"] = langs
    df["text_model"] = _maybe_translate(df["text"].tolist())
    return df

def business_disjoint_split(df: pd.DataFrame, cfg: SplitConfig):
    place_ids = df["place_id"].dropna().unique()
    rng = np.random.default_rng(cfg.random_state)
    rng.shuffle(place_ids)
    n = len(place_ids)
    n_test = max(1, int(n * cfg.test_size))
    n_val = max(1, int(n * cfg.val_size))
    test_ids = set(place_ids[:n_test])
    val_ids = set(place_ids[n_test:n_test+n_val])
    train_ids = set(place_ids[n_test+n_val:])
    train = df[df["place_id"].isin(train_ids)].reset_index(drop=True)
    val = df[df["place_id"].isin(val_ids)].reset_index(drop=True)
    test = df[df["place_id"].isin(test_ids)].reset_index(drop=True)
    return train, val, test
