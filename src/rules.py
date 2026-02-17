
from typing import Dict, List
from .features import PROMO_RE, NEVER_BEEN_RE
from .utils import extract_urls, repeat_char_ratio

def rule_scores(text: str) -> Dict[str, float]:
    text = text or ""
    scores = {}
    scores["has_url"] = 1.0 if extract_urls(text) else 0.0
    scores["promo_terms"] = 1.0 if PROMO_RE.search(text) else 0.0
    scores["never_been"] = 1.0 if NEVER_BEEN_RE.search(text) else 0.0
    scores["gibberish"] = repeat_char_ratio(text)
    return scores

def weak_labels(text: str) -> List[str]:
    s = rule_scores(text)
    labs: List[str] = []
    if s["has_url"] and s["promo_terms"]:
        labs.append("Advertisement/Promo")
    if s["never_been"]:
        labs.append("Rant (Likely Non-Visitor)")
    if s["gibberish"] > 0.0:
        labs.append("Spam/Low-quality")
    return labs
