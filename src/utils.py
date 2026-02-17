import re, unicodedata, html, math, pathlib
from typing import List, Set

URL_RE = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_RE = re.compile(r'<.*?>')

def normalize_text(s: str) -> str:
    s = s or ""
    s = html.unescape(s)
    s = re.sub(HTML_TAG_RE, ' ', s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    return re.sub(r"\s+", " ", s)

def extract_urls(s: str) -> List[str]:
    return URL_RE.findall(s or "")

def caps_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters: return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters)

def emoji_count(s: str) -> int:
    return sum(1 for c in s if ord(c) > 10000)

def repeat_char_ratio(s: str) -> float:
    return 1.0 if re.search(r'(.)\1{4,}', s or "") else 0.0

def char_entropy(s: str) -> float:
    if not s: return 0.0
    from collections import Counter
    c = Counter(s)
    total = sum(c.values())
    H = 0.0
    for v in c.values():
        p = v/total
        H -= p*math.log2(p)
    return H

def load_profanity_words() -> Set[str]:
    try:
        words = (pathlib.Path(__file__).resolve().parents[1] / "resources" / "profanity.txt").read_text(encoding="utf-8").splitlines()
        return set(w.strip().lower() for w in words if w.strip())
    except Exception:
        return {"damn","hell","crap","stupid"}

def profanity_count(s: str) -> int:
    if not s: return 0
    toks = re.findall(r"[A-Za-z']+", s.lower())
    bad = load_profanity_words()
    return sum(1 for t in toks if t in bad)

def experiential_score(s: str) -> float:
    if not s: return 0.0
    cues = [
        r"\bwe ordered\b", r"\bi ordered\b", r"\bwe visited\b", r"\bwe went\b",
        r"\bthe waiter\b", r"\btable\b", r"\breceipt\b", r"\bmenu\b",
        r"\bappointment\b", r"\bfront desk\b", r"\bparking\b", r"\bline\b",
        r"\bcheck(ing)? in\b", r"\bseated\b", r"\bord(er)ed\b"
    ]
    hits = sum(1 for pat in cues if re.search(pat, s.lower()))
    return min(1.0, hits/4.0)

def is_gibberish_name(name: str) -> float:
    if not name or not isinstance(name, str): return 1.0
    n = name.strip()
    if len(n) <= 1: return 1.0
    core = ''.join(ch for ch in n if ch.isalpha())
    if not core: return 1.0
    vowels = sum(1 for c in core.lower() if c in 'aeiou')
    vowel_ratio = vowels / max(1, len(core))
    digits = sum(1 for c in n if c.isdigit())
    nonalpha = sum(1 for c in n if not c.isalnum() and not c.isspace())
    long_consonant_run = 1 if re.search(r'(?i)[bcdfghjklmnpqrstvwxyz]{5,}', core) else 0
    score = 0.0
    score += 0.4 if vowel_ratio < 0.2 else 0.0
    score += 0.2 if digits / max(1, len(n)) > 0.3 else 0.0
    score += 0.2 if nonalpha / max(1, len(n)) > 0.2 else 0.0
    score += 0.2 * long_consonant_run
    return min(1.0, max(0.0, score))
