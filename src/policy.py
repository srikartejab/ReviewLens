from typing import Dict, List, Optional
from .config import LABELS, Config

def decision_layer(text: str,
                   model_probs: Dict[str, float],
                   relevancy: float,
                   thresholds: Dict[str, float],
                   rel_thresh: float = 0.3,
                   user_name_gibberish: Optional[float] = None,
                   image_relevancy: Optional[float] = None,
                   user_burst_24h: Optional[int] = None,
                   dup_across_places: Optional[bool] = None,
                   char_entropy: Optional[float] = None,
                   profanity_count: Optional[int] = None,
                   distance_km: Optional[float] = None,
                   never_been_cues: Optional[bool] = None,
                   experiential_score: Optional[float] = None) -> List[str]:

    cfg = Config()
    pri = cfg.priors
    opts = cfg.options

    flags: List[str] = []

    # === Advertisement/Promo ===
    if model_probs.get("Advertisement/Promo", 0) >= thresholds.get("Advertisement/Promo", 0.5):
        flags.append("Advertisement/Promo")

    # === Irrelevant ===
    irr = False
    if relevancy < rel_thresh and model_probs.get("Irrelevant Content", 0) >= thresholds.get("Irrelevant Content", 0.5):
        irr = True
    if opts.get("enable_image_relevance", True) and (image_relevancy is not None):
        if image_relevancy < float(pri.get("image_low_to_irrelevant", 0.2)):
            irr = True
    if irr:
        flags.append("Irrelevant Content")

    # === Rant (Likely Non-Visitor) ===
    nonvis = False
    if model_probs.get("Rant (Likely Non-Visitor)", 0) >= thresholds.get("Rant (Likely Non-Visitor)", 0.5):
        nonvis = True
    if opts.get("enable_device_location_prior", True) and distance_km is not None:
        far_km = float(cfg.device_location.get("far_km", 100))
        if distance_km > far_km and (never_been_cues or (experiential_score is not None and experiential_score < 0.2)):
            nonvis = True
    if nonvis:
        flags.append("Rant (Likely Non-Visitor)")

    # === Spam/Low-quality ===
    spam_prob = model_probs.get("Spam/Low-quality", 0.0)

    # Username
    if opts.get("enable_username_prior", True) and (user_name_gibberish is not None):
        if user_name_gibberish >= float(pri.get("username_gibberish_to_spam", 0.8)):
            spam_prob = max(spam_prob, 0.6)

    # Image mismatch
    if opts.get("enable_image_relevance", True) and (image_relevancy is not None) and image_relevancy < float(pri.get("image_irrelevant_to_spam", 0.1)):
        spam_prob = max(spam_prob, 0.7)

    # Burst / dup
    if opts.get("enable_dup_burst_prior", True):
        if (user_burst_24h is not None) and (user_burst_24h >= int(pri.get("burst_24h_to_spam", 5))):
            spam_prob = max(spam_prob, 0.6)
        if dup_across_places:
            spam_prob = max(spam_prob, 0.7)

    # Readability extremes
    if opts.get("enable_readability_prior", True):
        if (char_entropy is not None and char_entropy < float(pri.get("low_entropy_to_spam", 2.5))):
            spam_prob = max(spam_prob, 0.6)

    # Profanity (mild)
    if opts.get("enable_profanity_prior", True) and profanity_count is not None and profanity_count > 0:
        if (experiential_score is None or experiential_score < 0.2) or (relevancy < 0.3):
            spam_prob = max(spam_prob, 0.55)

    if spam_prob >= thresholds.get("Spam/Low-quality", 0.5):
        flags.append("Spam/Low-quality")

    # Preserve label order
    out = [f for f in LABELS if f in flags]
    return out

def default_thresholds() -> Dict[str, float]:
    return {k: 0.5 for k in LABELS}
