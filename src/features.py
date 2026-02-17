import re, json
import numpy as np, pandas as pd
from typing import List
from .utils import extract_urls, caps_ratio, emoji_count, repeat_char_ratio, is_gibberish_name, char_entropy, profanity_count, experiential_score, normalize_text
from .geo import haversine_km

PROMO_RE = re.compile(r"(?:discount|promo|coupon|deal|sale|%\s*off|dm\s*me|buy now)", re.I)
NEVER_BEEN_RE = re.compile(r"\b(?:never been|haven't been|didn't visit|did not go|have not gone)\b", re.I)

def parse_pics(val) -> List[str]:
    if isinstance(val, str):
        val = val.strip()
        if not val: return []
        try:
            arr = json.loads(val)
            if isinstance(arr, list): return [x for x in arr if isinstance(x, str)]
            return [val]
        except Exception:
            if ";" in val: return [x.strip() for x in val.split(";") if x.strip()]
            return [val]
    if isinstance(val, list):
        return [x for x in val if isinstance(x, str)]
    return []

def _parse_dt(s):
    # Handles ISO strings or epoch milliseconds
    try:
        # numeric epoch ms
        if isinstance(s, (int, float)):
            ts = pd.to_datetime(int(s), unit='ms', utc=True)
            return ts.tz_convert(None)
        s = str(s)
        if s.isdigit():
            # assume ms
            ts = pd.to_datetime(int(s), unit='ms', utc=True)
            return ts.tz_convert(None)
        ts = pd.to_datetime(s, utc=True, errors='coerce')
        if pd.isna(ts):
            return pd.NaT
        return ts.tz_convert(None)
    except Exception:
        return pd.NaT

def add_metadata_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    texts = df["text"].fillna("")
    df["text_norm"] = texts.apply(lambda s: normalize_text(str(s)).lower())
    df["char_len"] = texts.apply(len)
    df["word_len"] = texts.apply(lambda s: len(s.split()))
    df["url_count"] = texts.apply(lambda s: len(extract_urls(s)))
    df["has_url"] = df["url_count"] > 0
    df["caps_ratio"] = texts.apply(caps_ratio)
    df["emoji_count"] = texts.apply(emoji_count)
    df["repeat_ratio"] = texts.apply(repeat_char_ratio)
    df["promo_terms"] = texts.str.contains(PROMO_RE)
    df["never_been_cues"] = texts.str.contains(NEVER_BEEN_RE)
    df["nonword_ratio"] = texts.apply(lambda s: sum(1 for c in s if not (c.isalnum() or c.isspace())) / max(1, len(s)))
    df["char_entropy"] = texts.apply(char_entropy)
    df["profanity_count"] = texts.apply(profanity_count)
    df["experiential_score"] = texts.apply(experiential_score)

    # Username
    if "user_name" in df.columns:
        df["user_name_gibberish"] = df["user_name"].fillna("").apply(is_gibberish_name)
        df["user_name_suspicious"] = df["user_name_gibberish"] >= 0.6
    else:
        df["user_name_gibberish"] = 0.0
        df["user_name_suspicious"] = False

    # Pics parsing
    df["pics_list"] = df.get("pics", pd.Series([""]*len(df))).apply(parse_pics)
    df["image_count"] = df["pics_list"].apply(len)

    # Time parsing for burst detection
    df["dt"] = df.get("created_at", pd.Series([None]*len(df))).apply(_parse_dt)

    # Burstiness within 24h for same user
    df["user_burst_24h"] = 0
    if "user_id" in df.columns:
        for uid, g in df.groupby("user_id", dropna=False):
            if g["dt"].notna().sum() == 0:
                df.loc[g.index, "user_burst_24h"] = 0
                continue
            g = g.sort_values("dt")
            times = g["dt"].values
            counts = []
            for i, t in enumerate(times):
                if pd.isna(t):
                    counts.append(1); continue
                lo = t - pd.Timedelta(hours=24)
                hi = t
                cnt = ((g["dt"] >= lo) & (g["dt"] <= hi)).sum()
                counts.append(int(cnt))
            df.loc[g.index, "user_burst_24h"] = counts

    # Duplicate across places: same user + same normalized text but different place_id
    df["dup_across_places"] = False
    if "user_id" in df.columns and "place_id" in df.columns:
        grp = df.groupby(["user_id","text_norm"])["place_id"].nunique().reset_index(name="n_places")
        join = df.merge(grp, on=["user_id","text_norm"], how="left")
        df["dup_across_places"] = join["n_places"].fillna(0) > 1

    # Distance if user device coords & place coords present
    if {"user_lat","user_lon","place_lat","place_lon"}.issubset(df.columns):
        df["distance_km"] = df.apply(lambda r: haversine_km(r["user_lat"], r["user_lon"], r["place_lat"], r["place_lon"]), axis=1)
    else:
        df["distance_km"] = np.nan

    return df

def pack_meta_tokens(row: pd.Series) -> str:
    return f"meta: url={int(row['has_url'])} len={int(row['char_len'])} caps={row['caps_ratio']:.2f} emoji={int(row['emoji_count'])} uname_gib={row.get('user_name_gibberish',0):.2f} imgs={int(row.get('image_count',0))} burst24={int(row.get('user_burst_24h',0))} dup={int(bool(row.get('dup_across_places',False)))} ent={row.get('char_entropy',0):.2f} prof={int(row.get('profanity_count',0))} dist={row.get('distance_km',float('nan'))}"
