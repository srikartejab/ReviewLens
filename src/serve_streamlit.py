import streamlit as st
import pandas as pd
import numpy as np
import json, os, subprocess
from typing import Dict
from .config import Config, LABELS
from .models.multilabel import MultiLabelSklearn
from .models.relevancy_ce import RelevancyModel
from .features import add_metadata_feats, parse_pics
from .policy import decision_layer, default_thresholds
from .image_utils import ImageTextRelevance

st.set_page_config(page_title="Review Quality & Relevancy", layout="wide")

@st.cache_resource
def load_models():
    cfg = Config()
    model_path = None
    if os.path.isdir(cfg.models_dir):
        cands = [p for p in os.listdir(cfg.models_dir) if p.endswith(".joblib")]
        if cands:
            cands.sort()
            model_path = os.path.join(cfg.models_dir, cands[-1])
    if model_path is None:
        st.warning("No trained multilabel model found. Using a fresh baseline trained on sample data.")
        from .train import prepare_training_data, to_multi_list
        reviews = pd.read_csv("data/sample_reviews.csv")
        reviews = add_metadata_feats(reviews)
        reviews = prepare_training_data(reviews)
        model = MultiLabelSklearn(labels=LABELS)
        model.fit(reviews["text"].tolist(), to_multi_list(reviews))
    else:
        model = MultiLabelSklearn.load(model_path)
    rel = RelevancyModel(); rel.load()
    clip = ImageTextRelevance(); clip.load()
    return model, rel, clip

model, rel_model, clip_model = load_models()
st.title("🔎 Review Quality & Relevancy")
st.caption("Hybrid rules + ML with priors (username, images, burst/dup, location, profanity, readability)")

uploaded = st.file_uploader("Upload reviews CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("Using bundled sample dataset. Upload your own to override.")
    df = pd.read_csv("data/sample_reviews.csv")

required = ["review_id","place_id","place_name","place_category","city","text"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Merge place lat/lon if user provided file with place info
if os.path.exists("data/sample_places.csv"):
    places_df = pd.read_csv("data/sample_places.csv")[["place_id","place_lat","place_lon","place_name","place_category","city","description"]]
    df = df.merge(places_df, on="place_id", how="left")

df = add_metadata_feats(df)

# Build place descriptions
place_descs = df.apply(lambda r: f"{r.get('place_name','')} — {r.get('place_category','')}, {r.get('city','')}. {r.get('description','')}" , axis=1).tolist()

# Text classifier
probs = model.predict_proba(df["text"].fillna("").tolist())
thresholds = default_thresholds()

# Relevancy
pairs = list(zip(df["text"].fillna("").tolist(), place_descs))
rel_scores = rel_model.score_pairs(pairs) if pairs else np.zeros(len(df))

# Image relevance
image_rel = np.array([np.nan]*len(df), dtype=float)
if clip_model is not None:
    for i, row in df.iterrows():
        urls = row.get("pics_list", [])
        desc = place_descs[i]
        score = clip_model.score(urls, desc) if urls and desc else None
        image_rel[i] = score if score is not None else np.nan

outputs = []
for i, row in df.iterrows():
    mp = {lab: float(probs[i, j]) for j, lab in enumerate(LABELS)}
    rel = float(rel_scores[i]) if i < len(rel_scores) else 0.0
    ir = None if np.isnan(image_rel[i]) else float(image_rel[i])
    flags = decision_layer(
        row["text"], mp, rel, thresholds,
        user_name_gibberish=float(row.get("user_name_gibberish", 0.0)),
        image_relevancy=ir,
        user_burst_24h=int(row.get("user_burst_24h", 0)),
        dup_across_places=bool(row.get("dup_across_places", False)),
        char_entropy=float(row.get("char_entropy", 0.0)),
        profanity_count=int(row.get("profanity_count", 0)),
        distance_km=float(row.get("distance_km", np.nan)) if not pd.isna(row.get("distance_km", np.nan)) else None,
        never_been_cues=bool(row.get("never_been_cues", False)),
        experiential_score=float(row.get("experiential_score", 0.0))
    )
    outputs.append({
        "review_id": row.get("review_id", i),
        "labels": flags,
        "confidences": mp,
        "relevancy": rel,
        "image_relevancy": ir,
        "user_name_gibberish": float(row.get("user_name_gibberish", 0.0)),
        "user_burst_24h": int(row.get("user_burst_24h", 0)),
        "dup_across_places": bool(row.get("dup_across_places", False)),
        "char_entropy": float(row.get("char_entropy", 0.0)),
        "profanity_count": int(row.get("profanity_count", 0)),
        "distance_km": None if pd.isna(row.get("distance_km", np.nan)) else float(row.get("distance_km", 0.0)),
        "rationale": "Rules + model + priors (username, images, burst/dup, location, profanity, readability)."
    })

out_df = pd.DataFrame(outputs)
st.subheader("Results")
with st.expander("Raw predictions JSON", expanded=False):
    st.json(json.loads(out_df.to_json(orient="records")))

st.dataframe(out_df)

st.subheader("Filters")
label = st.selectbox("Filter by label", ["(all)"] + LABELS)
conf_min = st.slider("Minimum confidence (any class)", 0.0, 1.0, 0.0, 0.01)
if label != "(all)":
    mask = out_df["labels"].apply(lambda lst: label in lst)
else:
    mask = pd.Series([True]*len(out_df))
conf_mask = out_df["confidences"].apply(lambda d: max(d.values()) >= conf_min)
view = out_df[mask & conf_mask]
st.write(f"{len(view)}/{len(out_df)} shown")
st.dataframe(view)

st.download_button("Download predictions CSV", view.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")

eval_path = "data/processed/gmrpl_converted.csv"
if st.button("Evaluate on GMR-PL dataset"):
    if os.path.exists(eval_path):
        with st.spinner("Running evaluation..."):
            res = subprocess.run(
                ["python", "scripts/06_eval_on_gmrpl.py", "--input", eval_path, "--model", "models"],
                capture_output=True, text=True,
            )
        st.text(res.stdout)
        if res.stderr:
            st.error(res.stderr)
    else:
        st.warning(f"{eval_path} not found. Run `make convert_gmrpl` first.")
