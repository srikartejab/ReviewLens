"""
Evaluate the model's predictions vs. the GMR-PL fake/real labels.

We map our multi-label outputs to "fake" in two ways:
  A) spam_only:          predict fake if P(Spam/Low-quality) >= 0.5
  B) any_violation:      predict fake if max(P(Ad), P(Irrelevant), P(Rant), P(Spam)) >= 0.5

Usage:
  python scripts/06_eval_on_gmrpl.py --input data/processed/gmrpl_converted.csv --model models
"""
import argparse, os, pandas as pd, numpy as np, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from glob import glob

from src.config import LABELS
from src.models.multilabel import MultiLabelSklearn
from src.data_prep import clean_reviews
from src.features import add_metadata_feats

def load_latest_model(models_dir: str) -> str:
    cands = sorted(glob(os.path.join(models_dir, "*.joblib")))
    if not cands:
        raise FileNotFoundError(f"No model .joblib found under {models_dir}")
    return cands[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to gmrpl_converted.csv")
    ap.add_argument("--model", default="models", help="Directory containing trained .joblib model(s)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "is_fake" not in df.columns:
        raise ValueError("Expected 'is_fake' column in the converted GMR-PL CSV.")

    # Prepare text like the app does (clean, features)
    df = clean_reviews(df)
    df = add_metadata_feats(df)

    # Load model
    model_path = load_latest_model(args.model)
    model = MultiLabelSklearn.load(model_path)

    # Predict probabilities
    probs = model.predict_proba(df["text"].fillna("").tolist())
    idx = {lab:i for i,lab in enumerate(LABELS)}
    p_ad = probs[:, idx["Advertisement/Promo"]]
    p_irr = probs[:, idx["Irrelevant Content"]]
    p_rant = probs[:, idx["Rant (Likely Non-Visitor)"]]
    p_spam = probs[:, idx["Spam/Low-quality"]]

    y_true = df["is_fake"].astype(int).values

    # Strategy A: Spam only
    y_pred_a = (p_spam >= 0.5).astype(int)

    # Strategy B: Any violation
    y_pred_b = (np.maximum.reduce([p_ad, p_irr, p_rant, p_spam]) >= 0.5).astype(int)

    def summarize(name, y_pred):
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        return {"name": name, "accuracy": acc, "precision": p, "recall": r, "f1": f1}

    res_a = summarize("spam_only", y_pred_a)
    res_b = summarize("any_violation", y_pred_b)

    print("=== Evaluation on GMR-PL ===")
    print("Model:", os.path.basename(model_path))
    for res in [res_a, res_b]:
        print(f"{res['name']}: acc={res['accuracy']:.3f}  p={res['precision']:.3f}  r={res['recall']:.3f}  f1={res['f1']:.3f}")

if __name__ == "__main__":
    main()
