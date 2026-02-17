
import argparse, pandas as pd
from .config import LABELS
from .models.multilabel import MultiLabelSklearn
from .models.relevancy_ce import RelevancyModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()

    model = MultiLabelSklearn.load(args.model_path)
    df = pd.read_csv("data/sample_reviews.csv")
    probs = model.predict_proba(df["text"].tolist())

    print("Per-label avg prob:")
    for i, lab in enumerate(LABELS):
        print(lab, f"{probs[:,i].mean():.3f}")

    places = pd.read_csv("data/sample_places.csv").set_index("place_id")
    rel = RelevancyModel(); rel.load()
    pairs = []
    for _, r in df.iterrows():
        if r["place_id"] in places.index:
            p = places.loc[r["place_id"]]
            desc = f"{p['place_name']} — {p['place_category']}, {p['city']}. {p['description']}"
            pairs.append((r["text"], desc))
    rs = rel.score_pairs(pairs)
    print("Avg relevancy (demo):", float(rs.mean()) if len(rs) else "n/a")

if __name__ == "__main__":
    main()
