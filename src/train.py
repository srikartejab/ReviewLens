
import argparse, os, json
import pandas as pd
from typing import List
from .config import Config, LABELS
from .data_prep import load_samples, clean_reviews, business_disjoint_split
from .features import add_metadata_feats
from .rules import weak_labels
from .models.multilabel import MultiLabelSklearn
from .models.relevancy_ce import RelevancyModel
from .policy import default_thresholds

def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weak_labels"] = df["text"].apply(weak_labels)
    return df

def to_multi_list(df: pd.DataFrame) -> List[List[str]]:
    return [labs if isinstance(labs, list) else [] for labs in df["weak_labels"].tolist()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sklearn","hf"], default="sklearn")
    ap.add_argument("--run_name", default="run")
    args = ap.parse_args()

    cfg = Config()
    reviews, places = load_samples()
    reviews = clean_reviews(reviews)
    reviews = add_metadata_feats(reviews)

    tr, val, te = business_disjoint_split(reviews, cfg=type("tmp",(object,),{"test_size":0.2,"val_size":0.2,"random_state":42})())
    tr = prepare_training_data(tr)
    val = prepare_training_data(val)

    if args.backend == "sklearn":
        clf = MultiLabelSklearn(labels=LABELS)
        clf.fit(tr["text"].tolist(), to_multi_list(tr))
        os.makedirs(cfg.models_dir, exist_ok=True)
        clf_path = os.path.join(cfg.models_dir, f"multilabel_{args.run_name}.joblib")
        clf.save(clf_path)
        print(f"Saved multilabel model to {clf_path}")
    else:
        print("HF backend not implemented in this scaffold. Use sklearn baseline or extend models.")

    rel = RelevancyModel()
    rel.load()
    place_map = places.set_index("place_id")[["place_name","place_category","city","description"]].to_dict(orient="index")
    pairs = []
    for _, row in tr.iloc[:10].iterrows():
        pl = place_map.get(row["place_id"], {})
        place_desc = f"{pl.get('place_name','')} — {pl.get('place_category','')}, {pl.get('city','')}. {pl.get('description','')}"
        pairs.append((row["text"], place_desc))
    if pairs:
        scores = rel.score_pairs(pairs)
        os.makedirs(cfg.outputs_dir, exist_ok=True)
        with open(os.path.join(cfg.outputs_dir, f"relevancy_train_samples_{args.run_name}.json"), "w", encoding="utf-8") as f:
            json.dump({"scores": [float(x) for x in scores]}, f, indent=2)
        print("Relevancy scores sample saved.")

    with open(os.path.join(cfg.outputs_dir, f"thresholds_{args.run_name}.json"), "w", encoding="utf-8") as f:
        json.dump(default_thresholds(), f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
