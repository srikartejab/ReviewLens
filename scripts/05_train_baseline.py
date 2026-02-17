
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from src.config import Config, LABELS
from src.features import add_metadata_feats
from src.rules import weak_labels
from src.models.multilabel import MultiLabelSklearn

def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weak_labels"] = df["text"].apply(weak_labels)
    return df

def to_multi_list(df: pd.DataFrame):
    return [labs if isinstance(labs, list) else [] for labs in df["weak_labels"].tolist()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name', default='baseline')
    ap.add_argument('--input', default='data/processed/train.csv')
    args = ap.parse_args()

    cfg = Config()
    if os.path.exists(args.input):
        df = pd.read_csv(args.input)
    else:
        print("No processed train.csv found; falling back to sample data.")
        df = pd.read_csv("data/sample_reviews.csv")

    df = add_metadata_feats(df)
    df = prepare_training_data(df)

    clf = MultiLabelSklearn(labels=LABELS)
    clf.fit(df["text"].tolist(), to_multi_list(df))
    os.makedirs(cfg.models_dir, exist_ok=True)
    path = os.path.join(cfg.models_dir, f"multilabel_{args.run_name}.joblib")
    clf.save(path)
    print("Saved model to", path)

if __name__ == '__main__':
    main()
