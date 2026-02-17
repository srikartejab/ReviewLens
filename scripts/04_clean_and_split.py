
import os, argparse, pathlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_prep import clean_reviews, business_disjoint_split
from src.features import add_metadata_feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Folder with processed CSVs (converted)')
    ap.add_argument('--out', required=True, help='Output folder to write train/val/test CSVs')
    args = ap.parse_args()
    in_dir = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for p in in_dir.glob("*.csv"):
        if 'train' in p.name or 'val' in p.name or 'test' in p.name: continue
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            pass
    if not dfs:
        print("No converted CSVs found. Place a CSV in data/processed or run convert scripts first.")
        return
    df = pd.concat(dfs, ignore_index=True)

    df = clean_reviews(df)
    df = add_metadata_feats(df)
    tr, val, te = business_disjoint_split(df, cfg=type("tmp",(object,),{"test_size":0.1,"val_size":0.1,"random_state":42})())

    tr.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    te.to_csv(out_dir / "test.csv", index=False)
    print("Wrote train/val/test CSVs to", out_dir)

if __name__ == '__main__':
    main()
