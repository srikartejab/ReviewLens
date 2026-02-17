"""
Convert the GMR-PL fake/real Kaggle dataset into the unified CSV schema.

Input (unzipped under data/raw/):
- Tries to find a CSV with text + label columns. It supports several common names:
  text column candidates: ["text", "review", "content", "opinion"]
  label column candidates (1=fake, 0=real): ["label", "is_fake", "fake", "target", "y"]

Output:
- data/processed/gmrpl_converted.csv with columns:
  review_id, place_id, place_name, place_category, city, text, rating, created_at,
  user_id, user_name, pics, is_fake
"""
import argparse, pathlib, pandas as pd, json

TEXT_CANDS = ["text", "review", "content", "opinion"]
LABEL_CANDS = ["label", "is_fake", "fake", "target", "y"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with unzipped GMR-PL CSVs")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.input)
    csvs = list(in_dir.glob("*.csv"))
    if not csvs:
        print("No CSVs found in", in_dir)
        return
    # Heuristic: pick the largest CSV
    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
    df = pd.read_csv(csvs[0])

    text_col = next((c for c in TEXT_CANDS if c in df.columns), None)
    label_col = next((c for c in LABEL_CANDS if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns. Have: {df.columns.tolist()}")

    # Map label to 0/1
    lab = df[label_col]
    if lab.dtype == object:
        lab = lab.astype(str).str.lower().str.strip().map(
            {"fake":1, "true":1, "real":0, "false":0, "1":1, "0":0}
        ).fillna(0).astype(int)
    else:
        lab = (lab.astype(float) > 0.5).astype(int)

    out = pd.DataFrame({
        "review_id": df.index.map(lambda i: f"gmrpl_{i}"),
        "place_id": "",
        "place_name": "",
        "place_category": "",
        "city": "",
        "text": df[text_col].fillna(""),
        "rating": "",
        "created_at": "",
        "user_id": "",
        "user_name": "",
        "pics": "[]",
        "is_fake": lab
    })
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {out_path} with {len(out)} rows.")

if __name__ == "__main__":
    main()
