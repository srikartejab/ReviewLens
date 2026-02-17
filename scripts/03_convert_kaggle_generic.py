
import os, argparse, pathlib, json
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Folder with Kaggle CSVs')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()

    in_dir = pathlib.Path(args.input)
    outp = pathlib.Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    csvs = list(in_dir.glob('*.csv'))
    if not csvs:
        print("No CSV files found in input folder.")
        return
    dfs = [pd.read_csv(p) for p in csvs]
    df = pd.concat(dfs, ignore_index=True)

    # Map columns from generic schema to unified schema
    df_out = pd.DataFrame()
    df_out['review_id'] = df.index.map(lambda i: f"kaggle_{i}")
    df_out['place_id'] = df.get('business_id', pd.Series(['']*len(df)))
    df_out['place_name'] = df.get('business_name', '')
    df_out['place_category'] = df.get('rating_category', '')
    df_out['city'] = df.get('city','')
    df_out['text'] = df.get('text','')
    df_out['rating'] = df.get('rating', '')
    df_out['created_at'] = df.get('created_at', '')
    df_out['user_id'] = df.get('author_id', '')
    df_out['user_name'] = df.get('author_name', '')
    # photo path or URL can be in 'photo' column
    pics = []
    if 'photo' in df.columns:
        for v in df['photo']:
            if isinstance(v, str) and v.strip():
                pics.append(json.dumps([v]))
            else:
                pics.append("[]")
    else:
        pics = ["[]"]*len(df)
    df_out['pics'] = pics

    df_out.to_csv(outp, index=False)
    print(f"Wrote {args.out} with {len(df_out)} rows.")

if __name__ == '__main__':
    main()
