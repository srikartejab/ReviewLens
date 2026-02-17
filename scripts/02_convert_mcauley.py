
import os, json, argparse, pathlib, csv, re
from typing import Any, Dict, List

def coerce_pics(pics_field) -> List[str]:
    urls: List[str] = []
    if isinstance(pics_field, list):
        for item in pics_field:
            if isinstance(item, dict) and 'url' in item:
                u = item['url']
                if isinstance(u, list): urls.extend(u)
                elif isinstance(u, str): urls.append(u)
    return urls

def parse_jsonl(path: pathlib.Path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                fixed = re.sub(r"'", '"', line)
                try:
                    yield json.loads(fixed)
                except Exception:
                    continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Folder with McAuley JSON lines files')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    fields = ['review_id','place_id','place_name','place_category','city','text','rating','created_at','user_id','user_name','pics']
    w = open(outp, 'w', encoding='utf-8', newline='')
    writer = csv.DictWriter(w, fieldnames=fields)
    writer.writeheader()

    in_dir = pathlib.Path(args.input)
    for p in in_dir.glob('*.json*'):
        for obj in parse_jsonl(p):
            place_id = obj.get('gmap_id') or ''
            user_id = obj.get('user_id') or ''
            user_name = obj.get('name') or ''
            text = obj.get('text') or ''
            rating = obj.get('rating') or ''
            created_at = obj.get('time') or ''
            pics = coerce_pics(obj.get('pics', []))

            writer.writerow({
                'review_id': f"{place_id}_{user_id}_{created_at}",
                'place_id': place_id,
                'place_name': '',
                'place_category': '',
                'city': '',
                'text': text,
                'rating': rating,
                'created_at': created_at,
                'user_id': user_id,
                'user_name': user_name,
                'pics': json.dumps(pics, ensure_ascii=False)
            })

    w.close()
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()
