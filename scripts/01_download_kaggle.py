import os, pathlib, subprocess, urllib.request, zipfile, tarfile, gzip, shutil
from dotenv import load_dotenv

# Also read YAML config to include eval slugs like GMR-PL
def read_yaml_slugs():
    try:
        import yaml
        y = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
        base = (y.get("kaggle", {}) or {}).get("dataset_slugs", "") or ""
        eval_slug = (y.get("datasets", {}) or {}).get("eval_gmrpl_slug", "") or ""
        parts = []
        if base.strip():
            parts.append(base.strip())
        if eval_slug.strip():
            parts.append(eval_slug.strip())
        return ",".join(parts)
    except Exception:
        return ""

load_dotenv()
DATA_DIR = pathlib.Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Merge .env slugs + YAML slugs
env_slugs = os.getenv("KAGGLE_DATASET_SLUGS", "").strip()
yaml_slugs = read_yaml_slugs()
if env_slugs and yaml_slugs:
    slugs = env_slugs + "," + yaml_slugs
elif env_slugs:
    slugs = env_slugs
else:
    slugs = yaml_slugs

slugs = slugs.strip()
if slugs:
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except Exception:
        print("Kaggle CLI not found. Install with `pip install kaggle` and ensure credentials (kaggle.json) exist. Skipping Kaggle downloads.")
        slugs = ""

if slugs:
    for slug in [s.strip() for s in slugs.split(",") if s.strip()]:
        print(f"Downloading {slug} ...")
        code = subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(DATA_DIR), "-q"]).returncode
        if code != 0:
            print(f"Falling back to competitions download for {slug} (if applicable)...")
            subprocess.run(["kaggle", "competitions", "download", "-c", slug, "-p", str(DATA_DIR), "-q"])
else:
    print("No Kaggle dataset slugs found. Set KAGGLE_DATASET_SLUGS in .env or fill config/config.yaml if you need Kaggle data.")

# Download McAuley Pennsylvania dataset
MCAULEY_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/review-Pennsylvania.json.gz"
mcauley_path = DATA_DIR / "review-Pennsylvania.json.gz"
if not mcauley_path.exists():
    try:
        print("Downloading McAuley Pennsylvania dataset ...")
        urllib.request.urlretrieve(MCAULEY_URL, mcauley_path)
    except Exception as e:
        print(f"Failed to download McAuley dataset: {e}")


def unpack_archives(directory: pathlib.Path):

    """Unpack any archives in `directory` into their own subfolders."""
    for p in list(directory.iterdir()):
        if not p.is_file():
            continue
        base_name = p.name.replace(''.join(p.suffixes), '')
        subdir = directory / base_name
        if zipfile.is_zipfile(p):
            print(f"Unzipping {p.name} into {subdir} ...")
            subdir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, 'r') as z:
                z.extractall(subdir)
            p.unlink()
        elif tarfile.is_tarfile(p):
            print(f"Untarring {p.name} into {subdir} ...")
            subdir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(p, 'r:*') as t:
                t.extractall(subdir)
            p.unlink()
        elif p.suffix == '.gz' and not p.name.endswith('.tar.gz'):
            print(f"Gunzip {p.name} into {subdir} ...")
            subdir.mkdir(parents=True, exist_ok=True)
            new_path = subdir / p.with_suffix('').name

            with gzip.open(p, 'rb') as f_in, open(new_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            p.unlink()


print("Unpacking any archives in data/raw ...")
unpack_archives(DATA_DIR)
print("Downloads and unpacking complete.")
