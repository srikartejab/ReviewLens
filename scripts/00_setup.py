import os, yaml, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg_path = ROOT / "config" / "config.yaml"
env_path = ROOT / ".env"

if not cfg_path.exists():
    print("config/config.yaml not found. Please create it from the repo.")
    sys.exit(1)

cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

# Write .env from config
lines = []
lines.append(f"HUGGINGFACE_TOKEN={cfg.get('huggingface',{}).get('token','')}")
lines.append(f"WANDB_API_KEY={cfg.get('wandb',{}).get('api_key','')}")
k = cfg.get('kaggle',{})
lines.append(f"KAGGLE_USERNAME={k.get('username','')}")
lines.append(f"KAGGLE_KEY={k.get('key','')}")
lines.append(f"KAGGLE_DATASET_SLUGS={k.get('dataset_slugs','')}")
translate = cfg.get('options',{}).get('translate_non_en', True)
lines.append(f"TRANSLATE_NON_EN={'1' if translate else '0'}")

env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {env_path} from config/config.yaml")

print("\nSummary:")
print("  Kaggle slugs:", k.get('dataset_slugs','') or "(none)")
print("  HF token set:", bool(cfg.get('huggingface',{}).get('token','')))
print("  W&B enabled:", bool(cfg.get('wandb',{}).get('enable', False)))
