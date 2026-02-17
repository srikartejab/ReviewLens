import os, yaml
from dotenv import load_dotenv

load_dotenv()

LABELS = [
    "Advertisement/Promo",
    "Irrelevant Content",
    "Rant (Likely Non-Visitor)",
    "Spam/Low-quality"
]

class Config:
    data_dir = os.getenv("DATA_DIR", "data")
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    models_dir = os.getenv("MODELS_DIR", "models")
    outputs_dir = os.getenv("OUTPUTS_DIR", "outputs")

    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    wandb_api_key = os.getenv("WANDB_API_KEY", None)
    translate_non_en = os.getenv("TRANSLATE_NON_EN", "1") == "1"

    # Load config.yaml for model names & priors
    cfg_yaml = "config/config.yaml"
    models = {
        "classifier": "microsoft/deberta-v3-base",
        "relevancy_cross_encoder": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "image_text": "clip-ViT-B-32",
    }
    options = {}
    priors = {}
    device_location = {}
    if os.path.exists(cfg_yaml):
        y = yaml.safe_load(open(cfg_yaml, "r", encoding="utf-8"))
        models.update(y.get("models", {}))
        options = y.get("options", {})
        priors = y.get("priors", {})
        device_location = y.get("device_location", {})

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
