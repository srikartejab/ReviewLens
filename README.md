# Review Quality and Relevancy (Plug-and-Play)

Detect low-quality or policy-violating reviews and score relevancy to a place. The system combines light ML with rules and priors to flag:
- Advertisement/Promo
- Irrelevant Content
- Rant (Likely Non-Visitor)
- Spam/Low-quality

It also produces a relevancy score between review text and a place description, plus optional image-to-text relevance.

## Highlights
- One-file config: `config/config.yaml`
- Streamlit demo for CSV uploads
- Baseline training and evaluation scripts
- Offline-friendly: relevancy falls back to TF-IDF if the cross-encoder is not available

## Tech Stack
- Python 3.10+
- pandas, numpy, scikit-learn
- transformers, sentence-transformers, torch
- streamlit
- langdetect, nltk, spacy
- Pillow, requests
- datasketch, shap, matplotlib

## Project Structure
- `config/` config and switches
- `data/` sample inputs and processed outputs
- `scripts/` data prep, training, evaluation
- `src/` core logic (features, rules, models, Streamlit app)
- `resources/` lexicons

## Quickstart (Sample Data Only)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Writes .env from config/config.yaml
python scripts/00_setup.py

# Train a baseline model using bundled sample data
python scripts/05_train_baseline.py --run_name smoke

# Launch the demo app
streamlit run src/serve_streamlit.py
```

## Configuration
Edit `config/config.yaml` to enable features and set optional credentials.
- Kaggle username/key and dataset slugs are optional
- Hugging Face token is optional (helps with model downloads)
- W&B is optional


`python scripts/00_setup.py` reads `config/config.yaml` and writes a local `.env` with these values.

## Data Pipeline (Full)
```bash
# 1) Download datasets
python scripts/01_download_kaggle.py

# 2) Convert to unified schema
python scripts/02_convert_mcauley.py --input data/raw --out data/processed/mcauley_converted.csv
python scripts/03_convert_kaggle_generic.py --input data/raw --out data/processed/kaggle_converted.csv
python scripts/03b_convert_gmr_pl.py --input data/raw --out data/processed/gmrpl_converted.csv

# 3) Clean and split
python scripts/04_clean_and_split.py --input data/processed --out data/processed

# 4) Train baseline
python scripts/05_train_baseline.py --run_name baseline_sklearn

# 5) Evaluate on GMR-PL
python scripts/06_eval_on_gmrpl.py --input data/processed/gmrpl_converted.csv --model models
```

## How It Works
- **Text quality labels:** weak labels from rules + TF-IDF multi-label baseline
- **Relevancy:** cross-encoder model by default with TF-IDF fallback
- **Priors:** username gibberish, image mismatch, burst/dup behavior, profanity, readability

## Outputs
- `models/` trained baseline artifacts
- `outputs/` relevancy samples and thresholds

## Unified CSV Schema
Required columns:
- `review_id, place_id, place_name, place_category, city, text`

Optional columns:
- `rating, created_at, user_id, user_name, pics`
- `user_lat, user_lon, place_lat, place_lon` (for distance-based priors)

`pics` can be a JSON list or a single URL string.

## Notes on Model Downloads
The first run may download models from Hugging Face. If you want higher rate limits, set `HUGGINGFACE_TOKEN`.

## Ethics and Limitations
- This is a screening aid, not a final decision system.
- Negative reviews should not be suppressed solely for sentiment.
- Thresholds are configurable in `config/config.yaml`.

## References
- GMR-PL Fake Reviews dataset (Kaggle): https://www.kaggle.com/datasets/pawegryka/gmr-pl-fake-reviews-dataset
- "Detecting Fake Reviews in Google Maps - A Case Study" (Applied Sciences, 2023): https://doi.org/10.3390/app13106331
