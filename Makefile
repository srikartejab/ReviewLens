.PHONY: setup configure download convert_mcauley convert_kaggle convert_gmrpl clean_split baseline eval_gmrpl demo

PY=python

setup:
$(PY) -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

configure:
$(PY) scripts/00_setup.py

download:
$(PY) scripts/01_download_kaggle.py

convert_mcauley:
$(PY) scripts/02_convert_mcauley.py --input data/raw --out data/processed/mcauley_converted.csv

convert_kaggle:
$(PY) scripts/03_convert_kaggle_generic.py --input data/raw --out data/processed/kaggle_converted.csv

# NEW: Convert the GMR-PL fake/real dataset to our unified schema
convert_gmrpl:
$(PY) scripts/03b_convert_gmr_pl.py --input data/raw --out data/processed/gmrpl_converted.csv

clean_split:
$(PY) scripts/04_clean_and_split.py --input data/processed --out data/processed

baseline:
$(PY) scripts/05_train_baseline.py --run_name baseline_sklearn

# NEW: Evaluate on GMR-PL as a fake/real test set
eval_gmrpl:
$(PY) scripts/06_eval_on_gmrpl.py --input data/processed/gmrpl_converted.csv --model models

demo:
streamlit run src/serve_streamlit.py
