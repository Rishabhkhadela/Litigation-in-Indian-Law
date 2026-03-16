# Litigation-in-Indian-Law (Litigation Predictor)

Streamlit apps and training utilities for predicting Indian litigation outcomes, plus similarity search and PDF report generation.

## Quickstart

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run rf.py
```

## Data & model files (not committed)

This repo intentionally does **not** commit large datasets and model weights because GitHub rejects files over 100MB.

Place these locally (or adjust paths in code):

- `dataset/master_combined.csv`
- optional generated artifacts: `case_embeddings.npy`, `case_texts.pkl`
- model directory: `fine_tuned_model/` (see `fine_tuned_model/README.md`)

## Apps

- `rf.py` — full Streamlit app (prediction + similarity + ReportLab PDF)
- `app.py`, `app1.py`, `app_advanced.py`, `app_full.py` — alternate app variants

