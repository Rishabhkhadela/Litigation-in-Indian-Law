# Litigation-in-Indian-Law

Streamlit apps and training utilities for predicting litigation outcomes (classification) and retrieving similar cases, with optional PDF report generation.

## Features

- Outcome prediction using a local fine-tuned Transformers model (`fine_tuned_model/`).
- Similar-case retrieval using Sentence-Transformers embeddings (cosine similarity).
- Upload case facts as PDF / DOCX / TXT, or paste text directly (in `rf.py` / `d1.py` variants).
- PDF report generation:
  - ReportLab (higher fidelity) in `rf.py` / `d1.py`
  - FPDF (lighter) in some variants

## Repository contents

- Apps:
  - `rf.py` (recommended): prediction + similarity + ReportLab PDF report
  - `app.py`, `app1.py`, `app_advanced.py`, `app_full.py`, `d.py`, `d1.py`: alternate variants
- Training:
  - `train_model.py`: trains a classifier and saves it to `fine_tuned_model/`
- Utilities:
  - `fix_config.py`, `fix_expert.py`, `fix_tokenizer.py`, `regenerate_label_encoder.py`

## Setup

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU + CUDA for faster inference/training

### Install

PowerShell (Windows):

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Run (recommended app)

```powershell
streamlit run rf.py
```

## Data & model files (not committed to GitHub)

This repo intentionally does **not** commit very large datasets or model weights (GitHub rejects files > 100MB). See `.gitignore` for what is excluded.

### Dataset location

The apps resolve the dataset from:

1. `DATASET_PATH` environment variable (if set)
2. `master_combined.csv` (project root)
3. `dataset/master_combined.csv`

Examples:

PowerShell:

```powershell
$env:DATASET_PATH = "dataset\\master_combined.csv"
streamlit run rf.py
```

bash/zsh:

```bash
export DATASET_PATH="dataset/master_combined.csv"
streamlit run rf.py
```

### Embeddings cache (optional)

Some app variants generate and cache:

- `case_embeddings.npy`
- `case_texts.pkl`

If these are present locally, the app will load them; otherwise it will build them from the dataset CSV.

### Model files

Apps load a local HuggingFace-style folder at `fine_tuned_model/`.

- Tokenizer/config files can be committed.
- Large weights (e.g., `model.safetensors`) are excluded by `.gitignore`.

See `fine_tuned_model/README.md` for expected contents.

## Training (optional)

`train_model.py` trains a classifier and writes outputs to `fine_tuned_model/`.

Notes:

- It expects CSVs named `multi_train.csv`, `single_train.csv`, `expert_clean.csv` (either place them in the project root or update paths inside `train_model.py`).
- Training dependencies used by `train_model.py` include `datasets` and `evaluate` (install separately if needed):

```bash
pip install datasets evaluate
```

## Notes / troubleshooting

- If you see “Dataset CSV not found”, set `DATASET_PATH` or place `master_combined.csv` in `dataset/`.
- If GitHub push fails due to file size, keep large files out of git (or use Git LFS / Releases / external storage).

## License

No license file is included yet. Add a `LICENSE` file if you want this repository to be explicitly open-source.
