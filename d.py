# app.py
# Patched full Streamlit app (UI-B standard) + Safe Simple PDF generator (DejaVu Unicode)
# Author: Patched for user — fixes FPDF errors (width + unicode)
# Requirements: streamlit, torch, transformers, sentence-transformers, fpdf2, matplotlib, seaborn, plotly, PyPDF2, docx2txt, pandas, numpy

import os
import re
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx2txt

# ---------------------------
# CONFIG
# ---------------------------
MODEL_DIR = "fine_tuned_model"       # local HF-style model folder

def _resolve_dataset_path():
    env_path = os.getenv("DATASET_PATH")
    candidates = [
        env_path,
        "master_combined.csv",
        os.path.join("dataset", "master_combined.csv"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[-1]

DATASET_PATH = _resolve_dataset_path() # CSV with 'text' or 'facts'
EMB_PATH = "case_embeddings.npy"
TEXT_PATH = "case_texts.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# DejaVu font files must exist next to app.py
FONT_REGULAR = "DejaVuSans.ttf"
FONT_BOLD = "DejaVuSans-Bold.ttf"

# PDF settings
PDF_TEXT_WIDTH = 190  # mm safe width for A4 with default margins

# ---------------------------
# Utility: file reading
# ---------------------------
def read_pdf(file_obj):
    reader = PdfReader(file_obj)
    parts = []
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            parts.append(txt)
    return " ".join(parts)

def read_docx(file_obj):
    # docx2txt accepts filename or bytes, streamlit provides UploadedFile with read()
    # docx2txt.process expects path, but can process bytes via temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    try:
        tmp.write(file_obj.read())
        tmp.close()
        text = docx2txt.process(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return text

# ---------------------------
# Clean text for PDF (sanitize)
# ---------------------------
def clean_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Replace common problematic characters
    replacements = {
        "•": "-",
        "●": "-",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "\u2022": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # Remove surrogate/emoji ranges (safe)
    text = re.sub(r"[\U00010000-\U0010FFFF]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Load models (cached)
# ---------------------------
@st.cache_resource
def load_prediction_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_or_build_embeddings():
    if os.path.exists(EMB_PATH) and os.path.exists(TEXT_PATH):
        emb = np.load(EMB_PATH)
        texts = pickle.load(open(TEXT_PATH, "rb"))
        return emb, texts

    if not os.path.exists(DATASET_PATH):
        st.warning(
            f"Dataset CSV not found. Set DATASET_PATH env var or place it at "
            f"'master_combined.csv' or 'dataset/master_combined.csv'. Tried: {DATASET_PATH}"
        )
        return None, None

    df = pd.read_csv(DATASET_PATH)
    if "text" in df.columns:
        texts = df["text"].astype(str).tolist()
    elif "facts" in df.columns:
        texts = df["facts"].astype(str).tolist()
    else:
        raise ValueError("CSV must contain 'text' or 'facts' column.")
    embedder = load_embedding_model()
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMB_PATH, embeddings)
    pickle.dump(texts, open(TEXT_PATH, "wb"))
    return embeddings, texts

# ---------------------------
# Prediction & similarity
# ---------------------------
def predict_with_probs(text, tokenizer, model, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.cpu().numpy()[0]
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
    pred_idx = int(np.argmax(probs))
    label = "FAVOURABLE" if pred_idx == 1 else "NOT FAVOURABLE"
    return label, probs

def get_similar_cases(query, embeddings, case_texts, top_k=5):
    embedder = load_embedding_model()
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    dots = np.dot(embeddings, q_emb)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    scores = dots / (norms + 1e-12)
    idx = np.argsort(scores)[::-1][:top_k]
    results = [{"text": case_texts[i], "similarity": float(scores[i])} for i in idx]
    return results, scores[idx]

# ---------------------------
# Plot helpers (temp files)
# ---------------------------
def _save_fig_temp(fig):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return tmp.name

def generate_bar_image(labels, scores):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    sns.barplot(x=scores, y=labels, palette="viridis", ax=ax)
    ax.set_xlabel("Similarity")
    plt.tight_layout()
    return _save_fig_temp(fig)

def generate_heatmap_image(scores):
    fig, ax = plt.subplots(figsize=(6.5, 1.2))
    sns.heatmap(np.array(scores).reshape(1, -1), annot=True, fmt=".3f", cmap="YlOrBr", cbar=False, ax=ax)
    ax.set_yticks([])
    plt.tight_layout()
    return _save_fig_temp(fig)

def generate_prob_chart_image(confidence):
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    ax.barh([0], [confidence], color="#4CAF50" if confidence >= 50 else "#F44336", height=0.4)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Favourable Probability (%)")
    plt.tight_layout()
    return _save_fig_temp(fig)

# ---------------------------
# Simple PDF generator (safe) - Option B (compact)
# ---------------------------
class SimplePDF(FPDF):
    def header(self):
        self.set_font("DejaVuB", size=12)
        self.cell(0, 8, "Indian Litigation Predictor — Report", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", size=9)
        footer = f"Page {self.page_no()}  •  Generated {datetime.now().strftime('%d %b %Y %H:%M')}"
        self.cell(0, 10, clean_text(footer), align="C")

def generate_simple_pdf(case_text: str, prediction: str, confidence_percent: float, similar_cases: list,
                        out_path: str = "simple_report.pdf",
                        font_regular: str = FONT_REGULAR,
                        font_bold: str = FONT_BOLD) -> str:
    # Validate fonts
    if not (os.path.exists(font_regular) and os.path.exists(font_bold)):
        raise FileNotFoundError("DejaVu fonts not found. Place DejaVuSans.ttf and DejaVuSans-Bold.ttf next to app.py")

    pdf = SimplePDF()
    pdf.add_font("DejaVu", "", font_regular, uni=True)
    pdf.add_font("DejaVuB", "", font_bold, uni=True)
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("DejaVuB", size=16)
    pdf.multi_cell(PDF_TEXT_WIDTH, 10, clean_text(f"Final Prediction: {prediction}"))

    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(PDF_TEXT_WIDTH, 7, clean_text(f"Confidence (Favourable): {confidence_percent:.2f}%"))
    pdf.ln(4)

    pdf.set_font("DejaVuB", size=13)
    pdf.multi_cell(PDF_TEXT_WIDTH, 8, "Case Summary:")
    pdf.set_font("DejaVu", size=11)
    pdf.multi_cell(PDF_TEXT_WIDTH, 6, clean_text(case_text[:3500]) + ("…" if len(case_text) > 3500 else ""))
    pdf.ln(6)

    pdf.set_font("DejaVuB", size=13)
    pdf.multi_cell(PDF_TEXT_WIDTH, 8, f"Top {len(similar_cases)} Similar Cases:")
    pdf.ln(2)

    for i, sc in enumerate(similar_cases, start=1):
        title = clean_text(f"Case {i} — Similarity: {sc.get('similarity', sc.get('score', 0)):.4f}")
        pdf.set_font("DejaVuB", size=11)
        pdf.multi_cell(PDF_TEXT_WIDTH, 7, title)

        pdf.set_font("DejaVu", size=10)
        pdf.multi_cell(PDF_TEXT_WIDTH, 6, clean_text(sc.get("text", "")[:900]) + ("…" if len(sc.get("text", "")) > 900 else ""))
        pdf.ln(3)

    pdf.output(out_path)
    return out_path

# ---------------------------
# Streamlit UI (UI-B standard corporate)
# ---------------------------
st.set_page_config(page_title="Litigation Outcome Predictor (Standard)", layout="wide")
st.title("⚖️ Indian Litigation Outcome Predictor — Standard (UI-B)")
st.markdown("Model: INLegalBERT (fine-tuned) + Semantic Similarity")

# Load models & embeddings
with st.spinner("Loading models..."):
    tokenizer, clf_model = load_prediction_model()
    embeddings, case_texts = load_or_build_embeddings()
    if embeddings is None:
        st.stop()

# Input area
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload case (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    if uploaded:
        ext = uploaded.name.lower().split(".")[-1]
        if ext == "pdf":
            user_text = read_pdf(uploaded)
        elif ext == "docx":
            # streamlit UploadedFile is file-like; read_docx writes a temp file
            user_text = read_docx(uploaded)
        else:
            user_text = uploaded.read().decode("utf-8", errors="ignore")
    else:
        user_text = st.text_area("Or paste case details here", height=300)
with col2:
    top_k = st.slider("Number of similar cases", 1, 20, 5)
    st.write("Tip: provide 150+ words for better predictions")

# Predict
if st.button("Predict"):
    if not user_text or len(user_text.strip()) < 40:
        st.warning("Please provide more detailed case facts (at least 40 characters).")
    else:
        with st.spinner("Running model & similarity search..."):
            label, probs = predict_with_probs(user_text, tokenizer, clf_model)
            prob_fav = float(probs[1]) * 100 if len(probs) > 1 else 0.0
            similar_cases, top_scores = get_similar_cases(user_text, embeddings, case_texts, top_k)

        # Prediction display
        st.subheader("✅ Final Prediction")
        c1, c2 = st.columns([3, 2])
        with c1:
            color = "#16a34a" if label == "FAVOURABLE" else "#ef4444"
            st.markdown(f"<div style='padding:12px;border-radius:8px;background:#fff'>"
                        f"<h2 style='color:{color};margin:0'>{label}</h2>"
                        f"<p>Confidence: <b>{prob_fav:.2f}%</b></p></div>", unsafe_allow_html=True)
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_fav,
                title={'text': "Favourable Probability"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green" if prob_fav >= 50 else "red"}}
            ))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Similar cases
        st.subheader("📚 Most Similar Cases")
        for i, sc in enumerate(similar_cases, start=1):
            with st.expander(f"🔹 Case {i} — Similarity {sc['similarity']:.4f}"):
                st.write(sc['text'][:1400] + ("…" if len(sc['text']) > 1400 else ""))

        # Visuals
        st.subheader("📊 Similarity Visuals")
        bar_labels = [f"Case {i+1}" for i in range(len(similar_cases))]
        bar_scores = [s['similarity'] for s in similar_cases]

        fig_bar, ax_bar = plt.subplots(figsize=(8, 3.2))
        sns.barplot(x=bar_scores, y=bar_labels, palette="viridis", ax=ax_bar)
        ax_bar.set_xlabel("Similarity")
        st.pyplot(fig_bar)

        fig_heat, ax_heat = plt.subplots(figsize=(8, 1.2))
        sns.heatmap(np.array(bar_scores).reshape(1, -1), annot=True, fmt=".3f", cmap="YlOrBr", cbar=False, ax=ax_heat)
        ax_heat.set_yticks([])
        st.pyplot(fig_heat)

        # Generate Safe Simple PDF
        with st.spinner("Generating PDF report..."):
            try:
                pdf_file = generate_simple_pdf(
                    case_text=user_text,
                    prediction=label,
                    confidence_percent=prob_fav,
                    similar_cases=similar_cases,
                    out_path="simple_lit_report.pdf",
                    font_regular=FONT_REGULAR,
                    font_bold=FONT_BOLD
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")
                pdf_file = None

        if pdf_file and os.path.exists(pdf_file):
            st.success("📄 PDF report ready")
            with open(pdf_file, "rb") as f:
                st.download_button("📥 Download PDF report", f, file_name=os.path.basename(pdf_file), mime="application/pdf")
        else:
            st.error("PDF generation failed. Ensure DejaVu fonts exist next to app.py.")

# End
