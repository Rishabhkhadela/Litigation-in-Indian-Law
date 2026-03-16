# app.py
# Premium Litigation Predictor (UI A + UI B toggle) + Level-2 PDF (Unicode-safe)
# Requirements: see earlier requirements.txt (fpdf2, transformers, sentence-transformers, torch, streamlit, matplotlib, seaborn, PyPDF2, docx2txt)

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
from PIL import Image
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx2txt

# -----------------------------
# CONFIG - change if needed
# -----------------------------
MODEL_DIR = "fine_tuned_model"        # local HF-style model folder

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

DATASET_PATH = _resolve_dataset_path()  # CSV with 'text' or 'facts'
EMB_PATH = "case_embeddings.npy"
TEXT_PATH = "case_texts.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# DejaVu font files (must be in same folder)
FONT_REGULAR = "DejaVuSans.ttf"
FONT_BOLD = "DejaVuSans-Bold.ttf"

# PDF safe width (mm)
PDF_TEXT_WIDTH = 190

# -----------------------------
# Utility: read uploaded files
# -----------------------------
def read_pdf(file):
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            pages.append(txt)
    return " ".join(pages)

def read_docx(file):
    return docx2txt.process(file)

# -----------------------------
# Sanitizer: make PDF-safe text
# -----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Replace common problematic characters
    reps = {
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
    for k, v in reps.items():
        text = text.replace(k, v)
    # Remove non-printable / surrogate / emoji characters
    # Keep basic punctuation and common unicode by replacing outside latin1 with space
    # but allow other BMP characters: we already use unicode font, but removing odd surrogates is safe
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Model loaders (cached)
# -----------------------------
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
    # build from CSV
    if not os.path.exists(DATASET_PATH):
        st.error(
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
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMB_PATH, emb)
    pickle.dump(texts, open(TEXT_PATH, "wb"))
    return emb, texts

# -----------------------------
# Prediction & similarity helpers
# -----------------------------
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
    return label, probs  # probs array

def get_similar_cases(query, embeddings, case_texts, top_k=5):
    embedder = load_embedding_model()
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    dots = np.dot(embeddings, q_emb)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    scores = dots / (norms + 1e-12)
    idx = np.argsort(scores)[::-1][:top_k]
    results = [{"text": case_texts[i], "similarity": float(scores[i])} for i in idx]
    return results, scores[idx]

# -----------------------------
# PDF utilities (images) - save temp files
# -----------------------------
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

# -----------------------------
# Premium PDF Level-2 generator (Unicode-safe, fixed widths)
# -----------------------------
class PDF(FPDF):
    def header(self):
        # skip header on first cover page
        if self.page_no() == 1:
            return
        self.set_font("DejaVuB", size=12)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, "AI Litigation Analysis Report", ln=True, align="C")
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.3)
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("DejaVu", size=9)
        self.set_text_color(100, 100, 100)
        footer_text = clean_text(f"Page {self.page_no()} • Generated by Indian Litigation Outcome Predictor")
        self.cell(0, 10, footer_text, align="C")

def generate_pdf_level2(text, outcome, confidence, similar_cases, bar_labels, bar_scores):
    # Ensure fonts exist
    if not (os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)):
        raise FileNotFoundError("DejaVu font files not found. Place DejaVuSans.ttf and DejaVuSans-Bold.ttf next to app.py")

    pdf = PDF()
    # register fonts (fpdf2)
    pdf.add_font("DejaVu", "", FONT_REGULAR, uni=True)
    pdf.add_font("DejaVuB", "", FONT_BOLD, uni=True)
    pdf.set_auto_page_break(auto=True, margin=15)

    # COVER page
    pdf.add_page()
    pdf.set_font("DejaVuB", size=26)
    pdf.set_text_color(10, 32, 91)
    pdf.ln(30)
    pdf.cell(0, 12, "Indian Litigation Outcome Predictor", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("DejaVu", size=12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}", ln=True, align="C")
    pdf.ln(10)

    # EXECUTIVE SUMMARY + VISUALS
    pdf.add_page()
    pdf.set_font("DejaVuB", size=16)
    pdf.set_text_color(0, 45, 91)
    pdf.cell(0, 8, "Executive Summary", ln=True)
    pdf.ln(4)

    pdf.set_font("DejaVu", size=11)
    summ = (f"The AI (INLegalBERT) analyzed the provided case facts and predicts '{outcome}' "
            f"with confidence {confidence:.2f}%. The following pages include visual analytics and similar precedents.")
    pdf.multi_cell(PDF_TEXT_WIDTH, 7, clean_text(summ))
    pdf.ln(6)

    # Visuals: heatmap + bar + prob chart
    heatmap_img = generate_heatmap_image(bar_scores)
    bar_img = generate_bar_image(bar_labels, bar_scores)
    prob_img = generate_prob_chart_image(confidence)

    pdf.set_font("DejaVuB", size=14)
    pdf.cell(0, 8, "Risk Visualization", ln=True)
    pdf.ln(2)

    # insert heatmap
    pdf.image(heatmap_img, w=PDF_TEXT_WIDTH)
    pdf.ln(6)
    pdf.image(bar_img, w=PDF_TEXT_WIDTH)
    pdf.ln(6)
    pdf.image(prob_img, w=90)
    pdf.ln(8)

    # CASE SUMMARY
    pdf.set_font("DejaVuB", size=14)
    pdf.set_text_color(10, 32, 91)
    pdf.cell(0, 8, "Case Summary", ln=True)
    pdf.set_font("DejaVu", size=11)
    pdf.multi_cell(PDF_TEXT_WIDTH, 7, clean_text(text[:3500]) + ("…" if len(text) > 3500 else ""))
    pdf.ln(6)

    # SIMILAR CASES
    pdf.set_font("DejaVuB", size=14)
    pdf.cell(0, 8, "Top Similar Cases", ln=True)
    pdf.ln(2)
    pdf.set_font("DejaVu", size=10)
    for i, sc in enumerate(similar_cases, start=1):
        title = clean_text(f"Case {i} — Similarity: {sc['similarity']:.4f}")
        pdf.set_font("DejaVuB", size=11)
        pdf.multi_cell(PDF_TEXT_WIDTH, 7, title)
        pdf.set_font("DejaVu", size=10)
        pdf.multi_cell(PDF_TEXT_WIDTH, 6, clean_text(sc["text"][:900]) + ("…" if len(sc["text"]) > 900 else ""))
        pdf.ln(3)

    # watermark (simple)
    pdf.set_text_color(200, 200, 200)
    pdf.set_font("DejaVuB", size=40)
    # rotate not needed; simple text at bottom
    pdf.set_xy(20, pdf.h - 40)
    pdf.cell(0, 10, "AI-GENERATED", align="L")

    out_path = "Litigation_Report_Level2.pdf"
    pdf.output(out_path)

    # cleanup temp images
    for p in (heatmap_img, bar_img, prob_img):
        try:
            os.remove(p)
        except Exception:
            pass

    return out_path

# -----------------------------
# Streamlit UI (UI-A and UI-B toggle)
# -----------------------------
st.set_page_config(page_title="Indian Litigation Predictor — Premium", layout="wide")
st.title("⚖️ Indian Litigation Outcome Predictor — Premium")

# Theme toggle (A=light, B=dark)
ui_choice = st.sidebar.selectbox("UI Theme", ("UI-A (Light)", "UI-B (Dark)"))
if ui_choice == "UI-B (Dark)":
    # minimal dark styling
    st.markdown("""
    <style>
    .stApp { background: #0b1220; color: #e6eef6; }
    .glass { background: rgba(255,255,255,0.04); padding: 14px; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background: #ffffff; color: #111827; }
    .glass { background: rgba(245,247,250,0.9); padding: 14px; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='glass'><b>Model:</b> INLegalBERT (fine-tuned) + Semantic Similarity</div>", unsafe_allow_html=True)

# load models & embeddings
tokenizer, clf_model = load_prediction_model()
embeddings, case_texts = load_or_build_embeddings()
if embeddings is None:
    st.stop()

# File uploader / text area
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload case (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    if uploaded:
        ext = uploaded.name.lower().split(".")[-1]
        if ext == "pdf":
            user_text = read_pdf(uploaded)
        elif ext == "docx":
            user_text = read_docx(uploaded)
        else:
            user_text = uploaded.read().decode("utf-8", errors="ignore")
    else:
        user_text = st.text_area("Or paste case details here", height=300)
with col2:
    top_k = st.slider("Number of similar cases", 1, 20, 5)
    st.write("Tip: 150+ words improves similarity & prediction quality")

# Predict action
if st.button("🔮 Predict Outcome"):
    if not user_text or len(user_text.strip()) < 40:
        st.warning("Please provide more detailed case facts (at least ~40 characters).")
    else:
        with st.spinner("Running model & finding similar cases..."):
            label, probs = predict_with_probs(user_text, tokenizer, clf_model)
            prob_fav = float(probs[1]) * 100 if len(probs) > 1 else 0.0
            similar_cases, top_scores = get_similar_cases(user_text, embeddings, case_texts, top_k)

        # Prediction card
        st.subheader("✅ Final Prediction")
        c1, c2 = st.columns([3, 2])
        with c1:
            color = "#16a34a" if label == "FAVOURABLE" else "#ef4444"
            st.markdown(f"<div class='glass'><h2 style='color:{color}; margin:0'>{label}</h2>"
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

        # Similar cases display
        st.subheader("📚 Most Similar Cases")
        for i, sc in enumerate(similar_cases, start=1):
            with st.expander(f"🔹 Case {i} — Similarity {sc['similarity']:.4f}"):
                st.write(sc['text'][:1400] + ("…" if len(sc['text']) > 1400 else ""))

        # Charts
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

        # Generate Level-2 PDF
        with st.spinner("Generating premium PDF report..."):
            try:
                pdf_path = generate_pdf_level2(user_text, label, prob_fav, similar_cases, bar_labels, bar_scores)
            except Exception as e:
                st.exception(e)
                pdf_path = None

        if pdf_path and os.path.exists(pdf_path):
            st.success("📄 Premium PDF (Level-2) generated")
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Premium PDF (Level-2)", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
        else:
            st.error("Failed to generate PDF. Check logs.")

# -----------------------------
# End of app
# -----------------------------
