# app.py
"""
Full Streamlit app with:
- INLegalBERT classifier (local fine-tuned model)
- sentence-transformers semantic similarity
- charts (matplotlib/seaborn/plotly)
- Level-4 Ultra-Premium PDF generator (ReportLab) with full Unicode (DejaVu)
- Safe PDF image embedding and cleanup

Make sure DejaVuSans.ttf and DejaVuSans-Bold.ttf are next to this file.
"""

import os
import re
import tempfile
import shutil
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from PIL import Image

# ML libs
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# PDF/report libs
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# helper to read uploaded docs
from PyPDF2 import PdfReader
import docx2txt

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_DIR = "fine_tuned_model"         # folder with saved transformers model

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

DATASET_PATH = _resolve_dataset_path()   # dataset CSV with 'text' or 'facts'
EMB_PATH = "case_embeddings.npy"
TEXT_PATH = "case_texts.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fonts (must exist in same folder)
FONT_REGULAR = "DejaVuSans.ttf"
FONT_BOLD = "DejaVuSans-Bold.ttf"
REPORTLAB_FONT_NAME = "DejaVu"  # internal name after register

# Temp folder for chart images
TMP_DIR = "tmp_images"
os.makedirs(TMP_DIR, exist_ok=True)

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return " ".join(pages)

def read_docx(uploaded_file):
    # write temp file and use docx2txt
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    try:
        tmp.write(uploaded_file.read())
        tmp.close()
        txt = docx2txt.process(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return txt

def clean_text(text: str) -> str:
    """Sanitize text for PDF/ReportLab (remove control chars, replace bullets, etc.)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "•": "-", "●": "-", "–": "-", "—": "-", "’": "'", "“": '"', "”": '"', "…": "..."
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # remove undesired control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # remove surrogate pair emojis (optional safety)
    text = re.sub(r"[\U00010000-\U0010FFFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------------------------------
# Model loading (cached by streamlit)
# ------------------------------------------------------------
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
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMB_PATH, emb)
    pickle.dump(texts, open(TEXT_PATH, "wb"))
    return emb, texts

# ------------------------------------------------------------
# Prediction and similarity helpers
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Chart generation helpers (save figs to temp files)
# ------------------------------------------------------------
def _save_fig(fig, fname_prefix="chart"):
    path = os.path.join(TMP_DIR, f"{fname_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return path

def create_bar_chart(labels, scores):
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(x=scores, y=labels, palette="viridis", ax=ax)
    ax.set_xlabel("Similarity")
    plt.tight_layout()
    return _save_fig(fig, "bar")

def create_heatmap(scores):
    fig, ax = plt.subplots(figsize=(7, 1.2))
    sns.heatmap(np.array(scores).reshape(1, -1), annot=True, fmt=".3f", cmap="YlOrBr", cbar=False, ax=ax)
    ax.set_yticks([])
    plt.tight_layout()
    return _save_fig(fig, "heat")

def create_gauge_image(confidence_percent):
    # simple horizontal bar gauge as image
    fig, ax = plt.subplots(figsize=(4.2, 1.8))
    ax.barh([0], [confidence_percent], color="#4CAF50" if confidence_percent >= 50 else "#F44336", height=0.6)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Favourable Probability (%)")
    plt.tight_layout()
    return _save_fig(fig, "gauge")

# ------------------------------------------------------------
# Level-4 Ultra Premium PDF generator (ReportLab)
# ------------------------------------------------------------
def register_reportlab_font():
    # Register DejaVu font for ReportLab (Unicode-safe)
    if not (os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)):
        raise FileNotFoundError("Please place DejaVuSans.ttf and DejaVuSans-Bold.ttf next to app.py")
    try:
        pdfmetrics.registerFont(TTFont(REPORTLAB_FONT_NAME, FONT_REGULAR))
    except Exception:
        # already registered or other; ignore
        pass

def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(REPORTLAB_FONT_NAME, 9)
    canvas.setFillColor(colors.grey)
    footer_text = f"Indian Litigation Outcome Predictor • Generated on {datetime.now().strftime('%d %b %Y %H:%M')}"
    canvas.drawCentredString(A4[0] / 2.0, 20, footer_text)
    # subtle watermark
    canvas.setFont(REPORTLAB_FONT_NAME, 48)
    canvas.setFillColor(colors.Color(0.8, 0.8, 0.8, alpha=0.12))
    canvas.drawCentredString(A4[0] / 2.0, A4[1] / 2.0, "CONFIDENTIAL")
    canvas.restoreState()

def generate_ultra_premium_pdf(pred_label, prob_fraction, sims, case_text, gauge_img_path, bar_img_path):
    """
    Creates a high-quality PDF (Level-4).
    Returns: path to generated PDF.
    """
    register_reportlab_font()
    out_path = "Litigation_Level4_Report.pdf"
    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=60)
    styles = getSampleStyleSheet()

    # Custom paragraph styles using registered font
    title_style = ParagraphStyle(
        name="Title",
        parent=styles["Heading1"],
        fontName=REPORTLAB_FONT_NAME,
        fontSize=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1A237E"),
        spaceAfter=14,
    )
    h2 = ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        fontName=REPORTLAB_FONT_NAME,
        fontSize=16,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=8,
    )
    body = ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName=REPORTLAB_FONT_NAME,
        fontSize=11,
        leading=15,
    )

    story = []

    # Cover
    story.append(Paragraph("⚖️ Indian Litigation Outcome Predictor", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Final Outcome:</b> {pred_label}", h2))
    story.append(Paragraph(f"<b>Confidence:</b> {prob_fraction * 100:.2f}%", body))
    story.append(Spacer(1, 18))
    story.append(Paragraph("Prepared by AI legal analytics using INLegalBERT and semantic similarity.", body))
    story.append(PageBreak())

    # Table of contents (simple)
    story.append(Paragraph("📑 Table of Contents", title_style))
    for i, name in enumerate(["Prediction Summary", "Prediction Gauge", "Similar Case Analysis", "Input Case Facts"], start=1):
        story.append(Paragraph(f"{i}. {name}", body))
        story.append(Spacer(1, 6))
    story.append(PageBreak())

    # Prediction Summary
    story.append(Paragraph("1. Prediction Summary", title_style))
    box_color = colors.HexColor("#00723F") if pred_label == "FAVOURABLE" else colors.HexColor("#AB1010")
    table_data = [["Outcome", pred_label], ["Confidence", f"{prob_fraction * 100:.2f}%"]]
    t = Table(table_data, colWidths=[160, 300])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), REPORTLAB_FONT_NAME),
        ("BACKGROUND", (0, 0), (-1, -1), box_color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 13),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(t)
    story.append(PageBreak())

    # Gauge page
    story.append(Paragraph("2. Prediction Confidence Gauge", title_style))
    if gauge_img_path and os.path.exists(gauge_img_path):
        story.append(RLImage(gauge_img_path, width=400, height=240))
    else:
        story.append(Paragraph("Gauge image not available.", body))
    story.append(PageBreak())

    # Similar cases
    story.append(Paragraph("3. Similar Case Analysis", title_style))
    if bar_img_path and os.path.exists(bar_img_path):
        story.append(RLImage(bar_img_path, width=430, height=260))
        story.append(Spacer(1, 12))

    # Table of similar cases
    table_rows = [["Case", "Similarity"]]
    for i, s in enumerate(sims, start=1):
        table_rows.append([f"Case {i}", f"{s['similarity']:.4f}"])
    sim_table = Table(table_rows, colWidths=[300, 160])
    sim_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), REPORTLAB_FONT_NAME),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A237E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(sim_table)
    story.append(PageBreak())

    # Input case facts
    story.append(Paragraph("4. Input Case Facts", title_style))
    story.append(Paragraph(clean_text(case_text).replace("\n", "<br/>"), body))

    # Build PDF with header/footer
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

    return out_path

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Indian Litigation Predictor — Ultra Premium", layout="wide")
st.title("⚖️ Indian Litigation Outcome Predictor — Ultra Premium PDF (Level-4)")

st.sidebar.markdown("### Settings")
st.sidebar.info("Place DejaVuSans.ttf & DejaVuSans-Bold.ttf next to app.py before generating PDF.")

# load models & embeddings
with st.spinner("Loading models... (this may take a moment)"):
    tokenizer, clf_model = load_prediction_model()
    embeddings, case_texts = load_or_build_embeddings()
    if embeddings is None:
        st.warning(
            "Embeddings or dataset not ready. Set DATASET_PATH env var or place the dataset at "
            "'master_combined.csv' or 'dataset/master_combined.csv'."
        )
        st.stop()

# input area
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload case (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    if uploaded:
        ext = uploaded.name.lower().split(".")[-1]
        if ext == "pdf":
            user_text = read_pdf(uploaded)
        elif ext == "docx":
            user_text = read_docx(uploaded)
        else:
            user_text = uploaded.read().decode("utf-8", errors="ignore")
    else:
        user_text = st.text_area("Or paste case details here", height=320)

with col2:
    top_k = st.slider("Number of similar cases", 1, 20, 5)
    st.write("Tip: 150+ words improves predictions and similarity retrieval.")

# run prediction
if st.button("Generate Prediction & Report"):
    if not user_text or len(user_text.strip()) < 40:
        st.warning("Please provide at least ~40 characters of case facts.")
    else:
        with st.spinner("Running inference and similarity search..."):
            label, probs = predict_with_probs(user_text, tokenizer, clf_model)
            prob_fav = float(probs[1]) if len(probs) > 1 else 0.0
            similar_cases, top_scores = get_similar_cases(user_text, embeddings, case_texts, top_k)

        st.subheader("✅ Prediction")
        st.success(f"{label} — Confidence: {prob_fav * 100:.2f}%")

        # display similar cases
        st.subheader("📚 Most similar cases")
        for i, sc in enumerate(similar_cases, start=1):
            with st.expander(f"Case {i} — Similarity {sc['similarity']:.4f}"):
                st.write(sc['text'][:1600] + ("…" if len(sc['text']) > 1600 else ""))

        # generate charts (temp files)
        bar_labels = [f"Case {i+1}" for i in range(len(similar_cases))]
        bar_scores = [s['similarity'] for s in similar_cases]
        try:
            bar_img = create_bar_chart(bar_labels, bar_scores)
            heat_img = create_heatmap(bar_scores)
            gauge_img = create_gauge_image(prob_fav * 100)
        except Exception as e:
            st.error(f"Chart generation failed: {e}")
            bar_img = heat_img = gauge_img = None

        st.info("Generating ultra-premium PDF — this may take a few seconds.")
        try:
            pdf_path = generate_ultra_premium_pdf(label, prob_fav, similar_cases, user_text, gauge_img, bar_img)
            st.success("📄 PDF generated: " + pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Ultra-Premium PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
        except Exception as e:
            st.exception(e)
            st.error("PDF generation failed. Make sure DejaVu fonts are available and ReportLab is installed.")
        finally:
            # cleanup temp images
            try:
                for p in os.listdir(TMP_DIR):
                    fp = os.path.join(TMP_DIR, p)
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
            except Exception:
                pass

# End of app
