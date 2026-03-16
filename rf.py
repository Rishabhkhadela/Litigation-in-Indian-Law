"""
Indian Litigation Outcome Predictor — ReportLab Premium 
Full Streamlit app with prediction, similarity, charts and professional PDF (ReportLab).
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

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image as RLImage, PageBreak, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

from PyPDF2 import PdfReader
import docx2txt


# CONFIG

MODEL_DIR = "fine_tuned_model"

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

DATASET_PATH = _resolve_dataset_path()
EMB_PATH = "case_embeddings.npy"
TEXT_PATH = "case_texts.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Font files
ROBOTO_REG = "Roboto-Regular.ttf"
ROBOTO_BOLD = "Roboto-Bold.ttf"
DEJAVU_REG = "DejaVuSans.ttf"
DEJAVU_BOLD = "DejaVuSans-Bold.ttf"

TMP_DIR = "tmp_images"
os.makedirs(TMP_DIR, exist_ok=True)

PAGE_WIDTH, PAGE_HEIGHT = A4
CONTENT_WIDTH = PAGE_WIDTH - (40 * mm)  # left+right margins
LEFT_MARGIN = RIGHT_MARGIN = 20 * mm

# Utilities: file reading + cleaning

def read_pdf_file(uploaded_file):
    reader = PdfReader(uploaded_file)
    parts = []
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            parts.append(txt)
    return " ".join(parts)

def read_docx_file(uploaded_file):
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
    """Sanitize text for the PDF (replace bullets, remove control chars/emojis)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    reps = {
        "•": "-", "●": "-", "–": "-", "—": "-", "’": "'", "“": '"', "”": '"', "…": "..."
    }
    for k, v in reps.items():
        text = text.replace(k, v)
    # remove control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # remove surrogate emojis
    text = re.sub(r"[\U00010000-\U0010FFFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Register fonts for ReportLab

def register_reportlab_font():
    """Register Roboto or DejaVu for ReportLab. Returns the internal font name."""
    if os.path.exists(ROBOTO_REG) and os.path.exists(ROBOTO_BOLD):
        try:
            pdfmetrics.registerFont(TTFont("AppRoboto", ROBOTO_REG))
            pdfmetrics.registerFont(TTFont("AppRoboto-Bold", ROBOTO_BOLD))
            return ("AppRoboto", "AppRoboto-Bold")
        except Exception:
            pass
    if os.path.exists(DEJAVU_REG) and os.path.exists(DEJAVU_BOLD):
        pdfmetrics.registerFont(TTFont("AppDejaVu", DEJAVU_REG))
        pdfmetrics.registerFont(TTFont("AppDejaVu-Bold", DEJAVU_BOLD))
        return ("AppDejaVu", "AppDejaVu-Bold")
    raise FileNotFoundError("Roboto or DejaVu fonts not found. Place Roboto-Regular.ttf & Roboto-Bold.ttf or DejaVuSans.ttf & DejaVuSans-Bold.ttf next to app.py")

# Model loading (cached)

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
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMB_PATH, emb)
    pickle.dump(texts, open(TEXT_PATH, "wb"))
    return emb, texts

# Prediction & similarity

def predict_with_probs(text, tokenizer, model):
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

# Chart creation (high DPI)

def save_fig_temp(fig, prefix="chart"):
    path = os.path.join(TMP_DIR, f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path

def create_bar_chart(labels, scores):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    sns.barplot(x=scores, y=labels, palette="Blues_r", ax=ax)
    ax.set_xlabel("Similarity")
    plt.tight_layout()
    return save_fig_temp(fig, "bar")

def create_heatmap(scores):
    fig, ax = plt.subplots(figsize=(7.5, 1.3))
    sns.heatmap(np.array(scores).reshape(1, -1), annot=True, fmt=".3f", cmap="YlOrBr", cbar=False, ax=ax)
    ax.set_yticks([])
    plt.tight_layout()
    return save_fig_temp(fig, "heat")

def create_gauge_image(conf_percent):
    # use a nice semicircle gauge style approximation (horizontal bar is simplest & robust)
    fig, ax = plt.subplots(figsize=(4.8, 1.8))
    ax.barh([0], [conf_percent], color="#2E7D32" if conf_percent >= 50 else "#C62828", height=0.6)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Favourable Probability (%)")
    plt.tight_layout()
    return save_fig_temp(fig, "gauge")


# ReportLab PDF builder (Corporate Premium)

def header_footer(canvas, doc, title_text="INDIAN LITIGATION OUTCOME REPORT"):
    canvas.saveState()
    # Header: left small badge + centered title
    canvas.setFont(pdf_font_reg, 10)
    canvas.setFillColor(colors.HexColor("#0A3D62"))
    canvas.drawCentredString(PAGE_WIDTH/2.0, PAGE_HEIGHT - 30, title_text)
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.6)
    canvas.line(LEFT_MARGIN, PAGE_HEIGHT - 34, PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 34)
    canvas.setFont(pdf_font_reg, 8)
    footer_text = f"Generated by Indian Litigation Predictor • {datetime.now().strftime('%d %b %Y %H:%M')}"
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(PAGE_WIDTH/2.0, 18, footer_text)
    canvas.setFont(pdf_font_reg, 40)
    canvas.setFillColor(colors.Color(0.8, 0.8, 0.8, alpha=0.12))
    canvas.drawCentredString(PAGE_WIDTH/2.0, PAGE_HEIGHT/2.0, "CONFIDENTIAL")
    canvas.restoreState()

def build_premium_pdf(pdf_path, pred_label, prob_fraction, similar_cases, case_text, gauge_img=None, bar_img=None, heat_img=None):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Heading1"], alignment=TA_CENTER,
                              fontName=pdf_font_bold, fontSize=22, spaceAfter=12, textColor=colors.HexColor("#0A3D62")))
    styles.add(ParagraphStyle(name="Subhead", parent=styles["Heading2"], fontName=pdf_font_bold, fontSize=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontName=pdf_font_reg, fontSize=11, leading=15))

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN,
                            topMargin=40, bottomMargin=40)

    flow = []

    # Cover Page
    flow.append(Spacer(1, 40))
    flow.append(Paragraph("INDIAN LITIGATION OUTCOME REPORT", styles["TitleCenter"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph("Generated by Indian Litigation Predictor", styles["Body"]))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %b %Y %H:%M')}", styles["Body"]))
    flow.append(PageBreak())

    # Table of Contents
    flow.append(Paragraph("Table of Contents", styles["Subhead"]))
    toc_items = [
        "1. Prediction Summary",
        "2. Confidence Gauge & Visuals",
        "3. Similar Case Analysis",
        "4. Input Case Facts"
    ]
    for item in toc_items:
        flow.append(Paragraph(item, styles["Body"]))
    flow.append(PageBreak())

    # 1. Prediction Summary
    flow.append(Paragraph("1. Prediction Summary", styles["Subhead"]))
    # Summary box as Table for consistent layout
    box_data = [
        ["Outcome", pred_label],
        ["Confidence", f"{prob_fraction*100:.2f}%"]
    ]
    box_table = Table(box_data, colWidths=[CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.65], hAlign="LEFT")
    box_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), pdf_font_reg),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F6FB")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    flow.append(box_table)
    flow.append(Spacer(1, 12))
    flow.append(Paragraph("Executive summary", styles["Subhead"]))
    summary_text = f"The model predicts the case outcome as <b>{pred_label}</b> with estimated favourable probability of <b>{prob_fraction*100:.2f}%</b>. The following pages show similarity analytics and supporting excerpts."
    flow.append(Paragraph(clean_text(summary_text), styles["Body"]))
    flow.append(PageBreak())

    # 2. Gauge & Visuals
    flow.append(Paragraph("2. Confidence Gauge & Visuals", styles["Subhead"]))
    if gauge_img and os.path.exists(gauge_img):
        flow.append(Paragraph("Prediction Confidence", styles["Body"]))
        img = RLImage(gauge_img, width=160, height=90)
        flow.append(img)
        flow.append(Spacer(1, 8))
    if heat_img and os.path.exists(heat_img):
        flow.append(Paragraph("Similarity Heatmap", styles["Body"]))
        img2 = RLImage(heat_img, width=CONTENT_WIDTH, height=45)
        flow.append(img2)
        flow.append(Spacer(1, 8))
    if bar_img and os.path.exists(bar_img):
        flow.append(Paragraph("Similar Cases (Bar Chart)", styles["Body"]))
        img3 = RLImage(bar_img, width=CONTENT_WIDTH, height=180)
        flow.append(img3)
    flow.append(PageBreak())

    # 3. Similar Case Analysis (table + excerpts)
    flow.append(Paragraph("3. Similar Case Analysis", styles["Subhead"]))
    # Table header + rows
    table_data = [["Case", "Similarity"]]
    for i, sc in enumerate(similar_cases, start=1):
        table_data.append([f"Case {i}", f"{sc['similarity']:.4f}"])
    table = Table(table_data, colWidths=[CONTENT_WIDTH*0.7, CONTENT_WIDTH*0.25])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), pdf_font_reg),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0A3D62")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1, -1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#E0E0E0")),
        ("FONTSIZE", (0,0), (-1,-1), 10),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 8))

    # Add excerpts for each similar case
    for i, sc in enumerate(similar_cases, start=1):
        heading = Paragraph(f"<b>Case {i} — Similarity {sc['similarity']:.4f}</b>", ParagraphStyle("smallHeading", parent=styles["Heading4"], fontName=pdf_font_bold, fontSize=11))
        excerpt = Paragraph(clean_text(sc["text"][:1200]) + ("..." if len(sc["text"]) > 1200 else ""), styles["Body"])
        flow.append(KeepTogether([heading, Spacer(1,4), excerpt, Spacer(1,8)]))
    flow.append(PageBreak())

    # 4. Full case facts
    flow.append(Paragraph("4. Input Case Facts", styles["Subhead"]))
    flow.append(Paragraph(clean_text(case_text).replace("\n", "<br/>"), styles["Body"]))

    # Build PDF using header/footer callbacks
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN, topMargin=40, bottomMargin=40)
    def _on_first(canvas, doc_obj):
        header_footer(canvas, doc_obj, title_text="INDIAN LITIGATION OUTCOME REPORT")
    def _on_later(canvas, doc_obj):
        header_footer(canvas, doc_obj, title_text="INDIAN LITIGATION OUTCOME REPORT")
    doc.build(flow, onFirstPage=_on_first, onLaterPages=_on_later)
    return pdf_path

# Streamlit UI
st.set_page_config(page_title="Indian Litigation Predictor ", layout="wide")
st.title("⚖️ Indian Litigation Outcome Predictor")

with st.spinner("Loading models..."):
    tokenizer, clf_model = load_prediction_model()
    embeddings, case_texts = load_or_build_embeddings()
    if embeddings is None:
        st.warning(
            "Dataset/embeddings missing. Set DATASET_PATH env var or place the dataset at "
            "'master_combined.csv' or 'dataset/master_combined.csv'."
        )
        st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload case (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    if uploaded:
        ext = uploaded.name.lower().split(".")[-1]
        if ext == "pdf":
            user_text = read_pdf_file(uploaded)
        elif ext == "docx":
            user_text = read_docx_file(uploaded)
        else:
            user_text = uploaded.read().decode("utf-8", errors="ignore")
    else:
        user_text = st.text_area("Or paste case details here", height=320)
with col2:
    top_k = st.slider("Number of similar cases", 1, 20, 5)
    st.write("Tip: 150+ words yields better similarity retrieval.")

if st.button("Predict & Generate Premium PDF"):
    if not user_text or len(user_text.strip()) < 40:
        st.warning("Please provide at least ~40 characters of case facts.")
    else:
        with st.spinner("Running inference & similarity search..."):
            label, probs = predict_with_probs(user_text, tokenizer, clf_model)
            prob_fav = float(probs[1]) if len(probs) > 1 else 0.0
            similar_cases, top_scores = get_similar_cases(user_text, embeddings, case_texts, top_k)

        st.subheader("Prediction")
        st.success(f"{label} — Confidence: {prob_fav*100:.2f}%")

        st.subheader("Most similar cases")
        for i, sc in enumerate(similar_cases, start=1):
            with st.expander(f"Case {i} — Similarity {sc['similarity']:.4f}"):
                st.write(sc['text'][:1400] + ("…" if len(sc['text']) > 1400 else ""))

        # create charts
        try:
            bar_img = create_bar_chart([f"Case {i+1}" for i in range(len(similar_cases))], [s['similarity'] for s in similar_cases]) if len(similar_cases)>0 else None
            heat_img = create_heatmap([s['similarity'] for s in similar_cases]) if len(similar_cases)>0 else None
            gauge_img = create_gauge_image(prob_fav*100)
        except Exception as e:
            st.error(f"Chart generation failed: {e}")
            bar_img = heat_img = gauge_img = None

        # register fonts for PDF (global variables for header/footer)
        try:
            pdf_font_reg, pdf_font_bold = register_reportlab_font()
        except Exception as e:
            st.error(f"Font registration failed: {e}")
            st.stop()

        # generate pdf
        pdf_out = "Premium_Report_ReportLab.pdf"
        with st.spinner("Building PDF..."):
            try:
                build_premium_pdf(pdf_out, label, prob_fav, similar_cases, user_text, gauge_img=gauge_img, bar_img=bar_img, heat_img=heat_img)
                st.success("PDF generated")
                with open(pdf_out, "rb") as f:
                    st.download_button("📥 Download Premium PDF", f, file_name=os.path.basename(pdf_out), mime="application/pdf")
            except Exception as e:
                st.exception(e)
                st.error("PDF creation failed — paste the traceback here and I will fix it.")

        # cleanup temp images
        try:
            shutil.rmtree(TMP_DIR)
            os.makedirs(TMP_DIR, exist_ok=True)
        except Exception:
            pass
