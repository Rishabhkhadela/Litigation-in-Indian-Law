import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt

from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx2txt

# -------------------------------
# CONFIG
# -------------------------------
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


# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_clf_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


# -------------------------------
# FILE READERS
# -------------------------------
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        txt = p.extract_text()
        if txt:
            text += txt + " "
    return text

def read_docx(file):
    return docx2txt.process(file)


# -------------------------------
# EMBEDDINGS
# -------------------------------
def build_embeddings():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset CSV not found. Set DATASET_PATH env var or place it at: "
            f"'master_combined.csv' or 'dataset/master_combined.csv'. Tried: {DATASET_PATH}"
        )
    df = pd.read_csv(DATASET_PATH)

    if "text" in df.columns:
        texts = df["text"].astype(str).tolist()
    elif "facts" in df.columns:
        texts = df["facts"].astype(str).tolist()
    else:
        raise ValueError("CSV must contain 'text' or 'facts' column.")

    embedder = load_embedder()
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    np.save(EMB_PATH, emb)
    pickle.dump(texts, open(TEXT_PATH, "wb"))

    return emb, texts

@st.cache_resource
def load_embeddings():
    if os.path.exists(EMB_PATH) and os.path.exists(TEXT_PATH):
        return np.load(EMB_PATH), pickle.load(open(TEXT_PATH, "rb"))
    return build_embeddings()


# -------------------------------
# PREDICTION
# -------------------------------
def predict_with_prob(text, tokenizer, model):
    tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        logits = model(**tokens).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = np.argmax(probs)
    label = "FAVOURABLE" if pred == 1 else "NOT FAVOURABLE"

    return label, probs


# -------------------------------
# SIMILAR CASES
# -------------------------------
def get_similar_cases(query, embeddings, case_texts, top_k):
    embedder = load_embedder()
    q = embedder.encode([query], convert_to_numpy=True)[0]

    scores = np.dot(embeddings, q) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q)
    )

    idx = np.argsort(scores)[::-1][:top_k]

    results = [
        {"text": case_texts[i], "similarity": float(scores[i])}
        for i in idx
    ]
    return results


# -------------------------------
# PDF REPORT GENERATOR
# -------------------------------
def generate_pdf(prediction, prob, similar_cases, user_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Litigation Prediction Report", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, f"Prediction: {prediction}")
    pdf.multi_cell(0, 6, f"Probability Favourable: {prob[1]*100:.2f}%")
    pdf.multi_cell(0, 6, f"Probability Not Favourable: {prob[0]*100:.2f}%")

    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, "Similar Cases", ln=True)

    pdf.set_font("Arial", size=9)
    for i, case in enumerate(similar_cases, start=1):
        pdf.multi_cell(0, 5, f"{i}. Similarity: {case['similarity']:.3f}")
        pdf.multi_cell(0, 5, case["text"][:400] + "...")
        pdf.ln(2)

    filename = "litigation_report.pdf"
    pdf.output(filename)
    return filename


# -------------------------------
# UI START
# -------------------------------
st.set_page_config(page_title="Premium Litigation Predictor", layout="wide")

st.markdown("""
    <h1 style='text-align:center;color:#4A90E2;'>⚖️ Premium Litigation Outcome Predictor</h1>
""", unsafe_allow_html=True)

tokenizer, clf_model = load_clf_model()
embeddings, case_texts = load_embeddings()

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload Case (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

with col2:
    top_k = st.slider("Similar Cases to Display:", 1, 20, 5)


if uploaded:
    ext = uploaded.name.split(".")[-1].lower()
    if ext == "pdf":
        user_text = read_pdf(uploaded)
    elif ext == "docx":
        user_text = read_docx(uploaded)
    else:
        user_text = uploaded.read().decode("utf-8")
else:
    user_text = st.text_area("Or paste the case text here manually:")


if st.button("🔍 Predict Outcome"):
    if len(user_text.strip()) < 20:
        st.warning("Please enter detailed case facts!")
        st.stop()

    with st.spinner("Running prediction..."):
        prediction, probs = predict_with_prob(user_text, tokenizer, clf_model)

    st.success(f"✅ Prediction: **{prediction}**")

    # Chart
    fig, ax = plt.subplots()
    ax.bar(["Not Favourable", "Favourable"], probs)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Similar cases
    with st.spinner("Finding similar cases..."):
        sims = get_similar_cases(user_text, embeddings, case_texts, top_k)

    st.subheader("📚 Most Similar Cases:")
    for i, c in enumerate(sims, start=1):
        st.markdown(f"### 🔹 Case {i}")
        st.write(c["text"][:800] + "...")
        st.caption(f"Similarity: {c['similarity']:.4f}")

    # Report Download
    pdf_file = generate_pdf(prediction, probs, sims, user_text)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download PDF Report",
            f,
            file_name="litigation_report.pdf",
            mime="application/pdf"
        )
