# app_full.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import io
import time
import docx2txt
import PyPDF2
from sentence_transformers import SentenceTransformer

# -----------------------
# Config
# -----------------------
MODEL_DIR = Path("fine_tuned_model")       # local fine-tuned classification model
DATA_PATH = Path("cases_database.csv")  # combined dataset (option A)
EMBED_CACHE = Path("embeddings.npy")
ID_CACHE = Path("ids.npy")
BACKEND_CACHE_INFO = Path("embed_backend.txt")  # stores which backend used when embeddings were created
BATCH_SIZE_EMBED = 32
MAX_TOKENS = 512  # tokenizer max length (truncate long texts)
DEFAULT_SBERT = "all-mpnet-base-v2"  # SBERT model id (sentence-transformers package)

# -----------------------
# Helpers: load classifier model & tokenizer
# -----------------------
@st.cache_resource
def load_classifier(model_dir: str = str(MODEL_DIR)):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    clf_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    clf_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_model.to(device)

    # For encoder-based embeddings, we will try to use base_model if available
    if hasattr(clf_model, "base_model"):
        encoder = clf_model.base_model
    else:
        encoder = None  # fallback: we will not use encoder embedding if None

    return tokenizer, clf_model, encoder, device

# -----------------------
# Mean-pooling helper (for encoder output)
# -----------------------
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts

# -----------------------
# Encode texts using encoder (mean pooled) from classifier
# -----------------------
def encode_with_encoder(texts, tokenizer, encoder, device, batch_size=BATCH_SIZE_EMBED):
    all_emb = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=MAX_TOKENS,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = getattr(outputs, "last_hidden_state", None) or outputs[0]
            emb = mean_pooling(last_hidden, attention_mask)  # (batch, hidden)
            emb = emb.cpu().numpy()
            all_emb.append(emb)
    all_emb = np.vstack(all_emb)
    norms = np.linalg.norm(all_emb, axis=1, keepdims=True).clip(min=1e-9)
    all_emb = all_emb / norms
    return all_emb

# -----------------------
# Encode texts using Sentence-Transformers (SBERT)
# -----------------------
def encode_with_sbert(texts, sbert_model_name=DEFAULT_SBERT, batch_size=BATCH_SIZE_EMBED):
    sbert = SentenceTransformer(sbert_model_name)
    emb = sbert.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    emb = emb / norms
    return emb

# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_cases(csv_path: str = str(DATA_PATH)):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column with case text.")
    if "case_id" not in df.columns:
        df["case_id"] = df.index.astype(str)
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    else:
        df["label"] = ""
    # Create snippet for display
    df["snippet"] = df["text"].astype(str).str.replace("\n", " ").str.slice(0, 800)
    # Normalize citation fields if present
    for col in ["title", "court", "year"]:
        if col not in df.columns:
            df[col] = ""
    return df

# -----------------------
# Embedding cache handling
# -----------------------
def load_or_build_embeddings(df, tokenizer, encoder, device, backend="sbert", sbert_name=DEFAULT_SBERT, force_rebuild=False):
    """
    backend: "sbert" or "encoder"
    """
    need_build = force_rebuild or (not EMBED_CACHE.exists()) or (not ID_CACHE.exists()) or (not BACKEND_CACHE_INFO.exists())
    if not need_build:
        # check backend matches; if mismatch, force rebuild
        cached_backend = BACKEND_CACHE_INFO.read_text().strip() if BACKEND_CACHE_INFO.exists() else ""
        if cached_backend != backend:
            need_build = True

    if need_build:
        st.info("Building embeddings for corpus (this may take a few minutes)...")
        texts = df["text"].astype(str).tolist()
        if backend == "sbert":
            embeddings = encode_with_sbert(texts, sbert_model_name=sbert_name)
        else:  # encoder
            if encoder is None:
                raise ValueError("Encoder backend not available for this model.")
            embeddings = encode_with_encoder(texts, tokenizer, encoder, device)
        np.save(EMBED_CACHE, embeddings)
        np.save(ID_CACHE, np.array(df["case_id"].astype(str)))
        BACKEND_CACHE_INFO.write_text(backend)
        return embeddings, np.array(df["case_id"].astype(str))
    else:
        embeddings = np.load(EMBED_CACHE)
        ids = np.load(ID_CACHE)
        return embeddings, ids

# -----------------------
# Prediction function
# -----------------------
def predict_text(text, tokenizer, clf_model, device):
    enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_TOKENS, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = clf_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    id2label = getattr(clf_model.config, "id2label", None)
    if id2label:
        pred_label = id2label.get(pred, str(pred))
    else:
        pred_label = "WIN" if pred == 1 else "LOSS"
    return pred_label, float(probs[pred]), probs.tolist()

# -----------------------
# Similarity search
# -----------------------
def find_similar(query_emb, corpus_embs, top_k=5):
    sims = cosine_similarity(query_emb.reshape(1, -1), corpus_embs).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    return top_idx, sims[top_idx]

# -----------------------
# File reading helpers (txt, pdf, docx)
# -----------------------
def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    t = uploaded_file.type
    try:
        if "text" in t or uploaded_file.name.lower().endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        if uploaded_file.name.lower().endswith(".pdf") or "pdf" in t:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        if uploaded_file.name.lower().endswith(".docx") or "word" in t:
            file_bytes = uploaded_file.read()
            return docx2txt.process(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Failed to extract text from file: {e}")
        return ""
    st.warning("Unsupported file type; please upload .txt, .pdf or .docx")
    return ""

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Litigation Predictor — Full", layout="wide")
st.title("⚖️ Litigation Outcome Predictor — Full feature app")

# Sidebar: settings & controls
st.sidebar.header("App Settings")
backend_choice = st.sidebar.selectbox("Embedding backend", options=["sbert", "encoder"], index=0,
                                      help="sbert: Sentence-Transformers SBERT (recommended). encoder: mean-pooled encoder from fine-tuned model (if available).")
sbert_model_name = st.sidebar.text_input("SBERT model (if using sbert backend)", value=DEFAULT_SBERT)
rebuild_btn = st.sidebar.button("Regenerate embeddings (force rebuild)")
page_size = st.sidebar.number_input("Results per page", min_value=1, max_value=50, value=5)
confidence_threshold = st.sidebar.slider("Confidence threshold (filter similar cases by prediction confidence)", 0.0, 1.0, 0.0, 0.01)
show_citations = st.sidebar.checkbox("Show citation fields (case_id/title/court/year) if present", value=True)
pagination_enabled = st.sidebar.checkbox("Enable pagination for similar cases display", value=True)

# Load models
with st.spinner("Loading classifier..."):
    tokenizer, clf_model, encoder, device = load_classifier()

# Load dataset
try:
    df_cases = load_cases()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.sidebar.markdown(f"**Cases in DB:** {len(df_cases):,}")

# Build or load embeddings
with st.spinner("Preparing corpus embeddings..."):
    try:
        embeddings, ids = load_or_build_embeddings(df_cases, tokenizer, encoder, device,
                                                   backend=backend_choice, sbert_name=sbert_model_name,
                                                   force_rebuild=rebuild_btn)
    except Exception as e:
        st.error(f"Embedding build/load failed: {e}")
        st.stop()

# Main layout
left, right = st.columns([2, 1])

with left:
    st.subheader("Enter case text (paste or upload file)")
    user_text = st.text_area("Case facts / excerpt", height=300)
    uploaded = st.file_uploader("Or upload a case file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    if uploaded:
        extracted = read_uploaded_file(uploaded)
        if extracted:
            st.success(f"Extracted {len(extracted):,} characters from uploaded file.")
            user_text = extracted

    st.write("")  # spacer
    controls = st.columns([1, 1, 1])
    with controls[0]:
        n_similar = st.number_input("Number of similar cases to retrieve", min_value=1, max_value=50, value=5)
    with controls[1]:
        run_button = st.button("Predict & Search Similar Cases")
    with controls[2]:
        reset_cache = st.button("Clear embedding cache")

with right:
    st.subheader("Model & corpus")
    st.write(f"Classifier folder: `{MODEL_DIR}`")
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    st.write(f"Device: {device_name}")
    st.write(f"Embedding backend: **{backend_choice}**")
    if EMBED_CACHE.exists():
        mtime = time.ctime(EMBED_CACHE.stat().st_mtime)
        st.write("Embeddings cached:", EMBED_CACHE.name, f"(last updated {mtime})")
    if reset_cache:
        try:
            EMBED_CACHE.unlink(missing_ok=True)
            ID_CACHE.unlink(missing_ok=True)
            BACKEND_CACHE_INFO.unlink(missing_ok=True)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed clearing cache: {e}")

# Action
if run_button:
    if not user_text or not user_text.strip():
        st.warning("Please paste case text or upload a file.")
    else:
        with st.spinner("Running classifier..."):
            pred_label, pred_conf, all_probs = predict_text(user_text, tokenizer, clf_model, device)

        st.markdown("### 🔮 Prediction")
        if pred_conf >= 0.5:
            label_color = "green" if ("win" in pred_label.lower() or "allow" in pred_label.lower() or pred_label.upper() == "WIN") else "red"
        else:
            label_color = "orange"
        st.markdown(f"<div style='font-size:20px'>Predicted: <span style='color:{label_color}; font-weight:700'>{pred_label}</span></div>", unsafe_allow_html=True)
        st.write(f"Confidence: {pred_conf:.3f}")
        st.write("Full probability vector:", all_probs)

        # If user wants to filter by confidence threshold, check it
        if pred_conf < confidence_threshold:
            st.warning(f"Model confidence ({pred_conf:.3f}) is below threshold ({confidence_threshold}); you may want to interpret result cautiously.")

        # Encode query using chosen backend
        with st.spinner("Encoding query & computing similarity..."):
            try:
                if backend_choice == "sbert":
                    query_emb = encode_with_sbert([user_text], sbert_model_name)
                else:
                    if encoder is None:
                        st.error("Encoder backend not available for this model. Choose SBERT backend.")
                        st.stop()
                    query_emb = encode_with_encoder([user_text], tokenizer, encoder, device)
            except Exception as e:
                st.error(f"Failed to compute query embedding: {e}")
                st.stop()

            top_idx, top_scores = find_similar(query_emb[0], embeddings, top_k=n_similar*5)  # fetch extra for filtering/pagination

        # Build results list (respect confidence threshold for predicted class of corpus rows if desired)
        results = []
        for idx, score in zip(top_idx, top_scores):
            matched_id = ids[idx]
            row = df_cases[df_cases["case_id"].astype(str) == str(matched_id)].iloc[0]
            # optional: if you want to compute classifier prediction for each matched case (to show model label), you can
            # but usually label column already has the gold label; we will show that.
            result = {
                "case_id": row.get("case_id", ""),
                "title": row.get("title", ""),
                "court": row.get("court", ""),
                "year": row.get("year", ""),
                "outcome": row.get("label", ""),
                "similarity": float(score),
                "snippet": row.get("snippet", ""),
                "text": row.get("text", "")
            }
            results.append(result)

        # Optionally filter by similarity score or by classifier confidence of the matched case (we currently don't recompute classifier for all matches)
        # Sort and keep top n_similar
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:n_similar]

        # Display results with pagination
        st.markdown("### 📚 Similar Cases")
        if len(results) == 0:
            st.info("No similar cases found.")
        else:
            # Prepare CSV download
            df_out = pd.DataFrame(results)[["case_id", "title", "court", "year", "outcome", "similarity", "snippet"]]
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download similar cases (CSV)", csv_bytes, file_name="similar_cases.csv", mime="text/csv")

            # Pagination display
            if pagination_enabled and len(results) > page_size:
                total_pages = (len(results) + page_size - 1) // page_size
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                start = (page - 1) * page_size
                end = start + page_size
                display_slice = results[start:end]
                st.write(f"Showing {start+1}-{min(end, len(results))} of {len(results)} results (page {page}/{total_pages})")
            else:
                display_slice = results

            # Show each result with color-coded outcome
            for i, it in enumerate(display_slice, start=1):
                outcome = str(it.get("outcome", "")).lower()
                if "win" in outcome or "allow" in outcome or outcome == "1":
                    color = "green"
                elif "loss" in outcome or "dismiss" in outcome or outcome == "0":
                    color = "red"
                else:
                    color = "black"

                header = f"#{i} — Case ID: {it['case_id']} — sim: {it['similarity']:.3f}"
                exp = st.expander(header, expanded=False)
                with exp:
                    if show_citations:
                        meta = ""
                        if it.get("title"):
                            meta += f"**Title:** {it['title']}  \n"
                        if it.get("court"):
                            meta += f"**Court:** {it['court']}  \n"
                        if it.get("year"):
                            meta += f"**Year:** {it['year']}  \n"
                        if meta:
                            st.markdown(meta)
                    st.markdown(f"**Outcome (gold):** <span style='color:{color}; font-weight:700'>{it['outcome']}</span>", unsafe_allow_html=True)
                    st.markdown("**Snippet:**")
                    st.write(it["snippet"])
                    if st.button(f"Show full text for {it['case_id']}", key=f"full_{it['case_id']}"):
                        st.write(it["text"])

st.markdown("---")
st.write("Tips: Use SBERT backend for best semantic similarity out-of-the-box. Encoder backend (mean-pooled) can work, but SBERT is usually more robust.")
