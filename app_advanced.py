# app_advanced.py
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

# -----------------------
# Config
# -----------------------
MODEL_DIR = Path("fine_tuned_model")       # local fine-tuned model folder
DATA_PATH = Path("cases_database.csv")  # combined dataset (Option A)
EMBED_CACHE = Path("embeddings.npy")
ID_CACHE = Path("ids.npy")
BATCH_SIZE_EMBED = 32
MAX_TOKENS = 512  # tokenizer max length (truncate long texts)

# -----------------------
# Helpers: load model & tokenizer
# -----------------------
@st.cache_resource
def load_models_and_tokenizer(model_dir: str = str(MODEL_DIR)):
    # Load tokenizer and classification model (local)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    clf_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    clf_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_model.to(device)

    # Encoder for embeddings: use base model inside the classification model
    # Many HF models expose `clf_model.base_model` which is the encoder (bert/roberta/xlm-roberta)
    if hasattr(clf_model, "base_model"):
        encoder = clf_model.base_model
    else:
        # fallback: use full model (should work but may include pooler/head) 
        encoder = clf_model

    encoder.eval()
    encoder.to(device)

    return tokenizer, clf_model, encoder, device

# -----------------------
# Compute embedding (mean-pool)
# -----------------------
def mean_pooling(last_hidden_state, attention_mask):
    # last_hidden_state: (batch, seq_len, hidden)
    # attention_mask: (batch, seq_len)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts

def encode_texts(texts, tokenizer, encoder, device, batch_size=BATCH_SIZE_EMBED):
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
            last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
            emb = mean_pooling(last_hidden, attention_mask)  # (batch, hidden)
            emb = emb.cpu().numpy()
            all_emb.append(emb)
    all_emb = np.vstack(all_emb)
    # normalize (cosine similarity convenience)
    norms = np.linalg.norm(all_emb, axis=1, keepdims=True).clip(min=1e-9)
    all_emb = all_emb / norms
    return all_emb

# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_cases(csv_path: str = str(DATA_PATH)):
    df = pd.read_csv(csv_path)
    # ensure columns
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column with case text.")
    # optional case_id
    if "case_id" not in df.columns:
        df["case_id"] = df.index.astype(str)
    # normalize label column if exists
    if "label" in df.columns:
        # keep as-is for display; training label mapping assumed earlier
        df["label"] = df["label"].astype(str)
    else:
        df["label"] = ""
    # create preview shorter snippet
    df["snippet"] = df["text"].astype(str).str.replace("\n", " ").str.slice(0, 600)
    return df

# -----------------------
# Build or load cached embeddings
# -----------------------
def load_or_build_embeddings(df, tokenizer, encoder, device, force_rebuild=False):
    if force_rebuild or not EMBED_CACHE.exists() or not ID_CACHE.exists():
        st.info("Building embeddings for corpus (this may take a few minutes)...")
        texts = df["text"].astype(str).tolist()
        embeddings = encode_texts(texts, tokenizer, encoder, device)
        np.save(EMBED_CACHE, embeddings)
        np.save(ID_CACHE, np.array(df["case_id"].astype(str)))
        return embeddings, np.array(df["case_id"].astype(str))
    else:
        embeddings = np.load(EMBED_CACHE)
        ids = np.load(ID_CACHE)
        return embeddings, ids

# -----------------------
# Predict function: returns label + probability distribution
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
    # Attempt to infer label names if present in model config
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
# UI
# -----------------------
st.set_page_config(page_title="Litigation Predictor + Similar Cases", layout="wide")
st.title("⚖️ Litigation Outcome Predictor — Advanced UI (INLegalBERT + Semantic Search)")

# Sidebar controls
st.sidebar.header("Settings")
show_advanced = st.sidebar.checkbox("Show advanced options", value=True)
top_k_default = st.sidebar.slider("Default number of similar cases", min_value=1, max_value=20, value=5)
regenerate = st.sidebar.button("Regenerate embeddings (force)")

# Load model + tokenizer + encoder
with st.spinner("Loading model and tokenizer..."):
    tokenizer, clf_model, encoder, device = load_models_and_tokenizer()

# Load dataset
try:
    df_cases = load_cases()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.sidebar.markdown(f"**Cases loaded:** {len(df_cases):,}")

# Build or load embeddings
force_rebuild = regenerate
with st.spinner("Preparing corpus embeddings..."):
    embeddings, ids = load_or_build_embeddings(df_cases, tokenizer, encoder, device, force_rebuild=force_rebuild)

# Main input area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter case details")
    user_text = st.text_area("Paste case facts / judgment excerpt:", height=260)
    # Option: upload file (txt/pdf/docx) could be added later

    options_col = st.columns([1,1,1])
    with options_col[0]:
        num_similar = st.number_input("How many similar cases to show", min_value=1, max_value=50, value=int(top_k_default))
    with options_col[1]:
        show_prob = st.checkbox("Show model probability", value=True)
    with options_col[2]:
        do_search = st.button("Predict & Find similar cases")

with col2:
    st.subheader("Model info")
    st.write(f"Model folder: `{MODEL_DIR}`")
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    st.write(f"Device: {device_name}")
    st.write("Corpus size:", len(df_cases))
    if EMBED_CACHE.exists():
        mtime = time.ctime(EMBED_CACHE.stat().st_mtime)
        st.write("Embeddings cached:", EMBED_CACHE.name, "(last updated: " + mtime + ")")
    if st.button("Clear cached embeddings"):
        try:
            EMBED_CACHE.unlink(missing_ok=True)
            ID_CACHE.unlink(missing_ok=True)
            st.success("Cache cleared. Next search will rebuild embeddings.")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

# Action
if do_search:
    if not user_text.strip():
        st.warning("Please enter case details first.")
    else:
        with st.spinner("Running prediction..."):
            pred_label, pred_conf, all_probs = predict_text(user_text, tokenizer, clf_model, device)

        # Display prediction
        st.markdown("### 🔮 Prediction")
        pcol1, pcol2 = st.columns([2,3])
        with pcol1:
            st.metric("Predicted outcome", pred_label)
        with pcol2:
            if show_prob:
                st.write("Confidence (predicted class):", f"{pred_conf:.3f}")
                # show probability bar
                st.progress(min(max(pred_conf, 0.0), 1.0))

        # Generate embedding for user_text using same encoder
        st.markdown("### 📚 Similar cases")
        with st.spinner("Encoding query and ranking..."):
            query_emb = encode_texts([user_text], tokenizer, encoder, device)  # returns normalized
            top_idx, top_scores = find_similar(query_emb[0], embeddings, top_k=num_similar)

        # Build results table
        results = []
        for rank, idx in enumerate(top_idx, start=1):
            # map idx to dataframe row: we have ids array aligned with df_cases order
            matched_case_id = ids[idx]
            row = df_cases[df_cases["case_id"].astype(str) == str(matched_case_id)].iloc[0]
            results.append({
                "rank": rank,
                "case_id": row["case_id"],
                "similarity": float(top_scores[rank-1]),
                "outcome": row.get("label", ""),
                "snippet": row.get("snippet", ""),
                "text": row.get("text", "")
            })

        # Show results with expanders and a download option
        csv_buffer = io.StringIO()
        csv_df = pd.DataFrame(results)
        csv_df.to_csv(csv_buffer, index=False)

        # Layout results
        st.download_button("⬇️ Download similar cases (CSV)", csv_buffer.getvalue().encode("utf-8"),
                           file_name="similar_cases.csv", mime="text/csv")
        st.markdown("---")

        for item in results:
            exp = st.expander(f"#{item['rank']} — Case ID: {item['case_id']}  — similarity: {item['similarity']:.3f}")
            with exp:
                st.markdown(f"**Outcome**: {item['outcome']}")
                st.markdown(f"**Similarity**: {item['similarity']:.4f}")
                st.markdown("**Snippet:**")
                st.write(item['snippet'])
                with st.expander("Show full case text"):
                    st.write(item['text'])


# Footer / tips
st.markdown("---")
st.write("Tips:")
st.write("- If corpus is large, first-time embedding build may take minutes. Use a GPU if available.")
st.write("- Use 'Regenerate embeddings' when you update dataset CSV.")
st.write("- If you want higher-quality semantic search, we can switch from mean-pooled encoder embeddings to a supervised embedding head or use a dedicated sentence-transformer model (I can add this).")
