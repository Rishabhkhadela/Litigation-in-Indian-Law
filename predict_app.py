import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = "fine_tuned_model"       # Local model folder
CASES_DB = "cases_database.csv"      # CSV of previous cases


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_case_database():
    df = pd.read_csv(CASES_DB)
    df = df.dropna(subset=["text"])
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    return df, vectorizer, tfidf_matrix


def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    return "WIN" if pred == 1 else "LOSS"


def get_similar_cases(text, df, vectorizer, tfidf_matrix, top_k=3):
    query_vec = vectorizer.transform([text])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    return df.iloc[top_indices], similarities[top_indices]


# ----------------- STREAMLIT UI -----------------

st.title("⚖️ Legal Case Outcome Predictor with Similar Case Search")

# Load model once
tokenizer, model = load_model()
df_cases, vectorizer, tfidf_matrix = load_case_database()

text_input = st.text_area("Enter Case Description:", height=200)
top_k = st.slider("Number of similar cases to show:", 1, 10, 3)

if st.button("Predict & Find Similar Cases"):
    if text_input.strip() == "":
        st.warning("Please enter a case description first.")
    else:
        # Prediction
        result = predict(text_input, tokenizer, model)
        st.success(f"🔮 Prediction: **{result}**")

        # Similar case search
        st.subheader("📚 Most Similar Past Cases")

        similar_cases, scores = get_similar_cases(
            text_input, df_cases, vectorizer, tfidf_matrix, top_k=top_k
        )

        for i, (idx, row) in enumerate(similar_cases.iterrows()):
            st.markdown(f"### ✅ Case {i+1}")
            st.markdown(f"**Similarity:** {scores[i]:.2f}")
            st.markdown(f"**Outcome:** {row['outcome']}")
            st.markdown(f"**Case Text:**")
            st.write(row["text"])
            st.markdown("---")
