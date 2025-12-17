# app/streamlit_app.py

import streamlit as st
import pickle
import os
import sys

# Add src/ to sys.path so we can import project modules
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import normalize_reviews

# Paths to persisted model artifacts (update these if your filenames differ)
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join("models", "log_reg_model.pkl")

@st.cache_resource
def load_vectorizer():
    # Load the pretrained TF-IDF vectorizer
    with open(VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    # Load the trained Logistic Regression model
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def main():
    st.title("IMDB Sentiment Analysis Demo")

    st.write("Enter a movie review below and click 'Predict Sentiment':")
    user_text = st.text_area("Review", height=150)

    if st.button("Predict Sentiment"):
        if not user_text.strip():
            st.warning("Please enter some text before predicting.")
            return

        # Wrap single text into a DataFrame and apply normalization from pipeline
        import pandas as pd
        input_df = pd.DataFrame({"review": [user_text]})
        clean_df = normalize_reviews(input_df)

        # Transform using the pre-trained TF-IDF vectorizer
        vectorizer = load_vectorizer()
        X_input = vectorizer.transform(clean_df["review"])

        # Predict using the loaded logistic regression model
        model = load_model()
        pred = model.predict(X_input)[0]

        label = "Positive" if pred == 1 else "Negative"
        st.subheader(f"Predicted Sentiment: {label}")

if __name__ == "__main__":
    main()
