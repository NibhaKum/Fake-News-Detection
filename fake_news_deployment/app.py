import re
import string
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="AI-Powered Fake News Detection",
    page_icon="📰",
    layout="centered",
)

MODEL_PATH = Path("model/fake_news_model.pkl")
VECTORIZER_PATH = Path("model/tfidf_vectorizer.pkl")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run the model-export cell in the notebook "
            "and upload both files inside the model/ folder in GitHub."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def clean_text(text):
    # Keep this preprocessing identical to the training notebook.
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text


st.title("📰 AI-Powered Fake News Detection")
st.write(
    "Enter a news article to classify its text using the Random Forest "
    "model trained in this project."
)

st.info(
    "This tool identifies patterns learned from its training dataset. "
    "It does not independently verify whether a news claim is factually true."
)

try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

news_text = st.text_area(
    "News article text",
    placeholder="Paste the news article here...",
    height=240,
)

if st.button("Detect News", type="primary", use_container_width=True):
    if not news_text.strip():
        st.warning("Please enter some news text first.")
    else:
        cleaned = clean_text(news_text)
        features = vectorizer.transform([cleaned])
        prediction = int(model.predict(features)[0])

        st.subheader("Model prediction")
        if prediction == 1:
            st.success("Predicted category: REAL NEWS (Class 1)")
        else:
            st.error("Predicted category: FAKE NEWS (Class 0)")

        st.caption(
            "Prediction is based on patterns in the training dataset; "
            "please verify important news using reliable independent sources."
        )

st.divider()
st.caption(
    "Built with Python, Natural Language Processing, TF-IDF and Random Forest."
)
