"""
Interactive live sentiment prediction demo.
"""
import sys
from pathlib import Path

_APP_DIR      = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib

import numpy as np
import streamlit as st
from scipy.special import expit as _sigmoid

from _shared import get_classic_models, get_lstm, get_vectorizer
from src.preprocessing import preprocess_text

st.set_page_config(
    page_title="Live Demo | IMDB Sentiment",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Live Sentiment Prediction Demo")
st.markdown(
    "Enter any movie review and see all five trained models predict its sentiment simultaneously.  \n"
    "Models are loaded from the saved artifacts in `models/`. "
    "Preprocessing is applied live using the same pipeline as training."
)
st.divider()

# ── Load models ───────────────────────────────────────────────────────────────
vectorizer, classic_models, lstm_model, lstm_tokenizer = None, {}, None, None
load_ok = True

col_status1, col_status2, col_status3 = st.columns(3)

try:
    vectorizer = get_vectorizer()
    col_status1.success("TF-IDF vectorizer ready")
except Exception as e:
    col_status1.error(f"TF-IDF vectorizer failed: {e}")
    load_ok = False

try:
    classic_models = get_classic_models()
    col_status2.success(f"{len(classic_models)} classic models ready")
except Exception as e:
    col_status2.error(f"Classic models failed: {e}")
    load_ok = False

try:
    lstm_model, lstm_tokenizer = get_lstm()
    col_status3.success("LSTM model ready")
except Exception as e:
    col_status3.warning(f"LSTM not available: {e}")

if not load_ok:
    st.error(
        "Some models failed to load. Make sure you have run all notebooks first "
        "so the `models/` directory is populated."
    )

st.divider()

# ── Example selector ──────────────────────────────────────────────────────────
EXAMPLES = {
    "Custom input: write your own": "",
    "Clearly positive": (
        "This film was an absolute masterpiece. The performances were breathtaking, "
        "the cinematography stunning, and the story emotionally resonant throughout. "
        "One of the best movies I have ever seen. Highly recommended!"
    ),
    "Clearly negative": (
        "Terrible waste of two hours. The plot made no sense whatsoever, "
        "the acting was wooden, and the dialogue was painfully bad. "
        "I nearly fell asleep halfway through. Avoid at all costs."
    ),
    "Mixed / ambiguous sentiment": (
        "The special effects were impressive and the first half showed real promise. "
        "Unfortunately the ending was a complete disappointment and felt horribly rushed. "
        "Not sure if I would bother watching it again."
    ),
    "Negation-heavy (interesting edge case)": (
        "This movie was not bad at all. I was not expecting to enjoy it, "
        "but it was never boring and the acting was not wooden like I feared. "
        "Not my favourite genre, but not a waste of time either."
    ),
    "Sarcasm (hard for all models)": (
        "Oh yes, what a truly wonderful film. The plot holes were so refreshingly "
        "original, and the acting so charmingly wooden. I cannot imagine a better way "
        "to spend two hours of my life than watching this."
    ),
}

example_key = st.selectbox("Load an example or write your own:", list(EXAMPLES.keys()))
default_text = EXAMPLES[example_key]

user_text = st.text_area(
    "Movie review text:",
    value=default_text,
    height=120,
    placeholder="Type or paste a movie review here...",
    key="review_input",
)

analyse_btn = st.button(
    "Analyse sentiment",
    type="primary",
    disabled=not user_text.strip() or not load_ok,
)

# ── Prediction run ────────────────────────────────────────────────────────────
if analyse_btn and user_text.strip():
    st.divider()

    # Preprocessing
    with st.spinner("Preprocessing text..."):
        preprocessed = preprocess_text(user_text)

    with st.expander("Preprocessing output (what the models receive)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original text:**")
            st.write(user_text)
        with c2:
            st.markdown("**After preprocessing pipeline:**")
            if preprocessed.strip():
                st.write(preprocessed)
                st.caption(
                    f"{len(user_text.split())} words → {len(preprocessed.split())} words "
                    f"({100 * (1 - len(preprocessed.split()) / max(len(user_text.split()), 1)):.0f}% reduction)"
                )
            else:
                st.warning(
                    "The input was reduced to an empty string after preprocessing. "
                    "Try a longer review with more content words."
                )

    # ── Guard: stop if preprocessing emptied the input ────────────────────────
    if not preprocessed.strip():
        st.stop()

    # ── TF-IDF features ───────────────────────────────────────────────────────
    X_vec = vectorizer.transform([preprocessed]) if vectorizer else None

    # ── Classic model predictions ─────────────────────────────────────────────
    results: dict[str, tuple[int, float]] = {}

    for model_name, model in classic_models.items():
        if X_vec is None:
            continue
        try:
            pred = int(model.predict(X_vec)[0])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_vec)[0]
                confidence = float(proba[1])
            elif hasattr(model, "decision_function"):
                score = float(model.decision_function(X_vec)[0])
                confidence = float(_sigmoid(score))
            else:
                confidence = 1.0 if pred == 1 else 0.0
            results[model_name] = (pred, confidence)
        except Exception as e:
            st.warning(f"{model_name} prediction error: {e}")

    # ── LSTM prediction ───────────────────────────────────────────────────────
    if lstm_model is not None and lstm_tokenizer is not None and preprocessed.strip():
        try:
            try:
                pad_mod       = importlib.import_module("keras.utils")
                pad_sequences = getattr(pad_mod, "pad_sequences")
            except (ImportError, AttributeError):
                pad_mod       = importlib.import_module("tensorflow.keras.preprocessing.sequence")
                pad_sequences = getattr(pad_mod, "pad_sequences")

            sequences = lstm_tokenizer.texts_to_sequences([preprocessed])
            X_seq     = pad_sequences(sequences, maxlen=300, padding="post", truncating="post")
            lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
            lstm_pred = 1 if lstm_prob >= 0.5 else 0
            results["LSTM (Keras)"] = (lstm_pred, lstm_prob)
        except Exception as e:
            st.warning(f"LSTM prediction error: {e}")

    # ── Majority verdict ──────────────────────────────────────────────────────
    MODEL_ORDER = ["Logistic Regression", "Linear SVM", "Naive Bayes", "Random Forest", "LSTM (Keras)"]
    available   = [m for m in MODEL_ORDER if m in results]

    pos_votes = sum(1 for m in available if results[m][0] == 1)
    neg_votes = len(available) - pos_votes
    majority  = "Positive" if pos_votes > neg_votes else "Negative"

    st.subheader("Predictions")
    if majority == "Positive":
        st.success(
            f"**Overall verdict: POSITIVE** - "
            f"{pos_votes}/{len(available)} models agree"
        )
    else:
        st.error(
            f"**Overall verdict: NEGATIVE** - "
            
            f"{neg_votes}/{len(available)} models agree"
        )

    # ── Per-model cards ───────────────────────────────────────────────────────
    model_cols = st.columns(len(available))
    for col, model_name in zip(model_cols, available):
        pred, confidence = results[model_name]
        label      = "POSITIVE" if pred == 1 else "NEGATIVE"
        # Bar shows confidence in predicted class
        bar_val    = confidence if pred == 1 else (1.0 - confidence)

        with col:
            with st.container(border=True):
                st.markdown(f"**{model_name}**")
                if pred == 1:
                    st.success(f"**{label}**")
                else:
                    st.error(f"**{label}**")
                st.progress(float(bar_val))
                st.caption(f"P(positive) = {confidence:.1%}")

    st.divider()

    # ── Probability table ─────────────────────────────────────────────────────
    with st.expander("Detailed probability table", expanded=False):
        table_rows = []
        for model_name in available:
            pred, conf = results[model_name]
            table_rows.append({
                "Model":        model_name,
                "Prediction":   "Positive" if pred == 1 else "Negative",
                "P(positive)":  f"{conf:.4f}",
                "P(negative)":  f"{1-conf:.4f}",
                "Confidence":   f"{max(conf, 1-conf):.1%}",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── Notes ─────────────────────────────────────────────────────────────────
    with st.expander("Notes on model confidence", expanded=False):
        st.markdown(
            """
            **Logistic Regression, Naive Bayes, Random Forest**: `predict_proba()` returns
            calibrated class probabilities.

            **Linear SVM**: `LinearSVC` does not natively output probabilities.
            P(positive) is estimated by applying a sigmoid function to the decision boundary score:
            `σ(score) = 1 / (1 + e^{-score})`. The sign of the score determines the predicted class.

            **LSTM**: The sigmoid output layer directly produces P(positive) ∈ [0, 1].
            Threshold: 0.5 (balanced precision/recall).

            **DistilBERT** is not included in the live demo: CPU inference takes ~2–3 seconds
            per review. Run notebook 05 or use Colab for DistilBERT evaluation.
            """
        )
