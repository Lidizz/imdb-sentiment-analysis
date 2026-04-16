"""
IMDB Sentiment Analysis: Overview (home page).

Run from project root:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

# ── Path bootstrap (works regardless of where streamlit is launched) ──────────
_APP_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import plotly.express as px
import streamlit as st

from _shared import (
    FIGURES_DIR,
    load_comparison_metrics,
    pct,
    fmt_f1,
    fmt_time,
)

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 IMDB Sentiment Analysis")
st.markdown(
    "**Binary sentiment classification on 50,000 IMDB movie reviews.**  \n"
    "We compare four classic ML models (TF-IDF features) against an LSTM deep learning model "
    "and a pre-trained Transformer baseline."
)
st.caption(
    "AI3000R · Artificial Intelligence for Business Applications · Spring 2026 · "
    "Lidor Shachar & Christin Wøien Skattum"
)

st.divider()

# ── Key questions ─────────────────────────────────────────────────────────────
st.subheader("What we set out to answer")
q1, q2, q3 = st.columns(3)
q1.info(
    "**Can a simple TF-IDF + Logistic Regression baseline match a neural network?**\n\n"
    "Classic ML has much lower training cost, does it reach the same accuracy?"
)
q2.info(
    "**What does an LSTM capture that bag-of-words models cannot?**\n\n"
    "Sequence models preserve word order. Does this matter for sentiment?"
)
q3.info(
    "**How does a purpose-trained model compare to a pre-trained Transformer out of the box?**\n\n"
    "Zero-shot DistilBERT vs our trained models, and what breaks when input format doesn't match."
)

st.divider()

# ── Dataset summary ───────────────────────────────────────────────────────────
st.subheader("Dataset: IMDB Movie Reviews")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Reviews",    "50,000")
m2.metric("Positive Reviews", "25,000 (50%)")
m3.metric("Negative Reviews", "25,000 (50%)")
m4.metric("Train / Val / Test", "70 / 15 / 15 %")
m5.metric("Split sizes",     "35,000 / 7,500 / 7,500")

st.divider()

# ── Pipeline overview ─────────────────────────────────────────────────────────
st.subheader("Project Pipeline: 5 Notebooks")
nb_steps = [
    ("📊", "NB01",
     "Data Exploration",
     "Class balance · review length distributions · HTML prevalence · duplicates"),
    ("🔧", "NB02",
     "Preprocessing",
     "HTML removal · lowercase · special chars · stopwords · POS-aware lemmatisation"),
    ("📐", "NB03",
     "Classic ML Models",
     "TF-IDF (unigrams+bigrams, 10K features) · LogReg · Naive Bayes · SVM · Random Forest"),
    ("🧠", "NB04",
     "Deep Learning (LSTM)",
     "Keras Tokenizer (20K vocab) · Embedding 128-dim · LSTM 128 units · Dropout 0.3"),
    ("📈", "NB05",
     "Model Comparison",
     "All models + DistilBERT zero-shot · confusion matrices · ROC · error analysis"),
]
cols = st.columns(5)
for col, (icon, nb, title, desc) in zip(cols, nb_steps):
    with col:
        st.markdown(f"### {icon} {nb}")
        st.markdown(f"**{title}**")
        st.caption(desc)

st.divider()

# ── Results summary ───────────────────────────────────────────────────────────
st.subheader("Results Summary")

try:
    df = load_comparison_metrics()

    # Formatted display table
    disp = df.copy()
    disp["Accuracy"]       = disp["accuracy"].apply(pct)
    disp["F1"]             = disp["f1"].apply(fmt_f1)
    disp["Precision"]      = disp["precision"].apply(pct)
    disp["Recall"]         = disp["recall"].apply(pct)
    disp["Train Time"]     = disp["training_time_s"].apply(fmt_time)
    disp["Inf / sample"]   = disp["inference_per_sample_ms"].apply(
        lambda v: f"{float(v):.2f} ms" if pd.notna(v) else "-"
    )
    disp = disp.rename(columns={"model": "Model"})
    st.dataframe(
        disp[["Model", "Accuracy", "F1", "Precision", "Recall", "Train Time", "Inf / sample"]],
        use_container_width=True,
        hide_index=True,
    )

    # Training time vs accuracy scatter (only trained models with timing)
    timed = df[df["training_time_s"].notna()].copy()
    timed["Accuracy %"] = timed["accuracy"] * 100
    timed["Label"] = timed["model"].str.replace(r" \(.*\)", "", regex=True)

    fig = px.scatter(
        timed,
        x="training_time_s",
        y="Accuracy %",
        text="Label",
        color="model",
        size_max=18,
        log_x=True,
        labels={
            "training_time_s": "Training time (seconds, log scale)",
            "Accuracy %": "Accuracy (%)",
        },
        title="Training Time vs Test Accuracy (log scale)",
        height=420,
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=14, line=dict(width=1, color="white")),
    )
    fig.update_layout(showlegend=False, title_x=0.0)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Key finding:** Logistic Regression matches the LSTM in accuracy at a fraction of the training cost "
        "(0.5s vs 11 min). Word-choice signal dominates sequence-order signal for IMDB binary sentiment."
    )

except FileNotFoundError:
    st.warning(
        "Results CSVs not found. Run all notebooks first (`python run_pipeline.py`), "
        "then refresh this page."
    )

st.divider()
st.caption(
    "Navigate the project using the sidebar: Preprocessing → Classic ML → Deep Learning "
    "→ Model Comparison → Live Demo."
)
