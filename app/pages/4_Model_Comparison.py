"""
Full model comparison and error analysis page.
"""
import sys
from pathlib import Path

_APP_DIR      = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    FIGURES_DIR,
    load_comparison_metrics,
    load_misclassified,
    pct,
    fmt_f1,
    fmt_time,
)

st.set_page_config(
    page_title="Model Comparison | IMDB Sentiment",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Unified Model Comparison")
st.markdown(
    "All models evaluated on the same held-out test rows using identical preprocessing. "
    "Includes four classic TF-IDF models, LSTM, and a pre-trained DistilBERT baseline."
)
st.divider()

try:
    df = load_comparison_metrics()
except FileNotFoundError:
    st.error("Model comparison metrics not found. Run notebook 05 first.")
    st.stop()

# ── Results table ─────────────────────────────────────────────────────────────
st.subheader("Full Results Table")

disp = df.copy()
disp["Accuracy"]      = disp["accuracy"].apply(pct)
disp["F1"]            = disp["f1"].apply(fmt_f1)
disp["Precision"]     = disp["precision"].apply(pct)
disp["Recall"]        = disp["recall"].apply(pct)
disp["Train Time"]    = disp["training_time_s"].apply(fmt_time)
disp["Inf / sample"]  = disp["inference_per_sample_ms"].apply(
    lambda v: f"{float(v):.2f} ms" if pd.notna(v) else "N/A"
)
disp = disp.rename(columns={"model": "Model", "evaluated_on": "Evaluated on"})

st.dataframe(
    disp[["Model", "Evaluated on", "Accuracy", "F1", "Precision", "Recall", "Train Time", "Inf / sample"]],
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Accuracy / F1 figure ──────────────────────────────────────────────────────
st.subheader("Accuracy & F1 Comparison")
acc_f1_path = FIGURES_DIR / "accuracy_f1_comparison.png"
if acc_f1_path.exists():
    st.image(str(acc_f1_path), use_container_width=True)
else:
    # Plotly fallback
    sorted_df = df.sort_values("f1", ascending=True)
    fig = go.Figure()
    fig.add_bar(
        y=sorted_df["model"], x=sorted_df["accuracy"],
        orientation="h", name="Accuracy",
        text=sorted_df["accuracy"].apply(pct), textposition="outside",
    )
    fig.add_bar(
        y=sorted_df["model"], x=sorted_df["f1"],
        orientation="h", name="F1",
        text=sorted_df["f1"].apply(fmt_f1), textposition="outside",
    )
    fig.update_layout(barmode="group", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Precision / Recall breakdown ──────────────────────────────────────────────
pr_path = FIGURES_DIR / "precision_recall_f1_breakdown.png"
if pr_path.exists():
    st.subheader("Precision / Recall / F1 Breakdown")
    st.image(str(pr_path), use_container_width=True)
    st.caption(
        "DistilBERT on preprocessed text shows high precision (0.916) and low recall (0.596), "
        "it defaults to predicting 'negative' on malformed input, catching fewer positives."
    )
    st.divider()

# ── ROC curves ────────────────────────────────────────────────────────────────
roc_path = FIGURES_DIR / "roc_curves.png"
if roc_path.exists():
    st.subheader("ROC Curves")
    st.image(str(roc_path), use_container_width=True)
    st.caption(
        "Area under the ROC curve (AUC) measures discriminative ability independent of threshold. "
        "Logistic Regression and SVM show the highest AUC among trained models."
    )
    st.divider()

# ── Training & inference time ─────────────────────────────────────────────────
st.subheader("Training & Inference Time")
col1, col2 = st.columns(2)

train_path = FIGURES_DIR / "training_time_comparison.png"
inf_path   = FIGURES_DIR / "inference_time_comparison.png"

if train_path.exists():
    with col1:
        st.markdown("**Training time (5 trained models)**")
        st.image(str(train_path), use_container_width=True)

if inf_path.exists():
    with col2:
        st.markdown("**Inference time (all 7 models)**")
        st.image(str(inf_path), use_container_width=True)

if train_path.exists() or inf_path.exists():
    st.caption(
        "Linear models (LR, SVM, NB) train in under 1 second. "
        "LSTM requires ~11 minutes (CPU), but runs in ~2 minutes on T4 GPU. "
        "DistilBERT CPU inference is ~384–480 ms/review; GPU brings this to ~10 ms/review."
    )
    st.divider()

# ── Confusion matrices ────────────────────────────────────────────────────────
cm_path = FIGURES_DIR / "confusion_matrices.png"
if cm_path.exists():
    st.subheader("Confusion Matrices: Trained Models (NB05 Evaluation)")
    st.image(str(cm_path), use_container_width=True)
    st.divider()

# ── DistilBERT explanation ────────────────────────────────────────────────────
st.subheader("DistilBERT: Preprocessing Mismatch Explained")

bert_rows = df[df["model"].str.contains("DistilBERT", na=False)]
raw_row  = bert_rows[bert_rows["model"].str.contains("raw", na=False, case=False)]
pre_row  = bert_rows[~bert_rows["model"].str.contains("raw", na=False, case=False)]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**On raw text (as intended)**")
    if not raw_row.empty:
        r = raw_row.iloc[0]
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",  pct(r["accuracy"]))
        m2.metric("F1",        fmt_f1(r["f1"]))
    st.success("Raw sentences match the SST-2 fine-tuning data format.")

with c2:
    st.markdown("**On preprocessed text (mismatch)**")
    if not pre_row.empty:
        r = pre_row.iloc[0]
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",  pct(r["accuracy"]))
        m2.metric("F1",        fmt_f1(r["f1"]))
    st.error("Stopword-stripped input breaks BERT's contextual representations.")

with st.expander("Detailed explanation of the mismatch"):
    st.markdown(
        """
        **Why does preprocessing hurt DistilBERT so much?**

        `distilbert-base-uncased-finetuned-sst-2-english` was fine-tuned on the
        Stanford Sentiment Treebank (SST-2), a dataset of complete, grammatical English sentences.

        Our preprocessing pipeline strips stopwords and lemmatises all text.
        This transforms:
        > *"The acting in this film is not good at all."*

        into:

        > *"act film not good"*

        This violates the model's input expectations in two ways:
        1. **No grammatical structure**: BERT uses multi-head attention across all token positions.
           Removing function words destroys positional and syntactic context.
        2. **Negation broken**: `"not"` survives stopword removal, but the surrounding structure
           needed to interpret it correctly is gone.

        **Precision = 0.916, Recall = 0.596** on preprocessed text shows the model defaults
        to predicting 'negative', it can't parse the truncated input and plays it safe.

        **Bottom line:** DistilBERT on raw text (~92.7% accuracy) is the fair number.
        The 77.7% result is an artifact of input format incompatibility, not architectural weakness.
        """
    )

bert_cm_path = FIGURES_DIR / "confusion_matrix_bert_comparison.png"
if bert_cm_path.exists():
    st.image(str(bert_cm_path), use_container_width=True)

st.divider()

# ── Error analysis ────────────────────────────────────────────────────────────
st.subheader("Error Analysis: What Do Models Get Wrong?")
st.markdown(
    "Sampled misclassifications from the evaluation set. "
    "Common failure patterns: **negation**, **mixed sentiment**, **domain-specific vocabulary**, "
    "**long-context reversals**."
)

try:
    misclassified = load_misclassified()
    models_available = sorted(misclassified["model"].unique().tolist())
    selected_model = st.selectbox("Select a model to inspect:", models_available)

    model_errors = misclassified[misclassified["model"] == selected_model].head(5)

    for _, row in model_errors.iterrows():
        true_label = "Positive ✓" if row["true_label"] == 1 else "Negative ✗"
        pred_label = "Positive" if row["predicted_label"] == 1 else "Negative"
        score      = float(row["model_score"])

        header = f"True: **{true_label}** → Predicted: **{pred_label}** (score: {score:.3f})"
        with st.expander(header):
            text = str(row["original_text"])
            st.write(text[:600] + "..." if len(text) > 600 else text)

except FileNotFoundError:
    st.info("Misclassified samples file not found. Run notebook 05 first.")

st.divider()

# ── Summary ───────────────────────────────────────────────────────────────────
st.subheader("Summary of Findings")

with st.container(border=True):
    st.markdown(
        """
        | Finding | Details |
        |---|---|
        | **Best trained model** | Logistic Regression (TF-IDF): ~88.7% accuracy, 0.5s training |
        | **LSTM vs LogReg** | Comparable accuracy, ~1,000× more training time |
        | **TF-IDF dominance** | Word-choice signal > sequence-order signal for IMDB sentiment |
        | **Random Forest weakness** | Sparse TF-IDF features poorly suited to tree-based splitting |
        | **DistilBERT fair score** | ~92.7% on raw text (zero-shot, no fine-tuning) |
        | **DistilBERT mismatch** | ~77.7% on preprocessed text: input format incompatibility |
        | **Fastest inference** | Classic TF-IDF models: < 0.01 ms/review (batch) |
        """
    )
