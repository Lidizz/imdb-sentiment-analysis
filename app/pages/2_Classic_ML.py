"""
Feature engineering and Classic ML models page.
"""
import sys
from pathlib import Path

_APP_DIR      = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import plotly.express as px
import streamlit as st

from _shared import (
    FIGURES_DIR,
    load_classic_metrics,
    load_feature_metrics,
    pct,
    fmt_f1,
    fmt_time,
)

st.set_page_config(
    page_title="Classic ML | IMDB Sentiment",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Feature Engineering & Classic ML Models")
st.markdown(
    "TF-IDF features extracted from preprocessed text, then four classic model families "
    "trained on an identical feature matrix to isolate algorithm-specific differences."
)
st.divider()

# ── Feature engineering ───────────────────────────────────────────────────────
st.subheader("Feature Engineering: Finding the Best Representation")

try:
    feat = load_feature_metrics()
    bow_tfidf = feat[feat["experiment"] == "bow_vs_tfidf"].copy()
    ngram     = feat[feat["experiment"] == "tfidf_ngram_comparison"].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**BoW vs TF-IDF**: Logistic Regression, 10K features")
        fig1 = px.bar(
            bow_tfidf,
            x="representation",
            y="val_f1",
            color="representation",
            text=bow_tfidf["val_f1"].apply(lambda v: f"{v:.4f}"),
            labels={"val_f1": "Validation F1", "representation": ""},
            range_y=[0.855, 0.895],
            color_discrete_sequence=["#636EFA", "#EF553B"],
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(showlegend=False, height=360, margin=dict(t=30))
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(
            "TF-IDF outperforms BoW by **+1.7 pp F1** downweighting common words "
            "gives better per-review signal. TF-IDF adopted as default."
        )

    with col2:
        st.markdown("**Unigrams vs Bigrams**: TF-IDF + Logistic Regression")
        ngram_labels = ngram.copy()
        ngram_labels["label"] = ["Unigrams (1,1)", "Bigrams (1,2)"]
        fig2 = px.bar(
            ngram_labels,
            x="label",
            y="val_f1",
            color="label",
            text=ngram_labels["val_f1"].apply(lambda v: f"{v:.4f}"),
            labels={"val_f1": "Validation F1", "label": ""},
            range_y=[0.885, 0.892],
            color_discrete_sequence=["#00CC96", "#AB63FA"],
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, height=360, margin=dict(t=30))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Bigrams add **+0.2 pp F1** by capturing phrase-level sentiment cues like "
            "`'not good'`, `'very bad'`. Bigrams adopted as default."
        )

except FileNotFoundError:
    st.info("Feature engineering metrics not found. Run notebook 03 first.")

st.divider()

# ── Final TF-IDF config ───────────────────────────────────────────────────────
st.subheader("Final TF-IDF Configuration")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Max features",  "10,000")
c2.metric("N-gram range",  "(1, 2)")
c3.metric("Min doc freq",  "5 documents")
c4.metric("Max doc freq",  "95% of docs")
c5.metric("Feature matrix", "35,000 × 10,000")

st.caption(
    "All four models share the exact same feature matrix, "
    "performance differences are purely algorithm-specific, not feature-specific."
)

st.divider()

# ── Classic model results ─────────────────────────────────────────────────────
st.subheader("Classic Model Results: Full Validation Set (7,500 rows)")

try:
    metrics = load_classic_metrics()

    disp = metrics.copy()
    disp["Accuracy"]  = disp["accuracy"].apply(pct)
    disp["F1"]        = disp["f1"].apply(fmt_f1)
    disp["Precision"] = disp["precision"].apply(pct)
    disp["Recall"]    = disp["recall"].apply(pct)
    disp["Train Time"] = disp["training_time_s"].apply(fmt_time)
    disp = disp.rename(columns={"model": "Model"})
    st.dataframe(
        disp[["Model", "Accuracy", "F1", "Precision", "Recall", "Train Time"]],
        use_container_width=True,
        hide_index=True,
    )

    # F1 horizontal bar chart
    sorted_m = metrics.sort_values("f1", ascending=True)
    fig_bar = px.bar(
        sorted_m,
        x="f1",
        y="model",
        orientation="h",
        text=sorted_m["f1"].apply(fmt_f1),
        color="f1",
        color_continuous_scale="Blues",
        labels={"f1": "Validation F1", "model": ""},
        range_x=[0.84, 0.900],
        height=300,
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=200, r=60, t=30, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

except FileNotFoundError:
    st.info("Classic model metrics not found. Run notebook 03 first.")

st.markdown(
    "**Key insight:** Linear models (LogReg, SVM) dominate on sparse TF-IDF features. "
    "Random Forest is weakest, tree-based splitting struggles with 10,000-dimensional sparse vectors "
    "because individual features are rarely informative enough to split on."
)

st.divider()

# ── Confusion matrices ────────────────────────────────────────────────────────
st.subheader("Confusion Matrices")
cm_path = FIGURES_DIR / "classic_models_confusion_matrices.png"
if cm_path.exists():
    st.image(str(cm_path), use_container_width=True)
    st.caption(
        "Rows = true label (Positive top), columns = predicted label. "
        "All models show balanced FP/FN rates, confirming no systematic class bias."
    )
else:
    st.info("Confusion matrix figure not found. Run notebook 03 to generate it.")

st.divider()

# ── Top TF-IDF features ───────────────────────────────────────────────────────
st.subheader("Top TF-IDF Features: Logistic Regression Coefficients")
features_path = FIGURES_DIR / "tfidf_top_features.png"
if features_path.exists():
    st.image(str(features_path), use_container_width=True)
    st.caption(
        "High positive coefficients (right) drive positive predictions. "
        "High negative coefficients (left) drive negative predictions. "
        "The model learned meaningful sentiment vocabulary, not noise, confirming TF-IDF works as expected."
    )
else:
    st.info("Top features figure not found. Run notebook 03 to generate it.")

st.divider()

# ── Hyperparameter tuning ─────────────────────────────────────────────────────
st.subheader("Hyperparameter Tuning: Logistic Regression GridSearchCV")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Method",       "GridSearchCV")
c2.metric("CV folds",     "5-fold")
c3.metric("Best C found", "1.0  (default)")
c4.metric("Best CV F1",   "0.891")

st.info(
    "GridSearchCV over C ∈ {0.01, 0.1, **1.0**, 10.0, 100.0} found the default `C=1.0` to be optimal. "
    "No improvement over the untuned model, the default regularisation was already well-calibrated "
    "for this dataset size and feature dimensionality."
)

st.divider()

# ── RF feature importance ─────────────────────────────────────────────────────
st.subheader("Random Forest Feature Importance")
rf_fig = FIGURES_DIR / "random_forest_feature_importance.png"
if rf_fig.exists():
    st.image(str(rf_fig), use_container_width=True)
    st.caption(
        "Alternative interpretability view: tree-based importance ranking vs logistic regression coefficients. "
        "Both surfaces show similar top vocabulary, confirming consistent feature signal."
    )
