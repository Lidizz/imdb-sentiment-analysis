"""
Shared utilities for all app pages.

Every page inserts the app/ directory into sys.path before importing this module,
so the path resolution and cached loaders are available everywhere.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load as _jload

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
METRICS_DIR  = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR  = PROJECT_ROOT / "results" / "figures"


# ── Metric loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_comparison_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "model_comparison_metrics.csv")


@st.cache_data
def load_classic_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "classic_models_metrics.csv")


@st.cache_data
def load_lstm_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "dl_lstm_metrics.csv")


@st.cache_data
def load_feature_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "feature_engineering_metrics.csv")


@st.cache_data
def load_misclassified() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "misclassified_samples.csv")


# ── Model loaders ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TF-IDF vectorizer...")
def get_vectorizer():
    return _jload(MODELS_DIR / "tfidf_vectorizer_classic_models.joblib")


@st.cache_resource(show_spinner="Loading classic models...")
def get_classic_models() -> dict:
    return {
        "Logistic Regression": _jload(MODELS_DIR / "logistic_regression.joblib"),
        "Linear SVM":          _jload(MODELS_DIR / "linear_svm.joblib"),
        "Naive Bayes":         _jload(MODELS_DIR / "naive_bayes.joblib"),
        "Random Forest":       _jload(MODELS_DIR / "random_forest.joblib"),
    }


@st.cache_resource(show_spinner="Loading LSTM model (first load takes a moment)...")
def get_lstm():
    from keras.models import load_model as _keras_load
    model     = _keras_load(str(MODELS_DIR / "lstm_final.keras"))
    tokenizer = _jload(MODELS_DIR / "tokenizer_lstm.joblib")
    return model, tokenizer


# ── Formatting helpers ─────────────────────────────────────────────────────────
def pct(v: float) -> str:
    """Format 0.887 -> '88.7%'"""
    return f"{float(v) * 100:.1f}%"


def fmt_f1(v: float) -> str:
    """Format 0.887 -> '0.887'"""
    return f"{float(v):.3f}"


def fmt_time(s) -> str:
    """Format seconds to human-readable: 0.5 -> '0.5s', 669 -> '11m 09s'"""
    if pd.isna(s):
        return "N/A"
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s"


def fmt_ms(v) -> str:
    """Format ms per sample value."""
    if pd.isna(v):
        return "N/A"
    return f"{float(v):.2f} ms"
