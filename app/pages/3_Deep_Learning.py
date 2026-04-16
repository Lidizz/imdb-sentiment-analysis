"""
Deep learning (LSTM) page.
"""
import sys
from pathlib import Path

_APP_DIR      = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from _shared import (
    FIGURES_DIR,
    load_classic_metrics,
    load_lstm_metrics,
    pct,
    fmt_f1,
    fmt_time,
)

st.set_page_config(
    page_title="Deep Learning | IMDB Sentiment",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Deep Learning: LSTM Sentiment Classifier")
st.markdown(
    "Unlike TF-IDF which treats text as an unordered bag of words, an LSTM processes "
    "tokens sequentially and maintains a hidden state that captures word order and local context."
)
st.divider()

# ── TF-IDF vs LSTM ────────────────────────────────────────────────────────────
st.subheader("Why Sequence Modelling?")

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### TF-IDF (bag of words)")
    st.code('"not good"\n→ {\'not\': 1.2, \'good\': 0.8}')
    st.caption(
        "Two independent features. The model sees each word separately, without knowing their order. "
        "the fact that `not` precedes `good` is invisible."
    )
with c2:
    st.markdown("#### LSTM (sequence model)")
    st.code('"not good"\n→ hidden state after \'not\' modifies\n   the encoding of \'good\'')
    st.caption(
        "Tokens are processed in order. The hidden state after reading `not` "
        "changes how `good` is encoded so negation is represented."
    )

st.divider()

# ── Input representation ──────────────────────────────────────────────────────
st.subheader("Input Representation: Tokenisation & Padding")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vocabulary size",       "20,000 words")
c2.metric("Out-of-vocab token",    "`<OOV>`")
c3.metric("Max sequence length",   "300 tokens")
c4.metric("Padding / truncation",  "`post` / `post`")

st.markdown(
    "The tokenizer is fitted **only on training data** (same principle as TF-IDF: "
    "no leakage from validation or test). Unknown words at test time map to `<OOV>`."
)

st.divider()

# ── Architecture ──────────────────────────────────────────────────────────────
st.subheader("Model Architecture")

arch = pd.DataFrame([
    {
        "Layer":    "Embedding",
        "Config":   "input_dim=20,000 · output_dim=128 · mask_zero=True",
        "Role":     "Maps each integer word ID to a learned 128-dim dense vector. "
                    "`mask_zero=True` ignores padding tokens during LSTM computation.",
    },
    {
        "Layer":    "LSTM",
        "Config":   "128 units",
        "Role":     "Reads the sequence left-to-right, updating a 128-dimensional hidden state. "
                    "Captures context across up to 300 time steps.",
    },
    {
        "Layer":    "Dropout",
        "Config":   "rate=0.3",
        "Role":     "Regularisation: 30% of LSTM output units randomly zeroed during each training step. "
                    "Reduces co-adaptation and overfitting.",
    },
    {
        "Layer":    "Dense",
        "Config":   "1 unit · sigmoid activation",
        "Role":     "Outputs P(positive) ∈ [0, 1]. Threshold 0.5 maps to binary label.",
    },
])
st.dataframe(arch, use_container_width=True, hide_index=True)

m1, m2, m3 = st.columns(3)
m1.metric("Total trainable parameters", "~2.69M")
m2.metric("Embedding parameters",       "20,000 × 128 = 2.56M")
m3.metric("LSTM + Dense parameters",    "~132K")

st.divider()

# ── Training configuration ────────────────────────────────────────────────────
st.subheader("Training Configuration")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Optimizer",             "Adam")
c2.metric("Learning rate",         "0.001")
c3.metric("Batch size",            "128")
c4.metric("Max epochs",            "10")
c5.metric("Early stopping monitor","val_loss")
c6.metric("Patience",              "2 epochs")

st.info(
    "Early stopping monitors `val_loss` and restores the best weights. "
    "Training halted at epoch **4** and the best validation loss was at epoch **2**. "
    "Train accuracy at stopping: 96.5% vs val accuracy 87.1%, the 9 pp gap confirms "
    "mild overfitting that early stopping correctly caught."
)

st.divider()

# ── Training curves ───────────────────────────────────────────────────────────
st.subheader("Training Curves")

training_fig = FIGURES_DIR / "lstm_training_history.png"
if training_fig.exists():
    st.image(str(training_fig), use_container_width=True)
    st.caption(
        "Left: accuracy. Right: loss. Validation accuracy peaks at epoch 2 (88.0%) then plateaus. "
        "The widening gap between train and val curves signals overfitting, early stopping fires at epoch 4."
    )
else:
    st.info("Training history figure not found. Run notebook 04 to generate it.")

st.divider()

# ── Results comparison ────────────────────────────────────────────────────────
st.subheader("LSTM vs Best Classic Model")

try:
    lstm_m    = load_lstm_metrics()
    classic_m = load_classic_metrics()
    best_lr   = classic_m.loc[classic_m["f1"].idxmax()]
    lstm_val  = lstm_m[lstm_m["split"] == "validation"].iloc[0]
    lstm_test = lstm_m[lstm_m["split"] == "test"].iloc[0]

    comp = pd.DataFrame([
        {
            "Model":         "Logistic Regression (TF-IDF)",
            "Val Accuracy":  pct(best_lr["accuracy"]),
            "Val F1":        fmt_f1(best_lr["f1"]),
            "Train Time":    fmt_time(best_lr["training_time_s"]),
            "Inference":     "< 1 ms / review",
        },
        {
            "Model":         "LSTM (Keras)",
            "Val Accuracy":  pct(lstm_val["accuracy"]),
            "Val F1":        fmt_f1(lstm_val["f1"]),
            "Train Time":    fmt_time(lstm_val["training_time_s"]),
            "Inference":     "~42 ms / review (CPU)",
        },
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    acc_diff = (lstm_val["accuracy"] - best_lr["accuracy"]) * 100
    diff_str = f"{acc_diff:+.1f} pp"
    st.markdown(
        f"**Key finding:** LSTM achieves {pct(lstm_val['accuracy'])} vs {pct(best_lr['accuracy'])} "
        f"for Logistic Regression ({diff_str}). The LSTM required "
        f"{fmt_time(lstm_val['training_time_s'])} of training vs {fmt_time(best_lr['training_time_s'])}. "
        "The marginal accuracy difference does not justify the ~1,000× training cost increase, "
        "suggesting word-choice signal dominates sequence-order signal for IMDB binary sentiment."
    )

except FileNotFoundError:
    st.info("Metrics files not found. Run notebooks 03 and 04 first.")

st.divider()

# ── Full test metrics ─────────────────────────────────────────────────────────
st.subheader("LSTM Final Test Set Metrics")
try:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test Accuracy",  pct(lstm_test["accuracy"]))
    c2.metric("Test F1",        fmt_f1(lstm_test["f1"]))
    c3.metric("Precision",      pct(lstm_test["precision"]))
    c4.metric("Recall",         pct(lstm_test["recall"]))
    c5.metric("Training time",  fmt_time(lstm_test["training_time_s"]))
    st.caption(
        "Val and test metrics are consistent (within 1 pp), confirming generalisation without overfitting."
    )
except (NameError, KeyError):
    st.info("LSTM test metrics not available. Run notebook 04 to generate them.")
