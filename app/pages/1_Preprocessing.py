"""
Text preprocessing pipeline walkthrough.
"""
import sys
from pathlib import Path

_APP_DIR      = Path(__file__).resolve().parent.parent      # pages/ -> app/
_PROJECT_ROOT = _APP_DIR.parent                              # app/   -> project root
for _p in (str(_APP_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from _shared import FIGURES_DIR

from src.preprocessing import (
    remove_html_tags,
    to_lowercase,
    remove_special_characters,
    remove_stopwords,
    lemmatize_text,
)

st.set_page_config(
    page_title="Preprocessing | IMDB Sentiment",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Text Preprocessing Pipeline")
st.markdown(
    "Five sequential steps transform raw IMDB reviews into clean, normalised text ready "
    "for feature extraction. Each step is justified by a data-driven or linguistic reason."
)
st.divider()

# ── 5-step overview ───────────────────────────────────────────────────────────
st.subheader("The 5 Step Pipeline")

steps_info = [
    (
        "1. Remove HTML Tags",
        "58% of IMDB reviews contain `<br />` and similar markup.  \n"
        "Tags add noise and create spurious vocabulary tokens with no sentiment signal.",
    ),
    (
        "2. Lowercase",
        "`'Good'`, `'GOOD'`, and `'good'` should map to the same token.  \n"
        "Prevents vocabulary bloat from capitalisation variants.",
    ),
    (
        "3. Remove Special Characters",
        "Punctuation and digits carry no sentiment signal in a bag-of-words model.  \n"
        "Removing them cut unique vocabulary tokens by ~43% (94K → 54K).",
    ),
    (
        "4. Remove Stopwords",
        "~179 common English words removed (`the`, `is`, `at` …).  \n"
        "**Design choice:** negation words kept (`not`, `never`, `no`, `hardly`) -  \n"
        "`'not good'` is the opposite of `'good'`.",
    ),
    (
        "5. Lemmatisation",
        "Reduces inflected forms to base: `'running'` → `'run'`, `'loved'` → `'love'`.  \n"
        "POS-aware: uses grammatical role for the correct base form.  \n"
        "Chosen over stemming because lemmatisation returns real dictionary words.",
    ),
]

cols = st.columns(5)
for col, (title, desc) in zip(cols, steps_info):
    with col:
        st.markdown(f"**{title}**")
        st.markdown(desc)

st.divider()

# ── Interactive demo ──────────────────────────────────────────────────────────
st.subheader("Try the pipeline on any text")

EXAMPLE_REVIEWS = {
    "Positive review with HTML": (
        "I absolutely LOVED this movie! The acting was <br /> not bad at all, "
        "and the storyline was never boring. Definitely one of the best films I've seen in 2024!"
    ),
    "Negative review": (
        "Terrible waste of time. The plot made absolutely no sense, "
        "the acting was wooden, and I nearly fell asleep halfway through. Avoid!"
    ),
    "Negation test": (
        "This movie was not bad at all. I was not expecting to enjoy it, "
        "but it never felt too long and I was never bored."
    ),
    "Custom input": "",
}

example_choice = st.selectbox("Load an example or write your own:", list(EXAMPLE_REVIEWS.keys()))
default_val = EXAMPLE_REVIEWS[example_choice]

user_input = st.text_area(
    "Review text:",
    value=default_val,
    height=90,
    placeholder="Paste or type any movie review here...",
)

if st.button("Run preprocessing pipeline", type="primary", disabled=not user_input.strip()):
    with st.spinner("Processing..."):
        after_html  = remove_html_tags(user_input)
        after_lower = to_lowercase(after_html)
        after_chars = remove_special_characters(after_lower)
        after_sw    = remove_stopwords(after_chars)
        after_lemma = lemmatize_text(after_sw)

    pipeline_stages = [
        ("Original text",                  user_input,   None),
        ("① After HTML removal",            after_html,   user_input),
        ("② After lowercase",              after_lower,  after_html),
        ("③ After special char removal",   after_chars,  after_lower),
        ("④ After stopword removal",       after_sw,     after_chars),
        ("⑤ After lemmatisation (final)",  after_lemma,  after_sw),
    ]

    orig_words = len(user_input.split())

    for stage_label, result, prev in pipeline_stages:
        with st.container(border=True):
            col_label, col_text = st.columns([1, 4])
            with col_label:
                st.markdown(f"**{stage_label}**")
                if prev is not None:
                    words_now  = len(result.split())
                    words_prev = len(prev.split())
                    delta      = words_now - words_prev
                    delta_str  = f"{delta:+d} words" if delta != 0 else "no change"
                    st.caption(f"{words_now} words · {delta_str}")
            with col_text:
                if result.strip():
                    st.write(result)
                else:
                    st.caption("*(empty after this step)*")

    st.success(
        f"Pipeline complete: {orig_words} words in → {len(after_lemma.split())} words out "
        f"({100*(1 - len(after_lemma.split())/max(orig_words,1)):.0f}% reduction)"
    )

st.divider()

# ── Vocabulary shrinkage ──────────────────────────────────────────────────────
st.subheader("Vocabulary Shrinkage Impact (5,000 review sample)")
st.markdown(
    "Each preprocessing step reduces the vocabulary size. Smaller vocabulary → "
    "more compact feature matrix → faster training and less overfitting risk."
)

vcols = st.columns(6)
vocab_stages = [
    ("Raw text",             "94,889", None),
    ("After lowercase",      "63,064", "-31,825"),
    ("After special chars",  "54,355", "-8,709"),
    ("After stopwords",      "53,866", "-489"),
    ("After lemmatisation",  "46,268", "-7,598"),
    ("Total reduction",      "51%",    "of unique tokens"),
]
for col, (label, value, delta) in zip(vcols, vocab_stages):
    col.metric(label, value, delta, delta_color="inverse" if delta and delta.startswith("-") else "normal")

vocab_fig = FIGURES_DIR / "vocabulary_shrinkage.png"
if vocab_fig.exists():
    st.image(str(vocab_fig), use_container_width=True)

st.divider()

# ── Design decisions callout ──────────────────────────────────────────────────
st.subheader("Key Design Decisions")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Negation words preserved**")
    st.markdown(
        "Standard NLTK stopwords include `not`, `never`, `no`, `hardly`. "
        "We explicitly removed these from the stopword list.  \n\n"
        "Why: `'not good'` is the opposite of `'good'`. "
        "Stripping `not` would destroy the negation signal entirely."
    )
    st.code("_NEGATION_WORDS = {'not', 'no', 'nor', 'neither', 'never', ...}")

with d2:
    st.markdown("**Lemmatisation over stemming**")
    st.markdown(
        "Stemming (e.g. Porter) chops word endings with heuristic rules, "
        "producing non-words: `'running'` → `'run'` but `'better'` → `'better'`.  \n\n"
        "POS-aware lemmatisation uses grammatical role to find the correct base form: "
        "`'better'` (adj) → `'good'`, `'running'` (verb) → `'run'`."
    )
    st.code("lemmatize_text('running loved better') → 'run love good'")

st.divider()

# ── Full dataset run ──────────────────────────────────────────────────────────
st.subheader("Full Dataset Processing")
c1, c2, c3 = st.columns(3)
c1.metric("Input",  "50,000 raw reviews")
c2.metric("Output", "`data/imdb_preprocessed.csv`")
c3.metric("Runtime", "~25–30 min on CPU  \n(batch POS tagging)")
st.caption(
    "The preprocessed CSV is the single input for all downstream notebooks (NB03–NB05). "
    "NB02 only needs to be run once."
)
