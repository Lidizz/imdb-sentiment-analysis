"""Shared project configuration for paths and dataset schema."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_DATA_FILENAME = "IMDB Dataset.csv"
PREPROCESSED_DATA_FILENAME = "imdb_preprocessed.csv"

RAW_DATA_PATH = DATA_DIR / RAW_DATA_FILENAME
PREPROCESSED_DATA_PATH = DATA_DIR / PREPROCESSED_DATA_FILENAME

RAW_TEXT_COL = "review"
CLEAN_TEXT_COL = "review_clean"
SENTIMENT_COL = "sentiment"
LABEL_COL = "label"