"""Load and split IMDB datasets for downstream modeling notebooks."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COL,
    PREPROCESSED_DATA_PATH,
    CLEAN_TEXT_COL,
    SENTIMENT_COL,
)

_SENTIMENT_TO_LABEL = {"positive": 1, "negative": 0}


def default_preprocessed_path() -> Path:
    """Return the default path for the cleaned IMDB CSV."""
    return PREPROCESSED_DATA_PATH


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the preprocessed IMDB dataset produced by notebook 02.

    Expected columns:
    - sentiment
    - review_clean
    """
    path = Path(filepath) if filepath else default_preprocessed_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}. "
            "Run notebooks/02_preprocessing.ipynb to generate data/imdb_preprocessed.csv first."
        )

    df = pd.read_csv(path)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate required schema before splitting."""
    if SENTIMENT_COL not in df.columns:
        raise ValueError(
            f"Missing required column '{SENTIMENT_COL}'. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    if CLEAN_TEXT_COL not in df.columns:
        raise ValueError(
            "Missing cleaned text column. Expected: "
            f"'{CLEAN_TEXT_COL}'."
        )


def encode_labels(
    df: pd.DataFrame,
    sentiment_col: str = SENTIMENT_COL,
    output_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Map sentiment text labels to binary labels (positive=1, negative=0)."""
    if sentiment_col not in df.columns:
        raise ValueError(f"Column '{sentiment_col}' was not found in DataFrame.")

    encoded = df[sentiment_col].map(_SENTIMENT_TO_LABEL)
    unknown_count = int(encoded.isna().sum())
    if unknown_count:
        unique_values = sorted(df[sentiment_col].dropna().unique().tolist())
        raise ValueError(
            f"Found {unknown_count} unknown sentiment values. "
            f"Expected only {list(_SENTIMENT_TO_LABEL.keys())}, got: {unique_values}"
        )

    output = df.copy()
    output[output_col] = encoded.astype(int)
    return output


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    label_col: str = LABEL_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test splits.

    Defaults to 70/15/15 to support both model tuning and final unbiased testing.
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' was not found in DataFrame.")

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split sizes must sum to 1.0, got {total:.6f} "
            f"({train_size}, {val_size}, {test_size})."
        )

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    val_ratio_within_train_val = val_size / (train_size + val_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_within_train_val,
        random_state=random_state,
        stratify=train_val[label_col],
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def get_splits(
    filepath: Optional[str] = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    text_col: str = CLEAN_TEXT_COL,
    label_col: str = LABEL_COL,
    verbose: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load cleaned data and return X/y train/val/test splits.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    df = load_data(filepath=filepath)
    df = encode_labels(df, output_col=label_col)
    train, val, test = split_data(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        label_col=label_col,
    )

    for split_name, split_df in (("train", train), ("val", val), ("test", test)):
        if text_col not in split_df.columns:
            raise ValueError(
                f"Column '{text_col}' was not found in the {split_name} split. "
                f"Available columns: {sorted(split_df.columns.tolist())}"
            )

    X_train = train[text_col]
    y_train = train[label_col]
    X_val = val[text_col]
    y_val = val[label_col]
    X_test = test[text_col]
    y_test = test[label_col]

    if verbose:
        _print_split_summary(X_train, X_val, X_test, y_train, y_val, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _print_split_summary(
    X_train: pd.Series,
    X_val: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Print a compact split summary for notebook-friendly diagnostics."""
    total = len(X_train) + len(X_val) + len(X_test)
    print(f"Loaded {total:,} rows from preprocessed IMDB dataset.")
    print("Split sizes:")
    print(f"  train: {len(X_train):>6,} ({len(X_train) / total:.0%})")
    print(f"  val  : {len(X_val):>6,} ({len(X_val) / total:.0%})")
    print(f"  test : {len(X_test):>6,} ({len(X_test) / total:.0%})")
    print("Label distribution (positive class count):")
    print(f"  train: {int(y_train.sum()):>6,} / {len(y_train):,}")
    print(f"  val  : {int(y_val.sum()):>6,} / {len(y_val):,}")
    print(f"  test : {int(y_test.sum()):>6,} / {len(y_test):,}")
