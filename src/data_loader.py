"""
Data Loader for IMDB Sentiment Analysis
========================================
Ansvar:
    1. Laste inn IMDB_preprocessed.csv
    2. Kode sentiment-etiketter til binære tall  (positive -> 1, negative -> 0)
    3. Dele data i train / test (80/20)

Returnerte splits brukes av:
    - 03_classic_ml_models.ipynb   (TF-IDF + sklearn-modeller)
    - 04_deep_learning_model.ipynb (Keras Tokenizer + LSTM)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

_RANDOM_SEED = 42


def _get_data_dir() -> str:
    """Returnerer absolutt sti til data/-mappen uavhengig av hvor scriptet kalles fra."""
    return os.path.join(os.path.dirname(__file__), "..", "data")


def _encode_labels(series: pd.Series) -> pd.Series:
    """Konverterer 'positive'/'negative' til 1/0."""
    mapping = {"positive": 1, "negative": 0}
    encoded = series.map(mapping)
    unknown = encoded.isna().sum()
    if unknown > 0:
        raise ValueError(
            f"{unknown} rader har ukjente sentiment-verdier. "
            f"Forventet 'positive' eller 'negative'."
        )
    return encoded.astype(int)


def get_splits(
    verbose: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Laster IMDB_preprocessed.csv og deler inn i train / test (80/20).

    Returnerer:
        (X_train, y_train, X_test, y_test)

        X_* : pandas Series med renset tekst (clean_review)
        y_* : pandas Series med binære etiketter (1=positiv, 0=negativ)

    Eksempel:
        from src.data_loader import get_splits
        X_train, y_train, X_test, y_test = get_splits()
    """
    path = os.path.join(_get_data_dir(), "IMDB_preprocessed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fant ikke {path}\n"
            "Generer filen ved å kjøre: python src/preprocessing.py"
        )

    df = pd.read_csv(path)

    if verbose:
        print(f"Lastet {len(df):,} anmeldelser fra IMDB_preprocessed.csv")

    X = df["clean_review"].reset_index(drop=True)
    y = _encode_labels(df["sentiment"]).reset_index(drop=True)

    # Split dataset into training and testing (80/20 split)
    # stratify=y sikrer lik fordeling av positive/negative i begge splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=_RANDOM_SEED,
        stratify=y,
    )

    if verbose:
        total = len(X)
        print(f"\nDatasett-splits ({total:,} totalt):")
        print(f"  Train : {len(X_train):>6,}  ({len(X_train)/total:.0%})")
        print(f"  Test  : {len(X_test):>6,}  ({len(X_test)/total:.0%})")
        print(f"\nEtikettfordeling i train: "
              f"{y_train.sum():,} positive / {(y_train==0).sum():,} negative")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("=== Test av data_loader ===\n")
    X_train, y_train, X_test, y_test = get_splits()

    print("\nEksempel-anmeldelse fra treningssettet:")
    print(f"  Tekst  : {X_train.iloc[0][:120]}...")
    print(f"  Etikett: {y_train.iloc[0]}  (1=positiv, 0=negativ)")
