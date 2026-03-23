"""
Feature Engineering for IMDB Sentiment Analysis
=================================================
Konverterer renset tekst til tallmatriser som sklearn-modellene kan lære av.

To representasjoner:
    BoW  (Bag of Words) — teller hvor mange ganger hvert ord dukker opp
    TF-IDF              — vekter ord etter hvor viktige de er på tvers av alle anmeldelser

Begge bruker CountVectorizer / TfidfVectorizer fra sklearn med konfigurerbare parametere:
    max_features : maks antall ord å beholde (de mest frekvente)
    ngram_range  : (1,1) = unigrammer,  (1,2) = uni- + bigrammer
    min_df       : ignorer ord som forekommer i færre enn X dokumenter
    max_df       : ignorer ord som forekommer i mer enn X% av dokumentene (typisk støy)
"""

# --- Importer ---
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import spmatrix


# --- Standardverdier ---
# Disse verdiene er et godt utgangspunkt for IMDB og kan endres ved behov.
_MAX_FEATURES = 10_000   # beholder de 10 000 mest frekvente ordene
_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams: "good", "not good"
_MIN_DF       = 2        # ignorer ord som bare forekommer i én anmeldelse
_MAX_DF       = 0.95     # ignorer ord som finnes i >95% av anmeldelsene (for vanlige til å skille)


# ── Bag of Words ─────────────────────────────────────────────────────────────

def get_bow_features(
    X_train: pd.Series,
    X_test:  pd.Series,
    max_features: int        = _MAX_FEATURES,
    ngram_range:  tuple      = _NGRAM_RANGE,
    min_df:       int        = _MIN_DF,
    max_df:       float      = _MAX_DF,
    verbose:      bool       = True,
) -> tuple[spmatrix, spmatrix, CountVectorizer]:
    """
    Bag of Words: teller råfrekvensen av hvert ord i hver anmeldelse.

    Vektorisatoren trenes (fit) KUN på treningsdata, deretter transformeres
    både train og test. Dette hindrer data leakage — test-settet skal
    aldri påvirke hva vektorisatoren lærer.

    Returnerer:
        (X_train_bow, X_test_bow, vectorizer)
        X_*_bow er scipy sparse-matriser med form (n_samples, max_features)
    """
    vectorizer = CountVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,
        min_df       = min_df,
        max_df       = max_df,
    )

    # fit_transform på train: lærer vokabular OG transformerer i ett steg
    X_train_bow = vectorizer.fit_transform(X_train)

    # transform på test: bruker vokabularet fra train (ingen ny læring)
    X_test_bow  = vectorizer.transform(X_test)

    if verbose:
        print(f"BoW  — vokabularstørrelse : {len(vectorizer.vocabulary_):,}")
        print(f"       matrisestørrelse   : {X_train_bow.shape[0]:,} × {X_train_bow.shape[1]:,}")

    return X_train_bow, X_test_bow, vectorizer


# ── TF-IDF ───────────────────────────────────────────────────────────────────

def get_tfidf_features(
    X_train: pd.Series,
    X_test:  pd.Series,
    max_features: int        = _MAX_FEATURES,
    ngram_range:  tuple      = _NGRAM_RANGE,
    min_df:       int        = _MIN_DF,
    max_df:       float      = _MAX_DF,
    verbose:      bool       = True,
) -> tuple[spmatrix, spmatrix, TfidfVectorizer]:
    """
    TF-IDF (Term Frequency – Inverse Document Frequency):
    samme som BoW, men vekter ned ord som er vanlige i ALLE anmeldelser
    (f.eks. "film", "movie") og vekter opp ord som er spesifikke for noen få.

    Ord med høy TF-IDF er mer informative for å skille positiv fra negativ.

    Returnerer:
        (X_train_tfidf, X_test_tfidf, vectorizer)
        X_*_tfidf er scipy sparse-matriser med form (n_samples, max_features)
    """
    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,
        min_df       = min_df,
        max_df       = max_df,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    if verbose:
        print(f"TF-IDF — vokabularstørrelse : {len(vectorizer.vocabulary_):,}")
        print(f"         matrisestørrelse   : {X_train_tfidf.shape[0]:,} × {X_train_tfidf.shape[1]:,}")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ── Toppord per klasse ────────────────────────────────────────────────────────

def get_top_features_per_class(
    vectorizer,
    X_train:  spmatrix,
    y_train:  pd.Series,
    n:        int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finner de N mest karakteristiske ordene for positive og negative anmeldelser.
    Nyttig for visualisering i notebook.

    Beregner gjennomsnittlig TF-IDF / BoW-verdi per ord for hver klasse.
    Ord med høyest gjennomsnitt er mest representative for klassen.

    Returnerer:
        (top_positive, top_negative) — to DataFrames med kolonner: feature, score
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    y = np.array(y_train)

    # Gjennomsnittlig score per ord for positive (1) og negative (0) anmeldelser
    mean_positive = np.asarray(X_train[y == 1].mean(axis=0)).flatten()
    mean_negative = np.asarray(X_train[y == 0].mean(axis=0)).flatten()

    top_positive = pd.DataFrame({
        "feature": feature_names[np.argsort(mean_positive)[::-1][:n]],
        "score":   np.sort(mean_positive)[::-1][:n],
    })

    top_negative = pd.DataFrame({
        "feature": feature_names[np.argsort(mean_negative)[::-1][:n]],
        "score":   np.sort(mean_negative)[::-1][:n],
    })

    return top_positive, top_negative


# ── Kjør direkte for å bekrefte at alt fungerer ───────────────────────────────

if __name__ == "__main__":
    from data_loader import get_splits

    print("=== Test av features.py ===\n")
    X_train, y_train, X_test, y_test = get_splits(verbose=False)

    print("--- Bag of Words ---")
    X_train_bow, X_test_bow, bow_vec = get_bow_features(X_train, X_test)

    print("\n--- TF-IDF ---")
    X_train_tfidf, X_test_tfidf, tfidf_vec = get_tfidf_features(X_train, X_test)

    print("\n--- Topp 10 ord (TF-IDF) ---")
    top_pos, top_neg = get_top_features_per_class(tfidf_vec, X_train_tfidf, y_train, n=10)
    print("Positive:", top_pos["feature"].tolist())
    print("Negative:", top_neg["feature"].tolist())
