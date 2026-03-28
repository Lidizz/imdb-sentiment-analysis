"""Feature extraction utilities for classic NLP models."""

from typing import Iterable, Optional, Tuple, TypedDict, Union

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


VectorizerType = Union[CountVectorizer, TfidfVectorizer]


class VectorizedSplits(TypedDict, total=False):
    """Container for vectorized dataset splits and fitted vectorizer."""

    train: csr_matrix
    val: csr_matrix
    test: csr_matrix
    vectorizer: VectorizerType


def _fit_and_transform(
    vectorizer: VectorizerType,
    train_texts: Iterable[str],
    val_texts: Optional[Iterable[str]] = None,
    test_texts: Optional[Iterable[str]] = None,
) -> VectorizedSplits:
    """Fit a vectorizer on train texts and transform available splits."""
    X_train = vectorizer.fit_transform(train_texts)
    outputs: VectorizedSplits = {
        "train": X_train,
        "vectorizer": vectorizer,
    }

    if val_texts is not None:
        outputs["val"] = vectorizer.transform(val_texts)
    if test_texts is not None:
        outputs["test"] = vectorizer.transform(test_texts)

    return outputs


def get_bow_features(
    train_texts: Iterable[str],
    val_texts: Optional[Iterable[str]] = None,
    test_texts: Optional[Iterable[str]] = None,
    max_features: int = 10_000,
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 5,
    max_df: float = 0.95,
) -> VectorizedSplits:
    """Convert text into Bag-of-Words features using CountVectorizer."""
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    return _fit_and_transform(vectorizer, train_texts, val_texts, test_texts)


def get_tfidf_features(
    train_texts: Iterable[str],
    val_texts: Optional[Iterable[str]] = None,
    test_texts: Optional[Iterable[str]] = None,
    max_features: int = 10_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
) -> VectorizedSplits:
    """Convert text into TF-IDF features using TfidfVectorizer.

    Defaults include bigrams so phrases like "not good" can be captured.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    return _fit_and_transform(vectorizer, train_texts, val_texts, test_texts)
