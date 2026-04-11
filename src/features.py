"""Feature extraction utilities for classic and deep NLP models."""

from typing import Iterable, Optional, Tuple, TypedDict, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


VectorizerType = Union[CountVectorizer, TfidfVectorizer]


class VectorizedSplits(TypedDict, total=False):
    """Container for vectorized dataset splits and fitted vectorizer."""

    train: csr_matrix
    val: csr_matrix
    test: csr_matrix
    vectorizer: VectorizerType


class SequenceSplits(TypedDict, total=False):
    """Container for tokenized and padded sequence dataset splits."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    tokenizer: Tokenizer
    vocab_size: int
    max_sequence_length: int


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


def get_tokenized_padded_sequences(
    train_texts: Iterable[str],
    val_texts: Optional[Iterable[str]] = None,
    test_texts: Optional[Iterable[str]] = None,
    num_words: int = 20_000,
    max_sequence_length: int = 300,
    oov_token: str = "<OOV>",
) -> SequenceSplits:
    """Convert text into padded integer sequences for deep-learning models.

    The tokenizer is fit only on the training split to avoid leakage.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(train_texts)

    def _to_padded(texts: Iterable[str]) -> np.ndarray:
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences,
            maxlen=max_sequence_length,
            padding="post",
            truncating="post",
        )

    outputs: SequenceSplits = {
        "train": _to_padded(train_texts),
        "tokenizer": tokenizer,
        "vocab_size": min(num_words, len(tokenizer.word_index) + 1),
        "max_sequence_length": max_sequence_length,
    }

    if val_texts is not None:
        outputs["val"] = _to_padded(val_texts)
    if test_texts is not None:
        outputs["test"] = _to_padded(test_texts)

    return outputs
