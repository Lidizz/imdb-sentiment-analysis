"""Classic machine-learning training utilities for sentiment classification."""

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ModelResult = Tuple[BaseEstimator, np.ndarray, str]


def _fit_predict_report(
    model: BaseEstimator,
    X_train: csr_matrix,
    y_train: Iterable[int],
    X_eval: csr_matrix,
    y_eval: Iterable[int],
) -> ModelResult:
    """Train a model, predict on eval split, and return classification report text."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    report = classification_report(y_eval, y_pred, digits=4)
    return model, y_pred, report


def train_logistic_regression(
    X_train: csr_matrix,
    y_train: Iterable[int],
    X_eval: csr_matrix,
    y_eval: Iterable[int],
    C: float = 1.0,
    random_state: int = 42,
) -> ModelResult:
    """Train Logistic Regression on sparse TF-IDF features.

    Literature note (book Ch. 5): Logistic Regression is a strong linear baseline
    for classification and provides interpretable feature coefficients.
    """
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_state,
    )
    return _fit_predict_report(model, X_train, y_train, X_eval, y_eval)


def train_naive_bayes(
    X_train: csr_matrix,
    y_train: Iterable[int],
    X_eval: csr_matrix,
    y_eval: Iterable[int],
    alpha: float = 1.0,
) -> ModelResult:
    """Train Multinomial Naive Bayes baseline for text classification.

    Literature note (book Ch. 5 + Ch. 15): NB is commonly used for text features
    such as counts and TF-IDF despite the conditional-independence assumption.
    """
    model = MultinomialNB(alpha=alpha)
    return _fit_predict_report(model, X_train, y_train, X_eval, y_eval)


def train_svm(
    X_train: csr_matrix,
    y_train: Iterable[int],
    X_eval: csr_matrix,
    y_eval: Iterable[int],
    C: float = 1.0,
    random_state: int = 42,
) -> ModelResult:
    """Train Linear SVM for high-dimensional sparse text features.

    Literature note (book Ch. 5): SVM seeks a margin-maximizing hyperplane,
    which often performs well on sparse high-dimensional representations.
    """
    model = LinearSVC(
        C=C,
        dual="auto",
        max_iter=2000,
        random_state=random_state,
    )
    return _fit_predict_report(model, X_train, y_train, X_eval, y_eval)


def train_random_forest(
    X_train: csr_matrix,
    y_train: Iterable[int],
    X_eval: csr_matrix,
    y_eval: Iterable[int],
    n_estimators: int = 200,
    random_state: int = 42,
) -> ModelResult:
    """Train Random Forest as the ensemble-learning comparator.

    Literature note (book Ch. 6): Random Forest reduces variance via bagging and
    provides feature-importance signals that support interpretation.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    return _fit_predict_report(model, X_train, y_train, X_eval, y_eval)


def hyperparameter_tuning(
    X_train: csr_matrix,
    y_train: Iterable[int],
    param_grid: Optional[Dict[str, Iterable[Any]]] = None,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> GridSearchCV:
    """Run GridSearchCV for Logistic Regression hyperparameter tuning.

    Literature note (book Ch. 6): grid search is used to find stronger parameter
    settings than default values while controlling search space explicitly.
    """
    if param_grid is None:
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}

    search = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search
