"""Evaluation helpers for classic sentiment models."""

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Return standard binary classification metrics.

    Literature note (book Ch. 5): report precision, recall, and F1 alongside
    accuracy to avoid over-relying on a single metric.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def build_model_comparison_table(
    y_true: Iterable[int],
    predictions: Dict[str, Iterable[int]],
    split_name: str = "validation",
) -> pd.DataFrame:
    """Build a tidy model-comparison table from prediction arrays."""
    rows = []
    for model_name, y_pred in predictions.items():
        metric_values = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                **metric_values,
            }
        )
    return pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)


def save_metrics_table(df: pd.DataFrame, output_path: Path) -> None:
    """Persist comparison metrics to CSV for report reuse."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def plot_confusion_matrices_grid(
    y_true: Iterable[int],
    predictions: Dict[str, Iterable[int]],
    n_cols: int = 2,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot confusion matrices for multiple models in a single figure."""
    if not predictions:
        raise ValueError("predictions must contain at least one model.")

    model_names = list(predictions.keys())
    n_models = len(model_names)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, model_name in enumerate(model_names):
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            predictions[model_name],
            ax=axes_list[idx],
            colorbar=False,
            values_format="d",
        )
        axes_list[idx].set_title(model_name)

    for idx in range(n_models, len(axes_list)):
        axes_list[idx].axis("off")

    fig.tight_layout()
    return fig


def save_models(models: Dict[str, BaseEstimator], models_dir: Path) -> None:
    """Serialize fitted models to joblib files."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        filename = model_name.lower().replace(" ", "_") + ".joblib"
        dump(model, models_dir / filename)
