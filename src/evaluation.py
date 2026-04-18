"""Evaluation helpers for classic sentiment models."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
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
    predictions: dict[str, Iterable[int]],
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
    predictions: dict[str, Iterable[int]],
    n_cols: int = 2,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot confusion matrices for multiple models: green correct cells, red errors."""
    import matplotlib.patches as mpatches
    from sklearn.metrics import confusion_matrix as _cm

    if not predictions:
        raise ValueError("predictions must contain at least one model.")

    model_names = list(predictions.keys())
    n_models = len(model_names)
    n_rows = (n_models + n_cols - 1) // n_cols
    labels = ("Positive", "Negative")
    correct_color = "#27ae60"
    wrong_color = "#e74c3c"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, model_name in enumerate(model_names):
        ax = axes_list[idx]
        cm = _cm(y_true, predictions[model_name], labels=[1, 0])
        n = len(labels)
        for i in range(n):
            for j in range(n):
                color = correct_color if i == j else wrong_color
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j - 0.5, i - 0.5), 1, 1,
                    boxstyle="square,pad=0", fc=color, ec="white", lw=1.5,
                ))
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(model_name, fontsize=9)
        ax.tick_params(length=0)

    for idx in range(n_models, len(axes_list)):
        axes_list[idx].axis("off")

    fig.suptitle(
        "Confusion Matrices: Classic Models\nGreen = correct  |  Red = error",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def save_models(models: dict[str, BaseEstimator], models_dir: Path) -> None:
    """Serialize fitted models to joblib files."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        filename = model_name.lower().replace(" ", "_") + ".joblib"
        dump(model, models_dir / filename)
