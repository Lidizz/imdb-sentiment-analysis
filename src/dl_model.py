"""Deep-learning utilities for LSTM sentiment classification."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import Model, Sequential
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Input, LSTM
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


MetricsResult = Tuple[Dict[str, float], np.ndarray, np.ndarray]


def _to_numpy_1d(values: Iterable[int]) -> np.ndarray:
    """Convert labels to a 1D numpy array without unnecessary list materialization."""
    if isinstance(values, np.ndarray):
        return np.asarray(values).reshape(-1)
    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy()).reshape(-1)
    if hasattr(values, "__array__"):
        return np.asarray(values).reshape(-1)
    return np.fromiter(values, dtype=np.int64).reshape(-1)


def build_lstm_model(
    vocab_size: int,
    max_sequence_length: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """Build and compile an LSTM model for binary sentiment classification."""
    model = Sequential(
        [
            Input(shape=(max_sequence_length,)),
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm_model(
    model: Model,
    X_train: np.ndarray,
    y_train: Iterable[int],
    X_val: np.ndarray,
    y_val: Iterable[int],
    batch_size: int = 128,
    epochs: int = 10,
    patience: int = 2,
    monitor: str = "val_loss",
    mode: str = "min",
    min_delta: float = 0.0,
    model_path: Optional[Path] = None,
    verbose: int = 1,
) -> History:
    """Train an LSTM model with early stopping and optional checkpointing."""
    y_train_array = _to_numpy_1d(y_train)
    y_val_array = _to_numpy_1d(y_val)

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=True,
        ),
    ]

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=str(model_path),
                monitor=monitor,
                mode=mode,
                save_best_only=True,
            )
        )

    history = model.fit(
        X_train,
        y_train_array,
        validation_data=(X_val, y_val_array),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


def evaluate_lstm_model(
    model: Model,
    X_eval: np.ndarray,
    y_eval: Iterable[int],
    threshold: float = 0.5,
) -> MetricsResult:
    """Evaluate an LSTM model and return metrics, labels, and probabilities."""
    y_true = _to_numpy_1d(y_eval).astype(int)
    y_prob = model.predict(X_eval, verbose=0).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return metrics, y_pred, y_prob


def plot_training_history(history: History) -> plt.Figure:
    """Create a two-panel plot for training/validation accuracy and loss."""
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(hist.get("accuracy", []), label="train")
    axes[0].plot(hist.get("val_accuracy", []), label="validation")
    axes[0].set_title("LSTM Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(hist.get("loss", []), label="train")
    axes[1].plot(hist.get("val_loss", []), label="validation")
    axes[1].set_title("LSTM Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    return fig
