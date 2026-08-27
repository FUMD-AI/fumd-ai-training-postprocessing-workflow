"""
=============================================================================
FUMD-AI Training Workflow -- shared library: evaluation metrics
=============================================================================
Used by: notebooks/step_3_evaluate_model.ipynb

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Funding: This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

SPDX-License-Identifier: MIT
-----------------------------------------------------------------------------

Deliberately depends only on numpy/pandas/scikit-learn, not tensorflow --
it consumes plain numpy prediction arrays (whatever produced them), so it
can be exercised on its own without a GPU or a TensorFlow install.

Replaces the original exploratory notebook's per-dataset copy-pasted
metrics block (and `leer_metricas.ipynb`'s later regex-scraping of that
block's printed text back out of a `.txt` log) with one function that
returns structured results directly -- see metrics_io.py for writing them
to disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def infer_num_base_stations(df: pd.DataFrame, columns: list[str] | None = None) -> int:
    """
    Number of serving-cell classes to size the model's output heads for --
    derived from the data instead of hardcoded (the original notebook
    hardcoded 10, which silently mismatches on any run with a different
    base-station count, e.g. Alicante's 9). Covers every column the id
    could appear in (current, lagged, and destination), so classes that
    only ever show up as a *destination* aren't dropped.
    """
    if columns is None:
        from .data import LAG_SERVINGCELL_COLUMNS, TARGET_COLUMN
        columns = [TARGET_COLUMN, *LAG_SERVINGCELL_COLUMNS, "destination"]
    present = [c for c in columns if c in df.columns]
    max_id = max(int(df[c].max()) for c in present)
    return max_id + 1  # ids are 0-indexed classes


def evaluate_per_output(
    y_true: np.ndarray,
    y_pred_probs: list[np.ndarray],
    future_steps: list[int],
    num_base_stations: int,
) -> list[dict]:
    """
    Per-forecast-step precision/recall/F1 (weighted), top-2 accuracy, and
    confusion matrix.

    y_true: (n, len(future_steps)) ground-truth class ids.
    y_pred_probs: one (n, num_base_stations) softmax array per future step
        (i.e. exactly what `model.predict(X)` returns for this model).
    """
    results = []
    all_labels = np.arange(num_base_stations)

    for i, step in enumerate(future_steps):
        y_true_i = np.asarray(y_true[:, i]).astype(int)
        probs_i = np.asarray(y_pred_probs[i])
        y_pred_i = probs_i.argmax(axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_i, y_pred_i, average="weighted", zero_division=0, labels=all_labels,
        )
        top2 = top_k_accuracy_score(y_true_i, probs_i, k=min(2, num_base_stations), labels=all_labels)
        accuracy = float(np.mean(y_true_i == y_pred_i))

        cm = confusion_matrix(y_true_i, y_pred_i, labels=all_labels)
        present = np.unique(np.concatenate([y_true_i, y_pred_i]))
        p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
            y_true_i, y_pred_i, labels=present, average=None, zero_division=0,
        )
        total = cm.sum()
        per_class = []
        for j, cls in enumerate(present):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp
            tn = total - (tp + fp + fn)
            per_class.append({
                "class": int(cls),
                "precision": float(p_cls[j]),
                "recall": float(r_cls[j]),
                "f1": float(f1_cls[j]),
                "accuracy": float((tp + tn) / total) if total else float("nan"),
            })

        results.append({
            "future_step": int(step),
            "n_samples": int(len(y_true_i)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "top2_accuracy": float(top2),
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        })
    return results


def summarize_overall(per_output: list[dict]) -> dict:
    """Average precision/recall/F1/top-2/accuracy across all forecast steps."""
    def avg(key):
        return float(np.mean([r[key] for r in per_output]))
    return {
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1": avg("f1"),
        "top2_accuracy": avg("top2_accuracy"),
        "accuracy": avg("accuracy"),
    }


def evaluate_migration_breakdown(
    y_true_first_step: np.ndarray,
    y_pred_first_step: np.ndarray,
    migration_flags: np.ndarray,
) -> dict:
    """
    Accuracy of the *first* forecast step (the one `migration_flags` from
    `data.build_sequences` was sampled alongside), split by whether that
    window's forecast point falls inside a pre-handover warning window
    (`migration` 1 or 2) or in steady state (`migration` 0).

    This is only possible because the new preprocessing pipeline's
    `migration` label didn't exist for the old ad hoc training datasets --
    it directly answers "is the model actually any good right where it
    matters (around real handovers), not just on average", which a single
    aggregate accuracy figure can hide (steady-state rows dominate the
    dataset and a trivial "predict no change" model already scores well
    there).
    """
    y_true_first_step = np.asarray(y_true_first_step).astype(int)
    y_pred_first_step = np.asarray(y_pred_first_step).astype(int)
    migration_flags = np.asarray(migration_flags).astype(int)
    correct = y_true_first_step == y_pred_first_step

    steady_mask = migration_flags == 0
    warning_mask = migration_flags != 0

    def stats(mask):
        n = int(mask.sum())
        return {"n_samples": n, "accuracy": float(correct[mask].mean()) if n else float("nan")}

    return {"steady_state": stats(steady_mask), "warning_window": stats(warning_mask)}
