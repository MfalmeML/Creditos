"""Reject inference utilities for credit-risk experiments.

These functions support a controlled pseudo-labeling experiment.
Pseudo-labels are estimates, not observed outcomes for rejected applicants.
"""

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def fit_accept_model(X: pd.DataFrame, y) -> Pipeline:
    """Fit a binary default model on applicants with known outcomes."""
    if len(X) == 0:
        raise ValueError("Cannot fit a model with no accepted applicants.")

    y = np.asarray(y, dtype=int)

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if len(np.unique(y)) < 2:
        raise ValueError("Training labels must contain both classes (0 and 1).")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X, y)
    return model


def parcel_rejects(
    X_rejected: pd.DataFrame,
    model: Pipeline,
    k: int = 5,
) -> np.ndarray:
    """Assign deterministic pseudo-labels to rejected applicants.

    Applicants are ranked by predicted default probability and divided
    into risk parcels. Within each parcel, the predicted average default
    rate determines how many applicants receive a pseudo-label of 1.
    The highest-risk applicants in each parcel receive those labels first.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    if len(X_rejected) == 0:
        return np.array([], dtype=int)

    probabilities = model.predict_proba(X_rejected)[:, 1]
    probabilities = np.clip(probabilities, 0.0, 1.0)

    # Rank from lowest to highest predicted default risk.
    order = np.argsort(probabilities, kind="stable")
    parcels = np.array_split(order, min(k, len(order)))

    pseudo_labels = np.zeros(len(X_rejected), dtype=int)

    for parcel in parcels:
        if len(parcel) == 0:
            continue

        # Estimate the default rate within this risk parcel.
        estimated_rate = probabilities[parcel].mean()
        n_bad = int(round(estimated_rate * len(parcel)))

        # Assign positive labels to the highest-risk members of the parcel.
        ranked_parcel = parcel[
            np.argsort(probabilities[parcel], kind="stable")[::-1]
        ]
        pseudo_labels[ranked_parcel[:n_bad]] = 1

    return pseudo_labels