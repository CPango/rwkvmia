"""Logistic-regression attack classifier helpers."""

from __future__ import annotations

from typing import Any


def fit_logistic_regression(features: Any, labels: Any, *, random_state: int = 13):
    """Fit a simple logistic regression membership classifier."""
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(max_iter=1000, random_state=random_state)
    classifier.fit(features, labels)
    return classifier


def predict_membership_scores(classifier: Any, features: Any):
    """Return membership probabilities from a fitted classifier."""
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("classifier must expose predict_proba.")
    return classifier.predict_proba(features)[:, 1]
