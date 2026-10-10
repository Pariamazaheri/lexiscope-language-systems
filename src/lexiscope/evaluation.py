"""Threshold-fixed classification metrics and paired bootstrap contrasts."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def classification_metrics(y, p):
    pred = p >= 0.5
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def paired_bootstrap(y, left, right, seed=42, repeats=2000):
    delta = (right >= 0.5) == y
    delta = delta.astype(float) - ((left >= 0.5) == y).astype(float)
    rng = np.random.default_rng(seed)
    samples = np.array(
        [delta[rng.integers(len(y), size=len(y))].mean() for _ in range(repeats)]
    )
    return {
        "accuracy_delta": float(delta.mean()),
        "bootstrap_low": float(np.quantile(samples, 0.025)),
        "bootstrap_high": float(np.quantile(samples, 0.975)),
        "bootstrap_repeats": repeats,
    }
