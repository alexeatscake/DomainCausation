"""Model fitting, accuracy reporting, and decision boundary utilities."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def fit_models(
    X1: np.ndarray,
    y1: np.ndarray,
    X2: np.ndarray,
    y2: np.ndarray,
    random_seed: int = 42,
) -> tuple[LogisticRegression, LogisticRegression, LogisticRegression]:
    """Fit three logistic regression models.

    1. Domain 1 only
    2. Domain 2 only
    3. Combined dataset

    Parameters
    ----------
    X1 : ndarray, shape (n1, 2)
        Feature matrix for domain 1.
    y1 : ndarray of bool, shape (n1,)
        Labels for domain 1.
    X2 : ndarray, shape (n2, 2)
        Feature matrix for domain 2.
    y2 : ndarray of bool, shape (n2,)
        Labels for domain 2.
    random_seed : int
        Random state passed to LogisticRegression.

    Returns
    -------
    lr1 : LogisticRegression
        Model trained on domain 1 only.
    lr2 : LogisticRegression
        Model trained on domain 2 only.
    lr_all : LogisticRegression
        Model trained on the combined dataset.
    """
    lr1 = LogisticRegression(random_state=random_seed)
    lr1.fit(X1, y1)

    lr2 = LogisticRegression(random_state=random_seed)
    lr2.fit(X2, y2)

    X_all = np.vstack([X1, X2])
    y_all = np.concatenate([y1, y2])
    lr_all = LogisticRegression(random_state=random_seed)
    lr_all.fit(X_all, y_all)

    return lr1, lr2, lr_all


def report_accuracies(
    lr1: LogisticRegression,
    X1: np.ndarray,
    y1: np.ndarray,
    lr2: LogisticRegression,
    X2: np.ndarray,
    y2: np.ndarray,
    lr_all: LogisticRegression,
) -> None:
    """Print training accuracy for all three models."""
    X_all = np.vstack([X1, X2])
    y_all = np.concatenate([y1, y2])

    acc1 = accuracy_score(y1, lr1.predict(X1))
    acc2 = accuracy_score(y2, lr2.predict(X2))
    acc_all = accuracy_score(y_all, lr_all.predict(X_all))

    print(f"Domain-1 model  — training accuracy: {acc1:.3f}")
    print(f"Domain-2 model  — training accuracy: {acc2:.3f}")
    print(f"Combined model  — training accuracy: {acc_all:.3f}")


def decision_boundary_y(model: LogisticRegression, x_vals: np.ndarray) -> np.ndarray:
    """Compute y values along a logistic regression decision boundary.

    The decision boundary satisfies ``w · x + b = 0``, which gives
    ``y = -(w0 * x + b) / w1``.

    Returns an array of NaN if the boundary is vertical (w1 ≈ 0).
    """
    w = model.coef_[0]
    b = model.intercept_[0]
    if abs(w[1]) < 1e-10:
        return np.full_like(x_vals, np.nan)
    return -(w[0] * x_vals + b) / w[1]
