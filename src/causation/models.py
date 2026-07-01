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


# ---------------------------------------------------------------------------
# Control-collapse demo: classify by class while controlling for domain.
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def fit_class_classifier(X: np.ndarray, cls: np.ndarray) -> LogisticRegression:
    """Naive class classifier — best class split, ignores domain."""
    return LogisticRegression().fit(X, cls)


def fit_domain_classifier(X: np.ndarray, dom: np.ndarray) -> LogisticRegression:
    """Domain classifier — the direction the perpendicular collapse removes."""
    return LogisticRegression().fit(X, dom)


def fit_controlled_classifier(
    X: np.ndarray, cls: np.ndarray, dom: np.ndarray
) -> LogisticRegression:
    """Class classifier with domain as an effect-coded (-1/+1) covariate.

    The shared feature slope gives parallel per-domain boundaries; dropping the
    domain term at test time lands on the average of the two (see
    :func:`collapse_controlled`).
    """
    Xd = np.column_stack([X, 2.0 * dom - 1.0])
    return LogisticRegression().fit(Xd, cls)


def fit_control_classifiers(
    X: np.ndarray, cls: np.ndarray, dom: np.ndarray
) -> tuple[LogisticRegression, LogisticRegression, LogisticRegression]:
    """Fit the naive class, domain, and domain-controlled classifiers."""
    return (
        fit_class_classifier(X, cls),
        fit_domain_classifier(X, dom),
        fit_controlled_classifier(X, cls, dom),
    )


def domain_averaged_boundary(
    ctrl_clf: LogisticRegression,
) -> tuple[np.ndarray, float]:
    """Class boundary that averages over the two domains.

    The controlled classifier was trained on both domains pooled, with domain as
    an effect-coded (-1/+1) covariate, giving parallel per-domain boundaries
    offset by ±gamma. Dropping the domain term (its coefficient) collapses those
    two parallel boundaries onto their midpoint — a single boundary that sits
    halfway between the domains rather than on either one.

    Returns the boundary ``(w, b)`` over the feature axes (``w·x + b = 0``).
    """
    w = ctrl_clf.coef_[0][:2]
    b = ctrl_clf.intercept_[0]
    return w, b


def collapse_perpendicular_to_domain(
    X: np.ndarray, cls: np.ndarray, domain_clf: LogisticRegression
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project onto the axis perpendicular to the domain classifier.

    Projecting onto this axis zeroes out the component along the domain weight,
    so the two domains' projections overlap — the 1D feature carries no linear
    information about domain.

    Returns
    -------
    scores : ndarray
        1D projection of every point onto the domain-free axis.
    u : ndarray, shape (2,)
        Unit collapse direction (perpendicular to the domain weight).
    thr : float
        Class threshold along that axis (where a 1D class model crosses 0.5).
    """
    v = domain_clf.coef_[0]
    u = _unit(np.array([-v[1], v[0]]))  # perpendicular to domain direction
    scores = X @ u
    # Orient so the blue class (cls=1) sits on the positive side, for a tidy plot.
    if scores[cls == 1].mean() < scores[cls == 0].mean():
        u, scores = -u, -scores
    # Class threshold along the collapsed axis via a 1D logistic fit.
    clf1d = LogisticRegression().fit(scores[:, None], cls)
    thr = -clf1d.intercept_[0] / clf1d.coef_[0, 0]
    return scores, u, thr


def boundary_line(
    w: np.ndarray,
    b: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of the line ``{x : w·x + b = 0}`` across the *xlim*/*ylim* box.

    Parameterises along whichever axis the line is more aligned with, so it
    works for near-horizontal and near-vertical boundaries alike.

    Returns ``(xs, ys)``.
    """
    w0, w1 = w[0], w[1]
    if abs(w1) >= abs(w0):
        xs = np.linspace(*xlim, 200)
        ys = -(w0 * xs + b) / w1
    else:
        ys = np.linspace(*ylim, 200)
        xs = -(w1 * ys + b) / w0
    return xs, ys


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
