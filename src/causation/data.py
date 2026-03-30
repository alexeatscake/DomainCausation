"""Data generation for the two-Gaussian classification demo."""

import numpy as np


def _apply_label_loss(
    X: np.ndarray,
    y: np.ndarray,
    loss: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a fraction of one class according to *loss*.

    * ``loss < 0``: remove ``abs(loss)`` fraction of the **positive** (True) examples.
    * ``loss > 0``: remove ``loss`` fraction of the **negative** (False) examples.
    * ``loss == 0``: no change.

    Parameters
    ----------
    X : ndarray, shape (n, 2)
    y : ndarray of bool, shape (n,)
    loss : float in [-1, 1]
    rng : numpy Generator

    Returns
    -------
    X_out, y_out with the specified fraction removed.
    """
    if loss == 0.0:
        return X, y

    if loss < 0:
        # Remove fraction of positives
        fraction = abs(loss)
        target_mask = y
    else:
        # Remove fraction of negatives
        fraction = loss
        target_mask = ~y

    target_idx = np.where(target_mask)[0]
    n_remove = int(len(target_idx) * fraction)
    remove_idx = rng.choice(target_idx, size=n_remove, replace=False)

    keep = np.ones(len(y), dtype=bool)
    keep[remove_idx] = False
    return X[keep], y[keep]


def generate_two_gaussians(
    n1: int,
    n2: int,
    mean1,
    cov1,
    mean2,
    cov2,
    noise_std: float,
    random_seed: int,
    loss1: float = 0.0,
    loss2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample from two 2D Gaussians, assign binary labels, add noise, apply loss.

    Labels are assigned from the clean Gaussian samples (``True`` if raw y > 0),
    noise is added afterwards, then *loss* optionally removes a fraction of one
    class per domain:

    * ``loss < 0`` — remove ``abs(loss)`` fraction of **positive** examples.
    * ``loss > 0`` — remove ``loss`` fraction of **negative** examples.
    * ``loss == 0`` — no removal (default).

    Parameters
    ----------
    n1 : int
        Number of samples for domain 1.
    n2 : int
        Number of samples for domain 2.
    mean1 : array-like, shape (2,)
        Mean vector for Gaussian 1.
    cov1 : array-like, shape (2, 2)
        Covariance matrix for Gaussian 1.
    mean2 : array-like, shape (2,)
        Mean vector for Gaussian 2.
    cov2 : array-like, shape (2, 2)
        Covariance matrix for Gaussian 2.
    noise_std : float
        Standard deviation of Gaussian noise added to sampled points.
    random_seed : int
        Seed for the random number generator.
    loss1 : float, optional
        Class-removal fraction for domain 1 (default 0 = no removal).
    loss2 : float, optional
        Class-removal fraction for domain 2 (default 0 = no removal).

    Returns
    -------
    X1 : ndarray, shape (n1', 2)
        Noisy points for domain 1 after loss removal.
    y1 : ndarray of bool, shape (n1',)
        Binary labels for domain 1.
    X2 : ndarray, shape (n2', 2)
        Noisy points for domain 2 after loss removal.
    y2 : ndarray of bool, shape (n2',)
        Binary labels for domain 2.
    """
    rng = np.random.default_rng(random_seed)

    raw1 = rng.multivariate_normal(mean=mean1, cov=cov1, size=n1)
    raw2 = rng.multivariate_normal(mean=mean2, cov=cov2, size=n2)

    # Labels from clean samples — before noise is applied
    y1 = raw1[:, 1] > 0
    y2 = raw2[:, 1] > 0

    # Add noise to positions after labelling; some points will cross y=0
    X1 = raw1 + rng.normal(loc=0.0, scale=noise_std, size=raw1.shape)
    X2 = raw2 + rng.normal(loc=0.0, scale=noise_std, size=raw2.shape)

    # Apply per-domain label loss
    X1, y1 = _apply_label_loss(X1, y1, loss1, rng)
    X2, y2 = _apply_label_loss(X2, y2, loss2, rng)

    return X1, y1, X2, y2
