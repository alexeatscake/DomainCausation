"""Plotting utilities and main demo entry point."""

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from .data import generate_two_gaussians
from .models import decision_boundary_y, fit_models

# Colors: red = True label, blue = False label
# Domain 1 = light shades, Domain 2 = dark shades (slight hue shift)
_D1_TRUE = "#f4877a"  # light warm red
_D1_FALSE = "#7aaaf4"  # light sky blue
_D2_TRUE = "#8b1e0e"  # dark red, hue shifted toward brown-red
_D2_FALSE = "#0e2e8b"  # dark blue, hue shifted toward indigo


def plot_dataset_with_boundaries(
    X1: np.ndarray,
    y1: np.ndarray,
    X2: np.ndarray,
    y2: np.ndarray,
    boundary_specs: list[tuple],
    title: str = "Two-Gaussian Classification with Logistic Regression Boundaries",
    save_path: Path | None = None,
) -> None:
    """Plot the two-domain scatter and overlay decision boundaries.

    Parameters
    ----------
    X1 : ndarray, shape (n1, 2)
        Noisy points for domain 1.
    y1 : ndarray of bool, shape (n1,)
        Binary labels for domain 1.
    X2 : ndarray, shape (n2, 2)
        Noisy points for domain 2.
    y2 : ndarray of bool, shape (n2,)
        Binary labels for domain 2.
    boundary_specs : list of (model, label, linestyle, color)
        One entry per decision boundary to overlay.
    title : str
        Title displayed above the figure.
    save_path : Path or None
        If given, save the figure to this path before showing.
    """
    X_all = np.vstack([X1, X2])

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Domain 1: circles (light colors) ---
    ax.scatter(
        X1[y1, 0],
        X1[y1, 1],
        c=_D1_TRUE,
        marker="o",
        s=30,
        alpha=0.9,
        linewidths=0,
        label="_nolegend_",
    )
    ax.scatter(
        X1[~y1, 0],
        X1[~y1, 1],
        c=_D1_FALSE,
        marker="o",
        s=30,
        alpha=0.9,
        linewidths=0,
        label="_nolegend_",
    )

    # --- Domain 2: squares (dark colors) ---
    ax.scatter(
        X2[y2, 0],
        X2[y2, 1],
        c=_D2_TRUE,
        marker="s",
        s=30,
        alpha=0.9,
        linewidths=0,
        label="_nolegend_",
    )
    ax.scatter(
        X2[~y2, 0],
        X2[~y2, 1],
        c=_D2_FALSE,
        marker="s",
        s=30,
        alpha=0.9,
        linewidths=0,
        label="_nolegend_",
    )

    # --- Decision boundaries ---
    x_line = np.linspace(X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5, 300)
    for model, label, linestyle, color in boundary_specs:
        ax.plot(
            x_line,
            decision_boundary_y(model, x_line),
            linestyle=linestyle,
            color=color,
            linewidth=2.0,
            label=label,
        )

    # --- Legend ---
    # Color encodes label (red=True, blue=False); shade encodes domain (light=1, dark=2)
    def _m(color, marker, label):
        return mlines.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="None",
            markersize=7,
            markeredgewidth=0,
            label=label,
        )

    legend_handles = [
        _m(_D1_TRUE, "o", "Domain 1 — True  (light red)"),
        _m(_D1_FALSE, "o", "Domain 1 — False (light blue)"),
        _m(_D2_TRUE, "s", "Domain 2 — True  (dark red)"),
        _m(_D2_FALSE, "s", "Domain 2 — False (dark blue)"),
        *[
            mlines.Line2D([], [], color=c, linestyle=ls, linewidth=2, label=lbl)
            for _, lbl, ls, c in boundary_specs
        ],
    ]
    ax.legend(handles=legend_handles, loc="best", framealpha=0.85, fontsize=9)

    ax.set_xlabel("Low Audio", fontsize=14)
    ax.set_ylabel("High Audio", fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # Ticks: inward, mirrored on both axes, with minor ticks
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=12,
        length=6,
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True, length=3
    )
    ax.minorticks_on()

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_two_gaussian_logistic(
    mean1,
    cov1,
    mean2,
    cov2,
    n1: int = 500,
    n2: int = 500,
    noise_std: float = 0.8,
    random_seed: int | None = None,
    title: str = "Two-Gaussian Classification with Logistic Regression Boundaries",
    loss1: float = 0.0,
    loss2: float = 0.0,
    save_path: Path | None = None,
) -> None:
    """Generate data, fit three logistic regression models, and plot everything.

    Parameters
    ----------
    mean1 : array-like, shape (2,)
        Mean vector for Gaussian 1 (domain 1).
    cov1 : array-like, shape (2, 2)
        Covariance matrix for Gaussian 1.
    mean2 : array-like, shape (2,)
        Mean vector for Gaussian 2 (domain 2).
    cov2 : array-like, shape (2, 2)
        Covariance matrix for Gaussian 2.
    n1 : int
        Number of samples for domain 1 (circles).
    n2 : int
        Number of samples for domain 2 (squares).
    noise_std : float
        Standard deviation of Gaussian noise added to sampled points.
    random_seed : int | None
        Random seed for reproducibility. If None, uses a random seed.
    title : str
        Title displayed above the figure.
    loss1 : float, optional
        Class-removal fraction for domain 1. Negative removes positives,
        positive removes negatives (default 0 = no removal).
    loss2 : float, optional
        Class-removal fraction for domain 2. Same sign convention as loss1.
    save_path : Path or None
        If given, save the figure to this path before showing.
    """

    random_seed = random_seed if random_seed is not None else np.random.randint(0, 1000)

    X1, y1, X2, y2 = generate_two_gaussians(
        n1,
        n2,
        mean1,
        cov1,
        mean2,
        cov2,
        noise_std,
        random_seed,
        loss1=loss1,
        loss2=loss2,
    )

    lr1, lr2, lr_all = fit_models(X1, y1, X2, y2, random_seed)
    # report_accuracies(lr1, X1, y1, lr2, X2, y2, lr_all)

    boundary_specs = [
        (lr1, "Domain-1 model", "dotted", "#888888"),
        (lr2, "Domain-2 model", "dashed", "#555555"),
        (lr_all, "Combined model", "solid", "#333333"),
    ]
    plot_dataset_with_boundaries(
        X1, y1, X2, y2, boundary_specs, title=title, save_path=save_path
    )
