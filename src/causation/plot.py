"""Plotting utilities and main demo entry point."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .data import generate_two_gaussians
from .models import decision_boundary_y, fit_models

# Colors: red = True label, blue = False label
# Domain 1 = light shades, Domain 2 = dark shades (slight hue shift)
_D1_TRUE = "#f4877a"  # light warm red
_D1_FALSE = "#7aaaf4"  # light sky blue
_D2_TRUE = "#8b1e0e"  # dark red, hue shifted toward brown-red
_D2_FALSE = "#0e2e8b"  # dark blue, hue shifted toward indigo

# Control-collapse demo palette: colour = class, marker = domain.
RED = "#d1495b"  # class 0
BLUE = "#3a78c3"  # class 1
GREEN = "#2e8b57"  # held-out domain (never fit to)
DOMAIN_C = "#7b3fbf"  # domain classifier and the divider perpendicular to it


@dataclass
class Line:
    """A pre-computed line to draw on an axes (no calculation done at plot time)."""

    xs: np.ndarray
    ys: np.ndarray
    style: str = "-"
    color: str = "k"
    width: float = 2.0
    label: str | None = None


def feature_limits(
    X: np.ndarray,
    extra_X: np.ndarray | None = None,
    top_headroom: float = 0.1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits spanning *X* (and optional held-out *extra_X*), with headroom.

    The y-axis is extended at the top by ``top_headroom`` of the padded axis
    length, leaving room for legends in the top corners.

    Returns ``(xlim, ylim)``.
    """
    allX = X if extra_X is None or len(extra_X) == 0 else np.vstack([X, extra_X])
    xlim = (allX[:, 0].min() - 0.6, allX[:, 0].max() + 0.6)
    ylow = allX[:, 1].min() - 0.6
    yhigh = allX[:, 1].max() + 0.6
    ylim = (ylow, yhigh + top_headroom * (yhigh - ylow))
    return xlim, ylim


def draw_image_panel(ax: matplotlib.axes.Axes, image_path: str | Path) -> None:
    """Fill *ax* with a static image (e.g. a DAG), with no axes or frame."""
    ax.imshow(plt.imread(Path(image_path)))
    ax.axis("off")


def draw_feature_panel(
    ax: matplotlib.axes.Axes,
    X: np.ndarray,
    cls: np.ndarray,
    dom: np.ndarray,
    lines: Sequence[Line] = (),
    extra_X: np.ndarray | None = None,
    extra_cls: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Fill *ax* with the 2D feature space: scatter, held-out domain, and lines.

    Performs no model calculation — *lines* are already-computed :class:`Line`
    objects (see :func:`causation.models.boundary_line`).

    Parameters
    ----------
    ax : matplotlib Axes
    X, cls, dom : the fitted points and their class/domain labels
    lines : list of Line
        Boundaries / dividers to overlay.
    extra_X, extra_cls : optional held-out (green) points and their true class
    xlim : optional x-limits; computed from the data if omitted
    """
    if xlim is None or ylim is None:
        default_xlim, default_ylim = feature_limits(X, extra_X)
        xlim = xlim if xlim is not None else default_xlim
        ylim = ylim if ylim is not None else default_ylim

    # Fitted points: colour = class, marker = domain.
    for c, col in [(0, RED), (1, BLUE)]:
        for d, mk in [(0, "o"), (1, "s")]:
            m = (cls == c) & (dom == d)
            ax.scatter(
                X[m, 0],
                X[m, 1],
                c=col,
                marker=mk,
                s=28,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.5,
            )

    # Held-out domain: green fill, edge coloured by true class.
    if extra_X is not None and len(extra_X) and extra_cls is not None:
        for c, edge in [(0, RED), (1, BLUE)]:
            m = extra_cls == c
            if m.any():
                ax.scatter(
                    extra_X[m, 0],
                    extra_X[m, 1],
                    c=GREEN,
                    marker="^",
                    s=34,
                    alpha=0.85,
                    edgecolors=edge,
                    linewidths=1.3,
                )

    line_handles = []
    for line in lines:
        (artist,) = ax.plot(
            line.xs,
            line.ys,
            color=line.color,
            ls=line.style,
            lw=line.width,
            label=line.label,
        )
        line_handles.append(artist)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Low Audio", fontsize=13)
    ax.set_ylabel("High Audio", fontsize=13)

    # Two legends: one for the boundary lines, one for the point encoding
    # (colour = class, marker = domain). add_artist keeps the first when the
    # second legend is drawn.
    def _marker(color, marker, label, edge="black"):
        return mlines.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="None",
            markersize=7,
            markeredgecolor=edge,
            markeredgewidth=0.5,
            label=label,
        )

    data_handles = [
        _marker(RED, "o", "class 0 (red)"),
        _marker(BLUE, "o", "class 1 (blue)"),
        _marker("0.6", "o", "domain 0 (circle)"),
        _marker("0.6", "s", "domain 1 (square)"),
    ]
    if extra_X is not None and len(extra_X):
        data_handles.append(_marker(GREEN, "^", "held-out (not fit)"))

    if line_handles:
        ax.add_artist(ax.legend(handles=line_handles, loc="upper right", fontsize=8))
    ax.legend(handles=data_handles, loc="upper left", fontsize=8)

    # Inward, mirrored ticks with minor ticks (matches the house style).
    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True, length=6
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True, length=3
    )
    ax.minorticks_on()


def _density(ax, scores, mask, color, label):
    s = scores[mask]
    kde = gaussian_kde(s)
    grid = np.linspace(scores.min() * 1.05, scores.max() * 1.05, 400)
    ax.fill_between(grid, kde(grid), color=color, alpha=0.35)
    ax.plot(grid, kde(grid), color=color, lw=2.2, label=label)


def draw_collapse_panel(
    ax: matplotlib.axes.Axes,
    scores: np.ndarray,
    cls: np.ndarray,
    thr: float | None = None,
    xlabel: str = "Divider  (projection ⊥ to domain classifier)",
) -> None:
    """Fill *ax* with the per-class densities along the collapsed axis.

    Parameters
    ----------
    ax : matplotlib Axes
    scores : 1D projection of each point onto the collapse axis
    cls : class labels (0/1)
    thr : optional class threshold to mark with a vertical line
    xlabel : label for the collapse axis
    """
    _density(ax, scores, cls == 0, RED, "red class")
    _density(ax, scores, cls == 1, BLUE, "blue class")
    if thr is not None:
        ax.axvline(
            thr, color=DOMAIN_C, ls="--", lw=2, alpha=1.0, label="class threshold"
        )
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(scores.min() * 1.05, scores.max() * 1.05)
    ax.set_ylim(bottom=0.0)

    # Inward, mirrored ticks with minor ticks (matches the house style).
    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True, length=6
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True, length=3
    )
    ax.minorticks_on()


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
    ax.legend(handles=legend_handles, loc="best", framealpha=0.85, fontsize=9, ncol=2)

    ax.set_xlabel("Low Audio", fontsize=16)
    ax.set_ylabel("High Audio", fontsize=16)
    ax.set_title(title, fontsize=18)
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
