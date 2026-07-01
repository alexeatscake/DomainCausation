"""Control-collapse scatters at three noise levels, with shared axis limits.

A shift scenario: only the blue-square blob is pushed out along x, so the domain
shift affects one class far more than the other. Same blob centres in every
panel; only the Gaussian noise changes (0.5, 1.0, 2.0). Each panel shows the
scatter and the boundary lines (ERM, Controlling, Domain classifier, Domain
adversarial) fit on that panel's own data, titled with its noise. Axis limits
are shared so the panels are directly comparable.

All reusable logic lives in the `causation` package; this script only defines
the scenario (the blobs) and the figure layout.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causation import (
    Blob,
    Line,
    boundary_line,
    collapse_perpendicular_to_domain,
    domain_averaged_boundary,
    draw_feature_panel,
    feature_limits,
    fit_control_classifiers,
    generate_blobs,
)
from causation.plot import DOMAIN_C

# Same centres in every panel; only the noise changes. Domain shift hits one
# blob (blue square) much harder than the rest.
BLOBS = [
    Blob(cls=0, dom=0, centre=(-1.0, -1.0), n=200),  # red  circle
    Blob(cls=0, dom=1, centre=(-1.0, -1.0), n=200),  # red  square
    Blob(cls=1, dom=0, centre=(-1.0, +1.0), n=200),  # blue circle
    Blob(cls=1, dom=1, centre=(+3.0, +1.0), n=200),  # blue square
]

# (label, sigma) per panel.
NOISES = [("Small", 0.5), ("Medium", 1.0), ("Large", 2.0)]
SEED = 1337


def _boundary_lines(X, cls, dom, xlim, ylim):
    """The four boundary lines for one panel (no circle-only line)."""
    class_clf, domain_clf, ctrl_clf = fit_control_classifiers(X, cls, dom)
    wc, bc = domain_averaged_boundary(ctrl_clf)
    _scores, u, thr = collapse_perpendicular_to_domain(X, cls, domain_clf)
    return [
        Line(
            *boundary_line(class_clf.coef_[0], class_clf.intercept_[0], xlim, ylim),
            style="-",
            color="#222222",
            width=1.8,
            label="ERM class boundary",
        ),
        Line(
            *boundary_line(wc, bc, xlim, ylim),
            style=":",
            color="k",
            width=2.0,
            label="Controlling class boundary",
        ),
        Line(
            *boundary_line(domain_clf.coef_[0], domain_clf.intercept_[0], xlim, ylim),
            style="-",
            color=DOMAIN_C,
            width=2.0,
            label="Domain classifier",
        ),
        Line(
            *boundary_line(u, -thr, xlim, ylim),
            style="--",
            color=DOMAIN_C,
            width=1.8,
            label="Domain adversarial",
        ),
    ]


def main() -> None:
    datasets = [generate_blobs(BLOBS, noise=nz, seed=SEED) for _label, nz in NOISES]

    # Shared limits spanning every panel's points (widest at the largest noise).
    allX = np.vstack([X for X, _cls, _dom in datasets])
    xlim, ylim = feature_limits(allX, top_headroom=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, nz), (X, cls, dom) in zip(axes, NOISES, datasets, strict=True):
        lines = _boundary_lines(X, cls, dom, xlim, ylim)
        draw_feature_panel(ax, X, cls, dom, lines=lines, xlim=xlim, ylim=ylim)
        ax.set_title(rf"{label} noise $\sigma = {nz}$", fontsize=15)

    fig.tight_layout()
    save_path = Path(__file__).parent.parent / "figures" / "control_shift.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
