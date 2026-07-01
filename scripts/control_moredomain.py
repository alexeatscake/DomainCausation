"""Control-collapse demo where the domain shift is larger than the class shift.

Same machinery as ``control_simple`` but the domains are pushed far apart along
the vertical axis (and not perpendicular to the class split), so the domain
direction dominates the variation.

All reusable logic lives in the `causation` package; this script only defines
the scenario (the blobs) and the figure layout.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from causation import (
    Blob,
    Line,
    boundary_line,
    collapse_perpendicular_to_domain,
    domain_averaged_boundary,
    draw_collapse_panel,
    draw_feature_panel,
    draw_image_panel,
    feature_limits,
    fit_class_classifier,
    fit_control_classifiers,
    generate_blobs,
)
from causation.plot import DOMAIN_C

# Large domain shift along the vertical axis, not perpendicular to the classes.
BLOBS = [
    Blob(cls=0, dom=0, centre=(-2.0, -2.0), n=200),  # red  circle
    Blob(cls=0, dom=1, centre=(+2.0, -2.0), n=200),  # red  square
    Blob(cls=1, dom=0, centre=(-4.0, +2.0), n=200),  # blue circle
    Blob(cls=1, dom=1, centre=(+4.0, +2.0), n=200),  # blue square
]

NOISE = 1.5
SEED = 1339

# DAG illustrating the causal setup, shown as the leftmost panel.
DAG_IMAGE = Path(__file__).parent.parent / "figures" / "dag_images" / "MoreDomain.png"


def main() -> None:
    X, cls, dom = generate_blobs(BLOBS, noise=NOISE, seed=SEED)
    class_clf, domain_clf, ctrl_clf = fit_control_classifiers(X, cls, dom)
    wc, bc = domain_averaged_boundary(ctrl_clf)
    scores, u, thr = collapse_perpendicular_to_domain(X, cls, domain_clf)

    # Class classifier fit on the circle domain (dom == 0) only.
    circle = dom == 0
    circle_clf = fit_class_classifier(X[circle], cls[circle])

    # Limits span fitted (and any held-out) points, with top headroom for the
    # corner legends.
    xlim, ylim = feature_limits(X)

    # The domain classifier and the divider perpendicular to it share a colour.
    lines = [
        Line(
            *boundary_line(class_clf.coef_[0], class_clf.intercept_[0], xlim, ylim),
            style="-",
            color="#222222",
            width=1.8,
            label="ERM class boundary",
        ),
        Line(
            *boundary_line(circle_clf.coef_[0], circle_clf.intercept_[0], xlim, ylim),
            style="-.",
            color="#e07b00",
            width=1.8,
            label="Circle-only class boundary",
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), width_ratios=[1.0, 1.4, 1.4])
    draw_image_panel(axes[0], DAG_IMAGE)
    draw_feature_panel(axes[1], X, cls, dom, lines=lines, xlim=xlim, ylim=ylim)
    draw_collapse_panel(axes[2], scores, cls, thr=thr)

    fig.tight_layout()
    save_path = Path(__file__).parent.parent / "figures" / "control_moredomain.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
