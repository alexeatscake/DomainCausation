"""
causation: This shows correlation effects.
"""

from __future__ import annotations

from importlib.metadata import version

from .data import Blob, generate_blobs, generate_two_gaussians
from .models import (
    boundary_line,
    collapse_perpendicular_to_domain,
    decision_boundary_y,
    domain_averaged_boundary,
    fit_class_classifier,
    fit_control_classifiers,
    fit_controlled_classifier,
    fit_domain_classifier,
    fit_models,
    report_accuracies,
)
from .plot import (
    Line,
    draw_collapse_panel,
    draw_feature_panel,
    draw_image_panel,
    feature_limits,
    plot_dataset_with_boundaries,
    plot_two_gaussian_logistic,
)

__all__ = (
    "Blob",
    "Line",
    "__version__",
    "boundary_line",
    "collapse_perpendicular_to_domain",
    "decision_boundary_y",
    "domain_averaged_boundary",
    "draw_collapse_panel",
    "draw_feature_panel",
    "draw_image_panel",
    "feature_limits",
    "fit_class_classifier",
    "fit_control_classifiers",
    "fit_controlled_classifier",
    "fit_domain_classifier",
    "fit_models",
    "generate_blobs",
    "generate_two_gaussians",
    "plot_dataset_with_boundaries",
    "plot_two_gaussian_logistic",
    "report_accuracies",
)
__version__ = version(__name__)
