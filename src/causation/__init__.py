"""
causation: This shows correlation effects.
"""

from __future__ import annotations

from importlib.metadata import version

from .data import generate_two_gaussians
from .models import decision_boundary_y, fit_models, report_accuracies
from .plot import plot_dataset_with_boundaries, plot_two_gaussian_logistic

__all__ = (
    "__version__",
    "decision_boundary_y",
    "fit_models",
    "generate_two_gaussians",
    "plot_dataset_with_boundaries",
    "plot_two_gaussian_logistic",
    "report_accuracies",
)
__version__ = version(__name__)
