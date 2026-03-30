"""
Uncorrelated two-Gaussian classification demo.

Both domains share the same mean (0, 0) and identity covariance,
so there is no systematic separation between them — any decision boundary
learned from one domain should not transfer to the other.

All logic lives in the `causation` package:
  causation.data   — data generation
  causation.models — model fitting, accuracy reporting, decision boundary
  causation.plot   — plotting
"""

from pathlib import Path

from causation import plot_two_gaussian_logistic

_MEAN = (0.0, 0.0)
_COV = ((1.0, 0.0), (0.0, 1.0))

if __name__ == "__main__":
    plot_two_gaussian_logistic(
        mean1=_MEAN,
        cov1=_COV,
        mean2=_MEAN,
        cov2=_COV,
        title="Similar Domains",
        random_seed=1337,
        save_path=Path(__file__).parent.parent / "figures" / "uncorrelated.png",
    )
