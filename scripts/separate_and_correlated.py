"""
Separate two-Gaussian classification demo.
"""

from pathlib import Path

from causation import plot_two_gaussian_logistic

if __name__ == "__main__":
    plot_two_gaussian_logistic(
        mean1=(-2, 0),
        cov1=((1.0, 0.7), (0.7, 1.0)),
        mean2=(2, 0),
        cov2=((1.0, -0.7), (-0.7, 1.0)),
        title="Separate and Correlated Domains",
        random_seed=1337,
        save_path=(
            Path(__file__).parent.parent / "figures" / "separate_and_correlated.png"
        ),
    )
