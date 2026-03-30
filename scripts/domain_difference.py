"""
Different two-Gaussian classification demo.
"""

from pathlib import Path

from causation import plot_two_gaussian_logistic

COV = ((1.0, 0.0), (0.0, 1.0))

if __name__ == "__main__":
    plot_two_gaussian_logistic(
        mean1=(-2, 0),
        cov1=COV,
        mean2=(2, 0),
        cov2=COV,
        loss1=-0.5,
        loss2=0.5,
        title="With Label Loss",
        random_seed=1337,
        save_path=Path(__file__).parent.parent / "figures" / "domain_difference.png",
    )
