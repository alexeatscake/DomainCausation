from pathlib import Path

from causation import plot_two_gaussian_logistic

_MEAN = (0.0, 0.0)

if __name__ == "__main__":
    plot_two_gaussian_logistic(
        mean1=_MEAN,
        cov1=((1.0, 0.8), (0.8, 1.0)),
        mean2=_MEAN,
        cov2=((1.0, -0.8), (-0.8, 1.0)),
        title="Correlated Domains",
        random_seed=1337,
        save_path=Path(__file__).parent.parent / "figures" / "correlated.png",
    )
