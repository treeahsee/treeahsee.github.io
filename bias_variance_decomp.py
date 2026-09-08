"""Bias-variance decomposition on polynomial fits of sin(x).

Fitting the same target at several polynomial degrees over many resampled
label sets makes the tradeoff visible: low-degree fits are biased, high-degree
fits are high-variance, and a middle degree minimises total error.

Importable so a Quarto post can call ``run`` and ``plot_fits`` in a live code
chunk; runnable as a script to print the table and write the figure.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(42)

NOISE_SD = 0.1
N_TRAIN = 20
N_SETS = 300  # how many independent training sets we average over
DEGREES = (1, 3, 12)
N_CURVES_SHOWN = 40  # individual fits drawn per panel to convey the spread
PLOT_PATH = "bias-variance.png"


def true_f(x):
    return np.sin(x)


# Fixed design: same input locations every training set, only the label noise
# is resampled. This isolates variance as "sensitivity to noise" and keeps the
# high-degree fits numerically sane (resampling x too makes deg-12 explode).
X_raw_fixed = np.sort(rng.random(N_TRAIN) * 6).reshape(-1, 1)
X_train_fixed = (X_raw_fixed - 3) / 3
y_clean_train = true_f(X_raw_fixed.ravel())


# Fixed test grid. y_true is the clean target; the irreducible noise is NOISE_SD**2.
X_test_raw = np.linspace(0, 6, 100).reshape(-1, 1)
X_test = (X_test_raw - 3) / 3
y_true = true_f(X_test_raw.ravel())


def run() -> Sequence[dict]:
    """Fit each degree over many resampled label sets and report the split.

    Returns
    -------
    Sequence[dict]
        One entry per degree carrying its predictions and error terms.
    """
    print(
        f"{'deg':>3} | {'bias^2':>7} | {'var':>7} | {'noise':>7} | "
        f"{'sum':>7} | {'test MSE':>8}"
    )
    print("-" * 55)

    results = []
    for degree in DEGREES:
        poly = PolynomialFeatures(degree=degree)
        X_te_poly = poly.fit_transform(X_test)

        # Collect one prediction curve per training set.
        preds = np.empty((N_SETS, len(X_test)))
        test_mses = np.empty(N_SETS)

        X_tr_poly = poly.transform(X_train_fixed)
        for i in range(N_SETS):
            y_train = y_clean_train + rng.normal(0, NOISE_SD, N_TRAIN)

            model = LinearRegression().fit(X_tr_poly, y_train)

            preds[i] = model.predict(X_te_poly)
            # test error against noisy targets, to match the decomposition
            y_test_noisy = y_true + rng.normal(0, NOISE_SD, len(X_test))
            test_mses[i] = np.mean((preds[i] - y_test_noisy) ** 2)

        mean_pred = preds.mean(axis=0)  # average prediction per test point
        bias2 = np.mean((mean_pred - y_true) ** 2)  # avg pred vs truth
        variance = np.mean(preds.var(axis=0))  # spread of preds across sets
        noise = NOISE_SD**2
        test_mse = test_mses.mean()

        print(
            f"{degree:>3} | {bias2:>7.4f} | {variance:>7.4f} | {noise:>7.4f} | "
            f"{bias2 + variance + noise:>7.4f} | {test_mse:>8.4f}"
        )

        results.append(
            {
                "degree": degree,
                "preds": preds,
                "mean_pred": mean_pred,
                "bias2": bias2,
                "variance": variance,
            }
        )

    return results


def plot_fits(results: Sequence[dict], path: str | None = None) -> Figure:
    """Build a panel of model fits across increasing complexity.

    Each panel overlays a subset of the resampled fits so that the spread
    of curves shows variance directly, with the average fit and the true
    function drawn on top for reference.

    Parameters
    ----------
    results : Sequence[dict]
        One entry per degree, each holding ``degree``, ``preds``,
        ``mean_pred``, ``bias2`` and ``variance``.
    path : str or None, optional
        If given, save the rendered figure to this file.

    Returns
    -------
    Figure
        The rendered figure, so a notebook or Quarto chunk can display it inline.
    """
    x_plot = X_test_raw.ravel()
    x_train = X_raw_fixed.ravel()

    fig, axes = plt.subplots(1, len(results), figsize=(13, 4), sharey=True)
    for ax, result in zip(axes, results, strict=True):
        # Thin translucent lines: their vertical scatter *is* the variance.
        for pred in result["preds"][:N_CURVES_SHOWN]:
            ax.plot(x_plot, pred, color="#4c72b0", alpha=0.08, linewidth=0.8)
        ax.plot(
            x_plot,
            result["mean_pred"],
            color="#4c72b0",
            linewidth=2,
            label="average fit",
        )
        ax.plot(
            x_plot,
            y_true,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="true f(x) = sin(x)",
        )
        ax.scatter(
            x_train,
            y_clean_train,
            color="#c44e52",
            s=20,
            zorder=5,
            label="training points",
        )

        ax.set_title(
            f"degree {result['degree']}\n"
            f"bias$^2$={result['bias2']:.3f}  var={result['variance']:.3f}"
        )
        ax.set_xlabel("x")
        # Clip the view so the deg-12 blow-up stays legible instead of
        # rescaling every panel to its outliers.
        ax.set_ylim(-2.5, 2.5)

    axes[0].set_ylabel("y")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Same data, three complexities: underfit -> just right -> overfit",
        fontsize=13,
    )
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # Headless backend for CLI/CI so saving needs no display.
    plt.switch_backend("Agg")
    fits = run()
    plot_fits(fits, PLOT_PATH)
    plt.close("all")
    print(f"\nwrote {PLOT_PATH}")
