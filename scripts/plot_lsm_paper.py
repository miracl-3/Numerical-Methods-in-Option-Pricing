"""Plot the paper-parameter LSM table and Monte Carlo path convergence."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def save(figure, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.png", dpi=180)
    figure.savefig(output_dir / f"{name}.svg", metadata={"Date": None})
    plt.close(figure)
    print(f"Saved {name}.png and .svg")


def paper_table(data_dir, output_dir):
    data = pd.read_csv(data_dir / "lsm_paper_table.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharex=True)
    fig.suptitle("LSM on the Longstaff–Schwartz American-put parameter grid\n"
                 "K = 40, r = 6%; 50 exercise dates per year",
                 fontsize=15, weight="bold")
    for row, volatility in enumerate([0.2, 0.4]):
        for column, maturity in enumerate([1, 2]):
            axis = axes[row, column]
            frame = data[(data.volatility == volatility) & (data.maturity == maturity)]
            frame = frame.sort_values("spot")
            axis.errorbar(frame.spot, frame.price, yerr=1.96 * frame.standard_error,
                          fmt="o-", capsize=3, color="#00857d",
                          label="Paper-mode LSM (±1.96 pair-based SE)")
            axis.plot(frame.spot, frame.american_reference_crr, "s--",
                      color="#263b57", label="2,000-step Bermudan CRR reference")
            axis.set(title=f"Volatility={volatility:.0%}, maturity={maturity} year(s)",
                     xlabel="Initial stock price S0", ylabel="American put value")
            axis.set_xticks(sorted(frame.spot.unique()))
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, output_dir, "lsm_paper_table")


def convergence(data_dir, output_dir):
    data = pd.read_csv(data_dir / "lsm_paper_convergence.csv")
    grouped = data.groupby("paths")
    paths = np.array(sorted(data.paths.unique()))
    means = grouped.price.mean().reindex(paths).to_numpy()
    lows = grouped.price.min().reindex(paths).to_numpy()
    highs = grouped.price.max().reindex(paths).to_numpy()
    rmse = np.array([
        np.sqrt(np.mean((frame.price - frame.american_reference_crr) ** 2))
        for _, frame in grouped
    ])
    benchmark = float(data.american_reference_crr.iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    fig.suptitle("Paper-mode LSM: convergence with antithetic path pairs\n"
                 "S0 = K = 40, r = 6%, volatility = 20%, maturity = 1 year",
                 fontsize=15, weight="bold")
    axis = axes[0]
    axis.fill_between(paths, lows, highs, color="#00857d", alpha=0.17, label="Range across seeds")
    axis.plot(paths, means, "o-", color="#00857d", label="Mean estimate")
    axis.axhline(benchmark, color="#263b57", ls="--", label="2,000-step Bermudan CRR reference")
    axis.set(xscale="log", title="Price estimates as paths increase",
             xlabel="Total simulated paths", ylabel="American put value")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)

    axis = axes[1]
    axis.loglog(paths, rmse, "o-", color="#c16a2b", label="RMSE across seeds")
    guide = rmse[0] * np.sqrt(paths[0] / paths)
    axis.loglog(paths, guide, "--", color="#263b57", label=r"Reference $N^{-1/2}$ slope")
    axis.set(title="Observed error trend", xlabel="Total simulated paths",
             ylabel="RMSE vs Bermudan CRR reference")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    save(fig, output_dir, "lsm_paper_convergence")


def create_figures(data_dir=ROOT / "data" / "experiments", output_dir=ROOT / "figures"):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    paper_table(data_dir, output_dir)
    convergence(data_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "experiments")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    create_figures(args.data_dir, args.output_dir)
