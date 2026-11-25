"""Visualize C++ LSM estimates, original-paper prices, and SE calibration CSVs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment_files import plot_locations

ROOT = Path(__file__).resolve().parents[1]
TEAL, NAVY, ORANGE = "#00857d", "#263b57", "#bf6225"


def save(figure, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.png", dpi=180)
    svg = output_dir / f"{name}.svg"
    figure.savefig(svg, metadata={"Date": None})
    # Matplotlib appends spaces to path-data lines; keep generated text tidy.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                   + "\n", encoding="utf-8")
    plt.close(figure)
    print(f"Saved {name}.png and .svg")


def tree_label(data):
    return f"{int(data.reference_steps.iloc[0]):,}-step"


def paper_table(data_dir, output_dir):
    data = pd.read_csv(data_dir / "lsm_paper_table.csv")
    tree = tree_label(data)
    paths = int(data.paths.iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained", sharex=True)
    fig.suptitle(f"Longstaff-Schwartz Table 1: {paths:,} simulated paths\n"
                 "K = 40, r = 6%; 50 exercise dates per year", fontsize=15, weight="bold")
    for row, volatility in enumerate([0.2, 0.4]):
        for column, maturity in enumerate([1, 2]):
            axis = axes[row, column]
            frame = data[(data.volatility == volatility) & (data.maturity == maturity)].sort_values("spot")
            axis.plot(frame.spot, frame.american_reference_crr, ":", color="#8793a1",
                      label=f"{tree} American CRR, every tree date", lw=2)
            axis.plot(frame.spot, frame.american_grid_reference_crr, "--", color=NAVY,
                      label=f"{tree} American CRR, matched exercise dates")
            axis.errorbar(frame.spot - 0.10, frame.paper_lsm_american,
                          yerr=1.96 * frame.paper_standard_error, fmt="s", mfc="white",
                          ms=5, capsize=3, color=ORANGE, label="Published LSM (±1.96 printed SE)")
            axis.errorbar(frame.spot + 0.10, frame.price, yerr=1.96 * frame.standard_error,
                          fmt="o", ms=4, capsize=3, color=TEAL,
                          label="Our LSM (±1.96 pair SE)")
            axis.set(title=f"Volatility {volatility:.0%}, maturity {maturity} year(s)",
                     xlabel="Initial stock price", ylabel="Put value")
            axis.set_xticks(sorted(frame.spot.unique()))
            axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    save(fig, output_dir, "lsm_paper_table")


def convergence(data_dir, output_dir):
    data = pd.read_csv(data_dir / "lsm_paper_convergence.csv")
    summary = pd.read_csv(data_dir / "lsm_paper_se_summary.csv")
    paths = summary.paths.to_numpy()
    tree = tree_label(data)
    replicates = int(summary.replicates.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    fig.suptitle(f"LSM repeated from simulation through policy fitting: {replicates} seeds per budget\n"
                 "S0 = K = 40, r = 6%, volatility = 20%, maturity = 1 year",
                 fontsize=14, weight="bold")
    axis = axes[0]
    axis.scatter(data.paths, data.price, color=TEAL, s=12, alpha=0.25, label="Individual full fits")
    axis.errorbar(paths, summary.mean_price, yerr=1.96 * summary.replicate_price_sd,
                  fmt="o-", capsize=4, color=TEAL, label="Mean ± 1.96 x across-run SD")
    axis.axhline(data.american_grid_reference_crr.iloc[0], color=NAVY, ls="--",
                 label=f"{tree} American CRR, matched exercise dates")
    axis.axhline(data.paper_lsm_american.iloc[0], color=ORANGE, ls=":", label="Published LSM (rounded)")
    axis.set(xscale="log", title="Price and spread across fresh fits",
             xlabel="Total simulated paths", ylabel="Put value")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    axis = axes[1]
    axis.loglog(paths, summary.rmse_vs_american_grid_crr, "o-", color=ORANGE,
                label="Across-run RMSE vs American CRR (matched dates)")
    axis.loglog(paths, summary.replicate_price_sd, "s-", color=TEAL, label="Across-run SD")
    guide = summary.rmse_vs_american_grid_crr.iloc[0] * np.sqrt(paths[0] / paths)
    axis.loglog(paths, guide, "--", color=NAVY, label=r"$N^{-1/2}$ slope guide")
    axis.set(title="Sampling variability and total error", xlabel="Total simulated paths", ylabel="Price units")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    save(fig, output_dir, "lsm_paper_convergence")


def standard_errors(data_dir, output_dir):
    data = pd.read_csv(data_dir / "lsm_paper_table.csv")
    summary = pd.read_csv(data_dir / "lsm_paper_se_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    fig.suptitle("Standard errors: original table and repeated-fit check\n"
                 "Pathwise SE is a diagnostic; pair-based SE accounts for antithetic covariance",
                 fontsize=14, weight="bold")
    axis = axes[0]
    index = np.arange(len(data))
    axis.axhspan(0.8, 1.2, color="#dfe5ed", alpha=0.65, label="Within 20% of printed SE")
    axis.axhline(1, color=NAVY, ls="--", lw=1)
    axis.plot(index, data.naive_se_to_paper_se, "s-", ms=4, color=ORANGE, label="Pathwise SE / paper SE")
    axis.plot(index, data.pair_se_to_paper_se, "o-", ms=4, color=TEAL, label="Pair-based SE / paper SE")
    labels = [f"{int(row.spot)} / {int(100*row.volatility)} / {int(row.maturity)}"
              for row in data.itertuples()]
    axis.set_xticks(index, labels, rotation=90, fontsize=8)
    axis.set(title="All 20 Table 1 cases", xlabel="Spot / volatility (%) / maturity (years)",
             ylabel="Ratio to published standard error")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, loc="lower right")
    axis = axes[1]
    axis.loglog(summary.paths, summary.mean_naive_path_standard_error, "s-", color=ORANGE,
                label="Mean pathwise SE (diagnostic)")
    axis.loglog(summary.paths, summary.mean_pair_standard_error, "o-", color=TEAL,
                label="Mean pair-based plug-in SE")
    axis.loglog(summary.paths, summary.replicate_price_sd, "D--", color=NAVY,
                label=f"Across-run SD ({int(summary.replicates.iloc[0])} fits)")
    axis.set(title="At-the-money, one-year repeated fits", xlabel="Total simulated paths", ylabel="Price units")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    save(fig, output_dir, "lsm_paper_standard_errors")


def create_figures(data_dir=None, output_dir=None, quick=False):
    data_dir, output_dir = plot_locations("paper_replication", data_dir, output_dir, quick)
    paper_table(data_dir, output_dir)
    convergence(data_dir, output_dir)
    standard_errors(data_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    create_figures(args.data_dir, args.output_dir, args.quick)
