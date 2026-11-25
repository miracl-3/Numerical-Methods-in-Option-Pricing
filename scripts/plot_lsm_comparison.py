"""Plot the separate controlled LSM comparison datasets; no numerical pricing."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build import ROOT
from plot_lsm_paper import save
from experiment_files import REFERENCE_DIR, plot_locations

COLORS = {"plain": "#bf6225", "antithetic": "#00857d"}
LABELS = {"plain": "Plain LSM", "antithetic": "AV-LSM"}


def create_figures(data_dir=None, output_dir=None, quick=False):
    data_dir, output_dir = plot_locations("lsm_comparison", data_dir, output_dir, quick)
    table = pd.read_csv(data_dir / "table1.csv")
    repeats = pd.read_csv(data_dir / "replicates.csv")
    diagnostic = pd.read_csv(data_dir / "table2.csv")
    paths = int(table.paths.iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained", sharex=True)
    fig.suptitle(f"Plain LSM and AV-LSM: identical algorithms, {paths:,} paths each\n"
                 "Weighted Laguerre basis; in-sample valuation; 50 exercise dates/year plus time zero",
                 fontsize=14, weight="bold")
    for i, vol in enumerate((0.2, 0.4)):
        for j, maturity in enumerate((1, 2)):
            axis = axes[i, j]
            selected = table[(table.volatility == vol) & (table.maturity == maturity)]
            ref = selected[selected.sampling_method == "plain"].sort_values("spot")
            axis.plot(ref.spot, ref.comparison_reference_crr, "--", color="#263b57", label="American CRR, matched dates")
            axis.plot(ref.spot, ref.paper_lsm_price, "x", color="#333333", label="Published AV-LSM")
            for sampling, offset, marker in (("plain", -0.07, "s"), ("antithetic", 0.07, "o")):
                frame = selected[selected.sampling_method == sampling].sort_values("spot")
                axis.errorbar(frame.spot + offset, frame.price, yerr=1.96 * frame.standard_error,
                              fmt=marker, color=COLORS[sampling], ms=4, capsize=3,
                              label=f"{LABELS[sampling]} (±1.96 approximate SE)")
            axis.set(title=f"Volatility {vol:.0%}, maturity {maturity} year(s)",
                     xlabel="Initial stock price", ylabel="American put value", xticks=sorted(ref.spot.unique()))
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    save(fig, output_dir, "table1_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    fig.suptitle(f"At-the-money one-year put: repeated full fits, {paths:,} paths per method\n"
                 "S0 = K = 40, r = 6%, volatility = 20%", fontsize=14, weight="bold")
    for sampling in LABELS:
        frame = repeats[repeats.sampling_method == sampling].sort_values("seed")
        axes[0].plot(frame.seed, frame.price, "o-", ms=3, alpha=0.8, color=COLORS[sampling], label=LABELS[sampling])
    axes[0].axhline(repeats.comparison_reference_crr.iloc[0], ls="--", color="#263b57", label="American CRR, matched dates")
    axes[0].set(title="Price across fresh simulation and regression", xlabel="Master seed", ylabel="Price")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)
    for index, sampling in enumerate(LABELS):
        frame = repeats[repeats.sampling_method == sampling]
        values = [frame.price.std(ddof=1), frame.standard_error.mean(), frame.naive_path_standard_error.mean()]
        for k, (value, label, color) in enumerate(zip(values,
                ("Across-fit SD", "Mean reported SE", "Mean pathwise diagnostic"),
                ("#263b57", "#00857d", "#bf6225"))):
            axes[1].bar(index + (k - 1) * 0.23, value, width=0.21, color=color, label=label if index == 0 else None)
    axes[1].set(xticks=[0, 1], xticklabels=list(LABELS.values()), ylabel="Price units",
                title=f"SE calibration from {int(repeats.seed.nunique())} fits per method")
    axes[1].set_ylim(0, max(repeats.groupby("sampling_method").price.std().max(),
                          repeats.naive_path_standard_error.max()) * 1.35)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.2)
    save(fig, output_dir, "sampling_variability")

    keys = ["spot", "volatility", "maturity", "seed"]
    ins = diagnostic.loc[diagnostic.valuation_mode == "in-sample", keys + ["price"]]
    outs = diagnostic.loc[diagnostic.valuation_mode == "out-of-sample", keys + ["price"]]
    paired = ins.merge(outs, on=keys, suffixes=("_in", "_out"))
    paired["difference"] = paired.price_in - paired.price_out
    fig, axis = plt.subplots(figsize=(11, 5.2), layout="constrained")
    fig.suptitle("Table 2 diagnostic: one fitted AV-LSM policy, two valuation samples\n"
                 f"{paths:,} training paths and {paths:,} independent evaluation paths per fit",
                 fontsize=14, weight="bold")
    labels = []
    published = pd.read_csv(REFERENCE_DIR / "table2.csv")
    for index, ((spot, vol, maturity), frame) in enumerate(paired.groupby(keys[:3], sort=True)):
        labels.append(f"{int(spot)} / {vol:.0%} / {int(maturity)}y")
        x = index + np.linspace(-0.12, 0.12, len(frame))
        axis.scatter(x, frame.difference, color="#00857d", alpha=0.65, s=24,
                     label="Individual seed difference" if index == 0 else None)
        axis.errorbar(index, frame.difference.mean(), yerr=frame.difference.std(ddof=1),
                      fmt="D", capsize=4, color="#263b57", label="Mean ± across-fit SD" if index == 0 else None)
        original = published[(published.spot == spot) & (published.volatility == vol) & (published.maturity == maturity)]
        axis.plot(index, (original.paper_in_sample - original.paper_out_of_sample).mean(), "x",
                  color="#bf6225", ms=8, label="Published mean (from prices)" if index == 0 else None)
    axis.axhline(0, color="#555555", lw=1, ls="--")
    axis.set(xticks=range(len(labels)), xticklabels=labels, xlabel="Spot / volatility / maturity",
             ylabel="In-sample price minus out-of-sample price")
    axis.legend(fontsize=8); axis.grid(axis="y", alpha=0.2)
    save(fig, output_dir, "table2_diagnostic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    create_figures(args.data_dir, args.output_dir, args.quick)
