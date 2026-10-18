"""Create portfolio figures from C++ experiment CSVs (no pricing formulas in Python)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"plain_mc": "#263b57", "antithetic": "#00857d", "control_variate": "#c16a2b",
          "importance_sampling": "#8061a3"}
LABELS = {"plain_mc": "Plain MC", "antithetic": "Antithetic",
          "control_variate": "Control variate", "importance_sampling": "Importance sampling"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelcolor": "#263b57", "text.color": "#263b57", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18,
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(figure, output_dir, name):
    figure.savefig(output_dir / f"{name}.png", dpi=180)
    figure.savefig(output_dir / f"{name}.svg")
    plt.close(figure)
    print(f"Saved {name}.png and .svg")


def convergence(data_dir, output_dir):
    tree = pd.read_csv(data_dir / "tree_convergence.csv")
    mc = pd.read_csv(data_dir / "mc_convergence.csv")
    american = pd.read_csv(data_dir / "american_convergence.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle("Option pricing: convergence and the role of exercise policy", fontsize=18, weight="bold")
    ax = axes[0, 0]
    for method, label, color in [("binomial_european", "European put", "#00857d"),
                                  ("binomial_american", "American put", "#263b57")]:
        frame = tree[(tree.method == method) & (tree.option_type == "put")].sort_values("steps")
        ax.semilogx(frame.steps, frame.price, "o-", label=label, color=color)
        ax.axhline(frame.benchmark.iloc[0], color=color, ls="--", lw=1)
    ax.set(title="CRR backward induction", xlabel="Tree steps (log scale)", ylabel="Option value")
    ax.legend(loc="best")
    ax.text(0.98, 0.04, "Dashed: Black-Scholes / 2,000-step CRR",
            ha="right", transform=ax.transAxes, fontsize=8)

    ax = axes[0, 1]
    frame = mc[(mc.option_type == "put") & (mc.seed == 42)].sort_values("paths")
    ax.semilogx(frame.paths, frame.price, "o-", color="#00857d", label="European MC, seed 42")
    ax.fill_between(frame.paths, frame.ci_lower, frame.ci_upper, color="#00857d", alpha=0.16,
                    label="Pointwise 95% interval")
    ax.axhline(frame.benchmark.iloc[0], color="#263b57", ls="--", label="Black-Scholes")
    ax.set(title="Sampling uncertainty shrinks with paths", xlabel="Evaluation paths", ylabel="European put value")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for kind, color in [("call", "#263b57"), ("put", "#00857d")]:
        frame = mc[mc.option_type == kind].copy()
        frame["squared_error"] = (frame.price - frame.benchmark) ** 2
        rmse = np.sqrt(frame.groupby("paths").squared_error.mean())
        ax.loglog(rmse.index, rmse.values, "o-", color=color, label=f"{kind.title()} RMSE")
    counts = np.array(sorted(mc.paths.unique()))
    first = mc[(mc.option_type == "put") & (mc.paths == counts[0])]
    anchor = float(np.sqrt(((first.price - first.benchmark) ** 2).mean()))
    ax.loglog(counts, anchor * np.sqrt(counts[0] / counts), "--", color="#9c6c37", label=r"Reference $N^{-1/2}$")
    ax.set(title=f"Error across {mc.seed.nunique()} independent seeds",
           xlabel="Evaluation paths", ylabel="Root mean squared error")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for method, label, color in [("monte_carlo_american_fixed_policy", "Fixed rule (0.85 K)", "#c16a2b"),
                                  ("lsm_american", "LSM, independent test paths", "#00857d")]:
        frame = american[(american.method == method) & (american.seed == 42)].sort_values("paths")
        ax.errorbar(frame.paths, frame.price, yerr=1.959963984540054 * frame.standard_error,
                     fmt="o-", capsize=3, color=color, label=label)
    ax.axhline(american.benchmark.iloc[0], color="#263b57", ls="--", label="2,000-step American CRR")
    european_put = tree[(tree.method == "binomial_european") & (tree.option_type == "put")].benchmark.iloc[0]
    ax.axhline(european_put, color="#888888", ls=":", label="European Black-Scholes")
    ax.set(xscale="log", title="More paths do not remove policy bias", xlabel="Independent evaluation paths",
           ylabel="American put policy value")
    ax.legend(fontsize=8)
    save(fig, output_dir, "convergence")


def variance_reduction(data_dir, output_dir):
    data = pd.read_csv(data_dir / "variance_reduction.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle("European Monte Carlo: three variance reduction techniques", fontsize=18, weight="bold")
    for ax, kind in zip(axes[0], ["call", "put"]):
        for method in COLORS:
            frame = data[(data.option_type == kind) & (data.method == method)]
            group = frame.groupby("payoff_evaluations").standard_error.median()
            ax.loglog(group.index, group.values, "o-", color=COLORS[method], label=LABELS[method])
        ax.set(title=f"European {kind}: median standard error", xlabel="Terminal payoff evaluations (pilot included)",
               ylabel="Estimated standard error")
        ax.legend(fontsize=8)

    largest = data[data.paths == data.paths.max()]
    for ax, adjusted in zip(axes[1], [False, True]):
        for offset, kind, color in [(-0.18, "call", "#263b57"), (0.18, "put", "#00857d")]:
            frame = largest[largest.option_type == kind]
            baseline = frame[frame.method == "plain_mc"].set_index("seed")
            factors = []
            for method in ["antithetic", "control_variate", "importance_sampling"]:
                values = frame[frame.method == method].set_index("seed")
                ratio = baseline.standard_error ** 2 / values.standard_error ** 2
                if adjusted:
                    ratio *= baseline.payoff_evaluations / values.payoff_evaluations
                factors.append(float(ratio.median()))
            positions = np.arange(3) + offset
            bars = ax.bar(positions, factors, width=0.34, color=color, label=kind.title())
            ax.bar_label(bars, fmt="%.2fx", padding=3, fontsize=8)
        ax.axhline(1, color="#888888", ls="--", lw=1)
        ax.set_xticks(np.arange(3), ["Antithetic", "Control variate", "Importance sampling"])
        ax.set(title="Work-adjusted variance factor" if adjusted else "Estimator variance reduction factor",
               ylabel="Factor relative to plain MC (higher is better)")
        ax.margins(y=0.22)
        ax.legend(fontsize=8)
    save(fig, output_dir, "variance_reduction")


def exercise_grid(data_dir, output_dir):
    data = pd.read_csv(data_dir / "exercise_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    fig.suptitle("Longstaff-Schwartz: exercise-grid and training variability", fontsize=17, weight="bold")
    for seed, frame in data.groupby("seed"):
        frame = frame.sort_values("steps")
        axes[0].plot(frame.steps, frame.price, "o-", alpha=0.65, label=f"Seed {seed}")
    axes[0].axhline(data.benchmark.iloc[0], color="#263b57", ls="--", label="2,000-step CRR reference")
    axes[0].set(title="Independent policies and test samples", xlabel="Exercise intervals", ylabel="American put policy value")
    axes[0].legend(fontsize=8, ncols=2)
    grouped = data.groupby("steps").exercise_fraction
    axes[1].plot(grouped.mean().index, grouped.mean().values, "o-", color="#00857d")
    axes[1].fill_between(grouped.mean().index, grouped.min().values, grouped.max().values,
                         color="#00857d", alpha=0.15, label="Range across seeds")
    axes[1].set(title="Observed early-exercise behavior", xlabel="Exercise intervals", ylabel="Fraction exercised before maturity")
    axes[1].legend(fontsize=8)
    save(fig, output_dir, "lsm_exercise_grid")


def binomial_diagram(output_dir):
    # A four-step teaching diagram of CRR stock nodes; prices still come from C++.
    spot, strike, rate, sigma, maturity, steps = 100, 100, 0.05, 0.2, 1, 4
    up = np.exp(sigma * np.sqrt(maturity / steps))
    down = 1 / up
    probability = (np.exp(rate * maturity / steps) - down) / (up - down)
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for time in range(steps + 1):
        for ups in range(time + 1):
            height = 2 * ups - time
            stock = spot * up ** ups * down ** (time - ups)
            if time < steps:
                for movement in [-1, 1]:
                    ax.plot([time, time + 1], [height, height + movement], color="#c7d5df", zorder=1)
            label = f"S = {stock:.2f}"
            if time == steps:
                label += f"\nput = {max(strike - stock, 0):.2f}"
            ax.text(time, height, label, ha="center", va="center", fontsize=9,
                    bbox={"boxstyle": "round,pad=0.4", "fc": "#edf7f5" if time == steps else "white",
                          "ec": "#00857d" if time == steps else "#b2c4d1"}, zorder=2)
    ax.set(xlim=(-0.4, 4.5), ylim=(-4.7, 4.7))
    ax.axis("off")
    fig.suptitle("A recombining CRR tree: simulation is replaced by backward induction", fontsize=17, weight="bold")
    ax.text(0, 4.5, f"Four intervals | u = {up:.4f} | d = {down:.4f} | risk-neutral p = {probability:.4f}", fontsize=10)
    ax.text(0, -4.5, "European: discounted expected continuation.  American: max(exercise, continuation).", fontsize=10)
    save(fig, output_dir, "binomial_tree")


def create_figures(data_dir=ROOT / "data" / "experiments", output_dir=ROOT / "figures"):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style()
    convergence(data_dir, output_dir)
    variance_reduction(data_dir, output_dir)
    exercise_grid(data_dir, output_dir)
    binomial_diagram(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "experiments")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    options = parser.parse_args()
    create_figures(options.data_dir, options.output_dir)
