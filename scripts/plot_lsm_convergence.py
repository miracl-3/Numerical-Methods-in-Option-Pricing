"""Plot saved convergence summaries and timings; never invokes numerical pricing."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

from analyze_lsm import style, COLORS, LABELS, ATM_CONTRACT
from experiment_files import ROOT, prepare_results, plot_locations
from lsm_statistics import (DEFAULT_ANALYSIS, PATH_COUNTS, sha256, audit_existing_results,
                            load_completion, missing_convergence_keys, summarize_prices, convergence_slopes)
from plot_lsm_paper import save

T_975_DF29 = 2.045229642132703
OUTPUT = ROOT / "data/experiments/lsm_analysis/convergence"
REUSED_FIGURE = ROOT / "figures/lsm_analysis/full/variance_and_efficiency.png"


def load_saved(completion_dir=None):
    manifest_path = DEFAULT_ANALYSIS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # These are data/provenance checks only; no executable or environment checks.
    for name, digest in manifest["inputs_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Saved statistical inputs changed: {name}")
    files = {"summary": DEFAULT_ANALYSIS / "sampling_summary.csv",
             "efficiency": DEFAULT_ANALYSIS / "efficiency.csv",
             "missing": DEFAULT_ANALYSIS / "missing_configurations.csv",
             "paper_runs": ROOT / "data/experiments/paper_replication/full/lsm_paper_convergence.csv",
             "controlled_runs": ROOT / "data/experiments/lsm_comparison/full/replicates.csv",
             "reference": ROOT / "data/experiments/paper_replication/full/lsm_paper_table.csv"}
    tables = {name: pd.read_csv(path, float_precision="round_trip") for name, path in files.items()}
    frames, audit = audit_existing_results()
    summary = tables["summary"].copy()
    if not (summary.runs == 30).all() or summary.duplicated(["sampling_method", "paths"]).any():
        raise ValueError("Expected thirty unique repeated runs per saved summary group.")
    # Retain the saved means, SDs and RMSEs. Replace normal-approximation bounds
    # only in this derived table with explicitly labelled Student-t bounds.
    summary["mean_ci_lower"] = summary.mean_price - T_975_DF29 * summary.empirical_sd / np.sqrt(summary.runs)
    summary["mean_ci_upper"] = summary.mean_price + T_975_DF29 * summary.empirical_sd / np.sqrt(summary.runs)
    summary["ci_method"] = "pointwise Student t, df=29"
    summary["ci_critical_value"] = T_975_DF29
    reference = float(tables["reference"].query("spot==40 and volatility==.2 and maturity==1")
                      .american_grid_reference_crr.iloc[0])
    original = tables["paper_runs"].copy()
    original["sampling_method"] = "antithetic"
    combined = pd.concat([original[original.paths < 100000], tables["controlled_runs"]], ignore_index=True)
    for row in summary.itertuples():
        group = combined[(combined.paths == row.paths) & (combined.sampling_method == row.sampling_method)]
        if len(group) != 30 or group.seed.nunique() != 30:
            raise ValueError("Repeated-run counts do not match the saved summary.")
        measured = [group.price.mean(), group.price.std(ddof=1), np.sqrt(np.mean((group.price-reference)**2))]
        if not np.allclose(measured, [row.mean_price, row.empirical_sd, row.rmse], rtol=1e-12, atol=1e-12):
            raise ValueError("Saved convergence statistics disagree with their source CSVs.")
    extra = pd.DataFrame()
    extra_hashes = {}
    if completion_dir is not None:
        extra, extra_hashes = load_completion(completion_dir, frames, audit)
        added = []
        for paths, group in extra.groupby("paths", sort=True):
            row = summarize_prices(group.price.tolist(), group.standard_error.tolist(), reference)
            row.update(sampling_method="plain", paths=int(paths), mean_pathwise_se=group.naive_path_standard_error.mean())
            added.append(row)
        summary = pd.concat([summary, pd.DataFrame(added)], ignore_index=True)
        combined = pd.concat([combined, extra], ignore_index=True)
    if combined.duplicated(["sampling_method", "paths", "seed"]).any():
        raise ValueError("Duplicate observations across saved experiments.")
    summary["mean_ci_lower"] = summary.mean_price - T_975_DF29*summary.empirical_sd/np.sqrt(summary.runs)
    summary["mean_ci_upper"] = summary.mean_price + T_975_DF29*summary.empirical_sd/np.sqrt(summary.runs)
    summary["ci_method"] = "pointwise Student t, df=29"
    summary["ci_critical_value"] = T_975_DF29
    missing_keys = missing_convergence_keys(combined)
    missing = pd.DataFrame([{"sampling_method": method, "paths": paths,
                             "missing_runs": sum(m==method and n==paths for m,n,s in missing_keys),
                             "seeds": ",".join(str(s) for m,n,s in missing_keys if m==method and n==paths)}
                            for method, paths in sorted(set((m,n) for m,n,s in missing_keys))],
                           columns=["sampling_method", "paths", "missing_runs", "seeds"])
    timings = []
    for paths, group in original.groupby("paths", sort=True):
        timings.append({"experiment": "paper_replication", "execution_session": "original_manifest",
                        "sampling_method": "antithetic", "paths": int(paths),
                        "runs": len(group), "median_ms": group.runtime_ms.median(),
                        "q1_ms": group.runtime_ms.quantile(.25), "q3_ms": group.runtime_ms.quantile(.75)})
    for row in tables["efficiency"].itertuples():
        timings.append({"experiment": "controlled_comparison", "execution_session": "controlled_manifest", "sampling_method": row.sampling_method,
                        "paths": int(row.paths), "runs": int(row.runs), "median_ms": 1000*row.median_runtime_seconds,
                        "q1_ms": 1000*row.runtime_q1_seconds, "q3_ms": 1000*row.runtime_q3_seconds})
    if not extra.empty:
        for (session, paths), group in extra.groupby(["execution_session", "paths"], sort=True):
            timings.append({"experiment": "plain_completion", "execution_session": session, "sampling_method": "plain",
                            "paths": int(paths), "runs": len(group), "median_ms": group.runtime_ms.median(),
                            "q1_ms": group.runtime_ms.quantile(.25), "q3_ms": group.runtime_ms.quantile(.75)})
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files.values()}
    hashes[manifest_path.relative_to(ROOT).as_posix()] = sha256(manifest_path)
    hashes.update(extra_hashes)
    return summary, pd.DataFrame(timings), missing, reference, hashes, tables["efficiency"], audit


def budget_axis(axis):
    axis.set_xscale("log")
    axis.set_xlim(4000, 130000)
    axis.xaxis.set_major_locator(FixedLocator(PATH_COUNTS))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Total paths, N")
    axis.grid(which="major", alpha=.18)


def convergence_plot(summary, slopes, reference, output):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), layout="constrained")
    fig.suptitle("LSM convergence across repeated runs")
    for method in LABELS:
        group = summary[summary.sampling_method == method].sort_values("paths")
        marker = "s" if method == "plain" else "o"
        # Small display-only shifts separate overlapping intervals at 100k;
        # the CSV and axis budget values retain the exact observed counts.
        shift = (4 if method == "plain" else -4) / 72
        transform = axes[0].transData + ScaledTranslation(shift, 0, fig.dpi_scale_trans)
        axes[0].errorbar(group.paths, group.mean_price,
                         yerr=T_975_DF29*group.empirical_sd/np.sqrt(group.runs),
                         fmt=marker, color=COLORS[method], capsize=4, markersize=5,
                         transform=transform, label=LABELS[method], zorder=4)
        for axis, field in ((axes[1], "empirical_sd"), (axes[2], "rmse")):
            axis.plot(group.paths, group[field], marker, color=COLORS[method], markersize=6,
                      label=LABELS[method], zorder=4)
    axes[0].axhline(reference, color="#263b57", ls="--", lw=1.2,
                    label=f"Matched CRR: {reference:.6f}")
    axes[0].set(title="Mean price and 95% CI", ylabel="Mean estimated price")
    axes[0].margins(y=.22)
    anchor = summary.query("sampling_method=='antithetic' and paths==5000").empirical_sd.iloc[0]
    guide_paths = np.geomspace(5000, 100000, 100)
    axes[1].plot(guide_paths, anchor*np.sqrt(5000/guide_paths), color="#526273", ls="--", lw=1.2,
                 label=r"$N^{-1/2}$ slope guide")
    axes[1].set(title="Empirical standard deviation", ylabel="SD across runs", yscale="log")
    axes[2].set(title="RMSE against matched CRR", ylabel="RMSE across runs", yscale="log")
    slope_text = "; ".join(f"{LABELS[row.sampling_method]}: b={row.slope_b:.3f}" for row in slopes.itertuples()
                           if row.status == "descriptive_fit")
    axes[1].text(.04, .04, slope_text, transform=axes[1].transAxes, fontsize=8)
    for axis in axes:
        budget_axis(axis)
        axis.legend(fontsize=8, loc="upper right")
        axis.tick_params(labelsize=9)
    for axis in axes[1:]:
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
        axis.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:.3f}" if y in (.005,.006,.02,.03) else ""))
        axis.set_ylim(summary[["empirical_sd", "rmse"]].min().min()*.8,
                      summary[["empirical_sd", "rmse"]].max().max()*1.55)
    availability = "Both methods: all five budgets." if len(summary)==10 else "Plain LSM is missing below 100k."
    fig.supxlabel(ATM_CONTRACT + "\n"
        r"30 runs per point. Mean CI: $\bar P \pm t_{0.975,29}s/\sqrt{30}$; RMSE: $\sqrt{\mathrm{mean}[(P-P_{\mathrm{CRR}})^2]}$." + "\n"
        + availability + " CI markers are shifted for clarity. Slopes are descriptive; decreasing SD does not establish price accuracy.",
        fontsize=8.5)
    save(fig, output, "price_sd_rmse_convergence")


def runtime_plot(timings, output):
    completed = timings[timings.experiment == "plain_completion"]
    columns = 3 if not completed.empty else 2
    fig, axes = plt.subplots(1, columns, figsize=(5.4*columns, 4.9), layout="constrained")
    fig.suptitle("Recorded LSM runtimes")
    historical = timings[timings.experiment == "paper_replication"]
    axes[0].errorbar(historical.paths, historical.median_ms,
                     yerr=[historical.median_ms-historical.q1_ms, historical.q3_ms-historical.median_ms],
                     fmt="o", color=COLORS["antithetic"], capsize=4, label="AV-LSM")
    budget_axis(axes[0])
    axes[0].set(yscale="log", title="Original replication experiment", ylabel="Runtime (ms)")
    axes[0].yaxis.set_major_locator(FixedLocator([10,20,50,100,200,500]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=9)
    controlled = timings[timings.experiment == "controlled_comparison"].set_index("sampling_method")
    if not completed.empty:
        for i, (session, group) in enumerate(completed.groupby("execution_session", sort=True), 1):
            axes[1].errorbar(group.paths, group.median_ms,
                             yerr=[group.median_ms-group.q1_ms, group.q3_ms-group.median_ms],
                             fmt="s", color=COLORS["plain"], capsize=4, mfc="none" if i%2==0 else COLORS["plain"],
                             label=f"Plain LSM, session {i}")
        budget_axis(axes[1])
        axes[1].set(yscale="log", title="Completion sessions", ylabel="Runtime (ms)")
        axes[1].legend(fontsize=8)
    for i, method in enumerate(LABELS):
        row = controlled.loc[method]
        axes[-1].errorbar(i, row.median_ms, yerr=[[row.median_ms-row.q1_ms], [row.q3_ms-row.median_ms]],
                         fmt="s" if method=="plain" else "o", color=COLORS[method], capsize=6, markersize=6)
        axes[-1].annotate(f"{row.median_ms:.0f} ms", (i, row.q3_ms), xytext=(0,9), textcoords="offset points",
                         ha="center", fontsize=9, color=COLORS[method])
    axes[-1].set(title="Controlled comparison: 100,000 paths", ylabel="Runtime (ms)",
                xticks=[0,1], xticklabels=list(LABELS.values()), xlim=(-.6,1.6), ylim=(330,525))
    axes[-1].grid(axis="y", alpha=.18)
    fig.supxlabel(ATM_CONTRACT + "\n"
        "Median and interquartile range within each recorded session; counts are in runtime_summary.csv. Excludes CRR and CSV I/O.\n"
        "Panels use different recorded builds/sessions; compare methods within the right panel only. No cross-session efficiency ratios.",fontsize=8.5)
    save(fig, output, "runtime_convergence")


def variance_reduction(summary):
    rows = []
    for paths, group in summary.groupby("paths", sort=True):
        if set(group.sampling_method) != set(LABELS):
            continue
        variances = group.set_index("sampling_method").empirical_sd**2
        rows.append({"paths": int(paths), "variance_factor_plain_over_av": variances.plain/variances.antithetic,
                     "variance_reduction_percent": 100*(1-variances.antithetic/variances.plain)})
    return pd.DataFrame(rows)


def variance_efficiency_plot(reduction, efficiency, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8,4.9), layout="constrained")
    fig.suptitle("Variance reduction and computational efficiency")
    axes[0].plot(reduction.paths, reduction.variance_reduction_percent, "o", color=COLORS["antithetic"])
    axes[0].axhline(0, color="#526273", lw=1)
    budget_axis(axes[0])
    axes[0].set(title="Common statistical budgets", ylabel="AV-LSM variance reduction (%)")
    axes[0].margins(y=.3)
    data = efficiency.set_index("sampling_method").loc[list(LABELS)]
    axes[1].bar(range(2), data.relative_efficiency, color=list(COLORS.values()))
    axes[1].axhline(1, color="#526273", ls="--")
    for i, row in enumerate(data.itertuples()):
        axes[1].text(i, row.relative_efficiency+.08, f"{row.relative_efficiency:.2f}×", ha="center")
    axes[1].set(title="Comparable timings: 100,000 paths", ylabel="Relative efficiency (plain = 1)",
                xticks=[0,1], xticklabels=list(LABELS.values()), ylim=(0,data.relative_efficiency.max()*1.3))
    axes[1].grid(axis="y", alpha=.18)
    fig.supxlabel(ATM_CONTRACT+"\n"
        r"Variance reduction: $100(1-s_{AV}^2/s_{plain}^2)$. Efficiency: $s_{plain}^2 t_{plain}/(s^2 t)$, using median runtime $t$."+"\n"
        "Variance uses 30 full fits per method/budget. Cross-method runtime efficiency is supported only by the original controlled 100k experiment.",fontsize=8.5)
    save(fig, output, "variance_reduction_efficiency")


def create(output_dir, completion_dir=None, require_complete=False, check_only=False):
    summary, timings, missing, reference, hashes, efficiency, audit = load_saved(completion_dir)
    slopes = convergence_slopes(summary)
    missing_count = int(missing.missing_runs.sum())
    print(f"Compatible unique runs: {int(summary.runs.sum())}; missing runs: {missing_count}.")
    print(audit["statistical_compatibility"])
    print(slopes[["sampling_method", "budgets", "slope_b", "r_squared", "status"]].to_string(index=False))
    if require_complete and missing_count:
        raise ValueError("Complete the missing fits locally, then supply --completion-dir. No figures were written.")
    if check_only:
        return
    reduction = variance_reduction(summary)
    reused_hash = sha256(REUSED_FIGURE)
    output = prepare_results("lsm_analysis", output_dir, False)
    _, figures = plot_locations("lsm_analysis", output)
    summary.to_csv(output / "convergence_summary.csv", index=False, float_format="%.17g")
    timings.to_csv(output / "runtime_summary.csv", index=False, float_format="%.17g")
    missing.to_csv(output / "missing_configurations.csv", index=False)
    slopes.to_csv(output / "convergence_slopes.csv", index=False, float_format="%.17g")
    reduction.to_csv(output / "variance_reduction.csv", index=False, float_format="%.17g")
    efficiency.to_csv(output / "comparable_efficiency.csv", index=False, float_format="%.17g")
    style()
    convergence_plot(summary, slopes, reference, figures)
    runtime_plot(timings, figures)
    variance_efficiency_plot(reduction, efficiency, figures)
    manifest = {"analysis_only": True, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": hashes, "plotting_source_sha256": {name: sha256(ROOT/name) for name in
            ("scripts/plot_lsm_convergence.py", "scripts/analyze_lsm.py", "scripts/plot_lsm_paper.py")},
        "reference": {"grid_matched_crr": reference, "time_steps": 100000, "exercise_dates": 50},
        "mean_ci": {"distribution": "Student t", "df": 29, "critical_value": T_975_DF29, "level": .95,
                    "scope": "Pointwise CI for mean over repeated complete fits; approximate normality of run estimates."},
        "preserved_figure": {"path": REUSED_FIGURE.relative_to(ROOT).as_posix(), "sha256": reused_hash},
        "compatibility": audit["statistical_compatibility"],
        "slope_method": "Unweighted OLS log(SD)=a+b log(N). Descriptive slope/R-squared only; shared seeds correlate SDs across budgets, so no iid-regression confidence interval.",
        "missing_runs": missing_count, "simulations_run": 0, "crr_calculations_run": 0,
        "limitations": ["No interpolation of observations; absent configurations are listed separately.",
                        "Price CIs quantify uncertainty in the estimator mean, not uncertainty from exercise-grid or policy bias.",
                        "Empirical SD and RMSE use 30 runs; lower variance alone does not prove convergence to the true price.",
                        "The SD slope guide is anchored to the observed AV 5000-path SD, not fitted observations.",
                        "Runtime cohorts remain separate; lower-budget cross-method efficiency is unsupported."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    if sha256(REUSED_FIGURE) != reused_hash:
        raise ValueError("The retained efficiency figure changed unexpectedly.")
    print(f"Saved three multi-panel figures to {figures}; preserved earlier figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--completion-dir", type=Path, help="Completed missing-only run directory; all 120 fits must be valid.")
    parser.add_argument("--require-complete", action="store_true", help="Refuse incomplete five-budget comparisons.")
    parser.add_argument("--check-only", action="store_true", help="Check saved-data compatibility and print slopes without writing figures.")
    args = parser.parse_args()
    try:
        create(args.output_dir, args.completion_dir, args.require_complete, args.check_only)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, str(error)+"\n")
