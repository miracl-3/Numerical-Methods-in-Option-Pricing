"""Audit saved LSM evidence, complete only missing full-budget runs, and plot statistics.

No numerical pricing algorithm is implemented here. Existing research is read-only.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build import executable
from experiment_files import ROOT, prepare_results, plot_locations
from run_experiments import run, write_rows
from plot_lsm_paper import save
from lsm_statistics import (audit_existing_results, existing_convergence_runs, missing_convergence_keys,
    summarize_prices, standardized_difference, relative_efficiency, paired_difference_summary,
    sha256, DEFAULT_ANALYSIS, PATH_COUNTS, SEEDS)

COLORS = {"plain": "#bd6026", "antithetic": "#00857d"}
LABELS = {"plain": "Plain LSM", "antithetic": "AV-LSM"}
KEYS = ["spot", "volatility", "maturity"]


def complete_missing(keys, build_dir, output_dir, workers):
    metadata = json.loads((build_dir / "build_metadata.json").read_text())
    for name in ("src/lsm_american.cpp", "include/model.hpp", "include/experiment.hpp", "include/binomial.hpp"):
        if sha256(ROOT / name) != metadata["source_sha256"][name]: raise ValueError("LSM executable sources are stale.")
    binary = executable("lsm_american", build_dir)
    if sha256(binary) != metadata["binary_sha256"]["lsm_american"]: raise ValueError("LSM executable hash mismatch.")
    def estimate(key):
        sampling, paths, seed = key
        return run("lsm_american", build_dir, type="put", spot=40, strike=40, rate=.06,
                   volatility=.2, maturity=1, steps=50, paths=paths, train_paths=paths,
                   seed=seed, sampling=sampling, valuation="in-sample", reference_steps=50)[0]
    rows = [None] * len(keys)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(estimate, key): i for i, key in enumerate(keys)}
        for count, future in enumerate(as_completed(pending), 1):
            rows[pending[future]] = future.result()
            if count % 15 == 0 or count == len(keys): print(f"Completed {count}/{len(keys)} missing full-budget runs", flush=True)
    path = output_dir / "new_plain_runs.csv"
    write_rows(path, rows)
    manifest = {"profile": "full_missing_only", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "requested_keys": keys, "runs": len(keys), "paths_reduced": False,
                "protocol": "S=K=40,r=.06,sigma=.2,T=1,50 dates plus t=0; weighted Laguerre; in-sample",
                "reference_note": "C++ local reference is 50-step; statistical comparisons use the audited saved 100000-step matched-date value.",
                "build": metadata, "binary_sha256": sha256(binary), "csv_sha256": sha256(path),
                "workers": workers}
    (output_dir / "new_experiments_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def verify_supplement(path):
    manifest = json.loads((path.parent / "new_experiments_manifest.json").read_text())
    if sha256(path) != manifest["csv_sha256"]: raise ValueError("Supplemental run checksum mismatch.")
    for name in ("src/lsm_american.cpp", "include/model.hpp"):
        if sha256(ROOT / name) != manifest["build"]["source_sha256"][name]:
            raise ValueError("Supplemental LSM numerical source changed.")
    data = pd.read_csv(path)
    if not ((data.sampling_method == "plain") & (data.valuation_mode == "in-sample") &
            (data.basis == "weighted_laguerre_4") & (data.training_paths == data.paths)).all():
        raise ValueError("Supplemental convergence protocol mismatch.")


def build_tables(frames, runs, reference):
    published = frames["published_table1"][KEYS + ["paper_lsm_american", "paper_standard_error"]].rename(
        columns={"paper_lsm_american": "published_price", "paper_standard_error": "published_se"})
    trees = frames["paper_table"][KEYS + ["american_grid_reference_crr"]].rename(
        columns={"american_grid_reference_crr": "matched_crr"})
    errors = frames["control_table"].merge(published, on=KEYS, validate="many_to_one").merge(trees, on=KEYS, validate="many_to_one")
    if len(errors) != 40 or not np.allclose(errors.comparison_reference_crr, errors.matched_crr, atol=1e-12, rtol=0):
        raise ValueError("Controlled-study reference mapping mismatch.")
    errors["error_published"] = errors.price - errors.published_price
    errors["error_crr"] = errors.price - errors.matched_crr
    errors["z_published"] = [standardized_difference(p, ref, se, ref_se) for p,ref,se,ref_se in
                              zip(errors.price, errors.published_price, errors.standard_error, errors.published_se)]
    errors["z_crr"] = [standardized_difference(p, ref, se) for p,ref,se in
                        zip(errors.price, errors.matched_crr, errors.standard_error)]
    summary = []
    for (sampling, paths), group in runs.groupby(["sampling_method", "paths"], sort=True):
        row = summarize_prices(group.price.tolist(), group.standard_error.tolist(), reference)
        row.update(sampling_method=sampling, paths=int(paths), mean_pathwise_se=group.pathwise_se.mean())
        summary.append(row)
    convergence = pd.DataFrame(summary)
    reduction = []
    for paths in PATH_COUNTS:
        selected = convergence[convergence.paths == paths].set_index("sampling_method")
        var_plain, var_av = selected.loc["plain", "empirical_sd"] ** 2, selected.loc["antithetic", "empirical_sd"] ** 2
        reduction.append({"paths": paths, "runs_per_method": len(SEEDS), "variance_factor_plain_over_av": var_plain / var_av,
                          "variance_reduction_percent": 100 * (1 - var_av / var_plain)})
    efficiency = []
    controlled = frames["control_runs"]
    plain = controlled[controlled.sampling_method == "plain"]
    baseline_variance = plain.price.var(ddof=1)
    baseline_time = plain.runtime_ms.median() / 1000
    for sampling, group in controlled.groupby("sampling_method"):
        variance = group.price.var(ddof=1)
        duration = group.runtime_ms.median() / 1000
        efficiency.append({"sampling_method": sampling, "paths": 100000, "runs": len(group),
                           "empirical_variance": variance, "median_runtime_seconds": duration,
                           "runtime_q1_seconds": group.runtime_ms.quantile(.25) / 1000,
                           "runtime_q3_seconds": group.runtime_ms.quantile(.75) / 1000,
                           "variance_time_product": variance * duration,
                           "relative_efficiency": relative_efficiency(baseline_variance, baseline_time, variance, duration)})
    data = frames["table2"]
    columns = KEYS + ["seed", "price", "standard_error"]
    inside = data.loc[data.valuation_mode == "in-sample", columns]
    outside = data.loc[data.valuation_mode == "out-of-sample", columns]
    pairs = inside.merge(outside, on=KEYS + ["seed"], suffixes=("_in", "_out"), validate="one_to_one").merge(trees, on=KEYS)
    pairs["difference_in_minus_out"] = pairs.price_in - pairs.price_out
    pairs["error_in_crr"] = pairs.price_in - pairs.matched_crr
    pairs["error_out_crr"] = pairs.price_out - pairs.matched_crr
    table2 = []
    for key, group in pairs.groupby(KEYS, sort=True):
        row = dict(zip(KEYS, key))
        row.update(paired_difference_summary(group.price_in.tolist(), group.price_out.tolist()))
        row.update(runs=len(group), mean_error_in=group.error_in_crr.mean(), mean_error_out=group.error_out_crr.mean(),
                   sd_in=group.price_in.std(ddof=1), sd_out=group.price_out.std(ddof=1))
        table2.append(row)
    return {"pricing_errors": errors, "convergence_runs": runs, "convergence_summary": convergence,
            "variance_reduction": pd.DataFrame(reduction), "efficiency": pd.DataFrame(efficiency),
            "table2_pairs": pairs, "table2_summary": pd.DataFrame(table2)}


def error_plots(errors, output):
    cases = errors[KEYS].drop_duplicates().sort_values(KEYS)
    labels = [f"{int(r.spot)}/{int(r.volatility*100)}/{int(r.maturity)}" for r in cases.itertuples()]
    for standardized in (False, True):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
        fig.suptitle("Standardized pricing differences" if standardized else "Signed pricing errors: estimate minus reference",
                     fontsize=15, weight="bold")
        for i, target in enumerate(("published", "crr")):
            axis = axes[i]
            if standardized:
                axis.axhspan(-1.96, 1.96, color="#e7edf1")
                axis.axhline(-1.96, ls="--", color="#a54f57", lw=1)
                axis.axhline(1.96, ls="--", color="#a54f57", lw=1, label="±1.96 reference thresholds")
            axis.axhline(0, color="#526273", lw=1)
            for method, offset, marker in (("plain", -.10, "s"), ("antithetic", .10, "o")):
                frame = cases.merge(errors[errors.sampling_method == method], on=KEYS)
                axis.scatter(np.arange(len(frame)) + offset, frame[("z_" if standardized else "error_") + target],
                             color=COLORS[method], marker=marker, s=27, label=LABELS[method])
            axis.set(title="Against published LSM prices" if target == "published" else "Against 100000-step American CRR, matched exercise dates",
                     ylabel="Standardized difference" if standardized else "Price units")
            axis.grid(axis="y", alpha=.2)
            axis.legend(fontsize=8, ncol=3)
        axes[1].set_xticks(range(len(labels)), labels, rotation=65, fontsize=8)
        axes[1].set_xlabel("Initial stock price / volatility (%) / maturity (years)")
        if standardized:
            fig.supxlabel("Published: difference / combined SE. CRR: difference / reported SE. Descriptive thresholds; in-sample SEs are approximate.", fontsize=9)
        save(fig, output, "standardized_differences" if standardized else "signed_pricing_errors")


def convergence_plots(summary, output, reference):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    fig.suptitle("Convergence across path counts: 30 full LSM runs per point", fontsize=15, weight="bold")
    for method in LABELS:
        frame = summary[summary.sampling_method == method].sort_values("paths")
        axes[0].errorbar(frame.paths, frame.mean_price, yerr=1.96 * frame.se_of_mean, fmt="o-",
                         color=COLORS[method], capsize=3, label=LABELS[method])
        axes[1].loglog(frame.paths, frame.empirical_sd, "o-", color=COLORS[method], label=LABELS[method])
        axes[2].loglog(frame.paths, frame.rmse, "o-", color=COLORS[method], label=LABELS[method])
    axes[0].axhline(reference, ls="--", color="#263b57", label="Matched-date CRR")
    axes[0].set(xscale="log", title="Mean price ±1.96 SE of the mean", ylabel="Price")
    av = summary[summary.sampling_method == "antithetic"].sort_values("paths")
    axes[1].plot(av.paths, av.empirical_sd.iloc[0] * np.sqrt(av.paths.iloc[0] / av.paths), "--", color="#68717a", label=r"$N^{-1/2}$ guide")
    axes[1].set(title="Empirical SD of a single run", ylabel="Price units")
    axes[2].set(title="RMSE against matched-date CRR", ylabel="Price units")
    for axis in axes:
        axis.set_xlabel("Total simulated paths"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    save(fig, output, "price_sd_rmse_convergence")


def calibration_plot(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    fig.suptitle("Standard-error calibration against variation across 30 full fits", fontsize=14, weight="bold")
    for method in LABELS:
        frame = summary[summary.sampling_method == method].sort_values("paths")
        axes[0].plot(frame.paths, frame.reported_se_over_empirical_sd, "o-", color=COLORS[method], label=LABELS[method])
    axes[0].axhline(1, color="#263b57", ls="--")
    axes[0].set(xscale="log", xlabel="Total simulated paths", ylabel="Mean reported SE / empirical SD", title="Agreement would give a ratio of one")
    for i, method in enumerate(LABELS):
        row = summary[(summary.sampling_method == method) & (summary.paths == 100000)].iloc[0]
        for j, (field, color, label) in enumerate((("empirical_sd", "#263b57", "Empirical SD"),
                ("mean_reported_se", "#00857d", "Mean reported SE"), ("mean_pathwise_se", "#bd6026", "Mean pathwise SE"))):
            axes[1].bar(i + (j-1)*.23, row[field], width=.21, color=color, label=label if i == 0 else None)
    axes[1].set(xticks=[0,1], xticklabels=list(LABELS.values()), ylabel="Price units", title="100000 paths; AV reported SE uses pairs")
    axes[1].set_ylim(0, summary[summary.paths == 100000][["empirical_sd","mean_pathwise_se"]].to_numpy().max()*1.4)
    for axis in axes: axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
    save(fig, output, "standard_error_calibration")


def efficiency_plot(reduction, efficiency, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    fig.suptitle("Variance reduction and measured computational efficiency", fontsize=14, weight="bold")
    axes[0].plot(reduction.paths, reduction.variance_reduction_percent, "o-", color=COLORS["antithetic"])
    axes[0].axhline(0, color="#263b57", ls="--")
    axes[0].set(xscale="log", xlabel="Total simulated paths", ylabel="Variance reduction (%)", title="AV relative to plain LSM, equal path budgets")
    data = efficiency.set_index("sampling_method").loc[list(LABELS)]
    axes[1].bar(range(2), data.relative_efficiency, color=[COLORS[m] for m in LABELS])
    axes[1].axhline(1, color="#263b57", ls="--")
    for i, (_, row) in enumerate(data.iterrows()):
        axes[1].text(i, row.relative_efficiency+.04, f"{row.relative_efficiency:.2f}×\nmedian {1000*row.median_runtime_seconds:.0f} ms", ha="center", fontsize=9)
    axes[1].set(xticks=[0,1], xticklabels=list(LABELS.values()), ylabel="Relative efficiency (plain = 1)",
                title="100000 paths; same-build saved timings", ylim=(0, data.relative_efficiency.max()*1.3))
    for axis in axes: axis.grid(axis="y", alpha=.2)
    fig.supxlabel("Efficiency = 1 / (empirical variance × median runtime). Timings exclude CRR; machine/load dependent.", fontsize=9)
    save(fig, output, "variance_and_efficiency")


def table2_plot(pairs, summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
    fig.suptitle("In-sample and independent valuation of the same fitted AV-LSM policy", fontsize=14, weight="bold")
    labels = []
    for i, (key, group) in enumerate(pairs.groupby(KEYS, sort=True)):
        labels.append(f"{int(key[0])}/{int(100*key[1])}/{int(key[2])}")
        row = summary[(summary.spot == key[0]) & (summary.volatility == key[1]) & (summary.maturity == key[2])].iloc[0]
        axes[0].scatter(i+np.linspace(-.10,.10,len(group)), group.difference_in_minus_out, color="#00857d", alpha=.6,
                        label="Individual differences" if i == 0 else None)
        axes[0].errorbar(i, row.mean_difference, yerr=row.difference_sd, fmt="D", capsize=3, color="#263b57",
                         label="Mean ± SD across five runs" if i == 0 else None)
        for offset, mode, color in ((-.12,"in","#bd6026"),(.12,"out","#00857d")):
            axes[1].errorbar(i+offset, row["mean_error_"+mode], yerr=row["sd_"+mode], fmt="o", capsize=3, color=color,
                            label=("In-sample" if mode == "in" else "Out-of-sample") if i == 0 else None)
    axes[0].set(title="Paired in-sample minus out-of-sample differences", ylabel="Price difference")
    axes[1].set(title="Mean pricing error against matched-date CRR", ylabel="Mean error ± SD across five runs")
    for axis in axes:
        axis.axhline(0, color="#526273", ls="--"); axis.set_xticks(range(8), labels, rotation=45)
        axis.set_xlabel("Stock price / volatility (%) / maturity (years)"); axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
    fig.supxlabel("Each sample has 100000 paths. Whiskers show SD across five independent fits, not confidence intervals.", fontsize=9)
    save(fig, output, "in_sample_out_of_sample")


def analyze(output_dir=None, complete=False, build_dir=ROOT / "build", workers=2, audit_only=False):
    frames, audit = audit_existing_results()
    supplement = DEFAULT_ANALYSIS / "new_plain_runs.csv"
    if supplement.exists(): verify_supplement(supplement)
    runs = existing_convergence_runs(frames, audit["matched_crr_atm"], supplement if supplement.exists() else None)
    missing = missing_convergence_keys(runs)
    audit["missing_runs"] = len(missing)
    if audit_only:
        print(json.dumps(audit, indent=2)); return
    if missing and not complete:
        raise ValueError(f"Missing {len(missing)} convergence runs. Use --complete-missing; full budgets and 30 seeds are retained.")
    output = prepare_results("lsm_analysis", output_dir, False)
    if missing:
        supplement = complete_missing(missing, build_dir, output, workers)
        verify_supplement(supplement)
        runs = existing_convergence_runs(frames, audit["matched_crr_atm"], supplement)
    if missing_convergence_keys(runs): raise ValueError("Convergence data remain incomplete.")
    tables = build_tables(frames, runs, audit["matched_crr_atm"])
    for name, frame in tables.items(): write_rows(output / f"{name}.csv", frame.to_dict("records"))
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    summary = {"new_experiment_runs": len(missing), "reused_convergence_runs": 300-len(missing),
               "total_convergence_runs": len(runs),
               "standardized_threshold_counts": {method: {"published_outside_1_96": int((group.z_published.abs()>1.96).sum()),
                                                          "crr_outside_1_96": int((group.z_crr.abs()>1.96).sum())}
                   for method, group in tables["pricing_errors"].groupby("sampling_method")},
               "convergence": tables["convergence_summary"].to_dict("records"),
               "variance_reduction": tables["variance_reduction"].to_dict("records"),
               "efficiency": tables["efficiency"].to_dict("records"),
               "table2_mean_difference": tables["table2_pairs"].difference_in_minus_out.mean(),
               "table2_max_absolute_difference": tables["table2_pairs"].difference_in_minus_out.abs().max()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    inputs = dict(audit["inputs_sha256"])
    if supplement.exists(): inputs[supplement.relative_to(ROOT).as_posix()] = sha256(supplement)
    manifest = {"profile":"full", "generated_at_utc":datetime.now(timezone.utc).isoformat(),
                "inputs_sha256":inputs, "source_audit":audit["validation"], "path_counts":list(PATH_COUNTS),
                "runs_per_method_per_budget":30, "new_runs":len(missing), "preserved_budgets":True,
                "analysis_source_sha256":{name:sha256(ROOT/name) for name in ("scripts/analyze_lsm.py","scripts/lsm_statistics.py")},
                "limitations":["In-sample SEs are approximate.", "Published SE convention is unreported; standardized thresholds are descriptive.",
                    "CRR has discretization error; RMSE includes policy bias and sampling error.",
                    "Thirty-run SDs and variance ratios are uncertain.", "Efficiency is machine/load dependent and uses only same-build 100000-path timings.",
                    "Table 2 has five seeds per contract and paired policy dependence; its SD whiskers are not confidence intervals."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n",encoding="utf-8")
    _, figures = plot_locations("lsm_analysis", output)
    plt.rcParams.update({"font.size":10,"svg.hashsalt":"lsm-statistical-review"})
    error_plots(tables["pricing_errors"], figures)
    convergence_plots(tables["convergence_summary"], figures, audit["matched_crr_atm"])
    calibration_plot(tables["convergence_summary"], figures)
    efficiency_plot(tables["variance_reduction"], tables["efficiency"], figures)
    table2_plot(tables["table2_pairs"], tables["table2_summary"], figures)
    print(f"Reused {300-len(missing)} convergence runs; executed {len(missing)} missing runs. Saved six figures to {figures}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--complete-missing",action="store_true")
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--build-dir",type=Path,default=ROOT/"build")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    try:
        if args.workers < 1: raise ValueError("Need at least one worker.")
        analyze(args.output_dir,args.complete_missing,args.build_dir.resolve(),args.workers,args.audit_only)
    except (ValueError,RuntimeError,OSError) as error:
        parser.exit(1,str(error)+"\n")
