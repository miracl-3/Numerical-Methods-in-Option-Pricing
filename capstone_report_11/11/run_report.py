"""Run the native LSM paper table, convergence study, and boundary experiment."""

import argparse
import csv
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPOTS = (36, 38, 40, 42, 44)
VOLATILITIES = (0.20, 0.40)
MATURITIES = (1, 2)
PATH_COUNTS = (5000, 10000, 25000, 50000, 100000)
SEEDS = tuple(range(42, 72))
CRR_STEPS = 2000
CONFIDENCE_T_95_DF29 = 2.045229642132703
PUBLISHED_TABLE1 = {
    (36, 0.20, 1): (4.478, 4.472, 0.010),
    (36, 0.20, 2): (4.840, 4.821, 0.012),
    (36, 0.40, 1): (7.101, 7.091, 0.020),
    (36, 0.40, 2): (8.508, 8.488, 0.024),
    (38, 0.20, 1): (3.250, 3.244, 0.009),
    (38, 0.20, 2): (3.745, 3.735, 0.011),
    (38, 0.40, 1): (6.148, 6.139, 0.019),
    (38, 0.40, 2): (7.670, 7.669, 0.022),
    (40, 0.20, 1): (2.314, 2.313, 0.009),
    (40, 0.20, 2): (2.885, 2.879, 0.010),
    (40, 0.40, 1): (5.312, 5.308, 0.018),
    (40, 0.40, 2): (6.920, 6.921, 0.022),
    (42, 0.20, 1): (1.617, 1.617, 0.007),
    (42, 0.20, 2): (2.212, 2.206, 0.010),
    (42, 0.40, 1): (4.582, 4.588, 0.017),
    (42, 0.40, 2): (6.248, 6.243, 0.021),
    (44, 0.20, 1): (1.110, 1.118, 0.007),
    (44, 0.20, 2): (1.690, 1.675, 0.009),
    (44, 0.40, 1): (3.948, 3.957, 0.017),
    (44, 0.40, 2): (5.647, 5.622, 0.021),
}


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True)


def read_one_row(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one result row in {path}, got {len(rows)}.")
    return rows[0]


def run_case(executable, output_dir, *, spot, volatility, maturity, paths,
             sampling, seed, visualization=False):
    exercise_steps = 50 * maturity
    command = [
        str(executable), "--spot", str(spot), "--strike", "40", "--rate", "0.06",
        "--volatility", str(volatility), "--maturity", str(maturity),
        "--steps", str(exercise_steps), "--paths", str(paths),
        "--sampling", sampling, "--seed", str(seed),
        "--reference-steps", str(CRR_STEPS), "--output-dir", str(output_dir),
    ]
    if visualization:
        command.append("--visualization-data")
    run(command)
    return read_one_row(output_dir / "summary.csv")


def run_paper_table(executable, report_dir, temporary_dir, paths, seed):
    rows = []
    for maturity in MATURITIES:
        for volatility in VOLATILITIES:
            for spot in SPOTS:
                plain_dir = temporary_dir / f"paper_plain_{spot}_{volatility}_{maturity}"
                av_dir = temporary_dir / f"paper_av_{spot}_{volatility}_{maturity}"
                plain = run_case(
                    executable, plain_dir, spot=spot, volatility=volatility,
                    maturity=maturity, paths=paths, sampling="plain", seed=seed,
                )
                av = run_case(
                    executable, av_dir, spot=spot, volatility=volatility,
                    maturity=maturity, paths=paths, sampling="antithetic", seed=seed,
                )
                published_fd, published_lsm, published_se = PUBLISHED_TABLE1[
                    (spot, volatility, maturity)
                ]
                rows.append({
                    "spot": spot,
                    "strike": 40,
                    "rate": 0.06,
                    "volatility": volatility,
                    "maturity": maturity,
                    "exercise_dates_per_year": 50,
                    "steps": exercise_steps_for(maturity),
                    "paths": paths,
                    "seed": seed,
                    "crr_steps": CRR_STEPS,
                    "paper_fd_reference": published_fd,
                    "paper_lsm_estimate": published_lsm,
                    "paper_lsm_standard_error": published_se,
                    "plain_estimate": float(plain["estimate"]),
                    "plain_standard_error": float(plain["standard_error"]),
                    "plain_ci_lower": float(plain["ci_lower"]),
                    "plain_ci_upper": float(plain["ci_upper"]),
                    "av_estimate": float(av["estimate"]),
                    "av_standard_error": float(av["standard_error"]),
                    "av_ci_lower": float(av["ci_lower"]),
                    "av_ci_upper": float(av["ci_upper"]),
                    "crr_reference": float(plain["crr_reference"]),
                    "plain_absolute_error": float(plain["absolute_error"]),
                    "av_absolute_error": float(av["absolute_error"]),
                    "plain_exercise_fraction": float(plain["exercise_fraction"]),
                    "av_exercise_fraction": float(av["exercise_fraction"]),
                })
    table_path = report_dir / "paper_table.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False, float_format="%.10g")
    return rows


def exercise_steps_for(maturity):
    return 50 * maturity


def run_convergence(executable, report_dir, temporary_dir):
    observations = []
    for volatility in (0.20, 0.40):
        for sampling in ("plain", "antithetic"):
            for paths in PATH_COUNTS:
                for seed in SEEDS:
                    case_dir = temporary_dir / f"conv_{volatility}_{sampling}_{paths}_{seed}"
                    row = run_case(
                        executable, case_dir, spot=40, volatility=volatility,
                        maturity=1, paths=paths, sampling=sampling, seed=seed,
                    )
                    observations.append({
                        "spot": 40,
                        "strike": 40,
                        "rate": 0.06,
                        "volatility": volatility,
                        "maturity": 1,
                        "exercise_steps": 50,
                        "paths": paths,
                        "sampling": sampling,
                        "seed": seed,
                        "estimate": float(row["estimate"]),
                        "path_standard_error": float(row["standard_error"]),
                        "crr_reference": float(row["crr_reference"]),
                    })

    raw = pd.DataFrame(observations)
    raw.to_csv(report_dir / "convergence_replicates.csv", index=False,
               float_format="%.10g")
    summaries = []
    for (volatility, paths, sampling), frame in raw.groupby(
            ["volatility", "paths", "sampling"], sort=True):
        mean = frame.estimate.mean()
        between_seed_se = frame.estimate.std(ddof=1) / math.sqrt(len(frame))
        half_width = CONFIDENCE_T_95_DF29 * between_seed_se
        summaries.append({
            "spot": 40,
            "strike": 40,
            "rate": 0.06,
            "volatility": volatility,
            "maturity": 1,
            "exercise_steps": 50,
            "paths": paths,
            "sampling": sampling,
            "replications": len(frame),
            "mean_estimate": mean,
            "between_seed_standard_error": between_seed_se,
            "ci_lower": mean - half_width,
            "ci_upper": mean + half_width,
            "crr_reference": frame.crr_reference.iloc[0],
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(report_dir / "convergence_summary.csv", index=False,
                   float_format="%.10g")
    return raw, summary


def run_boundary_sample(executable, report_dir, temporary_dir, paths):
    frames = []
    for sampling in ("plain", "antithetic"):
        output = temporary_dir / f"boundary_{sampling}"
        run_case(
            executable, output, spot=40, volatility=0.20, maturity=1,
            paths=paths, sampling=sampling, seed=42, visualization=True,
        )
        boundary = pd.read_csv(output / "exercise_boundary.csv")
        boundary["method"] = "Longstaff--Schwartz"
        frames.append(boundary)
        paths_frame = pd.read_csv(output / "sample_paths.csv")
        paths_frame["sampling"] = sampling
        path_dest = report_dir / f"sample_paths_{sampling}.csv"
        paths_frame.to_csv(path_dest, index=False, float_format="%.10g")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(report_dir / "exercise_boundary.csv", index=False,
                    float_format="%.10g")
    return combined


def plot_confidence_intervals(summary, report_dir):
    colors = {"plain": "#2468a2", "antithetic": "#e47721"}
    markers = {"plain": "o", "antithetic": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False,
                             layout="constrained")
    for axis, volatility in zip(axes, (0.20, 0.40)):
        subset = summary[summary.volatility == volatility]
        reference = float(subset.crr_reference.iloc[0])
        axis.axhline(reference, color="#333333", linestyle=":", linewidth=1.5,
                     label=f"{CRR_STEPS:,}-step Bermudan CRR")
        for sampling in ("plain", "antithetic"):
            frame = subset[subset.sampling == sampling].sort_values("paths")
            centers = frame.mean_estimate.to_numpy()
            lower = centers - frame.ci_lower.to_numpy()
            upper = frame.ci_upper.to_numpy() - centers
            axis.errorbar(
                frame.paths, centers, yerr=np.vstack([lower, upper]),
                color=colors[sampling],
                marker=markers[sampling], linewidth=1.7, capsize=3,
                label=f"{sampling} (30-seed 95% $t$ interval)",
            )
        axis.set_xscale("log")
        axis.set_xticks(PATH_COUNTS)
        axis.set_xticklabels([f"{count // 1000}k" for count in PATH_COUNTS])
        axis.set_xlabel("Simulated paths per run")
        axis.set_title(fr"ATM put, $\sigma={volatility * 100:.0f}\%$")
        axis.set_ylabel("Mean LSM policy estimate")
        axis.grid(True, which="major", alpha=0.25, linestyle="--")
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("LSM: 95% confidence intervals across independent seeds")
    fig.savefig(report_dir / "lsm_confidence_intervals.png", dpi=180)
    fig.savefig(report_dir / "lsm_confidence_intervals.svg", metadata={"Date": None})
    plt.close(fig)


def plot_exercise_boundary(boundaries, report_dir, paths_per_run):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True,
                             layout="constrained")
    colors = {"plain": "#2474a6", "antithetic": "#e47721"}
    boundary_styles = {"plain": "-", "antithetic": "--"}
    for axis, sampling in zip(axes, ("plain", "antithetic")):
        path_data = pd.read_csv(report_dir / f"sample_paths_{sampling}.csv")
        exercise_marker_added = False
        for _, frame in path_data.groupby("path_id"):
            axis.plot(frame.time_years, frame.stock_price, color="#9aa5ae",
                      alpha=0.22, linewidth=0.65)
            if frame.exercised_early.iloc[0] == 1:
                stopping = frame[frame.time_years == frame.stopping_time_years]
                axis.scatter(stopping.time_years, stopping.stock_price, s=12,
                             color=colors[sampling], edgecolor="white",
                             linewidth=0.25, zorder=4,
                             label=("Early-exercise point"
                                    if not exercise_marker_added else None))
                exercise_marker_added = True
        frame = boundaries[boundaries.sampling == sampling].sort_values("time_years")
        axis.plot(frame.time_years, frame.boundary_stock,
                  color=colors[sampling], linestyle=boundary_styles[sampling],
                  linewidth=2.5, label="Estimated LSM boundary")
        axis.axhline(40, color="#303030", linestyle=":", linewidth=1.3,
                     label="Strike K=40")
        axis.set_xlim(0, 1)
        axis.set_ylim(25, 45)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_xlabel("Time (years)")
        axis.set_title(f"{sampling.capitalize()} MC")
        axis.grid(True, alpha=0.23, linestyle="--")
        axis.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Stock price")
    fig.suptitle(
        f"LSM exercise boundary with representative paths "
        f"({paths_per_run:,} paths; K=40, sigma=20%)"
    )
    fig.savefig(report_dir / "lsm_exercise_boundary.png", dpi=180)
    fig.savefig(report_dir / "lsm_exercise_boundary.svg", metadata={"Date": None})
    plt.close(fig)


def write_tex_tables(rows, report_dir):
    def fmt(value, digits=4):
        return f"{value:.{digits}f}"

    lines = [
        "% Generated by run_report.py from paper_table.csv.",
        "% Published LSM values transcribed from Longstaff-Schwartz (2001), Table 1.",
    ]
    for maturity in MATURITIES:
        rows_at_maturity = [row for row in rows if row["maturity"] == maturity]
        lines += [
            r"\begin{table}[!htbp]",
            r"    \centering\scriptsize",
            f"    \\caption{{Longstaff--Schwartz Table~1 comparison, \\(T={maturity}\\) year(s).}}",
            f"    \\label{{tab:lsm_paper_comparison_t{maturity}}}",
            r"    \setlength{\tabcolsep}{3pt}",
            r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rr|rrrrrr}",
            r"        \toprule",
            r"        \(S_0\) & \(\sigma\) & "
            r"\multicolumn{3}{c|}{Published comparison} & "
            r"\multicolumn{3}{c}{Our estimates and CRR errors} \\",
            r"        \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"        & & Paper LSM (SE) & AV LSM (SE) & AV $-$ paper & "
            r"CRR reference & Plain MC (SE) & \(|V_{\rm plain}-V_{\rm CRR}|\) \\",
            r"        \midrule",
        ]
        for row in rows_at_maturity:
            lines.append(
                f"        {row['spot']} & {row['volatility'] * 100:.0f}\\% & "
                f"{row['paper_lsm_estimate']:.3f} "
                f"({row['paper_lsm_standard_error']:.3f}) & "
                f"{fmt(row['av_estimate'])} ({fmt(row['av_standard_error'])}) & "
                f"{fmt(row['av_estimate'] - row['paper_lsm_estimate'], 4)} & "
                f"{fmt(row['crr_reference'])} & "
                f"{fmt(row['plain_estimate'])} ({fmt(row['plain_standard_error'])}) & "
                f"{fmt(row['plain_absolute_error'], 4)} \\\\"
            )
        lines += [
            r"        \bottomrule",
            r"    \end{tabular*}",
            r"    \begin{minipage}{0.96\linewidth}\footnotesize",
            r"    AV $-$ paper is our antithetic estimate minus the published LSM estimate. "
            r"Our CRR is a separate 2,000-step Bermudan reference, not the paper's finite-difference value.",
            r"    Plain MC is our added no-variance-reduction baseline; published prices and standard errors are rounded as in Table~1.",
            r"    \end{minipage}",
            r"\end{table}",
            "",
        ]
    (report_dir / "paper_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path,
                        default=Path(__file__).with_name("lsm_american.exe"))
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--table-paths", type=int, default=100000)
    parser.add_argument("--convergence-paths", type=int, nargs="+",
                        default=list(PATH_COUNTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--boundary-paths", type=int, default=100000)
    args = parser.parse_args()
    executable = args.executable.resolve()
    report_dir = args.report_dir.resolve()
    if not executable.is_file():
        parser.error(f"Native executable not found: {executable}")
    if args.table_paths % 2 or args.boundary_paths % 2:
        parser.error("Table and boundary path counts must be even.")
    if tuple(args.convergence_paths) != PATH_COUNTS:
        parser.error("The convergence plot currently expects 5k, 10k, 25k, 50k, 100k paths.")
    if tuple(args.seeds) != SEEDS:
        parser.error("The convergence plot currently expects seeds 42 through 71.")

    report_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lsm_report_", dir=report_dir) as temporary:
        temporary_dir = Path(temporary)
        table_rows = run_paper_table(
            executable, report_dir, temporary_dir, args.table_paths, seed=42
        )
        write_tex_tables(table_rows, report_dir)
        _, convergence = run_convergence(executable, report_dir, temporary_dir)
        boundaries = run_boundary_sample(
            executable, report_dir, temporary_dir, args.boundary_paths
        )
    plot_confidence_intervals(convergence, report_dir)
    plot_exercise_boundary(boundaries, report_dir, args.boundary_paths)
    print(f"Completed 20 paper-setting contract cases, plain and AV.")
    print(f"Generated report outputs in {report_dir}")


if __name__ == "__main__":
    main()
