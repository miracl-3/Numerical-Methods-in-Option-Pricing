"""Run original LSM Table 1 in C++; collect published-price and SE comparisons.

Python schedules processes, joins the printed paper reference, and summarizes
results. C++ computes paths, regressions, payoffs, standard errors, and trees.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

from build import ROOT, executable
from experiment_files import REFERENCE_DIR, prepare_results
from crr_cache import get_reference

PAPER_FILE = REFERENCE_DIR / "table1.csv"
PAPER_URL = ("https://www.kellogg.northwestern.edu/faculty/mcdonald/ftp/"
             "practicum/papers/longstaffschwartz.pdf")
STRIKE, RATE, EXERCISES_PER_YEAR = 40.0, 0.06, 50


def paper_cases():
    with PAPER_FILE.open(newline="", encoding="utf-8") as source:
        return [{name: float(value) for name, value in row.items()}
                for row in csv.DictReader(source)]


def run(build_dir, case, paths, seed, reference_steps):
    command = [
        str(executable("lsm_american", build_dir)),
        "--lsm-mode", "paper", "--type", "put",
        "--spot", str(case["spot"]), "--strike", str(STRIKE), "--rate", str(RATE),
        "--volatility", str(case["volatility"]), "--maturity", str(case["maturity"]),
        "--steps", str(int(EXERCISES_PER_YEAR * case["maturity"])),
        "--paths", str(paths), "--seed", str(seed),
        "--reference-steps", str(reference_steps),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    raw = next(csv.DictReader(io.StringIO(result.stdout)))
    text_fields = {"method", "option_type", "newton_status", "sampling_method",
                   "valuation_mode", "basis", "standard_error_scope"}
    return {name: (value if name in text_fields else int(value) if name.endswith("_seed")
                   else float(value)) for name, value in raw.items()}


def comparison(raw, case, reference=None):
    """Join C++ output with published values and optionally cached C++ trees."""
    reference = reference or raw
    pair_se, paper_se = raw["standard_error"], case["paper_standard_error"]
    difference = raw["price"] - case["paper_lsm_american"]
    combined_z = difference / math.hypot(pair_se, paper_se)
    row = {name: raw[name] for name in (
        "spot", "strike", "rate", "volatility", "maturity", "steps", "paths", "seed",
        "price", "standard_error", "ci_lower", "ci_upper", "naive_path_standard_error",
        "european_price", "european_standard_error", "european_naive_path_standard_error",
        "early_exercise_premium", "early_exercise_premium_standard_error",
        "early_exercise_premium_naive_path_standard_error", "early_exercise_premium_bs",
        "exercise_fraction", "independent_samples", "payoff_evaluations", "runtime_ms",
        "reference_runtime_ms")}
    row.update({name: raw[name] for name in (
        "sampling_method", "valuation_mode", "basis", "standard_error_scope",
        "training_paths", "evaluation_paths", "training_seed", "evaluation_seed",
        "antithetic_pairs", "training_antithetic_pairs", "unique_simulated_paths",
        "exercise_at_zero", "regression_rank_min", "regression_rank_max",
        "regression_condition_inf_max", "regression_fallback_dates", "regression_no_itm_dates")})
    row.update({
        "exercise_dates_per_year": EXERCISES_PER_YEAR,
        "reference_steps": reference["reference_steps"],
        "simulation_reference_steps": raw["reference_steps"],
        "american_grid_reference_crr": reference["benchmark"],
        "american_reference_crr": reference["american_reference_crr"],
        "analytic_european_price": raw["price"] - raw["early_exercise_premium_bs"],
        **{name: value for name, value in case.items() if name.startswith("paper_")},
        "difference_from_paper_lsm": difference,
        "combined_se_z_vs_paper": combined_z,
        "within_95_combined_se_of_paper": int(abs(combined_z) <= 1.959963984540054),
        "pair_se_to_paper_se": pair_se / paper_se,
        "naive_se_to_paper_se": raw["naive_path_standard_error"] / paper_se,
        "difference_from_american_grid_crr": raw["price"] - reference["benchmark"],
        "difference_from_american_crr": raw["price"] - reference["american_reference_crr"],
    })
    return row


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path.name} ({len(rows)} rows)", flush=True)


def parallel_runs(jobs, workers, label):
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run, *job): index for index, job in enumerate(jobs)}
        for count, future in enumerate(as_completed(pending), 1):
            index = pending[future]
            results[index] = future.result()
            if label == "Table 1" or count % 10 == 0 or count == len(jobs):
                print(f"{label}: {count}/{len(jobs)} complete", flush=True)
    return results


def standard_error_summary(convergence):
    result = []
    for paths in sorted({row["paths"] for row in convergence}):
        rows = [row for row in convergence if row["paths"] == paths]
        prices = [row["price"] for row in rows]
        empirical_sd = statistics.stdev(prices)
        pair_se = statistics.mean(row["standard_error"] for row in rows)
        naive_se = statistics.mean(row["naive_path_standard_error"] for row in rows)
        reference = rows[0]["american_grid_reference_crr"]
        result.append({
            "paths": int(paths), "replicates": len(rows),
            "mean_price": statistics.mean(prices), "replicate_price_sd": empirical_sd,
            "mean_pair_standard_error": pair_se, "mean_naive_path_standard_error": naive_se,
            "replicate_sd_to_pair_se": empirical_sd / pair_se,
            "replicate_sd_to_naive_se": empirical_sd / naive_se,
            "american_grid_reference_crr": reference,
            "mean_minus_american_grid_crr": statistics.mean(prices) - reference,
            "rmse_vs_american_grid_crr": math.sqrt(statistics.mean(
                (price - reference) ** 2 for price in prices)),
        })
    return result


def collect(build_dir, output_dir=None, quick=False, reference_steps=None, replicates=None, workers=2):
    build_dir = Path(build_dir).resolve()
    output_dir = prepare_results("paper_replication", output_dir, quick)
    binary = executable("lsm_american", build_dir)
    if not binary.is_file():
        raise RuntimeError(f"Missing {binary}; run python scripts/build.py first.")
    reference_steps = reference_steps if reference_steps is not None else (2000 if quick else 100000)
    replicates = replicates if replicates is not None else (2 if quick else 30)
    if reference_steps < 100 or reference_steps > 100000 or reference_steps % 100:
        raise ValueError("Reference steps must be a multiple of 100 in [100, 100000].")
    if replicates < 2 or workers < 1:
        raise ValueError("At least two replicates and one worker are required.")
    cases = paper_cases()
    def reference_for(case):
        contract = {name: case[name] for name in ("spot", "volatility", "maturity")}
        contract.update(strike=STRIKE, rate=RATE, type="put")
        return get_reference(build_dir, contract, reference_steps, int(EXERCISES_PER_YEAR * case["maturity"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        references = list(pool.map(reference_for, cases))
    matched_references = [{"reference_steps": ref["reference_steps"],
                           "benchmark": ref["american_grid_reference_crr"],
                           "american_reference_crr": ref["american_reference_crr"]} for ref in references]
    table_paths = 10000 if quick else 100000
    table_raw = parallel_runs([
        (build_dir, case, table_paths, 42, int(EXERCISES_PER_YEAR * case["maturity"])) for case in cases
    ], workers, "Table 1")
    table = [comparison(raw, case, reference) for raw, case, reference in zip(table_raw, cases, matched_references)]
    for row, ref in zip(table, references):
        row.update(reference_cache_key=ref["cache_key"], reference_cache_hit=int(ref["cache_hit"]))
    write_rows(output_dir / "lsm_paper_table.csv", table)

    atm_index = next(index for index, case in enumerate(cases)
                     if case["spot"] == 40 and case["volatility"] == 0.2 and case["maturity"] == 1)
    atm, atm_reference = cases[atm_index], matched_references[atm_index]
    path_counts = [1000, 2500, 5000] if quick else [5000, 10000, 25000, 50000, 100000]
    seeds = list(range(42, 42 + replicates))
    settings = [(paths, seed) for paths in path_counts for seed in seeds]
    # Reuse the same-contract C++ trees. Each convergence process calculates a
    # cheap local tree for its interface; the joined CSV uses the cached fine tree.
    convergence_raw = parallel_runs([
        (build_dir, atm, paths, seed, EXERCISES_PER_YEAR) for paths, seed in settings
    ], workers, "Convergence")
    convergence = [comparison(raw, atm, atm_reference) for raw in convergence_raw]
    write_rows(output_dir / "lsm_paper_convergence.csv", convergence)
    se_summary = standard_error_summary(convergence)
    write_rows(output_dir / "lsm_paper_se_summary.csv", se_summary)

    differences = [row["difference_from_paper_lsm"] for row in table]
    pair_ratios = [row["pair_se_to_paper_se"] for row in table]
    naive_ratios = [row["naive_se_to_paper_se"] for row in table]
    summary = {
        "profile": "smoke_only" if quick else "original_table1",
        "table_cases": len(table), "table_paths": table_paths, "reference_steps": reference_steps,
        "price_mean_difference_from_paper": statistics.mean(differences),
        "price_rmse_vs_paper": math.sqrt(statistics.mean(d*d for d in differences)),
        "price_max_absolute_difference_from_paper": max(map(abs, differences)),
        "cases_within_95_combined_se_of_paper": sum(row["within_95_combined_se_of_paper"] for row in table),
        "max_absolute_combined_se_z_vs_paper": max(abs(row["combined_se_z_vs_paper"]) for row in table),
        "pair_se_to_paper_se_range": [min(pair_ratios), max(pair_ratios)],
        "pair_se_to_paper_se_median": statistics.median(pair_ratios),
        "naive_se_to_paper_se_range": [min(naive_ratios), max(naive_ratios)],
        "naive_se_to_paper_se_median": statistics.median(naive_ratios),
        "naive_se_cases_within_20_percent_of_paper": sum(abs(r - 1) <= 0.2 for r in naive_ratios),
        "pair_se_cases_within_20_percent_of_paper": sum(abs(r - 1) <= 0.2 for r in pair_ratios),
        "atm_full_budget_calibration": se_summary[-1],
        "interpretation": [
            "Combined-SE comparisons are descriptive, not a multiple-testing-adjusted test.",
            "The paper does not specify its SE formula; any inferred convention is tentative.",
            "Naive path SE ignores antithetic covariance and is a diagnostic, not the reported estimator SE.",
            "Pair-based plug-in SE omits dependence induced by fitting on the same paths.",
            "Replicate SD repeats simulation and policy fitting; its estimate is itself noisy.",
            "American CRR references with matched exercise dates and every tree date have different exercise opportunities.",
        ],
    }
    (output_dir / "lsm_paper_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "Longstaff and Schwartz (2001), Table 1, journal pp. 126-127, normalization p. 143",
        "paper_url": PAPER_URL, "profile": summary["profile"],
        "platform": platform.platform(), "python": platform.python_version(),
        "contract_cases": {"strike": STRIKE, "rate": RATE, "spots": [36, 38, 40, 42, 44],
                           "volatilities": [0.2, 0.4], "maturities": [1, 2],
                           "exercise_dates_per_year": EXERCISES_PER_YEAR},
        "estimator": {"table_paths": table_paths, "table_seed": 42,
                      "sampling": "Z/-Z antithetic pairs",
                      "regressors": "constant + exp(-x/2) * (L0, L1, L2), x=S/K",
                      "cashflow_normalization": "divide payoffs by K for regression, restore for output",
                      "valuation": "in-sample realized stopped cashflows, time zero and positive exercise dates",
                      "standard_error": "SD of pair averages / sqrt(number of pairs)"},
        "reference": {"time_steps": reference_steps, "terminal_nodes": reference_steps + 1,
                      "total_lattice_nodes": (reference_steps + 1) * (reference_steps + 2) // 2,
                      "types": ["American CRR, every tree date", "American CRR, matched exercise dates (50 dates/year)"],
                      "paper_reference": "printed FD, 40,000 time steps/year and 1,000 stock steps",
                      "reuse": "Convergence CSV reuses table C++ trees; simulation_reference_steps records local tree."},
        "convergence": {"spot": 40, "volatility": 0.2, "maturity": 1,
                        "path_counts": path_counts, "seeds": seeds, "replicates": replicates},
        "workers": workers,
        "crr_cache": {"directory": ".cache/crr", "entries": references,
                      "invalidation": "contract, tree/exercise grids, and CRR driver/shared-header SHA256"},
        "paper_reference_sha256": hashlib.sha256(PAPER_FILE.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "limitations": summary["interpretation"] + [
            "Original RNG, seed and SE convention are unreported; bitwise reproduction is impossible.",
            "Published entries are rounded to 0.001; CRR is additional, not an FD reproduction.",
            "Reference wall times overlap with parallel workers; runtime_ms excludes references and I/O.",
            "Seeds reproduce draws within the same compiler and standard-library implementation."],
    }
    metadata = build_dir / "build_metadata.json"
    if metadata.is_file():
        manifest["build"] = json.loads(metadata.read_text(encoding="utf-8"))
    (output_dir / "lsm_paper_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary and manifest; {summary['cases_within_95_combined_se_of_paper']}/20 prices within combined 95% SE.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, help="Fresh result directory; smoke defaults to tmp/.")
    parser.add_argument("--quick", action="store_true", help="Smoke profile only; not published sample budget.")
    parser.add_argument("--reference-steps", type=int, help="Default 100000 (2000 with --quick).")
    parser.add_argument("--replicates", type=int, help="Convergence seeds, default 30 (2 with --quick).")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent C++ processes, default 2.")
    args = parser.parse_args()
    try:
        collect(args.build_dir, args.output_dir, args.quick, args.reference_steps, args.replicates, args.workers)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, (getattr(error, "stderr", "") or str(error)) + "\n")
