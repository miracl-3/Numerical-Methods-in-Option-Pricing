"""Controlled plain/AV LSM comparison and the paper's frozen-policy diagnostic.

Writes new datasets only. Reuses the saved fine CRR values, preserving their
provenance; C++ produces all estimates and per-date regression diagnostics.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from tempfile import TemporaryDirectory

from build import ROOT, executable
from run_experiments import run, write_rows
from run_lsm_paper import paper_cases, STRIKE, RATE, EXERCISES_PER_YEAR, PAPER_URL
from experiment_files import prepare_results
from crr_cache import get_reference


def case_key(row):
    return tuple(float(row[name]) for name in ("spot", "volatility", "maturity"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def execute_job(build_dir, job, diagnostic_path, references):
    case = job["case"]
    steps = int(EXERCISES_PER_YEAR * case["maturity"])
    rows = run("lsm_american", build_dir, type="put", spot=case["spot"], strike=STRIKE,
               rate=RATE, volatility=case["volatility"], maturity=case["maturity"],
               steps=steps, paths=job["paths"], train_paths=job["paths"], seed=job["seed"],
               sampling=job["sampling"], valuation=job["valuation"], reference_steps=steps,
               regression_diagnostics=diagnostic_path)
    reference = references[case_key(case)]
    for row in rows:
        row.update({"experiment": job["experiment"], "run_id": job["run_id"],
                    "comparison_reference_steps": reference["reference_steps"],
                    "comparison_reference_crr": reference["american_grid_reference_crr"],
                    "comparison_every_date_crr": reference["american_reference_crr"],
                    "reference_cache_key": reference["cache_key"],
                    "reference_cache_hit": int(reference["cache_hit"]),
                    "paper_lsm_price": case["paper_lsm_american"],
                    "paper_lsm_se": case["paper_standard_error"],
                    "paper_fd_price": case["paper_fd_american"]})
    diagnostics = read_rows(diagnostic_path)
    for row in diagnostics:
        row.update({"experiment": job["experiment"], "run_id": job["run_id"],
                    "sampling_method": job["sampling"], "seed": job["seed"],
                    "spot": case["spot"], "volatility": case["volatility"], "maturity": case["maturity"]})
    return rows, diagnostics


def summarize(table, replicates, table2):
    table_summary, calibration = {}, {}
    for sampling in ("plain", "antithetic"):
        cells = [row for row in table if row["sampling_method"] == sampling]
        errors = [float(row["price"]) - float(row["paper_lsm_price"]) for row in cells]
        table_summary[sampling] = {
            "cases": len(cells), "rmse_vs_published_lsm": math.sqrt(statistics.mean(x*x for x in errors)),
            "max_absolute_difference": max(map(abs, errors)),
            "mean_difference_vs_matched_crr": statistics.mean(
                float(row["price"]) - float(row["comparison_reference_crr"]) for row in cells),
        }
        selected = [row for row in replicates if row["sampling_method"] == sampling]
        prices = [float(row["price"]) for row in selected]
        mean_se = statistics.mean(float(row["standard_error"]) for row in selected)
        calibration[sampling] = {
            "replicates": len(prices), "mean_price": statistics.mean(prices),
            "replicate_sd": statistics.stdev(prices), "mean_plugin_se": mean_se,
            "replicate_sd_to_mean_plugin_se": statistics.stdev(prices) / mean_se,
        }
    by_sampling = {sampling: {int(row["seed"]): float(row["price"]) for row in replicates
                              if row["sampling_method"] == sampling} for sampling in calibration}
    price_differences = [by_sampling["plain"][seed] - by_sampling["antithetic"][seed]
                         for seed in sorted(by_sampling["plain"])]
    grouped = {}
    for row in table2:
        grouped.setdefault((case_key(row), row["seed"]), {})[row["valuation_mode"]] = row
    differences = [float(pair["in-sample"]["price"]) - float(pair["out-of-sample"]["price"])
                   for pair in grouped.values()]
    return {
        "table1": table_summary, "atm_replicate_calibration": calibration,
        "atm_empirical_variance_factor_plain_over_av": (calibration["plain"]["replicate_sd"] /
                                                         calibration["antithetic"]["replicate_sd"]) ** 2,
        "atm_mean_price_difference_plain_minus_av": statistics.mean(price_differences),
        "atm_paired_mean_difference_se": statistics.stdev(price_differences) / math.sqrt(len(price_differences)),
        "table2": {"fitted_policies": len(grouped), "mean_in_minus_out": statistics.mean(differences),
                   "max_absolute_in_minus_out": max(map(abs, differences)),
                   "positive_differences": sum(d > 0 for d in differences),
                   "negative_differences": sum(d < 0 for d in differences)},
        "limitations": [
            "In-sample plug-in SEs are approximate: the fitted policy couples cashflows.",
            "Out-of-sample SEs condition on one frozen fitted policy.",
            "Empirical variance factors from 30 replicates are noisy and contract-specific.",
            "Shared master seeds partially couple plain/AV runs; the mean contrast uses paired differences across seeds.",
            "Table 2 in/out values share fitted coefficients, so their SEs cannot simply be added in quadrature for an exact difference test.",
            "Original RNG, seeds and SE formula are unreported; printed-seed rows cannot be matched exactly.",
        ],
    }


def collect(build_dir, output_dir=None, quick=False, replicates=None, workers=2, reference_steps=None):
    paths = 2000 if quick else 100000
    repetitions = replicates if replicates is not None else (3 if quick else 30)
    if repetitions < 2 or workers < 1: raise ValueError("Need at least two replicates and one worker.")
    output_dir = prepare_results("lsm_comparison", output_dir, quick)
    reference_steps = reference_steps if reference_steps is not None else (2000 if quick else 100000)
    cases = paper_cases()
    def reference_for(case):
        contract = {name: case[name] for name in ("spot", "volatility", "maturity")}
        contract.update(strike=STRIKE, rate=RATE, type="put")
        return get_reference(build_dir, contract, reference_steps, int(EXERCISES_PER_YEAR * case["maturity"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cached = list(pool.map(reference_for, cases))
    references = {case_key(case): ref for case, ref in zip(cases, cached)}
    jobs = []
    def add(experiment, case, sampling, seed, valuation="in-sample"):
        jobs.append({"experiment": experiment, "case": case, "sampling": sampling,
                     "seed": seed, "paths": paths, "valuation": valuation, "run_id": len(jobs)})
    for case in cases:
        for sampling in ("plain", "antithetic"): add("table1", case, sampling, 42)
    atm = next(c for c in cases if case_key(c) == (40.0, 0.2, 1.0))
    for seed in range(42, 42 + repetitions):
        for sampling in ("plain", "antithetic"): add("replicates", atm, sampling, seed)
    for case in cases:
        if case["spot"] not in (36, 44): continue
        for seed in range(42, 44 if quick else 47): add("table2", case, "antithetic", seed, "both")
    outputs, diagnostics = [None] * len(jobs), [None] * len(jobs)
    with TemporaryDirectory(prefix="lsm-diagnostics-") as temp:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(execute_job, build_dir, job, Path(temp) / f"{job['run_id']}.csv", references): i
                       for i, job in enumerate(jobs)}
            for count, future in enumerate(as_completed(pending), 1):
                i = pending[future]
                outputs[i], diagnostics[i] = future.result()
                if count % 10 == 0 or count == len(jobs): print(f"Completed {count}/{len(jobs)} fits", flush=True)
    all_rows = [row for batch in outputs for row in batch]
    results = {name: [row for row in all_rows if row["experiment"] == name]
               for name in ("table1", "replicates", "table2")}
    for name, rows in results.items(): write_rows(output_dir / f"{name}.csv", rows)
    regressions = [row for batch in diagnostics for row in batch]
    write_rows(output_dir / "regressions.csv", regressions)
    summary = summarize(results["table1"], results["replicates"], results["table2"])
    summary["regressions"] = {
        "dates": len(regressions), "rank_deficient_or_constant_dates": sum(
            row["status"] in {"rank_deficient", "constant_fallback"} for row in regressions),
        "no_itm_dates": sum(row["status"] == "no_itm" for row in regressions),
        "max_condition_inf_retained_R": max(float(row["condition_inf_retained_R"]) for row in regressions),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    binary = executable("lsm_american", build_dir)
    sources = list((ROOT / "src").glob("*.cpp")) + list((ROOT / "include").glob("*.hpp"))
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "paper_url": PAPER_URL,
        "profile": "smoke_only" if quick else "full", "paths_per_sample": paths,
        "sampling_methods": ["plain", "antithetic"], "basis": "constant + three weighted Laguerre functions of S/K",
        "valuation": "in-sample comparison; Table 2 additionally freezes each fitted policy for independent evaluation",
        "exercise_grid": "t=0 and 50 equally spaced dates/year including maturity, shared with CRR",
        "atm_replicate_seeds": list(range(42, 42 + repetitions)),
        "table2_seeds": list(range(42, 44 if quick else 47)), "workers": workers,
        "crr_cache": {"directory": ".cache/crr", "entries": cached,
                      "invalidation": "contract, tree/exercise grids, and CRR driver/shared-header SHA256"},
        "reference_note": "Validated C++ cache entries; raw benchmark/reference_steps retain the cheap local tree, comparison_reference_* identifies the cached fine tree.",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limitations": summary["limitations"],
    }
    build_metadata = build_dir / "build_metadata.json"
    if build_metadata.exists(): manifest["build"] = json.loads(build_metadata.read_text(encoding="utf-8"))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Wrote comparison summary and provenance.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, help="Fresh result directory; smoke defaults to tmp/.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reference-steps", type=int, help="Default 100000 (2000 for smoke).")
    args = parser.parse_args()
    try:
        collect(args.build_dir.resolve(), args.output_dir, args.quick, args.replicates, args.workers, args.reference_steps)
    except (ValueError, RuntimeError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, (getattr(error, "stderr", "") or str(error)) + "\n")
