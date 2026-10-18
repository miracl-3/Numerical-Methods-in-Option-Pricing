"""Run C++ pricing experiments and collect their CSV output without reimplementing prices."""

import argparse
import csv
import io
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from statistics import median

from build import ROOT, METHODS, executable


def run(name, build_dir, **parameters):
    command = [str(executable(name, build_dir))]
    for key, value in parameters.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return list(csv.DictReader(io.StringIO(result.stdout)))


def write_rows(destination, rows):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {destination.relative_to(ROOT) if destination.is_relative_to(ROOT) else destination}"
          f" ({len(rows)} rows)", flush=True)


def summarize_variance(rows):
    largest = max(int(row["paths"]) for row in rows)
    summaries = []
    for kind in ["call", "put"]:
        selected = [row for row in rows if row["option_type"] == kind and int(row["paths"]) == largest]
        baseline = {row["seed"]: row for row in selected if row["method"] == "plain_mc"}
        for method in ["antithetic", "control_variate", "importance_sampling"]:
            values = [row for row in selected if row["method"] == method]
            variance_factors = [(float(baseline[row["seed"]]["standard_error"]) /
                                 float(row["standard_error"])) ** 2 for row in values]
            work_factors = [factor * int(baseline[row["seed"]]["payoff_evaluations"]) /
                            int(row["payoff_evaluations"]) for factor, row in zip(variance_factors, values)]
            reductions = [100 * (1 - 1 / factor) for factor in variance_factors]
            summaries.append({
                "option_type": kind, "method": method, "paths": largest, "seeds": len(values),
                "training_paths": int(values[0]["training_paths"]),
                "median_variance_reduction_pct": median(reductions),
                "min_variance_reduction_pct": min(reductions),
                "max_variance_reduction_pct": max(reductions),
                "median_variance_factor": median(variance_factors),
                "median_work_adjusted_factor": median(work_factors),
                "median_importance_shift": median(float(row["importance_shift"]) for row in values),
                "max_newton_iterations": max(int(row["newton_iterations"]) for row in values),
                "max_absolute_newton_gradient": max(abs(float(row["newton_gradient"])) for row in values),
            })
    return summaries


def collect(build_dir, output_dir, quick=False):
    build_dir, output_dir = Path(build_dir).resolve(), Path(output_dir).resolve()
    for method in METHODS:
        if not executable(method, build_dir).is_file():
            raise RuntimeError(f"Missing {method}; run python scripts/build.py first.")
    path_counts = [1000, 5000, 20000, 50000] if quick else [1000, 5000, 20000, 100000, 500000]
    seeds = [42, 43, 44] if quick else list(range(42, 50))
    references = {"reference_steps": 2000}
    trees, mc, variance, american, exercise_grid, newton = [], [], [], [], [], []
    for option_type in ["call", "put"]:
        for steps in [25, 50, 75, 100, 200, 400, 800]:
            for method in ["binomial_european", "binomial_american"]:
                trees.extend(run(method, build_dir, type=option_type, steps=steps, **references))
        for paths in path_counts:
            for seed in seeds:
                mc.extend(run("monte_carlo_european", build_dir, type=option_type, paths=paths, seed=seed))
                with TemporaryDirectory(prefix="pricing-newton-") as directory:
                    trace_path = Path(directory) / "trace.csv"
                    options = {"newton_trace": trace_path} if paths == path_counts[-1] else {}
                    variance.extend(run("variance_reduction", build_dir, type=option_type,
                                        paths=paths, train_paths=2000, seed=seed, **options))
                    if options:
                        with trace_path.open(newline="", encoding="utf-8") as source:
                            newton.extend(csv.DictReader(source))
    write_rows(output_dir / "tree_convergence.csv", trees)
    write_rows(output_dir / "mc_convergence.csv", mc)
    write_rows(output_dir / "variance_reduction.csv", variance)
    write_rows(output_dir / "variance_summary.csv", summarize_variance(variance))
    write_rows(output_dir / "newton_convergence.csv", newton)

    for paths in ([2000, 10000, 50000] if quick else [5000, 20000, 100000]):
        for seed in seeds:
            for method in ["monte_carlo_american", "lsm_american"]:
                policy_options = {"boundary_ratio": 0.75} if method == "monte_carlo_american" else {}
                american.extend(run(method, build_dir, paths=paths, train_paths=max(2000, paths // 2),
                                    steps=50, seed=seed, **references, **policy_options))
    write_rows(output_dir / "american_convergence.csv", american)
    for steps in [10, 25, 50, 100]:
        for seed in seeds:
            exercise_grid.extend(run("lsm_american", build_dir, paths=50000, train_paths=25000,
                                     steps=steps, seed=seed, **references))
    write_rows(output_dir / "exercise_grid.csv", exercise_grid)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "quick" if quick else "full",
        "platform": platform.platform(), "python": platform.python_version(),
        "seeds": seeds, "european_path_counts": path_counts,
        "contract": {"spot": 100, "strike": 100, "rate": 0.05, "volatility": 0.2,
                     "maturity": 1, "dividend_yield": 0},
        "american_reference_steps": 2000,
        "american_fixed_boundary_ratio": 0.75,
        "variance_reduction_pilot_paths": 2000,
        "importance_sampling": {"shift_selection": "basic Newton directly on the fixed-pilot second moment",
                                "pilot_seed_stream": 3, "evaluation_seed_stream": 0,
                                "initial_shift": "standardized exercise threshold, bounded below by 0 for calls and above by 0 for puts",
                                "gradient_tolerance": 1e-8, "shift_change_tolerance": 1e-8,
                                "maximum_absolute_correction": 1.0, "maximum_iterations": 50,
                                "optimality": "approximate shift obtained by the basic Newton stopping rules"},
        "notes": ["CSV prices come from the C++ executables.",
                  "Monte Carlo 95% intervals are asymptotic.",
                  "LSM intervals condition on one trained policy; retraining variability is excluded.",
                  "CRR is a numerical reference, not an exact American price.",
                  "Runtimes exclude reference-price computation and CSV I/O; include both pilots, Newton fitting, and LSM training.",
                  "Seed reproducibility is within the same compiler and standard-library implementation."],
    }
    build_metadata = build_dir / "build_metadata.json"
    if build_metadata.is_file():
        manifest["build"] = json.loads(build_metadata.read_text(encoding="utf-8"))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "experiments")
    parser.add_argument("--quick", action="store_true", help="Three seeds and smaller path counts.")
    args = parser.parse_args()
    try:
        collect(args.build_dir, args.output_dir, args.quick)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        parser.exit(1, detail + "\n")
