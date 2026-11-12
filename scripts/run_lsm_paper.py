"""Reproduce the Longstaff-Schwartz American-put experiment and path convergence."""

import argparse
import csv
import io
import json
from pathlib import Path
import platform
import subprocess

from build import ROOT, executable

STRIKE = 40.0
RATE = 0.06
EXERCISES_PER_YEAR = 50
REFERENCE_STEPS = 2000
TABLE_FIELDS = (
    "spot", "strike", "rate", "volatility", "maturity", "exercise_dates_per_year",
    "steps", "paths", "seed", "price", "standard_error", "ci_lower", "ci_upper",
    "american_reference_crr", "absolute_error", "exercise_fraction",
    "independent_samples", "payoff_evaluations", "runtime_ms",
)


def run(build_dir, spot, volatility, maturity, paths, seed):
    steps = EXERCISES_PER_YEAR * maturity
    command = [
        str(executable("lsm_american", build_dir)),
        "--lsm-mode", "paper", "--type", "put",
        "--spot", str(spot), "--strike", str(STRIKE), "--rate", str(RATE),
        "--volatility", str(volatility), "--maturity", str(maturity),
        "--steps", str(steps), "--paths", str(paths), "--seed", str(seed),
        "--reference-steps", str(REFERENCE_STEPS),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    row = next(csv.DictReader(io.StringIO(result.stdout)))
    return {
        "spot": spot,
        "strike": STRIKE,
        "rate": RATE,
        "volatility": volatility,
        "maturity": maturity,
        "exercise_dates_per_year": EXERCISES_PER_YEAR,
        "steps": steps,
        "paths": paths,
        "seed": seed,
        "price": float(row["price"]),
        "standard_error": float(row["standard_error"]),
        "ci_lower": float(row["ci_lower"]),
        "ci_upper": float(row["ci_upper"]),
        "american_reference_crr": float(row["benchmark"]),
        "absolute_error": float(row["absolute_error"]),
        "exercise_fraction": float(row["exercise_fraction"]),
        "independent_samples": int(row["independent_samples"]),
        "payoff_evaluations": int(row["payoff_evaluations"]),
        "runtime_ms": float(row["runtime_ms"]),
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    display_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"Wrote {display_path} ({len(rows)} rows)", flush=True)


def collect(build_dir, output_dir, quick=False):
    build_dir = Path(build_dir).resolve()
    output_dir = Path(output_dir).resolve()
    binary = executable("lsm_american", build_dir)
    if not binary.is_file():
        raise RuntimeError(f"Missing {binary}; run python scripts/build.py first.")

    table_paths = 10000 if quick else 100000
    table = []
    for spot in [36, 38, 40, 42, 44]:
        for volatility in [0.2, 0.4]:
            for maturity in [1, 2]:
                table.append(run(build_dir, spot, volatility, maturity, table_paths, 42))
    write_rows(output_dir / "lsm_paper_table.csv", table)

    path_counts = [1000, 2500, 5000] if quick else [5000, 10000, 25000, 50000, 100000]
    seeds = [42, 43] if quick else [42, 43, 44, 45, 46]
    convergence = [
        run(build_dir, STRIKE, 0.2, 1, paths, seed)
        for paths in path_counts
        for seed in seeds
    ]
    write_rows(output_dir / "lsm_paper_convergence.csv", convergence)

    manifest = {
        "paper": "Longstaff and Schwartz (2001), Table 1, American put experiment",
        "profile": "quick" if quick else "paper",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "contract_cases": {
            "strike": STRIKE,
            "rate": RATE,
            "spots": [36, 38, 40, 42, 44],
            "volatilities": [0.2, 0.4],
            "maturities": [1, 2],
            "exercise_dates_per_year": EXERCISES_PER_YEAR,
        },
        "estimator": {
            "table_paths": table_paths,
            "table_seed": 42,
            "sampling": "paired Z and -Z shocks; standard errors use antithetic pair averages",
            "regressors": "constant plus weighted Laguerre L0, L1, and L2 of spot/strike",
            "valuation": "in-sample stopping-policy fit and valuation, as in Table 1",
        },
        "convergence_case": {"spot": STRIKE, "volatility": 0.2, "maturity": 1},
        "convergence_paths": path_counts,
        "convergence_seeds": seeds,
        "reference": f"{REFERENCE_STEPS}-step Bermudan CRR tree aligned to the 50-per-year exercise grid; not the paper's finite-difference reference",
        "notes": [
            "Each maturity uses 50 exercise dates per year (50 or 100 total dates).",
            "Antithetic pair averages are the independent observations for standard errors.",
            "In-sample standard errors do not measure independent policy-evaluation uncertainty.",
            "In-sample estimates reproduce the paper protocol, not the independent-evaluation baseline.",
            "The paper's 100,000-path Table 1 uses 50,000 antithetic pairs.",
        ],
    }
    build_metadata = build_dir / "build_metadata.json"
    if build_metadata.is_file():
        manifest["build"] = json.loads(build_metadata.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lsm_paper_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "lsm_paper_manifest.json"
    display_path = manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path
    print(f"Wrote {display_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "experiments")
    parser.add_argument("--quick", action="store_true", help="Smaller sample for a smoke run.")
    args = parser.parse_args()
    try:
        collect(args.build_dir, args.output_dir, args.quick)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        parser.exit(1, detail + "\n")
