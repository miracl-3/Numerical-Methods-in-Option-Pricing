"""Run C++ pricing experiments and collect their CSV output without reimplementing prices."""

import argparse
import csv
import io
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

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


def collect(build_dir, output_dir, quick=False):
    build_dir, output_dir = Path(build_dir).resolve(), Path(output_dir).resolve()
    for method in METHODS:
        if not executable(method, build_dir).is_file():
            raise RuntimeError(f"Missing {method}; run python scripts/build.py first.")
    path_counts = [1000, 5000, 20000, 50000] if quick else [1000, 5000, 20000, 100000, 500000]
    seeds = [42, 43, 44] if quick else list(range(42, 50))
    references = {"reference_steps": 2000}
    trees, mc, variance, american, exercise_grid = [], [], [], [], []
    for option_type in ["call", "put"]:
        for steps in [25, 50, 75, 100, 200, 400, 800]:
            for method in ["binomial_european", "binomial_american"]:
                trees.extend(run(method, build_dir, type=option_type, steps=steps, **references))
        for paths in path_counts:
            for seed in seeds:
                mc.extend(run("monte_carlo_european", build_dir, type=option_type, paths=paths, seed=seed))
                variance.extend(run("variance_reduction", build_dir, type=option_type,
                                    paths=paths, train_paths=2000, seed=seed))
    write_rows(output_dir / "tree_convergence.csv", trees)
    write_rows(output_dir / "mc_convergence.csv", mc)
    write_rows(output_dir / "variance_reduction.csv", variance)

    for paths in ([2000, 10000, 50000] if quick else [5000, 20000, 100000]):
        for seed in seeds:
            for method in ["monte_carlo_american", "lsm_american"]:
                american.extend(run(method, build_dir, paths=paths, train_paths=max(2000, paths // 2),
                                    steps=50, seed=seed, **references))
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
        "notes": ["CSV prices come from the C++ executables.",
                  "Monte Carlo 95% intervals are asymptotic.",
                  "LSM intervals condition on one trained policy; retraining variability is excluded.",
                  "CRR is a numerical reference, not an exact American price.",
                  "Runtimes exclude reference-price computation and CSV I/O; include control pilot/LSM training.",
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
