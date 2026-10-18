"""Build the preserved advanced experiment independently of the main six-program project."""

import csv
import io
import json
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("Install g++ or clang++ and put it on PATH.")
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for name, metadata in provenance["original_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise RuntimeError(f"Preserved file has changed: {name}")
    destination = ROOT / "build"
    destination.mkdir(exist_ok=True)
    executable = destination / ("variance_reduction_advanced.exe" if sys.platform == "win32"
                                else "variance_reduction_advanced")
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                    "-I", str(ROOT / "include"), str(ROOT / "variance_reduction.cpp"),
                    "-o", str(executable)], check=True)
    result = subprocess.run([str(executable), "--type", "call", "--paths", "50000",
                             "--train-paths", "2000", "--seed", "42"],
                            capture_output=True, text=True, check=True)
    actual = {row["method"]: row for row in csv.DictReader(io.StringIO(result.stdout))}
    with (ROOT / "results" / "variance_reduction.csv").open(newline="", encoding="utf-8") as source:
        expected = [row for row in csv.DictReader(source) if row["option_type"] == "call"
                    and int(row["paths"]) == 50000 and int(row["seed"]) == 42]
    # Different standard libraries may generate different normal draws. Validate the
    # estimator and its convergence without requiring cross-platform bitwise identity.
    for row in actual.values():
        if abs(float(row["price"]) - float(row["benchmark"])) > 6 * float(row["standard_error"]):
            raise RuntimeError(f"Advanced price failed the benchmark check: {row['method']}")
    fit = actual["importance_sampling"]
    if fit["newton_status"] != "converged" or abs(float(fit["newton_gradient"])) > 1e-8:
        raise RuntimeError("Advanced Newton convergence check failed.")
    comparison = {row["method"]: float(actual[row["method"]]["price"]) - float(row["price"])
                  for row in expected}
    print(f"Built and checked {executable}")
    print("Price differences from the preserved snapshot (platform-dependent):", comparison)


if __name__ == "__main__":
    main()
