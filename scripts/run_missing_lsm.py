"""Plan/resume only missing LSM convergence fits; reuse saved fine CRR values.

The default is read-only. --run opts into executing the missing C++ fits.
Completed research inputs and figures are never overwritten.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from build import ROOT, executable
from run_experiments import run
from lsm_statistics import (audit_existing_results, existing_convergence_runs,
                            missing_convergence_keys, summarize_prices, sha256)

OUTPUT = ROOT / "data/experiments/lsm_comparison/convergence_completion"
EXPERIMENT = "missing_plain_lsm_convergence"
LSM_SOURCES = ("src/lsm_american.cpp", "include/model.hpp", "include/binomial.hpp",
               "include/experiment.hpp")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_name(key):
    method, paths, seed = key
    return f"{method}_{paths}_{seed}"


def check_result(rows, key):
    if len(rows) != 1:
        raise ValueError("Expected one in-sample valuation per fit.")
    row = rows[0]
    method, paths, seed = key
    expected = dict(sampling_method=method, paths=paths, seed=seed, training_paths=paths,
                    evaluation_paths=paths, spot=40, strike=40, rate=.06, volatility=.2,
                    maturity=1, steps=50, valuation_mode="in-sample", option_type="put",
                    basis="weighted_laguerre_4", antithetic_pairs=0)
    for field, value in expected.items():
        actual = row[field]
        valid = actual == value if isinstance(value, str) else math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-12)
        if not valid:
            raise ValueError(f"Unexpected {field} for {key}: {actual}")
    for field in ("price", "standard_error", "naive_path_standard_error", "runtime_ms"):
        if not math.isfinite(float(row[field])) or float(row[field]) < 0:
            raise ValueError(f"Invalid {field} for {key}.")


def build_provenance(build_dir):
    metadata_path = build_dir / "build_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    binary = executable("lsm_american", build_dir)
    if sha256(binary) != metadata["binary_sha256"]["lsm_american"]:
        raise ValueError("LSM executable differs from its recorded build; stopping without compiling.")
    # Git may translate LF to CRLF; only that byte-format difference is accepted.
    for name in LSM_SOURCES:
        content = (ROOT / name).read_bytes()
        candidates = {hashlib.sha256(content).hexdigest(),
                      hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()}
        if metadata["source_sha256"][name] not in candidates:
            raise ValueError(f"Build/source mismatch for {name}; stopping without compiling.")
    control = json.loads((ROOT / "data/experiments/lsm_comparison/full/manifest.json").read_text())
    for name in LSM_SOURCES[:3]:
        if metadata["source_sha256"][name] != control["source_sha256"][name]:
            raise ValueError(f"Numerical source differs from the controlled study: {name}")
    return {"binary_sha256": sha256(binary), "build_metadata_sha256": sha256(metadata_path),
            "build": metadata}


def execute_fit(build_dir, output, key, reference, reference_file):
    method, paths, seed = key
    destination = output / run_name(key)
    if destination.exists():
        raise ValueError(f"Preserving existing fit: {destination}")
    # A directory is committed only after both CSVs are complete; interruption
    # cannot make a partial fit look finished on the next invocation.
    with TemporaryDirectory(prefix=".pending-", dir=output) as temporary:
        stage = Path(temporary)
        parameters = dict(type="put", spot=40, strike=40, rate=.06, volatility=.2,
                          maturity=1, steps=50, paths=paths, train_paths=paths, seed=seed,
                          sampling=method, valuation="in-sample", reference_steps=50,
                          regression_diagnostics=stage / "regressions.csv")
        rows = run("lsm_american", build_dir, **parameters)
        check_result(rows, key)
        rows[0].update(comparison_reference_crr=reference, comparison_reference_steps=100000,
                       reference_source=reference_file, experiment=EXPERIMENT)
        write_rows(stage / "result.csv", rows)
        parameters["regression_diagnostics"] = "regressions.csv"
        write_json(stage / "record.json", {"parameters": parameters,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "sha256": {name: sha256(stage / name) for name in ("result.csv", "regressions.csv")}})
        stage.rename(destination)
    return key


def collect(build_dir, output, execute=False, workers=2):
    if workers < 1:
        raise ValueError("--workers must be positive.")
    if (not output.is_relative_to(ROOT / "data/experiments/lsm_comparison") or
            output.is_relative_to(ROOT / "data/experiments/lsm_comparison/full")):
        raise ValueError("Choose a separate completion directory under data/experiments/lsm_comparison/.")
    frames, audit = audit_existing_results()
    reference = audit["matched_crr_atm"]
    original = existing_convergence_runs(frames, reference)
    missing = missing_convergence_keys(original)
    if any(method != "plain" for method, _, _ in missing):
        raise ValueError("This completion runner expects only missing plain-LSM configurations.")
    provenance = build_provenance(build_dir)
    reference_file = "data/experiments/paper_replication/full/lsm_paper_table.csv"
    identity = {"experiment": EXPERIMENT, **provenance,
                "inputs_sha256": {**audit["inputs_sha256"], **audit["manifest_sha256"]},
                "matched_crr": reference, "reference_steps": 100000,
                "reference_source": reference_file, "planned_keys": missing, "workers": workers}
    # JSON normalizes tuples so the resume comparison is exact.
    identity = json.loads(json.dumps(identity))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity or previous["runner_sha256"] != sha256(Path(__file__)):
            raise ValueError("Completion manifest differs from inputs/build/settings; preserving all saved fits.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output without a completion manifest; preserving its contents.")
    completed = []
    pending = []
    for key in missing:
        directory = output / run_name(key)
        if not directory.exists():
            pending.append(key)
            continue
        record = json.loads((directory / "record.json").read_text())
        for name in ("result.csv", "regressions.csv"):
            if sha256(directory / name) != record["sha256"][name]:
                raise ValueError(f"Changed saved fit: {directory / name}")
        rows = read_rows(directory / "result.csv")
        check_result(rows, key)
        completed.extend(rows)
    print(f"Preserved convergence runs: {len(original)}; completed supplement: {len(completed)}; pending: {len(pending)}")
    for paths in sorted({n for _, n, _ in pending}):
        seeds = [s for _, n, s in pending if n == paths]
        print(f"  plain, {paths:,} paths: seeds {', '.join(map(str, seeds))}")
    print(f"Reuse saved 100,000-step matched CRR: {reference:.12f}. No fine-tree calculation or build.")
    if not execute:
        print("Plan only. Add --run to execute these missing fits.")
        return
    output.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        write_json(manifest_path, {"profile": "full_convergence_supplement", "identity": identity,
                   "runner_sha256": sha256(Path(__file__)),
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "timing_note": "New timings are a separate session; compare efficiency using the original controlled 100000-path runs.",
                   "reference_note": "Reuse saved fine CRR. The existing LSM executable also emits its cheap 50-step local reference."})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute_fit, build_dir, output, key, reference, reference_file) for key in pending]
        for i, future in enumerate(as_completed(futures), 1):
            key = future.result()
            print(f"Saved {i}/{len(pending)}: {run_name(key)}", flush=True)
    rows = [read_rows(output / run_name(key) / "result.csv")[0] for key in missing]
    if rows:
        write_rows(output / "convergence.csv", rows)
    combined = existing_convergence_runs(frames, reference, output / "convergence.csv")
    if missing_convergence_keys(combined):
        raise ValueError("Convergence grid remains incomplete.")
    combined.to_csv(output / "combined_convergence.csv", index=False, float_format="%.17g")
    summaries = []
    for (method, paths), group in combined.groupby(["sampling_method", "paths"], sort=True):
        summary = summarize_prices(group.price.tolist(), group.standard_error.tolist(), reference)
        # These outputs report variability and errors; no new confidence claims.
        for field in ("mean_ci_lower", "mean_ci_upper"):
            summary.pop(field)
        summary.update(sampling_method=method, paths=int(paths), empirical_variance=summary["empirical_sd"]**2)
        summaries.append(summary)
    write_rows(output / "convergence_summary.csv", summaries)
    print(f"Complete: {len(combined)} unique runs; summaries in {output}. Figures were not regenerated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run", action="store_true", help="Execute only pending fits; otherwise print a read-only plan.")
    args = parser.parse_args()
    try:
        collect(args.build_dir.resolve(), args.output_dir.resolve(), args.run, args.workers)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, (getattr(error, "stderr", "") or str(error)) + "\n")
