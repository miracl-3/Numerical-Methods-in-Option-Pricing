"""Cache C++ CRR references by contract, grid and pricing-source fingerprints.

This module stores/retrieves numbers; it contains no Python pricing algorithm.
Historical result tables remain evidence and are never treated as a valid cache
without the current source fingerprint.
"""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
from tempfile import NamedTemporaryFile
from time import perf_counter

from build import ROOT, executable

CACHE_DIR = ROOT / ".cache/crr"
SOURCES = ("src/binomial_american.cpp", "include/binomial.hpp", "include/model.hpp", "include/experiment.hpp")


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def request_for(contract, tree_steps, exercise_dates):
    names = ("spot", "strike", "rate", "volatility", "maturity")
    values = {name: float(contract[name]) for name in names}
    values["type"] = contract.get("type", "put")
    if not all(math.isfinite(values[name]) for name in names): raise ValueError("Non-finite CRR contract.")
    if values["spot"] <= 0 or values["strike"] <= 0 or values["volatility"] < 0 or values["maturity"] < 0:
        raise ValueError("Invalid CRR contract.")
    if values["type"] not in {"put", "call"}: raise ValueError("CRR type must be call or put.")
    if not 1 <= tree_steps <= 100000 or exercise_dates < 1 or tree_steps % exercise_dates:
        raise ValueError("Tree steps must be in [1,100000] and divisible by exercise dates.")
    return {"schema": 1, "contract": values, "tree_steps": tree_steps,
            "exercise_dates": exercise_dates, "exercise_at_zero": True, "source_sha256": source_hashes()}


def cache_key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_values(values):
    return all(math.isfinite(values[name]) and values[name] >= 0 for name in
               ("american_grid_reference_crr", "american_reference_crr")) and (
               values["american_reference_crr"] + 1e-10 >= values["american_grid_reference_crr"])


def get_reference(build_dir, contract, tree_steps, exercise_dates, cache_dir=CACHE_DIR):
    request = request_for(contract, tree_steps, exercise_dates)
    key = cache_key(request)
    path = Path(cache_dir) / f"{key}.json"
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if (stored["request"] == request and stored["key"] == key
                    and stored["values"]["reference_steps"] == tree_steps and valid_values(stored["values"])):
                return {**stored["values"], "cache_key": key, "cache_hit": True}
        except (ValueError, KeyError, TypeError):
            pass  # A malformed entry is a miss, never a benchmark.
    build_dir = Path(build_dir)
    metadata_path = build_dir / "build_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("CRR cache misses require build_metadata.json; run scripts/build.py (or the CMake build).")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if any(metadata["source_sha256"].get(name) != digest for name, digest in request["source_sha256"].items()):
        raise RuntimeError("CRR executable is stale relative to pricing sources; rebuild before calculating a reference.")
    binary = executable("binomial_american", build_dir)
    if metadata.get("binary_sha256", {}).get("binomial_american") != hashlib.sha256(binary.read_bytes()).hexdigest():
        raise RuntimeError("CRR binary does not match build metadata; rebuild before calculating a reference.")
    command = [str(binary), "--steps", str(tree_steps), "--reference-steps", str(tree_steps),
               "--exercise-dates", str(exercise_dates)]
    for name, value in request["contract"].items(): command.extend(["--" + name, str(value)])
    start = perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    rows = {row["method"]: row for row in csv.DictReader(io.StringIO(process.stdout))}
    for row in rows.values():
        for name in ("spot", "strike", "rate", "volatility", "maturity"):
            if float(row[name]) != request["contract"][name]: raise RuntimeError("CRR returned a different contract.")
        if row["option_type"] != request["contract"]["type"] or int(row["steps"]) != tree_steps:
            raise RuntimeError("CRR returned a different option type or tree resolution.")
    values = {"reference_steps": tree_steps,
              "american_reference_crr": float(rows["binomial_american"]["price"]),
              "american_grid_reference_crr": float(rows["binomial_american_grid"]["price"])}
    if not valid_values(values): raise RuntimeError("Invalid C++ CRR reference.")
    stored = {"key": key, "request": request, "values": values,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "runtime_seconds": perf_counter() - start, "command": command,
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "build": metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents interrupted writes from becoming valid cache hits.
    with NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", encoding="utf-8", delete=False) as out:
        temporary = Path(out.name)
        out.write(json.dumps(stored, indent=2) + "\n")
    temporary.replace(path)
    return {**values, "cache_key": key, "cache_hit": False}
