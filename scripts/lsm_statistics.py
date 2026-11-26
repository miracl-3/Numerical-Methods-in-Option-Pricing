"""Conventional statistics and read-only checks for the saved LSM experiments."""

import hashlib
import json
import math
from pathlib import Path
import statistics

from experiment_files import ROOT, REFERENCE_DIR

PATH_COUNTS = (5000, 10000, 25000, 50000, 100000)
SEEDS = tuple(range(42, 72))
T_975_DF4 = 2.7764451051977987
PAPER = ROOT / "data/experiments/paper_replication/full"
COMPARISON = ROOT / "data/experiments/lsm_comparison/full"
DEFAULT_ANALYSIS = ROOT / "data/experiments/lsm_analysis/full"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_prices(prices, reported_se, reference):
    if len(prices) < 2 or len(prices) != len(reported_se):
        raise ValueError("Need matching prices and SEs from at least two runs.")
    if not all(math.isfinite(x) for x in prices + reported_se + [reference]) or min(reported_se) < 0:
        raise ValueError("Prices/reference must be finite; SEs must be nonnegative.")
    mean = statistics.mean(prices)
    sd = statistics.stdev(prices)
    se_mean = sd / math.sqrt(len(prices))
    average_se = statistics.mean(reported_se)
    return {"runs": len(prices), "mean_price": mean, "empirical_sd": sd,
            "mean_reported_se": average_se, "se_of_mean": se_mean,
            "mean_ci_lower": mean - 1.96 * se_mean, "mean_ci_upper": mean + 1.96 * se_mean,
            "mean_error": mean - reference,
            "rmse": math.sqrt(statistics.mean((x - reference) ** 2 for x in prices)),
            "reported_se_over_empirical_sd": average_se / sd if sd else None}


def standardized_difference(price, reference, se, reference_se=0.0):
    if se < 0 or reference_se < 0: raise ValueError("SEs must be nonnegative.")
    scale = math.hypot(se, reference_se)
    return (price - reference) / scale if scale else (0.0 if price == reference else None)


def relative_efficiency(plain_variance, plain_time, av_variance, av_time):
    if min(plain_variance, plain_time, av_variance, av_time) <= 0:
        raise ValueError("Variances and times must be positive.")
    return plain_variance * plain_time / (av_variance * av_time)


def paired_difference_summary(first, second, critical_value=1.96):
    if len(first) != len(second) or len(first) < 2: raise ValueError("Need matching paired runs.")
    differences = [a - b for a, b in zip(first, second)]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    se = sd / math.sqrt(len(first))
    return {"mean_difference": mean, "difference_sd": sd, "se_of_mean_difference": se,
            "difference_ci_lower": mean - critical_value * se,
            "difference_ci_upper": mean + critical_value * se}


def audit_existing_results():
    import pandas as pd
    paper_manifest = json.loads((PAPER / "lsm_paper_manifest.json").read_text())
    control_manifest = json.loads((COMPARISON / "manifest.json").read_text())
    if paper_manifest["profile"] != "original_table1" or control_manifest["profile"] != "full":
        raise ValueError("Expected full research results, not smoke data.")
    files = {"paper_table": PAPER / "lsm_paper_table.csv", "paper_convergence": PAPER / "lsm_paper_convergence.csv",
             "control_table": COMPARISON / "table1.csv", "control_runs": COMPARISON / "replicates.csv",
             "table2": COMPARISON / "table2.csv", "published_table1": REFERENCE_DIR / "table1.csv",
             "published_table2": REFERENCE_DIR / "table2.csv"}
    frames = {name: pd.read_csv(path, float_precision="round_trip") for name, path in files.items()}
    expected_counts = dict(paper_table=20, paper_convergence=150, control_table=40,
                           control_runs=60, table2=80, published_table1=20, published_table2=40)
    for name, count in expected_counts.items():
        if len(frames[name]) != count: raise ValueError(f"{name}: expected {count} rows.")
    for name in ("control_table", "control_runs", "table2"):
        frame = frames[name]
        if not ((frame.paths == 100000) & (frame.basis == "weighted_laguerre_4") &
                (frame.strike == 40) & ((frame.rate - .06).abs() < 1e-12)).all():
            raise ValueError(f"{name}: incompatible budget, basis or contract.")
    if set(frames["control_table"].sampling_method) != {"plain", "antithetic"}:
        raise ValueError("Missing controlled sampling method.")
    for name in ("control_table", "control_runs"):
        if set(frames[name].valuation_mode) != {"in-sample"}: raise ValueError("Wrong main valuation mode.")
    paired = frames["table2"].groupby(["spot", "volatility", "maturity", "seed"])
    if paired.ngroups != 40 or not (paired.size() == 2).all(): raise ValueError("Incomplete Table 2 policy pairs.")
    for _, group in paired:
        if set(group.valuation_mode) != {"in-sample", "out-of-sample"} or group.training_seed.nunique() != 1:
            raise ValueError("Table 2 must reuse a single fitted policy.")
        if not (group.training_paths == 100000).all(): raise ValueError("Wrong Table 2 training budget.")
    keys = ["spot", "volatility", "maturity"]
    common = frames["control_table"].query("sampling_method == 'antithetic'").merge(frames["paper_table"], on=keys,
                                                                                 suffixes=("_control", "_paper"))
    av_overlap_error = float((common.price_control - common.price_paper).abs().max())
    repeat_overlap = frames["control_runs"].query("sampling_method == 'antithetic'").merge(
        frames["paper_convergence"].query("paths == 100000"), on="seed", suffixes=("_control", "_paper"))
    repeat_error = float((repeat_overlap.price_control - repeat_overlap.price_paper).abs().max())
    if len(repeat_overlap) != 30 or max(av_overlap_error, repeat_error) > 1e-12:
        raise ValueError("Original and controlled AV results are inconsistent.")
    for sampling, group in frames["control_runs"].groupby("sampling_method"):
        if set(group.seed) != set(SEEDS): raise ValueError(f"Incomplete 30-seed sample: {sampling}")
    if set(frames["paper_convergence"].paths) != set(PATH_COUNTS): raise ValueError("Wrong AV convergence budgets.")
    for _, group in frames["paper_convergence"].groupby("paths"):
        if set(group.seed) != set(SEEDS): raise ValueError("Incomplete AV convergence seeds.")
    reference = float(frames["paper_table"].query("spot == 40 and volatility == .2 and maturity == 1").american_grid_reference_crr.iloc[0])
    for name in ("paper_convergence", "control_runs"):
        frame = frames[name]
        for field, expected in dict(spot=40, strike=40, rate=.06, volatility=.2, maturity=1, steps=50).items():
            if not ((frame[field]-expected).abs() < 1e-12).all():
                raise ValueError(f"{name}: incompatible convergence contract/grid ({field}).")
        reference_field = "american_grid_reference_crr" if name == "paper_convergence" else "comparison_reference_crr"
        if not ((frame[reference_field]-reference).abs() < 1e-12).all():
            raise ValueError(f"{name}: inconsistent matched CRR.")
    if float((repeat_overlap.standard_error_control-repeat_overlap.standard_error_paper).abs().max()) > 1e-12:
        raise ValueError("Overlapping AV standard errors differ.")
    provenance = json.loads((REFERENCE_DIR / "provenance.json").read_text())
    reference_format_notes = []
    for table in provenance["tables"].values():
        path = REFERENCE_DIR / table["file"]
        if sha256(path) != table["sha256"]:
            normalized = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if normalized != table["sha256"]: raise ValueError("Published reference checksum mismatch.")
            reference_format_notes.append(f"{table['file']}: Windows line endings; LF-normalized checksum matches the recorded source.")
    audit = {"profile": "full", "row_counts": expected_counts, "path_counts": list(PATH_COUNTS),
             "seeds": list(SEEDS), "matched_crr_atm": reference, "crr_reference_steps": 100000,
             "max_av_table_overlap_difference": av_overlap_error, "max_av_repeat_overlap_difference": repeat_error,
             "reference_format_notes": reference_format_notes,
             "validation": "Read-only CSV/manifest checks: full budgets, valuation modes, sampling methods, published-table hashes and consistent overlapping AV estimates. No pricing code, executables or environment checks are run.",
             "runtime_scope": "Use only the same-build 100000-path controlled replicates for runtime efficiency; older AV convergence timings are not compared with new plain timings.",
             "statistical_compatibility": "Same ATM contract/grid and full-fit protocol; all 30 overlapping AV prices/SEs agree within 1e-12. Historical positive-date exercise is equivalent here because the time-zero put payoff is zero. Historical and controlled builds remain distinct timing cohorts.",
             "inputs_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in files.values()},
             "manifest_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in
                                 (PAPER / "lsm_paper_manifest.json", COMPARISON / "manifest.json")}}
    return frames, audit


def existing_convergence_runs(frames, reference, supplement=None):
    import pandas as pd
    rows = []
    def add(frame, sampling, source):
        for row in frame.to_dict("records"):
            rows.append({"paths": int(row["paths"]), "seed": int(row["seed"]), "sampling_method": sampling,
                         "price": float(row["price"]), "standard_error": float(row["standard_error"]),
                         "pathwise_se": float(row["naive_path_standard_error"]), "reference": reference,
                         "runtime_ms": float(row["runtime_ms"]), "source_file": str(source)})
    add(frames["paper_convergence"].query("paths < 100000"), "antithetic", (PAPER / "lsm_paper_convergence.csv").relative_to(ROOT))
    for sampling, frame in frames["control_runs"].groupby("sampling_method"):
        add(frame, sampling, (COMPARISON / "replicates.csv").relative_to(ROOT))
    if supplement is not None and Path(supplement).exists():
        add(pd.read_csv(supplement, float_precision="round_trip"), "plain", Path(supplement).relative_to(ROOT))
    result = pd.DataFrame(rows)
    if result.duplicated(["sampling_method", "paths", "seed"]).any(): raise ValueError("Duplicate convergence observations.")
    return result


def missing_convergence_keys(runs):
    available = set(zip(runs.sampling_method, runs.paths, runs.seed))
    return [(method, count, seed) for method in ("plain", "antithetic") for count in PATH_COUNTS for seed in SEEDS
            if (method, count, seed) not in available]


def load_completion(directory, frames, audit):
    """Read and authenticate a complete 120-fit supplement, without executing it."""
    import pandas as pd
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if identity["experiment"] != "missing_plain_lsm_convergence":
        raise ValueError("Not a plain-LSM completion manifest.")
    for name, digest in {**audit["inputs_sha256"], **audit["manifest_sha256"]}.items():
        if identity["inputs_sha256"].get(name) != digest:
            raise ValueError(f"Completion used different original inputs: {name}")
    control = json.loads((COMPARISON / "manifest.json").read_text())
    for name in ("src/lsm_american.cpp", "include/model.hpp", "include/binomial.hpp"):
        if identity["build"]["source_sha256"].get(name) != control["source_sha256"][name]:
            raise ValueError(f"Incompatible completion numerical source: {name}")
    reference = audit["matched_crr_atm"]
    if identity["reference_steps"] != 100000 or abs(identity["matched_crr"]-reference) > 1e-12:
        raise ValueError("Completion used a different reference.")
    required = set(missing_convergence_keys(existing_convergence_runs(frames, reference)))
    if set(map(tuple, identity["planned_keys"])) != required:
        raise ValueError("Completion manifest does not describe precisely the missing configurations.")
    for name, digest in manifest.get("outputs_sha256", {}).items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Changed completion output: {name}")
    csv_path = directory / "convergence.csv"
    data = pd.read_csv(csv_path, float_precision="round_trip",
                       dtype={"training_seed": "uint64", "evaluation_seed": "uint64"})
    keys = list(zip(data.sampling_method, data.paths, data.seed))
    if len(keys) != len(required) or set(keys) != required:
        raise ValueError("Need exactly 120 unique completed fits; rerun the missing-only runner to resume.")
    for field, value in dict(spot=40, strike=40, rate=.06, volatility=.2, maturity=1, steps=50,
                             comparison_reference_steps=100000, comparison_reference_crr=reference,
                             antithetic_pairs=0, exercise_at_zero=0).items():
        if not ((data[field]-value).abs() < 1e-12).all():
            raise ValueError(f"Incompatible completion {field}.")
    for field, value in dict(sampling_method="plain", valuation_mode="in-sample", basis="weighted_laguerre_4",
                             option_type="put", standard_error_scope="in_sample_approximate").items():
        if not (data[field] == value).all():
            raise ValueError(f"Incompatible completion {field}.")
    for field in ("training_paths", "evaluation_paths", "independent_samples", "unique_simulated_paths"):
        if not (data[field] == data.paths).all():
            raise ValueError(f"Completion path-count mismatch: {field}")
    if not (data.training_seed == data.evaluation_seed).all():
        raise ValueError("Completion is not in-sample valuation.")
    for field in ("price", "standard_error", "runtime_ms", "naive_path_standard_error"):
        if not data[field].map(lambda x: math.isfinite(x) and x >= 0).all():
            raise ValueError(f"Non-finite or negative {field}.")
    hashes = {manifest_path.relative_to(ROOT).as_posix(): sha256(manifest_path),
              csv_path.relative_to(ROOT).as_posix(): sha256(csv_path)}
    for row in data.itertuples(index=False):
        fit = directory / f"plain_{int(row.paths)}_{int(row.seed)}"
        record_path = fit / "record.json"
        record = json.loads(record_path.read_text())
        for name in ("result.csv", "regressions.csv"):
            if sha256(fit/name) != record["sha256"][name]:
                raise ValueError(f"Changed individual fit: {fit/name}")
        individual = pd.read_csv(fit/"result.csv", float_precision="round_trip",
                                  dtype={"training_seed": "uint64", "evaluation_seed": "uint64"})
        if len(individual) != 1 or individual.to_dict("records")[0] != row._asdict():
            raise ValueError(f"Aggregate differs from checkpoint: {fit}")
        if not row.execution_session or record["execution_session"] != row.execution_session:
            raise ValueError("Missing or inconsistent timing session.")
        hashes[record_path.relative_to(ROOT).as_posix()] = sha256(record_path)
    return data, hashes


def convergence_slopes(summary):
    """Descriptive OLS of log empirical SD on log path count (no iid-slope CI)."""
    import pandas as pd
    rows = []
    for method, group in summary.groupby("sampling_method", sort=True):
        group = group.sort_values("paths")
        row = {"sampling_method": method, "budgets": len(group), "intercept_a": None,
               "slope_b": None, "r_squared": None, "status": "insufficient_budgets"}
        if len(group) >= 2 and (group.empirical_sd > 0).all():
            x = [math.log(n) for n in group.paths]
            y = [math.log(s) for s in group.empirical_sd]
            xm, ym = statistics.mean(x), statistics.mean(y)
            b = sum((xi-xm)*(yi-ym) for xi, yi in zip(x,y)) / sum((xi-xm)**2 for xi in x)
            a = ym-b*xm
            total = sum((yi-ym)**2 for yi in y)
            residual = sum((yi-a-b*xi)**2 for xi, yi in zip(x,y))
            row.update(intercept_a=a, slope_b=b, r_squared=1-residual/total if total else None,
                       status="descriptive_fit")
        rows.append(row)
    return pd.DataFrame(rows)
