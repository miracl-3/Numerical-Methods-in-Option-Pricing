"""Conventional statistics and read-only checks for the saved LSM experiments."""

import hashlib
import json
import math
from pathlib import Path
import statistics

from experiment_files import ROOT, REFERENCE_DIR

PATH_COUNTS = (5000, 10000, 25000, 50000, 100000)
SEEDS = tuple(range(42, 72))
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


def paired_difference_summary(first, second):
    if len(first) != len(second) or len(first) < 2: raise ValueError("Need matching paired runs.")
    differences = [a - b for a, b in zip(first, second)]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    return {"mean_difference": mean, "difference_sd": sd, "se_of_mean_difference": sd / math.sqrt(len(first))}


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
    frames = {name: pd.read_csv(path) for name, path in files.items()}
    expected_counts = dict(paper_table=20, paper_convergence=150, control_table=40,
                           control_runs=60, table2=80, published_table1=20, published_table2=40)
    for name, count in expected_counts.items():
        if len(frames[name]) != count: raise ValueError(f"{name}: expected {count} rows.")
    for name in ("control_table", "control_runs", "table2"):
        frame = frames[name]
        if not ((frame.paths == 100000) & (frame.basis == "weighted_laguerre_4") &
                (frame.strike == 40) & (frame.rate == .06)).all():
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
    source_history = ROOT / "archive/organization_2026-11-26/verified_crr_source_history.json"
    proof = json.loads(source_history.read_text())
    historical_model = proof["historical_model_source"].encode()
    if hashlib.sha256(historical_model).hexdigest() != paper_manifest["build"]["source_sha256"]["include/model.hpp"]:
        raise ValueError("Historical model snapshot does not match the original reference manifest.")
    if historical_model.split(b"// SplitMix64")[0] != (ROOT / "include/model.hpp").read_bytes().split(b"// SplitMix64")[0]:
        raise ValueError("CRR contract/payoff definitions changed; saved references need revalidation.")
    if sha256(ROOT / "include/binomial.hpp") != proof["crr_header_sha256"]:
        raise ValueError("CRR algorithm changed; saved references need revalidation.")
    for name in ("src/lsm_american.cpp", "include/model.hpp"):
        if sha256(ROOT / name) != control_manifest["source_sha256"][name]:
            raise ValueError(f"LSM numerical source changed: {name}; do not mix current runs without revalidation.")
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
    provenance = json.loads((REFERENCE_DIR / "provenance.json").read_text())
    for table in provenance["tables"].values():
        if sha256(REFERENCE_DIR / table["file"]) != table["sha256"]: raise ValueError("Published reference checksum mismatch.")
    audit = {"profile": "full", "row_counts": expected_counts, "path_counts": list(PATH_COUNTS),
             "seeds": list(SEEDS), "matched_crr_atm": reference, "crr_reference_steps": 100000,
             "max_av_table_overlap_difference": av_overlap_error, "max_av_repeat_overlap_difference": repeat_error,
             "validation": "Current LSM/model sources match the controlled run. CRR header and contract/payoff prefix match the original source snapshot. Original AV convergence shares the same disclosed normalized Laguerre procedure; all 30 overlapping estimates agree to roundoff.",
             "runtime_scope": "Use only the same-build 100000-path controlled replicates for runtime efficiency; older AV convergence timings are not compared with new plain timings.",
             "inputs_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in files.values()},
             "manifest_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in
                                 (PAPER / "lsm_paper_manifest.json", COMPARISON / "manifest.json", source_history)}}
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
        add(pd.read_csv(supplement), "plain", Path(supplement).relative_to(ROOT))
    result = pd.DataFrame(rows)
    if result.duplicated(["sampling_method", "paths", "seed"]).any(): raise ValueError("Duplicate convergence observations.")
    return result


def missing_convergence_keys(runs):
    available = set(zip(runs.sampling_method, runs.paths, runs.seed))
    return [(method, count, seed) for method in ("plain", "antithetic") for count in PATH_COUNTS for seed in SEEDS
            if (method, count, seed) not in available]
