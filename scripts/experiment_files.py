"""Shared locations and overwrite rules for the three small research workflows."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data/reference/longstaff_schwartz_2001"
EXPERIMENTS = ROOT / "data/experiments"
FIGURES = ROOT / "figures"
STUDIES = ("paper_replication", "lsm_comparison", "pricing_methods", "lsm_analysis")


def results_dir(study, quick=False):
    if study not in STUDIES: raise ValueError(f"Unknown study: {study}")
    return ROOT / "tmp/smoke" / study if quick else EXPERIMENTS / study / "full"


def profile_of(directory):
    for path in Path(directory).glob("*manifest.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        if "profile" in document: return document["profile"]
    return None


def is_smoke(profile):
    return profile in {"quick", "smoke", "smoke_only"}


def prepare_results(study, directory=None, quick=False):
    path = Path(directory or results_dir(study, quick)).resolve()
    protected = (EXPERIMENTS, FIGURES, REFERENCE_DIR, ROOT / "archive")
    if quick and any(path.is_relative_to(parent) for parent in protected):
        raise ValueError("Smoke output must be outside research results, figures, references and archives; use tmp/.")
    allowed = (EXPERIMENTS, ROOT / "tmp", ROOT / "data/custom")
    if path.is_relative_to(ROOT) and not any(path.is_relative_to(parent) for parent in allowed):
        raise ValueError("Write results under data/experiments/, data/custom/, or tmp/.")
    if any(path == parent or parent.is_relative_to(path) for parent in protected):
        raise ValueError("Choose a study/run directory, not a repository or category root.")
    previous = profile_of(path) if path.is_dir() else None
    if previous and not is_smoke(previous):
        raise ValueError(f"Preserving existing research in {path}. Choose a new --output-dir for another run.")
    if not quick and path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new --output-dir.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_locations(study, data_dir=None, output_dir=None, quick=False):
    data = Path(data_dir or results_dir(study, quick)).resolve()
    if not data.is_dir(): raise ValueError(f"Missing result directory: {data}")
    smoke = quick or is_smoke(profile_of(data))
    if output_dir is not None:
        output = Path(output_dir).resolve()
    elif smoke and data.is_relative_to(EXPERIMENTS):
        output = ROOT / "tmp/figures" / study / data.name
    elif smoke or not data.is_relative_to(EXPERIMENTS):
        output = data / "figures"
    else:
        output = FIGURES / data.relative_to(EXPERIMENTS)
    if smoke and any(output.is_relative_to(parent) for parent in (FIGURES, EXPERIMENTS, REFERENCE_DIR, ROOT / "archive")):
        raise ValueError("Smoke figures cannot overwrite research figures or data; use tmp/.")
    if output.is_relative_to(EXPERIMENTS) or output.is_relative_to(REFERENCE_DIR) or output.is_relative_to(ROOT / "archive"):
        raise ValueError("Write figures under figures/ or tmp/, not research data or archives.")
    if output == FIGURES or (output.is_relative_to(ROOT) and not any(
            output.is_relative_to(parent) for parent in (FIGURES, ROOT / "tmp", ROOT / "data/custom"))):
        raise ValueError("Choose a figure study/run directory under figures/ or tmp/.")
    output.mkdir(parents=True, exist_ok=True)
    return data, output
