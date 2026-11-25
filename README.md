# Numerical option pricing

C++17 computes prices, stock paths, exercise policies and standard errors.
Python runs experiments, analyzes exported CSVs and creates figures.

- [Mathematics, equation-to-function map and eight-path example](docs/methodology.md)
- [Original-paper replication](docs/results/paper_replication.md)
- [Plain LSM versus AV-LSM](docs/results/lsm_comparison.md)
- [Repository tree and file/cache rules](docs/repository.md)

## Build and test

Use Python 3.10+ and GCC or Clang with C++17. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/build.py
python tests/test_pricing.py -v
python tests/test_lsm_procedure.py -v
python tests/test_experiment_files.py -v
```

Reuse an existing working `.venv`. On Linux/macOS activate `.venv/bin/activate`.
`build.py` is primary; `--compiler PATH` selects a compiler. CMake is optional.

## Research runs

Preserved runs live in `data/experiments/STUDY/full/`. Runners refuse to overwrite
research directories. Choose a fresh name for another full run:

```text
python scripts/run_lsm_paper.py --output-dir data/experiments/paper_replication/rerun
python scripts/plot_lsm_paper.py --data-dir data/experiments/paper_replication/rerun
python scripts/run_lsm_comparison.py --output-dir data/experiments/lsm_comparison/rerun
python scripts/plot_lsm_comparison.py --data-dir data/experiments/lsm_comparison/rerun
```

Both full studies use 100000 paths per sample and 50 exercise dates/year.
American CRR references use 100000 **time steps**, with every-tree-date and
matched-date exercise. Cold trees are expensive; compatible references are
reused from `.cache/crr/`. Contract, grid or pricing-source changes invalidate
entries. Historical tables remain evidence, not an unchecked cache.

Small installation checks are separate:

```text
python scripts/run_lsm_comparison.py --quick
python scripts/plot_lsm_comparison.py --quick
```

Smoke data and figures go under ignored `tmp/smoke/STUDY/`; they cannot target
research results or figures. The broader study uses `run_experiments.py` and
`plot_results.py`, with the same protections.

## Files

`src/`, `include/`, `scripts/` and `tests/` are maintained code.
`data/reference/longstaff_schwartz_2001/` is the canonical published-data bundle.
`data/experiments/` and `figures/` retain generated evidence by study and run.
`docs/` contains methodology, interpretation and organization; `archive/` and
`extensions/` preserve historical work. `build/`, `.cache/` and `tmp/` are local.

Notebooks are optional: install `requirements-notebooks.txt` if needed.
They read CSVs and invoke C++, without duplicating numerical pricing algorithms.
Executed notebook copies default to `tmp/notebooks/`.
