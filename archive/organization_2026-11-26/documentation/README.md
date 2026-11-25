# Numerical option pricing: C++ calculations, Python visualization

Six C++17 programs simulate stock paths, fit exercise policies, calculate prices
and standard errors, and export CSV. Python builds and runs the programs, collects
the exported results, and creates PNG/SVG figures. The main reproduction is the
20-case American-put benchmark in Longstaff and Schwartz (2001), with published
prices and standard errors and an additional 100,000-step CRR reference.

Start with [the workflow guide](docs/quickstart.md). See
[the benchmark findings](docs/lsm_benchmark.md) for measured results and
[the methodology](docs/methodology.md) for equations and interpretation.

The current controlled comparison adds plain LSM alongside AV-LSM through one
normalized algorithm. Read [the equations, function map and eight-path walkthrough](docs/lsm_implementation.md)
and [the new comparison results](docs/lsm_comparison.md). Existing datasets are
preserved; new runs and figures use separate `lsm_comparison/` subdirectories.

```text
python scripts/build.py
python tests/test_pricing.py -v
python tests/test_lsm_procedure.py --example-dir data/experiments/lsm_comparison/example -v
python scripts/run_lsm_comparison.py
python scripts/plot_lsm_comparison.py
```

## Reproduce the LSM benchmark

Requirements: Python 3.10+, a C++17 compiler, and the plotting packages in
[`requirements.txt`](requirements.txt). C++ pricing and numerical tests have no
third-party library dependencies. From the repository root:

```text
python -m pip install -r requirements.txt
python scripts/build.py
python tests/test_pricing.py -v
python tests/test_lsm_procedure.py -v
python scripts/run_lsm_paper.py
python scripts/plot_lsm_paper.py
```

On Windows, a local environment can be created and used with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/build.py
```

The build script finds `g++` or `clang++` on `PATH`; `--compiler PATH` selects
one explicitly. It writes optimized executables and compiler/source metadata
to `build/`. Use `--debug` for an unoptimized diagnostic build.

The full paper runner uses:

- `K=40`, `r=0.06`, no dividends; initial stock prices 36, 38, 40, 42, 44;
  volatilities 0.20 and 0.40; maturities one and two years.
- 100,000 paths per Table 1 case, comprising 50,000 antithetic pairs, seed 42,
  and 50 exercise dates per year.
- In-sample LSM with a constant and three weighted Laguerre functions of `S/K`;
  regressions use only in-the-money paths and discounted realized cashflows.
- 100,000-step CRR references with exercise at every tree date and with exercise
  restricted to the same 50 dates per year as LSM. C++ calculates each contract's
  references once; the runner reuses them for repeated simulations.
- A one-year at-the-money convergence study at 5,000, 10,000, 25,000, 50,000,
  and 100,000 paths, using 30 seeds (42-71) per budget.

Here **100,000 means time steps**: the CRR tree has 100,001 terminal nodes and
5,000,150,001 nodes in total. It requires quadratic computation, so the full
reference calculation is substantially slower than the simulations. A rolling
vector keeps memory linear in the number of steps.

`--workers` defaults to 2; `--replicates` defaults to 30; `--reference-steps`
defaults to 100000. Use `--quick` for a smoke run with 10,000 paths per table
case, 2,000-step references, two convergence seeds, and three path budgets.
Quick results do not reproduce the paper's sample budget. Choose `--output-dir`
to keep a smoke run separate from curated full results.

## Results and uncertainty

The runner writes these files under `data/experiments/`:

| File | Contents |
|---|---|
| `lsm_paper_table.csv` | Our 20 prices, uncertainty diagnostics, published Table 1 values, and CRR comparisons |
| `lsm_paper_convergence.csv` | Individual repeated-seed results by path budget |
| `lsm_paper_se_summary.csv` | Across-seed dispersion compared with reported standard errors |
| `lsm_paper_summary.json` | Aggregate benchmark comparisons |
| `lsm_paper_manifest.json` | Parameters, random seeds, build metadata, and reference provenance |

The plotting script reads these CSVs and creates `lsm_paper_table`,
`lsm_paper_convergence`, and `lsm_paper_standard_errors` figures in `figures/`,
each as PNG and SVG.

The primary within-run standard error uses antithetic **pair averages**. A
separate diagnostic calculates the marginal path standard deviation divided by
the square root of the path count, for comparison with the paper's printed
errors. The paper does not state its standard-error formula, random generator,
or seeds. Matching a printed error supports a numerical comparison; the
original formula remains unreported. Repeated simulations measure the
variability of the complete fitted LSM estimator across seeds.

The [published reference CSV](data/reference/longstaff_schwartz_table1.csv) was
transcribed and visually verified from Table 1, journal page 127. Its prices
and errors are printed to three decimal places. The paper's reference uses an
implicit finite difference scheme; our CRR references are additional numerical
comparisons. The [source record](data/reference/longstaff_schwartz_table1_provenance.json)
documents this distinction and the unreported implementation details.

## Six pricing methods

| C++ source in `src/` | Purpose |
|---|---|
| `binomial_european.cpp` | European CRR tree, checked against Black-Scholes |
| `binomial_american.cpp` | CRR tree with early exercise |
| `monte_carlo_european.cpp` | Exact-terminal GBM simulation and sampling uncertainty |
| `monte_carlo_american.cpp` | Simulation of a prescribed exercise policy |
| `variance_reduction.cpp` | Plain European MC, antithetic/control variates, and Newton importance sampling |
| `lsm_american.cpp` | Common weighted-Laguerre LSM; antithetic/in-sample by default; explicit plain sampling and independent evaluation |

The model assumes constant risk-neutral GBM parameters, continuous compounding,
and no dividends. Black-Scholes is the European reference. A fine CRR tree is
an American numerical reference. More simulation paths reduce sampling noise;
they do not remove exercise-grid or policy-approximation error.

Each executable accepts `--help` and writes CSV to standard output or
`--output PATH`. For example:

```powershell
.\build\lsm_american.exe --lsm-mode paper --type put --spot 40 --strike 40 `
    --rate 0.06 --volatility 0.20 --maturity 1 --steps 50 --paths 100000 `
    --reference-steps 100000 --output data/custom/lsm.csv
```

On macOS/Linux use `./build/lsm_american`. The executable's ordinary
`--reference-steps` default remains 2000; the full paper runner selects 100000.

The broader six-method study remains available through
`python scripts/run_experiments.py --quick` followed by
`python scripts/plot_results.py`. This produces the European convergence,
American policy, variance reduction, and exercise-grid figures.

## Build with CMake

```text
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake --config Release
ctest --test-dir build/cmake -C Release --output-on-failure
```

Pass `--build-dir build/cmake` to a runner for single-configuration generators,
or `--build-dir build/cmake/Release` for Visual Studio. Numerical validation
checks pricing identities, tree convergence, simulation uncertainty, LSM
cashflow decisions, reproducibility, edge cases, and input validation.

## Repository layout

```text
src/                     Six C++ pricing entry points
include/                 Shared pricing, statistics, command-line and CSV code
scripts/                 Build, collect C++ output, and plot CSVs
tests/                   Numerical and input-validation checks
data/reference/          Published Table 1 values and provenance
data/experiments/        Curated calculation results and manifests
figures/                 Generated PNG and SVG figures
docs/                    Workflow, methodology, and benchmark findings
extensions/              Separate historical research implementations
```

The separate importance-sampling extension is retained as historical research.
A single `.venv` supplies the Python packages for the CSV/plot workflow.
