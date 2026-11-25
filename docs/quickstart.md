# Run the C++ to CSV to Python workflow

C++ performs the numerical work: stock-path generation, regression, cashflow
decisions, CRR backward induction, prices, standard errors, and CSV export.
Python compiles and launches the programs, collects their rows, and visualizes
the exported results.

```text
src/*.cpp + include/*.hpp
            |
            | scripts/build.py
            v
      build executables
            |
            | scripts/run_lsm_paper.py
            v
  data/experiments/lsm_paper_*.csv
            |
            | scripts/plot_lsm_paper.py
            v
     figures/*.png and *.svg
```

## Set up and run

Use Python 3.10+ and a C++17 compiler. In this Windows workspace:

```powershell
Set-Location "D:\Projects\Capstone Thesis"
python -m venv .venv
$pricingPython = ".\.venv\Scripts\python.exe"
& $pricingPython -m pip install -r requirements.txt
& $pricingPython scripts/build.py
& $pricingPython tests/test_pricing.py -v
& $pricingPython tests/test_lsm_procedure.py -v
& $pricingPython scripts/run_lsm_paper.py
& $pricingPython scripts/plot_lsm_paper.py
```

Reuse an existing environment instead of creating it again. The build script
finds GCC or Clang; pass `--compiler PATH` when the compiler is not on `PATH`.
For MSVC, use the CMake commands in the README and pass the resulting executable
directory as `--build-dir` to the runners and tests.

The full paper run uses 100,000 paths for each of 20 contracts and 100,000-step
CRR references. The latter are expensive: 100,000 steps mean 100,001 terminal
nodes and 5,000,150,001 total tree nodes. C++ computes American CRR reference
values at every tree date and with matched exercise dates, then reuses them
for repeated runs.
The runner defaults to two worker processes.

For a short installation check, save separate outputs:

```powershell
& $pricingPython scripts/run_lsm_paper.py --quick --output-dir data/custom/lsm_smoke
```

`--quick` uses 10,000 table paths and 2,000 reference steps, with a smaller
convergence study. Run the default profile to use the paper's 100,000-path
budget. `--replicates 30` controls the convergence seeds; the default study
uses seeds 42-71 at five path budgets. `--reference-steps` changes tree
resolution, and `--workers` changes concurrency.

## Read the results

Open the generated `figures/lsm_paper_table.png`,
`figures/lsm_paper_convergence.png`, and
`figures/lsm_paper_standard_errors.png`.
The [benchmark findings](lsm_benchmark.md) explain the measured comparisons.
The underlying CSVs and JSON summaries are under `data/experiments/`.

The table compares our LSM prices against both the published simulation and
published finite difference values. It also reports American CRR values at
every tree date and with the same exercise dates as LSM. These references help
separate exercise-date effects from simulation and policy errors.

Standard-error comparisons distinguish antithetic pair-based uncertainty,
the marginal path-level diagnostic, and the empirical dispersion across
independently seeded complete LSM runs. See [the methodology](methodology.md)
for formulas and limitations.

## The six programs

| Source | Calculation |
|---|---|
| `binomial_european.cpp` | European CRR value |
| `binomial_american.cpp` | CRR value with early exercise |
| `monte_carlo_european.cpp` | European exact-terminal Monte Carlo |
| `monte_carlo_american.cpp` | American prescribed-policy Monte Carlo |
| `variance_reduction.cpp` | Plain European MC, antithetic/control variates, and importance sampling |
| `lsm_american.cpp` | Longstaff-Schwartz exercise-policy regression and valuation |

Each file builds as its own executable. Shared numerical functions live in
`include/`; programs call them directly. The executables do not launch one
another. The standard LSM executable trains and evaluates on independent paths
by default. The paper runner selects `--lsm-mode paper` for the original
in-sample Table 1 procedure.

Run a single paper-mode contract after building:

```powershell
.\build\lsm_american.exe --lsm-mode paper --type put --spot 40 --strike 40 `
    --rate 0.06 --volatility 0.20 --maturity 1 --steps 50 --paths 100000 `
    --reference-steps 100000 --output data/custom/lsm.csv
```

Paper mode uses 50,000 antithetic pairs and a constant plus three weighted
Laguerre functions of `S/K`. The same paths fit and value the stopping rule.
For the independent mode, omit `--lsm-mode paper` and set `--train-paths`
separately from `--paths`.

Any executable can print a row for inspection:

```powershell
.\build\monte_carlo_european.exe --type put --paths 100000 |
    ConvertFrom-Csv |
    Format-Table method, price, standard_error, benchmark
```

Use `--help` for all accepted parameters. Changing command-line values does
not require compilation; editing C++ sources or headers does.

## Supporting files and other experiments

| File | Role |
|---|---|
| `include/model.hpp` | Contracts, payoffs, Black-Scholes, GBM paths, random streams, statistics |
| `include/binomial.hpp` | Shared CRR calculation |
| `include/experiment.hpp` | Command-line parameters, runtime accounting, CSV fields |
| `scripts/build.py` | Compilation and build metadata |
| `scripts/run_lsm_paper.py` | Table 1 reproduction, reference reuse, and repeated-seed convergence |
| `scripts/plot_lsm_paper.py` | Figures from exported LSM CSVs |
| `data/reference/` | Visually verified published Table 1 values and source record |
| `tests/test_pricing.py` | Numerical and input-validation checks |

The broader European/American comparison is still available:

```powershell
& $pricingPython scripts/run_experiments.py --quick
& $pricingPython scripts/plot_results.py
```

For an importance-sampling Newton trace, append
`--newton-trace data/custom/newton.csv` to a `variance_reduction` command.
The separate historical implementation under `extensions/` is retained.
Use the project's single `.venv` for the build, experiment, and plotting scripts.
