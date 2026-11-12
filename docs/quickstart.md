# How the project works

The six C++ files are **six independent programs**. Each one prices a contract
with a different method. They are compiled separately and can run in any order.
Every program supports both `--type call` and `--type put`.

Python automates repeated runs and makes charts from the C++ results.

```text
src/*.cpp + include/*.hpp
          |
          | scripts/build.py compiles each program
          v
build/*.exe
          |
          | scripts/run_experiments.py runs different settings and seeds
          v
data/experiments/*.csv
          |
          | scripts/plot_results.py draws charts
          v
figures/*.png and *.svg

notebooks/option_pricing.ipynb reads the CSVs and figures,
explains the results, and can call an executable for a custom contract.
```

## The six programs

| C++ file | What it does | How it relates to the others |
|---|---|---|
| `binomial_european.cpp` | European pricing by CRR backward induction | Compare with European MC and Black-Scholes |
| `binomial_american.cpp` | CRR backward induction with early exercise | A fine tree provides the numerical reference for both American simulation programs |
| `monte_carlo_european.cpp` | European pricing by exact terminal-stock simulation | Baseline for the variance reduction study |
| `monte_carlo_american.cpp` | Simulates a prescribed exercise rule | Simple American baseline; its expected policy value is a lower bound |
| `variance_reduction.cpp` | Runs plain European MC, antithetic/control variates, and basic Newton importance sampling | Fits control and IS parameters on independent pilots, then compares sampling errors |
| `lsm_american.cpp` | Learns an exercise rule, then values it on independent paths | More flexible American simulation method; compare with the fixed rule and American tree |

The American simulation programs calculate their tree references by calling
shared C++ functions. They do not launch `binomial_american.exe`.
Similarly, the variance reduction program contains its own plain-MC estimator;
it does not launch `monte_carlo_european.exe`.

## The supporting files

| File | Role |
|---|---|
| `include/model.hpp` | Shared contract parameters, call/put payoffs, Black-Scholes, random streams, GBM paths, and sample statistics |
| `include/binomial.hpp` | Shared CRR tree algorithm, including American numerical references |
| `include/experiment.hpp` | Reads command-line parameters, measures runtime, and writes consistent CSV output |
| `scripts/build.py` | Compiles the six programs into `build/` |
| `scripts/run_experiments.py` | Calls those executables with different resolutions, path counts, and seeds; saves CSV datasets |
| `scripts/plot_results.py` | Reads the datasets and produces five figures, including Newton convergence |
| `scripts/verify_notebook.py` | Executes the notebook and saves its outputs |
| `scripts/run_lsm_paper.py` | Reproduces the 20 paper Table 1 put cases and a sample-size convergence study |
| `scripts/plot_lsm_paper.py` | Plots the paper-mode parameter grid and LSM path convergence |
| `tests/test_pricing.py` | Runs numerical and input-validation checks against the compiled programs |
| `docs/methodology.md` | Explains the mathematics and interpretation of the results |

Headers (`.hpp`) are included when compiling. They are not programs to run.
Black-Scholes lives in the shared model header, so it does not need a seventh
C++ source file. Each `.cpp` has its own `main()`; compile each separately.

## Run the complete workflow in this Windows workspace

The project environment and example results have already been created. Open
PowerShell and run:

```powershell
Set-Location "D:\Projects\Capstone Thesis"
$pricingPython = ".\.venv-portfolio\Scripts\python.exe"

# 1. Build the six C++ programs.
& $pricingPython scripts/build.py

# 2. Run the quick convergence and comparison experiments.
& $pricingPython scripts/run_experiments.py --quick

# 3. Generate the figures from those experiment results.
& $pricingPython scripts/plot_results.py
```

Read the generated PNGs in `figures/`, or open
`notebooks/option_pricing.ipynb` in VS Code and select the
`.venv-portfolio` Python interpreter. The notebook has saved outputs for review.
For a fresh clone or another computer, follow the environment setup in the README.

## Run just one pricing method

After building, choose any of these commands:

```powershell
.\build\binomial_european.exe --type put --steps 500
.\build\binomial_american.exe --type put --steps 500
.\build\monte_carlo_european.exe --type put --paths 100000
.\build\monte_carlo_american.exe --type put --paths 100000 --steps 50
.\build\variance_reduction.exe --type put --paths 100000 --train-paths 5000
.\build\lsm_american.exe --type put --train-paths 50000 --paths 100000 --steps 50
```

These print CSV. To show the main fields as a readable PowerShell table:

```powershell
.\build\monte_carlo_european.exe --type put --paths 100000 |
    ConvertFrom-Csv |
    Format-Table method, price, standard_error, benchmark
```

Use `--help` on any executable for its options. Append
`--output data/custom/my_result.csv` to save a result to a file.

The existing independently trained/evaluated LSM remains the default. To run
its paper-reproduction mode for the canonical Table 1 case, use:

```powershell
.\build\lsm_american.exe --lsm-mode paper --type put --spot 40 --strike 40 `
    --rate 0.06 --volatility 0.20 --maturity 1 --steps 50 --paths 100000
```

Paper mode uses antithetic pairs, the constant plus weighted `L0`, `L1`, and
`L2` Laguerre basis, and in-sample policy valuation. Run the 20-case table and
the 1-year at-the-money sample-size study with `scripts/run_lsm_paper.py`,
followed by `scripts/plot_lsm_paper.py`. `--quick` is for smoke testing only.

## What to change, and when to rerun

- **Change the contract:** use `--spot`, `--strike`, `--rate`, `--volatility`,
  `--maturity`, and `--type`. You can also edit the notebook's exploration cell.
- **Refine a tree:** increase `--steps`.
- **Increase the MC sample:** increase `--paths`.
- **Increase LSM training data:** increase `--train-paths`.
- **Increase control-variate or IS pilot data:** use `--train-paths` on `variance_reduction.exe`.
- **Change the plot appearance:** edit `scripts/plot_results.py`.

Changing command-line values does not require compilation. After editing C++
or a shared header, rebuild. To update the full figures after a pricing change,
run experiment collection again, then plotting. A single custom run does not
replace the convergence datasets or automatically update the figures.

To save the Newton iterations for importance sampling:

```powershell
.\build\variance_reduction.exe --type call --paths 100000 --train-paths 5000 --newton-trace data/custom/newton.csv
```

The basic Newton updates give an approximate shift from the direct pilot
second moment; the shift is then plugged into the pricing formula. It is learned independently of the evaluation paths. Both
pilot-based methods include their additional payoff evaluations and fitting
time. The saved `variance_summary.csv` gives the measured reductions; the
`newton_convergence.csv` file and figure show how the optimizer converged.

To run the numerical checks or refresh the notebook outputs:

```powershell
& $pricingPython tests/test_pricing.py -v
& $pricingPython scripts/verify_notebook.py
```
