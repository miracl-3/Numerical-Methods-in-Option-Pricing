# Longstaff--Schwartz American put experiment

This is the consolidated report workspace for the LSM-only numerical study.
The valuation, least-squares regressions, stopping decisions, and CRR
benchmark are implemented in the standalone [`lsm_american.cpp`](./lsm_american.cpp).
[`run_report.py`](./run_report.py) runs the experiments and creates report
tables and figures. [`report.tex`](./report.tex) is the report fragment.
The report mentions the Tsitsiklis--Van Roy recursion only as mathematical
context; no TVR code, numerical results, tables, or figures are included.

## Experiment settings

The paper-parameter table covers all 20 combinations:

- Strike \(K=40\), continuously compounded rate \(r=6\%\).
- Initial spot \(S_0\in\{36,38,40,42,44\}\).
- Volatility \(\sigma\in\{20\%,40\%\}\).
- Maturity \(T\in\{1,2\}\) years, with 50 exercise dates per year.
- 100,000 paths per contract case, seed 42, run once with plain MC and once
  with antithetic variates.
- Basis in the unnormalized stock-price state \(X=S\):
  \[
  \left\{1,\ e^{-X/2},\ e^{-X/2}(1-X),\
  e^{-X/2}\left(1-2X+\frac{X^2}{2}\right)\right\}.
  \]
- The stock state is used directly, with no division by strike. The C++
  regression solver rescales columns only internally for numerical stability
  and transforms fitted coefficients back; this does not change the
  regression basis or stock state.
- Benchmark: a 2,000-step Bermudan CRR tree, with exercise dates aligned to
  the Monte Carlo exercise grid. This is a numerical benchmark, not the
  finite-difference reference in the original Longstaff--Schwartz paper.
- The paper's finite-difference values use 40,000 time steps per year and
  1,000 stock-price steps. The report tables show the published LSM values
  alongside this study's estimates and CRR benchmark.

The separate convergence experiment uses the at-the-money one-year put,
volatilities 20% and 40%, path counts 5,000 through 100,000, and 30 seeds
(42--71) for each sampling scheme. The boundary visualization uses 100,000
paths and shows 32 paths per panel.

Antithetic standard errors are calculated from the independent pair-average
payoffs; plain-MC standard errors use individual path payoffs. The paper
table standard errors are in-sample diagnostics because the same paths fit
and evaluate the exercise policy. The convergence intervals instead show
run-to-run variation across 30 seeds.

The current benchmark intentionally evaluates the Laguerre basis at the
unscaled stock price \(S\), per the requested direct-state comparison. This
produces LSM estimates materially below the published Table 1 LSM values in
these runs; the paper comparison table and CRR reference expose that
discrepancy rather than hiding it with state normalization. The estimated
exercise boundary is consequently diagnostic and should not be interpreted
as an accurate economic boundary.

## Build and run

From PowerShell in this directory:

```powershell
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic lsm_american.cpp -o lsm_american.exe
..\..\.venv-portfolio\Scripts\python.exe run_report.py
```

The repository virtual environment supplies NumPy, pandas, and Matplotlib.
The runner writes:

- `paper_table.csv` and `paper_tables.tex`: published LSM values alongside
  native plain/AV estimates, standard errors, CRR references, and comparison
  errors for all 20 cases. The paper's finite-difference reference is
  documented in the report but not included in this simulation comparison.
- `convergence_replicates.csv` and `convergence_summary.csv`: raw seeded
  runs and 30-seed interval summaries.
- `lsm_confidence_intervals.png` / `.svg`: estimate intervals by path count.
- `exercise_boundary.csv`, `sample_paths_plain.csv`, and
  `sample_paths_antithetic.csv`: visualization data.
- `lsm_exercise_boundary.png` / `.svg`: fitted boundaries over simulated
  sample paths.

The report fragment inputs `paper_tables.tex` and the two PNG figures.
Include it in a LaTeX document with `amsmath`, `booktabs`, `graphicx`, and
`hyperref` available.
