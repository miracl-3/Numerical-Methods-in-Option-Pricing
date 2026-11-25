# Common LSM: equations, code and deterministic walkthrough

The implementation keeps the six executables and existing directories. The
LSM functions remain in `src/lsm_american.cpp`; exact GBM path generation is
shared in `include/model.hpp`. Python orchestrates experiments, summarizes
repeated runs, and plots the C++ CSVs. No American control variate or importance
sampling estimator is introduced. The existing same-path European premium
columns are diagnostics; they never adjust the American price.

The method follows [Longstaff and Schwartz (2001)](https://www.kellogg.northwestern.edu/faculty/mcdonald/ftp/practicum/papers/longstaffschwartz.pdf),
especially the worked example, Tables 1-2 and the strike normalization on p. 143.

## Inputs and independent choices

The contract supplies `spot`, `strike`, `rate`, `volatility`, `maturity`, and
`type`. `steps` is the number of exercise intervals. The paper experiments use
`K=40`, `r=.06`, zero dividends, and 50 dates/year plus time zero. Parameter
grids and paper constants come from `run_lsm_paper.py` and the reference CSVs.

| Choice | Values | Default |
|---|---|---|
| Sampling | `--sampling plain` or `antithetic` | antithetic |
| Valuation | `--valuation in-sample`, `out-of-sample`, or `both` | in-sample |
| Basis in the numerical experiments | Constant plus three weighted Laguerre functions | Same for both samplers |
| Training size | `--train-paths` | Equals `--paths` in in-sample mode |
| Evaluation size | `--paths` | 100000 |
| Random streams | `stream_seed(seed,1)` for fitting; stream 2 for independent evaluation | Master seed 42 |

`both` fits once using `--train-paths`, reports its in-sample valuation, then
values the frozen policy on `--paths` new paths. Pass 100000 for both counts
to run the Table 2 diagnostic. The legacy `paper` mode preserves the AV/in-sample
defaults. The legacy `independent` mode selects plain/out-of-sample unless
explicit sampling or valuation options override it. All modes now use the
same normalized Laguerre algorithm.

## Equations and C++ functions

Let dates be $t_j=j\Delta t$, $j=0,\ldots,M$, with $\Delta t=T/M$.
Write $g(S)=\max(K-S,0)$ for a put (or the corresponding call payoff).

1. `simulate_paths` in `include/model.hpp` generates

   $$S_{i,j+1}=S_{i,j}\exp[(r-\sigma^2/2)\Delta t+\sigma\sqrt{\Delta t}Z_{i,j}].$$

   Plain sampling uses independent Gaussian increments. AV sampling draws one
   increment per pair and uses $Z$ and $-Z$. The transition formula, storage,
   exercise dates and all downstream operations are shared. The existing
   four-argument function still selects plain paths, preserving its callers.

2. `normalized_payoff` and `basis_values` use $x=S/K$, $h(S)=g(S)/K$ and

   $$\phi(x)=(1,e^{-x/2},e^{-x/2}(1-x),e^{-x/2}(1-2x+x^2/2)).$$

3. `train_policy` initializes $Y_i=h(S_{i,M})$ and $\tau_i=M$. At each date
   $j=M-1,\ldots,1$, it first sets $Y_i\leftarrow e^{-r\Delta t}Y_i$.
   On ITM paths, `fit_continuation` solves

   $$\hat\beta_j=\arg\min_\beta\sum_{i:g(S_{i,j})>0}
   (Y_i-\phi(S_{i,j}/K)^\top\beta)^2.$$

   Reorthogonalized modified Gram-Schmidt QR solves this least-squares problem.
   No normal equations are formed. Basis values are computed once per ITM row.

4. `Regression::continuation` returns the normalized prediction

   $$\hat C_j(x)=\max(0,\phi(x)^\top\hat\beta_j).$$

   Clipping a negative prediction to zero does not change the exercise choice
   for a strictly positive ITM payoff. `should_exercise` compares **normalized
   payoff with normalized continuation**. If $h(S_{i,j})>\hat C_j(S_{i,j}/K)$,
   `train_policy` sets $Y_i=h(S_{i,j})$ and $\tau_i=j$.

   The target is the selected future realized cashflow, not the fitted
   continuation value. Targets on surviving paths keep their realized values.

5. At time zero, `train_policy` computes

   $$\hat C_0=e^{-r\Delta t}\frac1{N_{train}}\sum_iY_i.$$

   It compares $h(S_0)$ with $\hat C_0$, freezes that decision in the policy,
   and exports $K e^{-r\Delta t}Y_i$ or the immediate dollar payoff. This gives
   LSM the same time-zero opportunity as CRR. It does not change the paper's
   20-case results, where immediate exercise is inferior.

6. `evaluate_policy` applies those coefficients and the frozen time-zero
   decision to supplied independent paths. It chooses the first exercise date
   and returns $A_i=e^{-rt_{\tau_i}}g(S_{i,\tau_i})$. It does not refit models,
   rechoose time-zero exercise from the test mean, or compare dollar exercise
   values with normalized coefficients. In-sample and out-of-sample valuation
   use the same `should_exercise` function.

7. `summarize_cashflows` computes $\hat V=N^{-1}\sum_i A_i$. For plain paths,

   $$\widehat{SE}=s_A/\sqrt N.$$

   For AV, it forms $B_k=(A_{2k-1}+A_{2k})/2$, $m=N/2$, and computes

   $$\widehat{SE}_{AV}=s_B/\sqrt m.$$

   In-sample estimates are plug-in approximations: fitted regressions couple
   cashflows across paths. Out-of-sample SEs condition on the frozen policy.
   Repeated full fits assess variability including policy estimation.

## Pseudocode and cost

```text
paths = exact_GBM(contract, training_count, dates, sampling, training_seed)
cashflow = payoff_at_maturity / strike; stopping_time = maturity_index
for date = last_exercise_date down to first_positive_date:
    cashflow *= discount_one_interval
    fit QR on ITM stock/strike and current cashflow
    if normalized exercise > normalized fitted continuation:
        cashflow = normalized exercise; stopping_time = date
discount one more interval; make time-zero decision from training mean
report in-sample cashflows if requested; release training path matrix
if independent evaluation requested:
    generate paths with a different stream
    follow the frozen policy forward; discount selected realized payoffs
group observations by path or antithetic pair; calculate price and SE
```

For $p=4$, training takes $O(MNp^2)$ time and $O(MN+Np+Mp)$ memory.
Forward valuation takes $O(MN)$ time. Training paths are released before an
evaluation path matrix is allocated; both matrices are never retained together.
CRR remains its validated $O(N_{tree}^2)$ algorithm. The new experiment runner
reuses saved fine-tree values instead of recomputing them for every seed.

## Regression diagnostics and fallback behavior

`--regression-diagnostics PATH` writes one row per positive regression date:
ITM sample count, retained rank, coefficients, QR diagonal extrema, status, and
`condition_inf_retained_R`. The latter is $\|R\|_\infty\|R^{-1}\|_\infty$
for the retained triangular factor. It is a conditioning diagnostic, not an
SVD condition number of the full original matrix. A rank-deficient full basis
is not declared well-conditioned merely because its retained subset is.

- No ITM observations: `no_itm`, rank zero; continue without a fitted exercise rule.
- Fewer observations than basis columns: `constant_fallback`, rank one, mean target.
- Numerically dependent columns: `rank_deficient`, retaining the leading independent columns.
- Otherwise: `full_rank`, with the actual conditioning diagnostic still recorded.

The QR rank threshold is $10^{-10}\sqrt{N_{ITM}}$. Aggregates appear in each
pricing row. The known optimal hold-to-maturity rule for non-dividend calls
with nonnegative rates is marked `optimal_hold_call`; no regression is fitted.
The weighted Laguerre columns can be highly correlated on narrow ITM ranges.
Tests therefore check accurate predictions in a deliberately ill-conditioned
case as well as recovery of coefficients in a better-conditioned case.

## CSV accounting

`sampling_method`, `valuation_mode`, `basis`, and `standard_error_scope` identify
the estimator. `paths`/`evaluation_paths` count priced paths. `training_paths`
counts paths used to fit the policy. `antithetic_pairs` and
`training_antithetic_pairs` count pairs only for AV (zero for plain).
`independent_samples` counts input path/pair groups; fitted in-sample outputs
are not literally independent. Exact integer training/evaluation seeds are
recorded separately.

In sample, fitting and valuation share paths: `unique_simulated_paths=N`, not
`2N`. Out of sample it is `Ntrain+Neval`. In `both` mode the two rows share one
training sample, so summing their counts or runtimes would double-count it.
`training_runtime_ms` includes training simulation, fitting and any requested
in-sample summary; `valuation_runtime_ms` records independent evaluation.
`runtime_ms` excludes separately measured references and pricing/regression CSV
output. Optional backward-trace writes occur inside the training pass and are
included in its time; the controlled experiments do not enable tracing.

The new runner preserves raw C++ `benchmark`/`reference_steps` for its cheap
local tree. `comparison_reference_crr` and `comparison_reference_steps` identify
the saved fine American CRR on the same exercise dates; the manifest hashes
its source CSV. Original experimental results and reference datasets are not
overwritten.

## The paper's deterministic eight paths

The paper's worked example uses $1,S,S^2$, unlike the four-column Laguerre
basis used in Tables 1-2. `Basis::Quadratic` is retained only for that learning
fixture; neither sampling method in the controlled experiments uses it.
It still uses the common normalized cash-flow update and valuation functions.

For $K=1.1$, $r=.06$, dates 1, 2, 3, the fitted continuations, converted back
to dollar polynomials in raw stock price, are:

| Date | Constant | Stock coefficient | Stock-squared coefficient |
|---|---:|---:|---:|
| 2 | -1.069987655 | 2.983410626 | -1.813576183 |
| 1 | 2.037512342 | -3.335443403 | 1.356456588 |

Selected intermediate rows in dollars (CSV values are normalized by 1.1):

| Date/path | Stock | Discounted future target | Fitted continuation | Immediate payoff | Decision |
|---|---:|---:|---:|---:|---|
| 2 / 3 | 1.07 | 0.065924 | 0.045898 | 0.030000 | Continue to date 3 |
| 2 / 4 | 0.97 | 0.169518 | 0.117527 | 0.130000 | Select date 2 payoff |
| 1 / 4 | 0.93 | 0.122429 | 0.108749 | 0.170000 | Replace date 2 by date 1 |
| 1 / 8 | 0.88 | 0.000000 | 0.152762 | 0.220000 | Select date 1 payoff |

Path 4 illustrates why the future realized payoff is a regression target,
not an exercise oracle: the decision compares immediate payoff with fitted
continuation, although its realized future target happens to be larger.
Paths 4, 6, 7, 8 finally stop at date 1; path 3 receives 0.07 at date 3;
paths 1, 2, 5 receive zero. Time-zero continuation is 0.11443433, exceeding
the immediate payoff 0.1. Therefore

$$\hat V=(0.91e^{-.06}+0.07e^{-.18})/8=0.114434330045.$$

Generate the complete trace, coefficients and final cash flows with:

```text
python tests/test_lsm_procedure.py --example-dir data/experiments/lsm_comparison/example -v
```

## Run and test

```text
python scripts/build.py
python tests/test_pricing.py -v
python tests/test_lsm_procedure.py -v
python scripts/run_lsm_comparison.py
python scripts/plot_lsm_comparison.py
```

Use `--quick --output-dir data/custom/lsm_comparison_smoke` for a separate
smoke run. A direct independent policy evaluation is:

```text
build/lsm_american.exe --sampling antithetic --valuation both --spot 40 --strike 40 --rate 0.06 --volatility 0.2 --maturity 1 --steps 50 --train-paths 100000 --paths 100000 --regression-diagnostics data/custom/qr.csv
```

The [measured comparison](lsm_comparison.md) records results and remaining
statistical limitations. On systems without GCC/Clang the Python procedure
fixture reports a skip; local validation here compiled and executed it.
