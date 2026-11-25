# Numerical methodology

The underlying is a non-dividend-paying stock under the risk-neutral measure:

$$dS_t=rS_t\,dt+\sigma S_t\,dW_t,\qquad q=0.$$

All rates are continuously compounded, volatility is annualized, and maturity is
measured in years. Parameters are constant. Call and put payoffs are
$(S-K)^+$ and $(K-S)^+$. This is a numerical study under a specified model;
there is no market calibration or empirical trading-performance claim.

## Equations and their implementations

| Mathematical operation | C++ function / location |
|---|---|
| Contract payoff | `Contract::payoff`, `include/model.hpp` |
| Black-Scholes value and Gaussian CDF | `black_scholes`, `normal_cdf`, `include/model.hpp` |
| Exact terminal GBM and European MC | `terminal_stock`; `european_monte_carlo`, `src/monte_carlo_european.cpp` |
| Exact pathwise GBM, plain or antithetic | `simulate_paths`, `include/model.hpp` |
| CRR continuation and exercise maximum | `binomial_price`, `include/binomial.hpp` |
| Sample variance, mean and SE | `Moments::add`, `variance`, `estimate`, `include/model.hpp` |
| Prescribed American stopping policy | `fixed_policy_monte_carlo`, `src/monte_carlo_american.cpp` |
| European AV/CV/IS estimators | `estimate_method`, `fit_control_beta`, `src/variance_reduction.cpp` |
| LSM basis, QR and normalized decisions | `basis_values`, `fit_continuation`, `should_exercise`, `src/lsm_american.cpp` |
| LSM policy estimation and valuation | `train_policy`, `evaluate_policy`, `summarize_cashflows`, `src/lsm_american.cpp` |

The C++ algorithms produce prices and within-run SEs. Python performs orchestration,
across-run statistics and visualization. Its CRR cache stores C++ output and does
not implement backward induction. `binomial_american --exercise-dates M` exports
both every-tree-date and matched-date American values through the same existing
`binomial_price` function. The primary build remains `scripts/build.py`.

## European benchmark

Black-Scholes provides the closed-form reference:

$$C=S_0\Phi(d_1)-Ke^{-rT}\Phi(d_2),\qquad
P=Ke^{-rT}\Phi(-d_2)-S_0\Phi(-d_1),$$

$$d_1=\frac{\log(S_0/K)+(r+\sigma^2/2)T}{\sigma\sqrt T},\qquad
d_2=d_1-\sigma\sqrt T.$$

Expiry is handled directly by intrinsic value. When volatility is zero, the
European price is the discounted payoff of the deterministic terminal stock.

## CRR trees

For $N$ intervals, $\Delta t=T/N$,
$u=e^{\sigma\sqrt{\Delta t}}$, $d=1/u$, and
$p=(e^{r\Delta t}-d)/(u-d)$. An invalid probability produces an error asking for
more steps; it is never silently clipped.

Backward induction uses
$e^{-r\Delta t}[pV_{i+1,j+1}+(1-p)V_{i+1,j}]$.
For American exercise, each node instead takes the maximum of continuation and
immediate exercise. A rolling vector gives $O(N)$ memory and $O(N^2)$ time.
Tree convergence may oscillate with odd/even resolution; successive prices are
not guaranteed to improve monotonically. Zero-volatility American contracts are
valued by maximizing discounted deterministic exercise payoffs over the grid.

The default American benchmark is a 2,000-step CRR tree. It is a numerical
approximation to continuous exercise. A 50-date Monte Carlo grid and a 2,000-step
tree do **not** have the same exercise grid or stock-state discretization.

## Monte Carlo and uncertainty

European simulation samples the exact terminal distribution:

$$S_T=S_0\exp[(r-\sigma^2/2)T+\sigma\sqrt T Z],\qquad Z\sim N(0,1).$$

For independent observations $Y_i$, Welford's algorithm calculates the sample
variance $s^2$. Reported standard error is $s/\sqrt n$, and the asymptotic,
pointwise 95% interval is $\bar Y\pm1.95996398454\,s/\sqrt n$.
These intervals can be unreliable with very small samples or rare positive
payoffs. A zero estimated error when every observed payoff is zero does not
establish that the true option value is zero.

Convergence figures show repeated seeds and RMSE against Black-Scholes.
Prefixes within one seed share random draws; points along a curve are therefore
correlated. The quick profile uses only three seeds and is a demonstration, not
a precise convergence-rate or confidence-coverage study.

### American baseline: prescribed stopping policy

The baseline simulates exact GBM transitions on an exercise grid and exercises
an in-the-money put when $S_t\leq0.85K$ by default. The boundary is configurable;
the curated comparison uses $0.75K$ to expose a deliberately suboptimal policy.
For calls with $r<0$, the optional prescribed boundary is $S_t\geq bK$.
For non-dividend-paying calls with $r\geq0$, both American simulation programs
use the known optimal policy of holding until maturity.

This is a valid stopping time, so its **expected** discounted payoff is a lower
bound on the optimal American value. A finite-sample estimate need not lie
below a numerical tree benchmark. More paths reduce sampling error but cannot
make a fixed suboptimal rule optimal. The baseline is deliberately separated
from the learned LSM policy.

## Three European variance reduction methods

1. **Antithetic variates.** For each $Z$, average the discounted payoffs from
   $Z$ and $-Z$. With $N$ payoff evaluations there are $N/2$ independent pair
   averages. Standard error is computed from those pair averages, never from
   $N$ correlated individual payoffs. The payoff budget must be even.
2. **Control variate.** Use $X=e^{-rT}S_T$, whose known expectation is $S_0$.
   The adjusted observation is $Y-\hat\beta(X-S_0)$, where
   $\hat\beta=\widehat{\operatorname{Cov}}(Y,X)/\widehat{\operatorname{Var}}(X)$
   is fitted on an independent pilot sample. Conditional on that pilot the
   estimator is unbiased. Pilot work is included in the payoff count and
   runtime; its randomness is not included in the conditional standard error.
3. **Importance sampling.** Draw $Z\sim N(\mu,1)$ and multiply the discounted
   payoff by $\exp(-\mu Z+\mu^2/2)$. An independent pilot selects the mean
   shift by the basic Newton procedure on an empirical second moment. Freeze
   the fitted shift before valuation. This is ordinary likelihood-ratio
   weighting, not self-normalized sampling. Nonzero-shift weighted payoffs
   are formed in log space.

All methods have the same main-sample terminal payoff budget. The control
and importance-sampling methods each have an additional pilot of the same
size, reported separately. Methods share main random
draws to make comparisons less noisy; each method still uses independent
observations internally. The plots report both estimator variance ratios and
ratios adjusted by terminal payoff counts. Counts are a work proxy: they do
not equate actual CPU costs, random-number generation costs, or memory use.

### Basic Newton procedure for the importance-sampling shift

The procedure is: calculate a pilot objective, apply Newton's method to obtain
an approximate shift, then plug that fixed shift into the pricing estimator.
Let $h(z)$ be the discounted European payoff and $z_i\sim N(0,1)$ the fixed
pilot sample. The second moment of the importance-weighted payoff is estimated by

$$\widehat M(\mu)=\frac1m\sum_{i=1}^m
 h(z_i)^2\exp(\mu^2/2-\mu z_i).$$

The option price does not depend on the shift. Its squared value is therefore
constant in the variance expression, so the second moment is the objective
used to choose the shift. Differentiating the fixed-sample objective gives

$$\widehat M'(\mu)=\frac1m\sum_i
 h(z_i)^2(\mu-z_i)\exp(\mu^2/2-\mu z_i),$$

$$\widehat M''(\mu)=\frac1m\sum_i
 h(z_i)^2[1+(\mu-z_i)^2]\exp(\mu^2/2-\mu z_i).$$

Newton's correction is $d_k=\widehat M'(\mu_k)/\widehat M''(\mu_k)$.
The original implementation limits this correction to $[-1,1]$, so the
implemented update is

$$\mu_{k+1}=\mu_k-\operatorname{clip}(d_k,-1,1).$$

For a call, the initial shift is $\max(0,a)$, where

$$a=\frac{\log(K/S_0)-(r-\sigma^2/2)T}{\sigma\sqrt T}.$$

For a put, use the corresponding lower-tail initialization $\min(0,a)$.
As in the original routine, stop when either
$|\widehat M'(\mu_k)|<10^{-8}$ or
$|\mu_{k+1}-\mu_k|<10^{-8}$, allowing up to 50 Newton updates.
The resulting shift is an approximation obtained from a finite pilot sample.

With the shift fixed at $\widehat\mu$, generate independent
$\epsilon_j\sim N(0,1)$, put $Z_j=\epsilon_j+\widehat\mu$, and price using

$$\widehat V_N=\frac1N\sum_{j=1}^N
 h(Z_j)\exp(-\widehat\mu Z_j+\widehat\mu^2/2).$$

The likelihood ratio corrects for the shifted sampling density. The pilot
uses stream 3 and pricing uses stream 0; the control-variate pilot uses
stream 1. Both pilot generation and Newton iterations are included in runtime.
Repeated scans reuse the same pilot payoffs and are not counted as additional
simulated payoff evaluations.

`--newton-trace PATH` saves the initial point and every Newton update. Columns
`second_moment`, `gradient`, and `hessian` are the direct objective and its
first two derivatives. `step` is the signed shift change, and `step_limited`
records whether the original correction limit was used. Final status is
`converged_gradient` or `converged_step`, identifying the stopping rule.
Pricing rows report `newton_iterations`, `newton_gradient`,
`newton_second_moment_initial`, `newton_second_moment_final`, and
`newton_status`. Other methods use zero/`not_applicable` in those fields.

An all-zero pilot cannot identify a shift and produces a clear error asking
for more pilot paths. Failed convergence or non-finite direct calculations
also produce an error. Zero-volatility and expired contracts skip fitting
and report `deterministic`; their trace contains only the CSV header.

Numerical checks verify the original initialization, direct Newton correction,
step limit, stopping rules, and pricing accuracy. An independently integrated
objective provides a further software check on the fitted shift.

## Common Longstaff-Schwartz algorithm

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
the source-aware cache request. Original experimental results and reference datasets are not
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
python tests/test_lsm_procedure.py --example-dir tmp/eight_path_example -v
```

## Experimental evidence and file management

Published Tables 1 and 2 have one canonical dataset bundle:
`data/reference/longstaff_schwartz_2001/`, with `provenance.json` and two table
schemas. Original-paper replication and the plain/AV experiments are separate
under `data/experiments/`; their figures and reports use matching study names.
Historical results and their manifests were relocated without numerical edits.
The organization inventory resolves old paths retained in historical manifests.

In fresh runs, fine CRR values are validated through `scripts/crr_cache.py`.
Its key includes the full contract, tree resolution, exercise-date count, time-zero
convention, and SHA256 of the CRR driver and shared headers. A changed key is a
cache miss. A miss requires build metadata matching the current sources and
binary. Conservative shared-header fingerprinting can invalidate a reference
after a non-pricing header edit; it never silently re-labels historical values.
Every cache entry records its calculation command, build, runtime and values.
Experiment manifests retain the used keys and values when local cache files
are later removed. Cache hits do not require rerunning the tree.

Full result folders are immutable to the runners. Use a fresh `--output-dir`
for another full run. `--quick` defaults to ignored `tmp/smoke/STUDY`; guards
also reject explicit smoke destinations under research data, figures, references
or archives. Figure scripts infer smoke status from the input manifest.

See [repository organization](repository.md) for the maintained/generated tree,
archive decisions, notebook usage, source provenance and optional CMake workflow.
The research findings remain in [paper replication](results/paper_replication.md)
and [plain/AV comparison](results/lsm_comparison.md).

## References

- Cox, J. C., Ross, S. A., and Rubinstein, M. (1979).
  [Option Pricing: A Simplified Approach](https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/36891/files/2017/07/CRR79-1yy8av8.pdf).
  *Journal of Financial Economics*, 7(3), 229–263.
- Longstaff, F. A., and Schwartz, E. S. (2001).
  [Valuing American Options by Simulation: A Simple Least-Squares Approach](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf).
  *Review of Financial Studies*, 14(1), 113–147.
- Owen, A. B. *Monte Carlo Theory, Methods and Examples*,
  [Chapter 8: Variance Reduction](https://artowen.su.domains/mc/Ch-var-basic.pdf).
- Glasserman, P. (2003).
  [*Monte Carlo Methods in Financial Engineering*](https://doi.org/10.1007/978-0-387-21617-1).
  Springer.
- Huynh, H. T., Lai, V. S., and Soumaré, I. (2008).
  [*Stochastic Simulation and Applications in Finance with MATLAB Programs*](https://doi.org/10.1002/9781118467374).
  John Wiley & Sons.
