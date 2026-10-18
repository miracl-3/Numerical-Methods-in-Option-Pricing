# Numerical methodology

The underlying is a non-dividend-paying stock under the risk-neutral measure:

$$dS_t=rS_t\,dt+\sigma S_t\,dW_t,\qquad q=0.$$

All rates are continuously compounded, volatility is annualized, and maturity is
measured in years. Parameters are constant. Call and put payoffs are
$(S-K)^+$ and $(K-S)^+$. This is a numerical study under a specified model;
there is no market calibration or empirical trading-performance claim.

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

## Longstaff-Schwartz least-squares Monte Carlo

Training proceeds backward through exercise dates. Only in-the-money training
paths enter the regression of discounted selected future cashflows onto
$1,x,x^2$, where $x=S_t/K$. Regression states are centered and scaled;
reorthogonalized modified Gram-Schmidt QR avoids the conditioning penalty of
normal equations. Rank-deficient data fall back to a lower-degree fit, and
near-constant or very small samples use a constant continuation estimate.
Dates with no in-the-money observations have no exercise model.

The training pass chooses exercise when intrinsic value exceeds the fitted
nonnegative continuation value. The time-zero exercise decision is also made
on training data. Evaluation then follows the frozen rule **forward on a
separate random stream**. Training paths are released before evaluation;
test paths are generated one at a time.

The expected value of any frozen feasible policy is a lower bound on the
optimal exercise value. Independent evaluation removes the look-ahead effect
from reusing the training paths for valuation. It does not remove regression,
finite-training-sample, or exercise-grid bias. Reported intervals condition on
one trained policy and quantify evaluation sampling uncertainty only. Repeated
training/evaluation seeds expose additional variability, but do not create a
certified upper/lower confidence bracket for the true American price.

## Reproducibility and accounting

The master seed is recorded in every row. SplitMix64 derives separate seed
streams for pilot fitting, LSM training, and LSM evaluation. Stream identifiers
are local to each independent executable. The engine is
`std::mt19937_64`; `std::normal_distribution` is implementation-dependent, so
identical seeds do not imply identical samples across compilers or C++ standard
libraries. Numerical tests use financial identities and tolerances rather than
cross-platform bitwise Monte Carlo snapshots.

`runtime_ms` measures each method using a steady clock. It includes pilot or
policy training, and excludes benchmark calculation and CSV I/O. For tree
rows, `paths`, `independent_samples`, and `payoff_evaluations` are zero;
standard error is zero and the CI columns equal the deterministic price.
Those CI columns are a schema convenience, not statistical tree error bounds.
For simulation rows, `payoff_evaluations` counts terminal/selected cashflow
observations, plus training or pilot observations. It excludes intermediate
exercise tests and regression operations, so it is chiefly useful for the
European variance-reduction comparisons.

CSV `benchmark` means Black-Scholes for European methods and a CRR American
reference for American methods. `absolute_error` for an American policy mixes
policy/grid error, reference approximation error, and sampling noise; it is
not a pure Monte Carlo error. `reference_steps` records the numerical American
reference resolution. `steps=1` for all exact-terminal European simulations.

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
