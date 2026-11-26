# Longstaff–Schwartz and plain/AV-LSM: final numerical results

**Completed:** 300 distinct convergence fits: two methods × five budgets ×
30 seeds (42–71), plus the 20-contract Table 1 and 40-policy Table 2 studies.
This concludes the numerical investigation; no new model runs were needed.

## Consolidated convergence results

American put: S₀ = K = 40, r = 6%, σ = 20%, T = 1; 50 exercise dates plus
time zero. Both methods use normalized weighted Laguerre regression, QR and
in-sample valuation. Saved grid-matched CRR: **2.314066325146** (100,000 steps).
N counts total paths; AV uses N/2 antithetic pairs.

| N | Method | Mean price | Empirical SD | Estimated bias* | RMSE* | Mean SE / SD | AV variance reduction |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 5,000 | Plain | 2.324806 | 0.043388 | +0.010739 | 0.043990 | 0.892 | — |
| 5,000 | AV | 2.320802 | 0.023444 | +0.006736 | 0.024014 | 1.018 | 70.80% |
| 10,000 | Plain | 2.320744 | 0.031578 | +0.006677 | 0.031758 | 0.874 | — |
| 10,000 | AV | 2.316776 | 0.016725 | +0.002709 | 0.016666 | 1.009 | 71.95% |
| 25,000 | Plain | 2.314231 | 0.016375 | +0.000165 | 0.016101 | 1.063 | — |
| 25,000 | AV | 2.315370 | 0.010377 | +0.001304 | 0.010286 | 1.028 | 59.84% |
| 50,000 | Plain | 2.311899 | 0.013061 | −0.002167 | 0.013023 | 0.942 | — |
| 50,000 | AV | 2.314339 | 0.006497 | +0.000272 | 0.006394 | 1.160 | 75.26% |
| 100,000 | Plain | 2.313862 | 0.008793 | −0.000204 | 0.008648 | 0.990 | — |
| 100,000 | AV | 2.313504 | 0.005459 | −0.000562 | 0.005397 | 0.975 | 61.46% |

*Estimated bias = mean price − CRR; RMSE = √mean((price − CRR)²).
These estimate error relative to a numerical reference, not the unknown exact
price. SD uses denominator 29. Reduction = 100(1 − SD_AV²/SD_plain²).
Source: [completed summary](../../data/experiments/lsm_comparison/convergence_completion/convergence_summary.csv).

**Accuracy:** AV has lower RMSE at every budget; neither method's estimated
bias decreases monotonically. All ten pointwise mean-price 95% Student-t
intervals, mean ± 2.04523 × SD/√30, include matched CRR. This is uncertainty
about the estimator mean, not the dispersion of individual runs or proof of zero bias.

## Empirical order, variance constant and SE calibration

OLS log(SD) = a + b log(N), using five observed budgets, gives
**b_plain = −0.5384** (R² = 0.9903) and **b_AV = −0.5089** (R² = 0.9888),
both close to the Monte Carlo reference **−0.5**.

The evidence supports **a lower variance constant with approximately preserved
convergence order**. Under Var ≈ C/N, the observed N × SD² ranges are
**6.70–9.97 plain** and **2.11–2.98 AV**. These finite-budget estimates do not
prove identical asymptotic orders or convergence to the true American price.

Reported-SE/empirical-SD ratios span **0.874–1.063 plain** and **0.975–1.160 AV**.
Plain SEs understate observed spread by about 11–13% at 5k/10k; AV overstates it
by 16% at 50k. At 100k, agreement is within 2.5%. Thirty runs leave appreciable
uncertainty in SD and calibration; AV SEs correctly use pair averages.

## Agreement with Longstaff–Schwartz

**Table 1:** AV RMSE against the 20 published AV prices is **0.009021**
(maximum absolute difference **0.016837**); plain RMSE is **0.015097**.
All 20 AV and 19/20 plain discrepancies fall within ±1.96 combined reported
SEs. Against matched CRR across these contracts, RMSE is **0.008108 AV**
and **0.008998 plain**. This is close numerical agreement, with descriptive
thresholds rather than a simultaneous test or exact sample replication.

**Table 2:** the eight-contract, five-fit frozen-policy procedure is reproduced.
Our mean in-minus-out difference is **+0.005699**, versus **−0.000900** calculated
from the paper's printed prices. Maximum absolute individual differences are
**0.048475** and **0.035000**, respectively. Seven of our eight paired 95% t
intervals include zero; S₀ = 36, σ = 40%, T = 2 gives **0.017141**
with CI **[0.000579, 0.033702]**. Both datasets contain differences of both
signs, but our diagnostic does not establish universally negligible fitting bias.
Paper rows and our seeds cannot be paired across studies.
[Published Table 2](../../data/reference/longstaff_schwartz_2001/table2.csv);
[our paired summary](../../data/experiments/lsm_analysis/full/table2_summary.csv).

## Comparable computational efficiency

Use only the original controlled 100k experiment: median **465 ms plain**
versus **379 ms AV**; variance × median-runtime products are **3.593×10⁻⁵**
and **1.130×10⁻⁵ price²·s**. AV relative efficiency is **3.18×**.
This is a variance–time comparison, not a bias-adjusted ranking.
The new 120 fits span two execution sessions (119 + 1); their timings are not
pooled with historical AV timings for lower-budget efficiency claims.
[Saved timing summary](../../data/experiments/lsm_analysis/full/efficiency.csv).

## Thesis figures and remaining limits

Use these existing SVGs in the main results chapter:

1. [Signed pricing errors](../../figures/lsm_analysis/full/signed_pricing_errors.svg):
   Table 1 agreement with published AV prices and matched CRR.
2. [Table 2 paired comparisons](../../figures/lsm_analysis/full/in_sample_out_of_sample.svg):
   mean differences and correctly paired confidence intervals.
3. [Variance reduction and efficiency](../../figures/lsm_analysis/full/variance_and_efficiency.svg):
   the controlled 100k result; label this budget explicitly.

**Figure gap:** the saved [convergence panel](../../figures/lsm_analysis/convergence/price_sd_rmse_convergence.svg)
still represents the earlier 180-run dataset. It is not the final five-budget
plain/AV comparison. Use the consolidated table above for the completed
investigation; no figures were regenerated in this documentation-only update.

- Exact GBM transitions avoid path-discretization bias, but finite exercise
  dates, a fixed regression basis and finite training samples leave policy/grid
  error. The matched CRR tree also retains numerical error.
- In-sample regression couples payoffs, making within-run SEs approximate.
  Out-of-sample within-run SEs condition on one fitted policy.
- Five budgets and thirty fits give descriptive slopes. Shared seeds correlate
  budgets; no independent-error slope CI or equality-of-orders test is claimed.
  Table 2 has only five pairs per contract; t intervals assume approximate
  normality and are pointwise, without multiplicity adjustment.
- Published prices are rounded; RNG, seeds and the original SE formula are
  undisclosed. The printed SEs resemble individual-path SEs, but that ignores
  antithetic covariance. The known Table 2 sign typo is preserved; differences
  above are calculated from printed prices.
- Results concern this contract and path range; lower variance alone proves
  neither lower bias nor convergence to the true continuous-exercise value.

Provenance: [completion manifest](../../data/experiments/lsm_comparison/convergence_completion/manifest.json)
records 120 completed fits. Original-input and aggregate-output hashes match;
all 300 aggregate rows have the expected method/budget/seed keys and reproduce
the saved summaries. Individual checkpoint folders were inaccessible in this
session, so their per-file hashes were not rechecked. Only results documentation
was updated; no simulations, CRR calculations, builds or tests were run.
