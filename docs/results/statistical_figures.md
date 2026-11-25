# Statistical figures from saved LSM results

These figures reuse the preserved full research CSVs. No simulations, CRR
calculations, compilation, tests or environment checks were run for this update.
Pricing algorithms and the repository structure were unchanged.

## Figure guide

Each figure has PNG and SVG versions. Method comparisons use orange for plain
LSM and teal for AV-LSM. Panels comparing SE definitions or valuation modes
identify their colors explicitly in their legends.

| Figure | Interpretation |
|---|---|
| [Signed pricing errors](../../figures/lsm_analysis/full/signed_pricing_errors.png) · [SVG](../../figures/lsm_analysis/full/signed_pricing_errors.svg) | Twenty contracts; separate panels for published AV-LSM prices and matched-grid CRR |
| [Standardized pricing discrepancies](../../figures/lsm_analysis/full/standardized_differences.png) · [SVG](../../figures/lsm_analysis/full/standardized_differences.svg) | Differences in SE units, with descriptive ±1.96 thresholds |
| [Table 2: in-sample versus out-of-sample](../../figures/lsm_analysis/full/in_sample_out_of_sample.png) · [SVG](../../figures/lsm_analysis/full/in_sample_out_of_sample.svg) | Individual paired differences, paired means with 95% confidence intervals, and mean errors against CRR |
| [Sampling variability](../../figures/lsm_analysis/full/sampling_variability.png) · [SVG](../../figures/lsm_analysis/full/sampling_variability.svg) | Thirty fitted prices per method and mean ± empirical SD; whiskers describe spread, not a confidence interval |
| [Standard-error calibration](../../figures/lsm_analysis/full/standard_error_calibration.png) · [SVG](../../figures/lsm_analysis/full/standard_error_calibration.svg) | Mean reported SE compared with empirical SD; also shows individual-path SEs that ignore antithetic covariance |
| [Variance reduction and efficiency](../../figures/lsm_analysis/full/variance_and_efficiency.png) · [SVG](../../figures/lsm_analysis/full/variance_and_efficiency.svg) | Empirical variance and inverse variance–runtime product, relative to plain LSM |

## Data and statistical definitions

Inputs are the canonical [published references](../../data/reference/longstaff_schwartz_2001/provenance.json),
[paper replication CSVs](../../data/experiments/paper_replication/full), and
[controlled comparison CSVs](../../data/experiments/lsm_comparison/full).
Overlapping AV results agree within 1.8e-15. The
[analysis manifest](../../data/experiments/lsm_analysis/full/manifest.json)
records input and plotting-source SHA-256 hashes. Original data and earlier
figures are preserved.

All contracts are American puts with strike 40, rate 6%, and 50 equally spaced
exercise dates per year plus time zero. Table 1 uses seed 42 and 100,000 paths
per estimate. Repeated-run comparisons use S = K = 40, volatility 20%, maturity
one year, and thirty seeds (42–71). For AV-LSM, 100,000 paths means 50,000 pairs.

- **Signed pricing error:** P − R, where P is our estimate and R is the reference.
  Positive means our estimate is higher. CRR uses 100,000 time steps and the
  same exercise dates as LSM. Its remaining discretization error is not included
  in statistical intervals.
- **Standardized discrepancy:** (P − R) / sqrt(SE² + SE_paper²) for the published
  price, treating the simulations as independent; (P − R) / SE for deterministic
  CRR. The ±1.96 lines are descriptive reference thresholds. In-sample SEs are
  approximate, the paper's SE convention is undisclosed, and published rounding
  contributes to the discrepancies. These are not simultaneous confidence tests.
- **Empirical SD:** sqrt(sum((Pᵣ − mean(P))²)/(R − 1)) across R complete runs.
  It measures variability of a new simulation and policy fit. **Mean reported
  SE** averages the within-run SEs. Their ratio should be near one when reported
  SEs describe observed variability. With thirty runs, the empirical SD itself
  is uncertain; small departures from one do not establish miscalibration.
- **Antithetic SE:** for M pairs, form discounted payoff averages
  Aᵢ = (Yᵢ⁺ + Yᵢ⁻)/2 and use sd(A)/sqrt(M). The individual-path SE ignores pair
  covariance. In-sample SEs remain approximate because fitted regressions couple
  payoffs; repeated-run SD also captures variation in the fitted policy.
- **Table 2 mean-difference interval:** pair in-sample and out-of-sample prices
  by contract and seed, then form dᵣ = P_in,ᵣ − P_out,ᵣ. Use
  mean(d) ± 2.776445 × sd(d)/sqrt(5), a 95% Student-t interval with four degrees
  of freedom. This includes the observed covariance of paired prices; adding
  within-run SEs in quadrature would not. The right panel applies the same
  five-run t procedure separately to each valuation's errors against CRR.
- **Variance-reduction factor:** s_plain² / s_AV²; percentage reduction is
  100 × (1 − s_AV²/s_plain²). **Relative computational efficiency:**
  (s_plain² × median runtime_plain) / (s_method² × median runtime_method).
  Higher efficiency means more precision per recorded computation time. Only
  same-build controlled timings are compared; this metric excludes bias and
  depends on machine load.

Table 2 contains five policies per contract, each trained on 100,000 antithetic
paths and valued on both those paths and 100,000 independent antithetic paths.
Its pointwise intervals assume approximately normal run estimates/differences;
five runs provide limited information about that assumption. Shared seeds
correlate contracts, and no multiple-comparison adjustment is applied. The
paper does not disclose its random streams, so its printed rows cannot be
paired with our runs.
Inclusion of zero does not establish equality or absence of fitting bias.

## Findings and missing configurations

At 100,000 paths, empirical SDs are 0.00879290 (plain) and 0.00545889 (AV).
Mean reported SEs are 0.00870650 and 0.00532478; calibration ratios are 0.9902
and 0.9754. AV-LSM has **61.46% lower observed variance** (factor 2.5945) and
**3.18× relative computational efficiency**, using saved median runtimes of
465 ms (plain) and 379 ms (AV). These estimates describe this contract and
thirty runs, rather than a universal ranking.

Across twenty Table 1 cases, zero AV and one plain published-price discrepancies
exceed ±1.96; against matched CRR, two AV and zero plain discrepancies exceed
them. These descriptive counts do not establish a ranking by bias.

Seven of eight Table 2 paired intervals contain zero. For S = 36, volatility
40%, maturity two years, the mean in-minus-out difference is 0.017141, with
pointwise 95% CI [0.000579, 0.033702]. The
[paired summary](../../data/experiments/lsm_analysis/full/table2_summary.csv)
contains all eight cases. These limited data do not justify a general claim
of negligible fitting bias.

AV-LSM has thirty runs at each of 5,000, 10,000, 25,000, 50,000 and 100,000 paths;
plain LSM has thirty runs only at 100,000. Variance and efficiency comparisons
therefore use that common budget. The
[missing configurations](../../data/experiments/lsm_analysis/full/missing_configurations.csv)
are plain LSM at 5,000, 10,000, 25,000 and 50,000 paths, seeds 42–71 at each
budget, for the same one-year at-the-money contract: **120 missing runs**.
The calibration plot leaves those points absent. None were generated.

## Redrawing

The [analysis script](../../scripts/analyze_lsm.py) reads CSVs, computes statistics
and renders figures; it contains no simulation, build or CRR invocation.
Redraw the delivered figures from their saved derived tables:

```text
python scripts/analyze_lsm.py --figures-only
```

To recompute statistical summaries from the original saved inputs, choose a
fresh output directory; this also runs no numerical pricing code:

```text
python scripts/analyze_lsm.py --output-dir data/experiments/lsm_analysis/review
```

The [analysis directory](../../data/experiments/lsm_analysis/full) contains derived
statistics, missing configurations, summary, CSV audit and manifest. Those files
and figure exports are generated; this guide and the plotting scripts are
maintained manually.
