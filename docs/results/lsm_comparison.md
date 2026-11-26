# Plain LSM and AV-LSM: implementation review and measured comparison

For the completed five-budget investigation and consolidated numerical table,
see the [final results](reproduction_conclusion.md). This document records the
earlier implementation review and 100k controlled experiment.

Run on 26 November 2026, C++17/GCC 16.1.0 with portable `-O2`.
The [implementation guide](../methodology.md) explains the equations,
function mapping, pseudocode, complexity, CSV schema and deterministic example.

## Changes and reasons

| Change | Reason |
|---|---|
| One exact GBM generator with plain/antithetic sampling | The sampling mechanism is the only methodological difference |
| One normalized weighted-Laguerre training pass | Identical basis, targets and backward cash-flow rules for both methods |
| One normalized exercise comparison used during training and evaluation | Correct the prior dollar-payoff versus normalized-continuation inconsistency |
| Explicit in-sample/out-of-sample/both modes | Separate original-paper valuation from the frozen-policy diagnostic |
| Time-zero exercise in both modes, fixed from training | Align LSM and CRR exercise opportunities without maximizing over a test mean |
| Per-date rank, ITM count, retained-R conditioning and fallback status | Make sparse and degenerate regressions inspectable |
| Explicit path, pair, stream and valuation metadata | Prevent confusing antithetic pairs with paths or double-counting shared samples |
| Separate new datasets and figures | Preserve existing experiments and reference data |

The six executable entry points remain. European pricing, European variance
reduction, and CRR numerical algorithms were not changed. The original
reorthogonalized QR solver remains; its diagnostics and basis-value reuse were
added. No American control variate or importance sampling method was added.
Repository cleanup and environment consolidation are deferred.

## Controlled comparison

Both methods use 100000 paths, the same contract/grid, four weighted Laguerre
regressors, strike-normalized states and targets, and in-sample valuation.
The 20-case table uses seed 42. The one-year at-the-money contract uses 30
master seeds, 42-71, with a complete new simulation and policy fit each time.
The saved 100000-step American CRR with matched exercise dates is 2.31406633.
The paper's rounded LSM value is 2.313.

| At-the-money, one year | Plain LSM | AV-LSM |
|---|---:|---:|
| Mean of 30 full-fit prices | 2.31386219 | 2.31350400 |
| SD across 30 prices | 0.00879290 | 0.00545889 |
| Mean within-run plug-in SE | 0.00870650 | 0.00532478 |
| Across-fit SD / mean plug-in SE | 1.0099 | 1.0252 |
| RMSE against the published LSM prices across 20 cases | 0.01509683 | 0.00902140 |

At the same path budget, the observed plain/AV estimator-variance factor is
**2.5945**, or approximately **61.46% lower variance** for AV in this contract.
The mean plain-minus-AV difference is 0.00035819; its across-seed paired SE is
0.00165657. These runs do not show a meaningful price separation. Sharing
master seeds partly couples the two methods; the mean contrast therefore
uses the SD of paired differences, rather than assuming independence.

This is a contract-specific variance estimate from 30 replicates, not a
universal ranking. Lower error in one 20-case, single-seed table is also not
a proof of lower bias. AV Table 1 prices match the preserved historical
100000-path prices to 1.8e-15; adding time-zero exercise has no effect on these
contracts.

## Table 2 frozen-policy diagnostic

The diagnostic reproduces the disclosed procedure in
[Longstaff and Schwartz (2001), pp. 127-129](https://www.kellogg.northwestern.edu/faculty/mcdonald/ftp/practicum/papers/longstaffschwartz.pdf):
eight contracts (`S=36` or `44`, volatility `.20` or `.40`, maturity one or two
years), five seeds each, 100000 antithetic training paths and 100000 independent
antithetic evaluation paths. Each fit reports both valuations with identical
coefficients. There are 40 fitted policies and 80 valuation rows.

The 40 in-minus-out differences have mean **0.00569861**, with **21 positive**
and **19 negative** differences. The maximum absolute gap is **0.04847451**.
The result is reported directly; it does not establish negligible fitting bias
or exact agreement with the paper's seed rows. Five seeds per contract are a
limited diagnostic, and reusing master seeds across contracts correlates
their errors. The two valuation errors also share fitted coefficients, so
adding within-run SEs in quadrature is not an exact difference test.

The new [Table 2 reference](../../data/reference/longstaff_schwartz_2001/table2.csv)
was checked visually against the original PDF. Its
[provenance](../../data/reference/longstaff_schwartz_2001/provenance.json)
records a printed inconsistency: for `S=36`, volatility `.20`, maturity 2,
the third row prints `+.009`, while `4.820 - 4.829 = -.009`. The CSV retains
the printed sign; plots compute mean differences from the printed prices.

## Numerical diagnostics and validation

All **31 executable pricing tests and 15 deterministic procedure fixtures
passed**, including the original 29 pricing tests. Both CTest targets passed.
The fixtures cover exact GBM and antithetic shocks, normalization and scale
homogeneity, the original stopping decisions, backward updates, coefficient
reuse, both valuation modes, pair-based SEs, time-zero/grid alignment,
degeneracy, reproducibility, expiry and zero-volatility cases, and a narrow
stock range with ill-conditioned Laguerre columns.

The paper's eight-path example gives **0.114434330045**. Its intermediate
coefficients, continuation values, decisions and selected cashflows are
exported under [the example directory](../../data/experiments/lsm_comparison/full/example).

All **8860 positive-date regression records** in the full experiments retained
rank 4; none required a constant fallback or had no ITM observations. The
maximum retained-R infinity-norm condition diagnostic was **1.5003e9**.
This high correlation is why the QR solver and conditioning records matter;
individual coefficients should not be interpreted as stable economic effects.
The tests verify stable continuation predictions in a deliberately narrow
stock range. Full rank alone does not certify accuracy in every possible
contract or input regime.

## Files and remaining limits

- [20-case plain/AV table](../../data/experiments/lsm_comparison/full/table1.csv)
- [60 repeated full fits](../../data/experiments/lsm_comparison/full/replicates.csv)
- [80 Table 2 valuation rows](../../data/experiments/lsm_comparison/full/table2.csv)
- [Per-date regression diagnostics](../../data/experiments/lsm_comparison/full/regressions.csv)
- [Aggregate summary](../../data/experiments/lsm_comparison/full/summary.json)
- [Build, seeds, source hashes and cached-reference provenance](../../data/experiments/lsm_comparison/full/manifest.json)
- [Plain/AV price figure](../../figures/lsm_comparison/full/table1_comparison.png)
- [Sampling variability and SE calibration](../../figures/lsm_comparison/full/sampling_variability.png)
- [Frozen-policy diagnostic figure](../../figures/lsm_comparison/full/table2_diagnostic.png)

In-sample SEs remain approximate because fitted coefficients couple payoffs.
Independent-evaluation SEs condition on one policy. Neither removes grid,
policy approximation, or reference discretization error. The paper does not
publish its RNG, seeds, or SE formula, so exact random-sample replication is
not possible. Its printed SEs resemble the pathwise diagnostic; AV uncertainty
here uses pair averages. The variance factor and Table 2 conclusions retain
finite-replicate uncertainty. No remaining implementation failure was observed
in the executed tests; these statistical limits are not resolved by passing them.
