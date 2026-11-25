# Standard LSM reproduction: measured findings

This document records the historical 25 November run. The common plain/AV
engine and its new experiments are described in [the implementation](../methodology.md)
and [comparison results](lsm_comparison.md); the historical CSVs remain unchanged.

Run on 25 November 2026 using GCC 16.1.0, C++17, portable `-O2`, and the
original Longstaff-Schwartz Table 1 American-put protocol. C++ performs the
simulation, regression, cashflow valuation, within-run SE calculations, and
CRR backward induction. Python collects CSV output and draws the figures.

The rerun uses all 20 contracts, 100,000 paths (50,000 antithetic pairs),
50 exercise dates per year, a constant plus three weighted Laguerre functions,
and in-sample valuation. Stock prices and regression cashflows are normalized
by strike. The original procedure and printed reference values are from
[Longstaff and Schwartz (2001), pp. 126-127 and 143](https://www.kellogg.northwestern.edu/faculty/mcdonald/ftp/practicum/papers/longstaffschwartz.pdf).

## Prices resemble the published results

Across the 20 cases, our prices have **RMSE 0.00902** against the published LSM
prices, mean difference **-0.00127**, and maximum absolute difference
**0.01684**. All 20 differences are within
`1.96 * sqrt(our_pair_SE^2 + published_SE^2)`; the largest standardized
difference is 1.664. This is a descriptive comparison using the paper's
rounded SEs, not a simultaneous coverage test.

Against the published finite-difference values, our price RMSE is **0.00886**.
The fine matched-date CRR references differ from the printed finite-difference
values by 0.00095 on average in absolute value, with maximum difference
0.00575. Both numerical references retain discretization error.

For the one-year at-the-money case (`S0=K=40`, `r=0.06`, `sigma=0.20`):

| Quantity | Value |
|---|---:|
| Published LSM price | 2.313 |
| Our LSM price, seed 42 | 2.30793933 |
| Published finite-difference price | 2.314 |
| 100,000-step American CRR with matched exercise dates | 2.31406633 |
| 100,000-step American CRR at every tree date | 2.31957142 |
| Mean LSM price across 30 complete fits | 2.31350400 |

The CRR calculation uses **100,000 time steps**, meaning 100,001 terminal
nodes and 5,000,150,001 total lattice nodes. Both references use full backward
induction without tail truncation or small-value cutoffs. One American CRR
reference allows exercise at every tree date; the American CRR reference with
matched exercise dates and the paper's comparison allow 50 exercise dates per
year. Their values answer different
exercise-schedule questions. CRR also permits time-zero exercise, which does
not affect any of the 20 benchmark cases.

## Standard errors: close agreement requires distinguishing conventions

The paper does not state its SE formula. Our **pathwise diagnostic**,
`SD(discounted path cashflows) / sqrt(100000)`, is between **0.892 and 1.097**
times the printed SE across all cases, with median ratio **0.988**. Thus every
case is within about 11%. This resemblance is consistent with a marginal-path
convention; it does not establish how the authors calculated their SEs.

Our primary SE uses the sample SD of **antithetic pair averages** divided by
the square root of the number of pairs. It accounts for within-pair
covariance. These SEs are lower: 0.420-0.828 times the printed values, with
median ratio 0.570. We retain the pair-based SE for uncertainty reporting and
export the pathwise calculation as a diagnostic.

| At-the-money, 100,000 paths | SE or across-run SD |
|---|---:|
| Published SE | 0.009 |
| Our pathwise diagnostic, seed 42 | 0.00873927 |
| Our pair-based SE, seed 42 | 0.00537392 |
| Mean pair-based SE across 30 full fits | 0.00532478 |
| SD of prices across those 30 full fits | 0.00545889 |

The observed across-run SD is **2.52% above** the mean pair-based SE at this
budget. Across all five budgets, the SD/mean-pair-SE ratios range from 0.862
to 1.025. This supports the pair calculation in this one contract diagnostic;
30 runs still leave substantial uncertainty in the estimated SD. The repeated
fits include fresh simulation and regression, while within-run plug-in SEs
omit dependence induced by fitting on the same paths.

The 100,000-path across-run mean is 0.000562 below the matched-date CRR price.
The observed across-run SD, 0.005459, describes a single fitted run. The SE of
the 30-run mean is smaller, approximately 0.000997. Neither SE measures
exercise-grid bias or numerical reference error.

## Files and validation

- [Full 20-case comparison](../../data/experiments/paper_replication/full/lsm_paper_table.csv)
- [150 individual convergence fits](../../data/experiments/paper_replication/full/lsm_paper_convergence.csv)
- [SE calibration at five path budgets](../../data/experiments/paper_replication/full/lsm_paper_se_summary.csv)
- [Aggregate numerical summary](../../data/experiments/paper_replication/full/lsm_paper_summary.json)
- [Parameters, seeds, compiler and source provenance](../../data/experiments/paper_replication/full/lsm_paper_manifest.json)
- [Published table transcription and source record](../../data/reference/longstaff_schwartz_2001/provenance.json)
- [Price comparison figure](../../figures/paper_replication/full/lsm_paper_table.png)
- [Convergence figure](../../figures/paper_replication/full/lsm_paper_convergence.png)
- [Standard-error figure](../../figures/paper_replication/full/lsm_paper_standard_errors.png)

All 29 pricing tests and three procedure fixtures passed, including the
original eight-path stopping decisions and price rounded to **0.1144**.
Both CTest targets passed. A separate 216-case old/new CRR comparison had
maximum difference 1.56e-12. The full run produced 20 fine-reference cases
and 150 repeated fits (30 seeds at each of five budgets).

The paper does not publish its generator, seeds, or SE formula. This rerun
reproduces its disclosed procedure and sample budget; it cannot recreate the
original random sample exactly. Printed prices and SEs have precision 0.001.
See [the workflow](../../README.md) to rebuild and rerun.

`capstone_report_11`, the notebook folder, its execution script, and the
duplicate Python environment were removed. The project uses a single `.venv`
for the C++/CSV/Python workflow. The six shared-header C++ programs remain the
calculation source; the separate historical extension is retained.
