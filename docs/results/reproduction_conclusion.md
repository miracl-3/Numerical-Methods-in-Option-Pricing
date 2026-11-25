# Longstaff–Schwartz reproduction: numerical conclusion

**Status:** Tables 1 and 2 are complete. The convergence comparison has
180/300 unique runs; 120 plain-LSM runs remain. This report uses existing
results only. No simulations, figures or builds were repeated.

## Accuracy and variability

American put: S = K = 40, r = 6%, volatility 20%, maturity one year;
50 exercise dates plus time zero. Matched-grid CRR (100,000 time steps):
**2.314066325**. Each method below uses 100,000 paths and 30 seeds.

| Metric | Plain LSM | AV-LSM |
|---|---:|---:|
| Mean price | 2.31386219 | 2.31350400 |
| Mean error against CRR | −0.00020413 | −0.00056232 |
| Empirical SD | 0.00879290 | 0.00545889 |
| Empirical variance | 0.0000773151 | 0.0000297995 |
| RMSE against CRR, across 30 runs | 0.00864752 | 0.00539652 |
| Mean reported SE / empirical SD | 0.9902 | 0.9754 |
| Recorded median runtime | 465 ms | 379 ms |
| Relative computational efficiency | 1.00 | 3.18 |
| Table 1 RMSE vs published AV prices, 20 contracts | 0.01509683 | 0.00902140 |
| Table 1 RMSE vs matched CRR, 20 contracts | 0.00899781 | 0.00810792 |

AV-LSM reduces observed variance by **61.46%** (factor **2.5945**).
All 20 AV published-price discrepancies lie within ±1.96 combined SEs;
19/20 plain discrepancies do. This supports numerical agreement, not exact
random-sample reproduction. Reported SEs closely match repeated-run SD here.

## Available convergence results

Thirty independent fits per row; AV path counts include both members of each pair.

| AV paths | Mean price | Empirical SD | RMSE vs CRR | Mean SE / SD |
|---:|---:|---:|---:|---:|
| 5,000 | 2.320802 | 0.023444 | 0.024014 | 1.0180 |
| 10,000 | 2.316776 | 0.016725 | 0.016666 | 1.0091 |
| 25,000 | 2.315370 | 0.010377 | 0.010286 | 1.0283 |
| 50,000 | 2.314339 | 0.006497 | 0.006394 | 1.1599 |
| 100,000 | 2.313504 | 0.005459 | 0.005397 | 0.9754 |

AV SD falls by 4.29× as paths increase 20×, broadly consistent with the
sqrt(20) = 4.47 Monte Carlo scaling. RMSE also declines. The 50,000-path SE
exceeds observed SD by 16%; thirty runs give uncertain calibration estimates.
A comparison of convergence rates awaits the missing plain runs.

## Table 2 and limitations

All eight contracts have five paired in/out valuations: **40 fitted policies,
80 prices**. Mean in-minus-out difference across cases/runs is **0.005699**.
Seven of eight paired 95% t intervals include zero. For S = 36, volatility
40%, maturity two years: difference **0.017141**, CI **[0.000579, 0.033702]**.

- In-sample SEs are approximate because regressions couple payoffs. AV SEs use
  pair averages. Out-of-sample within-run SEs condition on the fitted policy.
- Five-run paired t intervals assume approximately normal differences; they are
  pointwise, with no multiple-comparison adjustment. Shared seeds correlate cases.
- A finite exercise grid, fixed Laguerre basis and finite training sample leave
  discretization and policy errors. CRR also has numerical error. Thirty-run
  variance/RMSE estimates and one-seed Table 1 comparisons do not establish bias.
- The paper omits RNG, seeds and its SE formula. Its printed SEs resemble the
  individual-path diagnostic, which ignores antithetic covariance; that convention
  remains an inference. Exact printed rows cannot be reproduced.
- Efficiency is 1/(variance × median runtime), relative to plain LSM, using the
  same saved build/session. It excludes bias and depends on machine load; later
  completion timings must not be compared directly with historical AV timings.

## Execute only the missing runs

Missing: **plain LSM × {5,000, 10,000, 25,000, 50,000} paths × seeds 42–71**.
Use the same contract above, weighted Laguerre basis and in-sample valuation;
training and valuation path counts both equal the stated budget.

```powershell
Set-Location 'D:\Projects\Capstone Thesis'
# Preview only (120 pending on the current saved state):
& .\.venv\Scripts\python.exe .\scripts\run_missing_lsm.py
# Execute; use this same command to resume after interruption:
& .\.venv\Scripts\python.exe .\scripts\run_missing_lsm.py --run --workers 2
```

The runner preserves full results and checkpoints each fit under
`data/experiments/lsm_comparison/convergence_completion/`. It reuses the saved
100,000-step CRR from the replication CSV; the existing executable additionally
emits its cheap 50-step local diagnostic. No fine CRR is recomputed. Completion
writes individual CSVs, regression diagnostics, a manifest, all 300 combined
convergence rows and `convergence_summary.csv`; it does not redraw figures.
Only the preview and isolated checkpoint checks were executed during handover.

Sources: [saved statistical summaries](../../data/experiments/lsm_analysis/full),
[replication summary](../../data/experiments/paper_replication/full/lsm_paper_summary.json),
[comparison manifest](../../data/experiments/lsm_comparison/full/manifest.json).
[Existing figures and metric definitions](statistical_figures.md).
