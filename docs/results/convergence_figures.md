# Convergence results and thesis figures

**The numerical dataset is complete:** 300 distinct fits, thirty seeds per
method at 5k, 10k, 25k, 50k and 100k paths. No configurations remain missing.

The [final numerical report](reproduction_conclusion.md) is the authoritative
source for the consolidated table, slopes, SE calibration and limitations.
It uses the [completed summary](../../data/experiments/lsm_comparison/convergence_completion/convergence_summary.csv)
and saved CRR **2.314066325146**.

- SD slopes: **−0.5384 plain**, **−0.5089 AV**, versus theoretical **−0.5**.
- AV variance reduction: **59.84%–75.26%**; RMSE is lower at every budget.
- Evidence supports a smaller variance constant and approximately unchanged
  order; it does not prove convergence to the true continuous-exercise price.
- Comparable runtime efficiency is restricted to the original controlled
  100k experiment: **3.18× AV**. New plain timings belong to later sessions.

For the thesis, retain the existing
[signed pricing errors](../../figures/lsm_analysis/full/signed_pricing_errors.svg),
[Table 2 paired intervals](../../figures/lsm_analysis/full/in_sample_out_of_sample.svg)
and [100k variance/efficiency comparison](../../figures/lsm_analysis/full/variance_and_efficiency.svg).

**Saved convergence figures are historical:** the
[price/SD/RMSE panel](../../figures/lsm_analysis/convergence/price_sd_rmse_convergence.svg)
still contains AV at all budgets and plain at 100k only. Its
[manifest](../../data/experiments/lsm_analysis/convergence/manifest.json) records
180 runs and 120 then-missing runs. Those are historical counts, not the current
dataset status. Do not present this panel as the completed two-method study.

No completed five-budget figure export is present in the current figure
directory. The final report's table covers the complete comparison.
This update changed documentation only; existing figures were preserved.
