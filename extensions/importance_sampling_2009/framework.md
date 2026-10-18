# Optional study: log-objective Newton importance sampling

This directory preserves the additional mathematical framework and C++
implementation developed during the refactor. It is separate from the main
six-program build and the thesis's basic Newton-and-plug-in procedure.
Its source, private headers, and saved results were copied before that
procedure was restored. `provenance.json` records their original paths and
SHA-256 hashes. This is an extension for later study, not a claim that its
additional analysis has already been developed in the thesis.

The Gaussian sample-average optimization framework is informed by Jourdain
and Lelong (2009), *Robust Adaptive Importance Sampling for Normal Random
Vectors*, [The Annals of Applied Probability, 19(5), 1687–1718](https://doi.org/10.1214/09-AAP595).
The paper is also available as [arXiv:0811.1496](https://arxiv.org/pdf/0811.1496).
This implementation uses independent pilot and evaluation samples. It does
not reproduce the paper's sample-reuse estimator or claim its asymptotic
theorems as new results. Both the basic and extended routines target the same
pilot second moment; the extension changes its numerical formulation and
iteration, rather than defining a different option price.

## Pricing and Gaussian change of density

For a non-dividend-paying stock under the risk-neutral measure,

$$S_T(z)=S_0\exp[(r-\sigma^2/2)T+\sigma\sqrt T z],\qquad Z\sim N(0,1).$$

Let $h(z)$ be the discounted call or put payoff. The price is
$V=\mathbb E[h(Z)]$. For proposal density $q_\mu(z)=\phi(z-\mu)$,

$$L_\mu(z)=\frac{\phi(z)}{\phi(z-\mu)}=e^{-\mu z+\mu^2/2},$$

so independent standard normals $\epsilon_j$ yield observations

$$Y_j(\mu)=h(\epsilon_j+\mu)e^{-\mu\epsilon_j-\mu^2/2}.$$

The estimator is $\widehat V_N=N^{-1}\sum_jY_j$.

## Second-moment objective and its empirical approximation

Because $V$ does not depend on the proposal mean,

$$N\operatorname{Var}(\widehat V_N(\mu))=M(\mu)-V^2,\qquad
M(\mu)=\mathbb E[h(Z)^2e^{\mu^2/2-\mu Z}].$$

For fixed pilot normals $z_1,\ldots,z_m$,

$$\widehat M_m(\mu)=\frac1m\sum_i h(z_i)^2e^{\mu^2/2-\mu z_i}.$$

The basic Newton procedure applies its update directly to this function and
its derivatives. The preserved extension instead minimizes
$\ell(\mu)=\log\widehat M_m(\mu)$, which has the same minimizer but different
Newton steps.

## Logarithmic reformulation

Define normalized optimization weights

$$w_i(\mu)=\frac{h(z_i)^2e^{-\mu z_i}}
 {\sum_j h(z_j)^2e^{-\mu z_j}},\qquad
\bar z_w=\sum_i w_i z_i.$$

Then

$$\ell'(\mu)=\mu-\bar z_w,\qquad
\ell''(\mu)=1+\sum_iw_i(z_i-\bar z_w)^2\geq1.$$

These weights are used to calculate the objective's derivatives. Pricing
still uses ordinary likelihood ratios, without self-normalization.
For a finite pilot with at least one positive payoff, the logarithmic
objective is strongly convex and coercive, so its minimizer is unique.
This is an empirical optimum within the $N(\mu,1)$ family, not the exact
population optimum or the unrestricted zero-variance importance density.

For numerical stability, evaluate

$$a_i=2\log h(z_i)-\mu z_i,\qquad a_* = \max_i a_i,$$

$$\ell(\mu)=\mu^2/2+a_*+\log\sum_i e^{a_i-a_*}-\log m.$$

Zero-payoff terms are omitted from the sum, while $m$ remains the total pilot
size. A pilot without positive payoffs fails explicitly.

## Damped Newton update

Starting at $\mu_0=0$, define $g_k=\ell'(\mu_k)$ and $H_k=\ell''(\mu_k)$.
The preserved implementation uses

$$\mu_{k+1}=\mu_k-\alpha_k g_k/H_k.$$

An Armijo line search tries $\alpha_k=1,1/2,1/4,\ldots$ and accepts a step if

$$\ell(\mu_k-\alpha_k g_k/H_k)
\leq\ell(\mu_k)-10^{-4}\alpha_k g_k^2/H_k$$

up to a floating-point rounding allowance. The limits are 40 backtracking
trials per step and 100 Newton steps. Stop when

$$|g_k|\leq10^{-10}(1+|\mu_k|).$$

The IS pilot uses stream 3, the control pilot stream 1, and main valuation
stream 0. The fitted parameters are frozen before main-sample valuation.
For a frozen fitted shift, conditional unbiasedness follows from the
likelihood-ratio identity. Pilot fitting and objective scans count toward
runtime; payoff counts include pilot observations but not repeated scans of
already stored payoffs.

## Files and independent use

- `variance_reduction.cpp`: preserved extended experiment, including its
  plain-MC, antithetic, and control-variate comparison methods.
- `include/`: private copies of its model, tree reference, and CSV interface.
- `results/`: preserved advanced CSVs and figures.
- `provenance.json`: snapshot hashes and the original benchmark manifest.
- `build.py`: builds this experiment and checks its pricing and convergence.

From the repository root:

```powershell
.\.venv-portfolio\Scripts\python.exe extensions/importance_sampling_2009/build.py
.\extensions\importance_sampling_2009\build\variance_reduction_advanced.exe --type call --paths 50000 --train-paths 2000 --seed 42
```

Use `--newton-trace extensions/importance_sampling_2009/results/custom_trace.csv`
to write an additional trace. This executable uses the preserved schema with
`log_second_moment` and `step_scale`; the main executable uses the raw second
moment, actual Newton step, and original stopping rules. Default build,
experiment, plot, notebook, and CI commands use only the main implementation.
