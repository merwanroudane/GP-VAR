# Methodology notes: what `gpvar` implements and how it maps to the paper

This document lists, equation by equation, how the Python code corresponds to
Hauzenberger, Huber, Marcellino and Petz (2022), *Gaussian Process Vector
Autoregressions and Macroeconomic Uncertainty* (arXiv:2112.01995v3), and where
the implementation makes choices that the paper leaves open.

## 1. Model (paper Section 3.1)

Structural form, eq. (2):

$$y_t = F(x_t) + G(z_t) + Q y_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, H_t),\; H_t = \mathrm{diag}(\omega_{1t},\dots,\omega_{Mt}),$$

with $x_{jt}$ the $p$ own lags of $y_{jt}$, $z_{jt}$ the $(M-1)p$ lags of the other variables and $Q$ lower triangular with zero diagonal.
Equation $j$ in full-data form:

$$Y_j = f_j + g_j + \sum_{k<j} q_{jk} Y_k + \epsilon_j, \qquad \epsilon_j \sim N(0, \Omega_j).$$

Code: `gpvar.kernels.lag_matrix` builds $y$ and $X = [y_{t-1}', \dots, y_{t-p}']$;
`own_other_index` selects $x_{jt}$ (columns $j, j+M, \dots$) and $z_{jt}$.
`GPVAR.fit` loops over equations; the contemporaneous term uses the observed
$Y_k$, $k<j$, exactly as in eq. (2).

Log-volatilities, eq. (3): $h_{jt} = \rho_{hj} h_{jt-1} + \nu_{jt}$, $h_{j0} \sim N(0, \sigma^2_{hj}/(1-\rho^2_{hj}))$.
The code adds an (optional, default on) unconditional mean $\mu_{hj}$,
$h_{jt} - \mu = \rho(h_{jt-1} - \mu) + \nu_{jt}$, with prior $N(0, 10)$.  Set
`SVPriors(estimate_mu=False)` to obtain eq. (3) literally.  (The authors' own R
code estimates the level through the `stochvol` package as well.)

## 2. Priors (Section 3.2)

* $f_j \sim N(0, \sqrt{\Omega_j} K_{\vartheta_{j1}}(X_j,X_j)\sqrt{\Omega_j})$ and likewise for $g_j$ (eq. 4): `KernelBlock.posterior_draw` uses exactly this SV-scaled kernel.
* Gaussian kernel with scaling matrix $D$ of empirical column variances: `kernels.scaled_sq_dists`, `gaussian_kernel`.
* Horseshoe on the free elements of $Q$: `priors.HorseshoeState.update` (Makalic and Schmidt, 2015; one global $\tau$ for all elements, as in the authors' R code).
* $(\rho+1)/2 \sim B(25,5)$, $\sigma^2_h \sim IG$ with mean 0.1 and variance 0.01 (i.e. $IG(3, 0.2)$): `priors.SVPriors`.

## 3. Hyperparameters (Section 3.3)

* Median heuristic $\bar\kappa = \mathrm{median}_{t\ne\tau}(1/\|x_{jt}-x_{j\tau}\|)$ on $D^{-1/2}$-scaled inputs: `kernels.median_heuristic`.
* Grid $\kappa \in [0.1\bar\kappa, 2\bar\kappa]$, $\xi \in [0.04, 4]$ with Gamma$(1/2, 1/(2c))$ hyperpriors: `kernels.make_hyper_grid`.
  Defaults `n_kappa=20`, `n_xi=50` (1000 combinations, "around 1000" in the paper).  The three specifications of
  Table 2 are `hyper_scheme="semi-automatic"`, `"semi-automatic-noscale"` and `"naive"`.
* Prior means $c_\kappa$ and $c_\xi$ are set by cross-validation in the paper; the code exposes them
  (`c_kappa=0.1`, `c_xi=1.0` by default, "small $c_\kappa$" as recommended).

Because $K_{\xi,\kappa} = \xi K_\kappa$, the code stores one eigendecomposition per $\kappa$ (not per grid point) and
obtains all posterior quantities for any $\xi$ in $O(T^2)$ — the "pre-compute inverses and Cholesky factors" idea of Section 3.4 in a
form that also avoids the $O(T^3)$ Cholesky per draw:

$$\bar V_{f_j} = \sqrt{\Omega_j}\,U\,\mathrm{diag}\!\Big(\tfrac{\xi\lambda}{1+\xi\lambda}\Big)U'\sqrt{\Omega_j}, \qquad
\bar f_j = \sqrt{\Omega_j}\,U\,\mathrm{diag}\!\Big(\tfrac{\xi\lambda}{1+\xi\lambda}\Big)U'\,\Omega_j^{-1/2}\Big(Y_j - g_j - \textstyle\sum_{k<j} q_{jk}Y_k\Big).$$

## 4. Posterior simulation (Section 3.4, Appendix A.4)

| Paper step | Code |
|---|---|
| 1. $q_{j\cdot}$ from its Gaussian conditional (heteroskedastic regression with horseshoe prior) | `GPVAR.fit`, "Step 1" |
| 2. $h_j$ marginally of $f_j, g_j$ via independence MH (Appendix A.3) | `sv.sample_logvol_imh` (banded Hessian, as in the paper) / `sv.sample_logvol_imh_dense` (exact Hessian, default) |
| 3. $f_j$ from its conjugate Gaussian conditional | `KernelBlock.posterior_draw` |
| 4. $g_j$ with the restriction $\iota' g_j = 0$ (Cong, Chen and Zhou, 2017, Alg. 2; Appendix A.2) | `GPVAR.fit`, "Steps 4 & 6" |
| 5./6. $\vartheta_{j1}, \vartheta_{j2}$ by multinomial sampling on the grid | `KernelBlock.log_marginal_grid` / `log_conditional_grid` + `_sample_grid` |
| 7. SV state-equation parameters | `sv.sample_sv_params` |
| 8. Horseshoe hyperparameters (inverse-Gamma Gibbs) | `HorseshoeState.update` |

Two implementation choices:

* **`hyper_sampling="conditional"`** samples $\vartheta$ from $p(\vartheta_{j1} \mid f_j, \Omega_j) \propto N(f_j; 0, \sqrt{\Omega_j}\xi K_\kappa\sqrt{\Omega_j})\,p(\vartheta)$ — the paper's Step 5 literally.  This requires $K_\kappa^{-1}$, which is numerically fragile for smooth kernels (tiny eigenvalues); a jitter of $10^{-8}$ is added.
* **`hyper_sampling="marginal"` (default)** draws the block $(\vartheta_{j1}, f_j)$ jointly: $\vartheta_{j1}$ from $p(\vartheta_{j1}\mid Y_j - g_j - \sum q Y_k, \Omega_j)$ with $f_j$ integrated out, i.e. $N(\Omega_j^{-1/2} r;\, 0, \xi K_\kappa + I_T)$, followed by $f_j \mid \vartheta_{j1}$.  This is a partially collapsed Gibbs step (Van Dyk and Park, 2008) that leaves the same posterior invariant, mixes better and only involves $\xi K_\kappa + I$ (always well conditioned).

The IMH step for $h_j$ follows Appendix A.3: Newton–Raphson to the mode of $\log p(h_j\mid \tilde Y_j, S_j, \theta_{hj})$
using the gradient $-\tfrac12\iota + \tfrac12 (S_j^{-1}\hat Y_j)\odot \hat Y_j$ and negative Hessian
$\tfrac14\mathrm{diag}(\hat Y_j)S_j^{-1}\mathrm{diag}(\hat Y_j) + \tfrac14\mathrm{diag}((S_j^{-1}\hat Y_j)\odot\hat Y_j)$
(plus the AR(1) prior precision $D'D/\sigma^2_h$), then a Gaussian proposal $N(\hat h_j, \hat R(\hat h_j)^{-1})$ and the
independence-MH acceptance ratio.  Two practical additions: (i) the first `sv_init_sweeps` burn-in sweeps accept the
proposal unconditionally so that the chain starts at the conditional mode (an independence sampler started far from
the mode can reject for thousands of iterations), and (ii) `sv_tries` accept/reject attempts are made per sweep from
the same proposal (the proposal is state-independent, so the expensive mode search is shared).

## 5. Forecasts and GIRFs (Section 3.5, Appendix A.5)

`predict.DrawContext` implements the predictive mean
$\bar m_{j,t+1} = \sigma_{j,t+1} K^*_{\vartheta_j}(W_{j,t+1}, W_j)(K^*_{\vartheta_j}(W_j,W_j)+I_T)^{-1}\Omega_j^{-1/2}(Y_j - \sum_{k<j}q_{jk}Y_k)$
with $K^* = \xi_1 K_1 + \xi_2 K_2$ (the sum-of-kernels representation of $m_j$, Section 3.2), the predictive variance
$\sigma^2_{j,t+1}[\xi_1+\xi_2 - K^*(\cdot)(K^*+I)^{-1}K^*(\cdot)']$ (optional, `sample_m=True`) and
$y_{t+1} = (I-Q)^{-1}(m_{t+1} + \varepsilon_{t+1})$ with $\varepsilon_{j,t+1}\sim N(0,\omega_{j,t+1})$, $\omega$ simulated from the SV law of motion.

`girf.girf` computes $\delta_{h,t} = E[y_{t+h}\mid I_t, \varepsilon_{jt}=\varsigma] - E[y_{t+h}\mid I_t]$ for every history $t$ (or a subset),
with the shock entering as $\varsigma q_j$ ($q_j$ = $j$-th column of $(I-Q)^{-1}$) and common future shocks in both branches.
`shock_scale="unit"` normalises the impact response of the shocked variable to `size` (the Figure 7 normalisation);
`shock_scale="sd"` uses `size` $\times \exp(h_{jt}/2)$.  Averages over $t$ (Figures 7–10), sub-samples (Figure 11),
yearly medians (Figure 12) and NBER recession/expansion splits (Figure C.6) are methods of `GIRFResult`.

## 6. Forecast evaluation (Section 5.2)

`forecast.recursive_forecast_evaluation` re-estimates a model at each origin of an expanding window and scores the
realisations with log predictive likelihoods (Gaussian approximation to the simulated predictive, or a KDE for
marginals) and CRPS; `forecast.lpbf_table` forms log predictive Bayes factors relative to a benchmark.  The
benchmark shipped with the package is a natural-conjugate Minnesota BVAR (`benchmarks.BVAR`, homoskedastic); the
paper's benchmark additionally has SV.

## 7. Synthetic data (Section 4)

`simulate.simulate_paper_dgp` reproduces the DGP of Section 4 (linear equation with $t_3$ errors, structural break at
$t = 100$, sine/quadratic equation, random-walk SV, horseshoe-like $Q$).  `simulate.simulate_replication_dgp`
reproduces `simulate_data.R` of the authors' replication archive.

## References

* Chan, J. C. C. (2017). The stochastic volatility in mean model with time-varying parameters. *JBES* 35, 17–28.
* Cong, Y., Chen, B. and Zhou, M. (2017). Fast simulation of hyperplane-truncated multivariate normal distributions. *Bayesian Analysis* 12, 1017–1037.
* Hauzenberger, N., Huber, F., Marcellino, M. and Petz, N. (2022). Gaussian process vector autoregressions and macroeconomic uncertainty. arXiv:2112.01995.
* Koop, G., Pesaran, M. H. and Potter, S. M. (1996). Impulse response analysis in nonlinear multivariate models. *Journal of Econometrics* 74, 119–147.
* Makalic, E. and Schmidt, D. F. (2015). A simple sampler for the horseshoe estimator. *IEEE Signal Processing Letters* 23, 179–182.
* Van Dyk, D. A. and Park, T. (2008). Partially collapsed Gibbs samplers. *JASA* 103, 790–796.
