This page explains the ideas behind `gpvar` in the order in which they appear in the paper of Hauzenberger, Huber, Marcellino and Petz (2022). Each concept ends with the corresponding function of the library.

## 1. Why a non-parametric VAR?

A linear VAR assumes that today's value of every variable is a fixed linear combination of past values. Macroeconomic relations, however, change with the state of the economy (recessions versus expansions, high versus low uncertainty, before and after structural breaks). Parametric extensions (threshold, Markov-switching, time-varying-parameter VARs) impose a specific form of change. The GP-VAR replaces the linear conditional mean by an **unknown function** with a Gaussian-process prior and lets the data decide the shape and degree of non-linearity, while keeping the VAR structure that makes impulse responses and forecasts interpretable.

## 2. Gaussian process regression in one paragraph

For a scalar series $y_t$ and regressors $x_t$,

$$y_t = f(x_t) + \varepsilon_t,\qquad f \sim \mathcal{GP}(0, k_\vartheta),$$

the prior says that any collection $f = (f(x_1),\dots,f(x_T))'$ is multivariate normal with covariance matrix $K_\vartheta(X,X)$ whose $(t,\tau)$ element is the kernel $k_\vartheta(x_t, x_\tau)$. With the Gaussian (squared-exponential) kernel

$$k_\vartheta(x_t,x_\tau)=\xi\,\exp\!\Big(-\frac{\kappa}{2}\,\|x_t-x_\tau\|^2\Big),$$

two periods whose regressors are similar get similar function values. The **inverse length scale** $\kappa$ governs how quickly the function can change ($\kappa \to 0$: linear-like, very smooth; large $\kappa$: wiggly, prone to over-fitting) and the **scaling** $\xi$ is the prior variance of $f$. The posterior of $f$ is Gaussian with mean $K(K+\sigma^2 I)^{-1}y$ — a weighted average of the observed $y$'s with weights determined by regressor similarity. *Library:* `gpvar.GPRegression`, `plots.plot_gp_illustration` (Figures 1 and 2 of the paper).

## 3. The GP-VAR

Let $y_t$ be $M$ (standardised) macroeconomic variables and $p$ the number of lags. The paper separates, for each equation $j$, the **own lags** $x_{jt} = (y_{jt-1},\dots,y_{jt-p})'$ from the **other variables' lags** $z_{jt}$ and gives each block its own Gaussian process with its own hyperparameters:

$$y_{jt} = f_j(x_{jt}) + g_j(z_{jt}) + \sum_{k<j} q_{jk}\, y_{kt} + \varepsilon_{jt},\qquad \varepsilon_{jt}\sim N(0,\omega_{jt}).$$

* $f_j$ captures the (possibly non-linear) own dynamics, $g_j$ the cross-variable effects — mirroring the Minnesota-prior tradition of shrinking own and other lags differently.
* $Q = (q_{jk})$ is lower triangular with zero diagonal, so the structural shocks $\varepsilon_t$ are orthogonal and the equations can be estimated one at a time (recursive identification: variable $j$ reacts contemporaneously only to variables ordered before it).
* $\omega_{jt} = e^{h_{jt}}$ is a **stochastic volatility** process: $h_{jt} = \mu_j + \rho_j (h_{jt-1}-\mu_j) + \nu_{jt}$. It absorbs large shocks (1975, 1980–82, 2008–09) so that they are not misread as changes in the conditional mean, and it yields fat-tailed, asymmetric predictive distributions.

*Library:* `gpvar.GPVAR(Y, p=5, sv=True)`.

## 4. The SV-scaled kernel: why the model is fast

The paper's key device is to scale the GP prior by the volatilities:

$$f_j \sim N\big(0, \sqrt{\Omega_j}\,K_{\vartheta_{j1}}\sqrt{\Omega_j}\big),\qquad \Omega_j = \mathrm{diag}(\omega_{j1},\dots,\omega_{jT}).$$

Two consequences. Economically, the prior variance of the unknown function is larger in high-volatility periods, so large shifts of the conditional mean are permitted exactly when the economy is in turmoil (Figure 5 of the paper shows $\omega_{jt}\xi_j$ spiking in recessions). Computationally, the volatilities factor out of the posterior: conditional on the hyperparameters, $f_j \mid \cdot \sim N(\bar f_j, \bar V_{f_j})$ with

$$\bar V_{f_j} = \sqrt{\Omega_j}\big(K - K(K+I)^{-1}K\big)\sqrt{\Omega_j},\qquad \bar f_j = \sqrt{\Omega_j}K(K+I)^{-1}\Omega_j^{-1/2}\big(Y_j - g_j - \textstyle\sum_{k<j}q_{jk}Y_k\big),$$

where $(K+I)^{-1}$ does **not** depend on $\Omega_j$ and can be pre-computed once for every point of the hyperparameter grid. `gpvar` goes one step further and stores an eigendecomposition $K_\kappa = U\Lambda U'$ per $\kappa$, so that every quantity for any $(\kappa,\xi)$ costs $O(T^2)$ and the sampler's cost is independent of the number of regressors — the property behind the paper's Figure 4. *Library:* `gpvar.kernels.KernelBlock`.

## 5. Choosing the hyperparameters: median heuristic and grid

$\kappa$ is crucial (Section 2 of the paper) and is inferred rather than fixed. The **median heuristic** $\bar\kappa = \mathrm{median}_{t\neq\tau}\,1/\|x_{jt}-x_{j\tau}\|$ adapts the length scale to the persistence of each series (persistent series have close regressor vectors, hence a smooth function). Around it the paper defines a grid $\kappa \in [0.1\bar\kappa, 2\bar\kappa]$, $\xi\in[0.04, 4]$ (about 1000 combinations) with informative Gamma hyperpriors, and samples the pair by evaluating the posterior ordinate at each grid point (inverse-transform sampling). `gpvar` implements the three grid variants compared in the paper's Table 2 (`hyper_scheme="semi-automatic" | "semi-automatic-noscale" | "naive"`) and, by default, samples the hyperparameters with the latent function integrated out (a blocked step that mixes better; the paper's conditional step is available with `hyper_sampling="conditional"`). *Library:* `gpvar.kernels.median_heuristic`, `make_hyper_grid`.

## 6. Posterior simulation

One sweep of the Gibbs sampler cycles, equation by equation, through: (1) the contemporaneous coefficients $q_{j\cdot}$ under a horseshoe prior; (2) the whole log-volatility path $h_j$ *marginally of* $f_j$ and $g_j$ by an independence Metropolis–Hastings step whose Gaussian proposal is built from a Newton–Raphson search of the conditional mode (Chan, 2017); (3)–(4) $f_j$ and $g_j$ from their conjugate Gaussian conditionals, with the restriction that $g_j$ has zero grand mean (Cong, Chen and Zhou, 2017) so that the level of the conditional mean is attributed to $f_j$; (5)–(6) the hyperparameters on the grid; (7) the SV state-equation parameters; (8) the horseshoe hyperparameters. Sampling $h_j$ marginally of the latent functions is what makes the collapsed sampler valid despite the volatilities appearing in the priors of $f_j$ and $g_j$. *Library:* `GPVAR.fit`, `gpvar.sv.sample_logvol_imh_dense`, `gpvar.priors.HorseshoeState`.

## 7. Forecasts

Given a posterior draw, the one-step-ahead conditional mean at a new regressor vector $w$ is the GP predictive mean $\sigma_{j,T+1}\,k^*(w, W_j)(K^*+I)^{-1}\Omega_j^{-1/2}\tilde Y_j$, where $K^* = \xi_1 K_1 + \xi_2 K_2$ combines the own- and other-lag kernels and $\sigma_{j,T+1}$ is the simulated volatility. Adding structural shocks and rotating with $(I-Q)^{-1}$ gives a draw of $y_{T+1}$; iterating gives multi-step predictive paths that are non-Gaussian and heteroskedastic. Density forecasts are scored with log predictive likelihoods; differences across models are **log predictive Bayes factors** (Table 2 of the paper). *Library:* `gpvar.predict`, `gpvar.log_predictive_score`, `gpvar.recursive_forecast_evaluation`, `gpvar.lpbf_table`.

## 8. Generalized impulse responses

Because the conditional mean is non-linear, the effect of a shock depends on the history, its sign and its size. Following Koop, Pesaran and Potter (1996), the response at horizon $h$ conditional on history $t$ is

$$\delta_{h,t} = E\big[y_{t+h}\mid \mathcal I_t, \varepsilon_{jt}=\varsigma\big] - E\big[y_{t+h}\mid \mathcal I_t\big],$$

computed by simulating two futures from the same posterior draw and the same future shocks, one with the structural shock $\varsigma$ added at $t$ (it moves $y_t$ by $\varsigma q_j$, the $j$-th column of $(I-Q)^{-1}$) and one without, and differencing. Averaging $\delta_{h,t}$ over all $t$ gives the *average GIRF* (paper's Figure 7); averaging over sub-samples or years gives the time-variation figures (11, 12); computing it for $\varsigma>0$ and $\varsigma<0$, or for one and two standard deviations, gives the sign and size asymmetries (Figures 9, 10). *Library:* `gpvar.girf`, `GIRFResult.average / by_periods / yearly_medians / peak_table`.

## 9. What the paper finds — and what the library reproduces on the bundled data

* **Own lags need more flexibility than other lags**: the inverse length scale is larger for own-lag kernels; cross-variable effects are close to linear (Figure 6).
* **Less shrinkage in recessions**: $\omega_{jt}\xi_j$ peaks in 1974–75, 1980–82 and 2008–09 (Figure 5).
* **Uncertainty shocks** lower GDP growth, employment, hours, stock returns and the policy rate; compared with a linear BVAR the real-activity responses peak later and are more persistent (Figure 7).
* **Sign asymmetry**: adverse uncertainty shocks have much larger effects than favourable ones (Figure 9); **size asymmetry**: large shocks are less than proportional for real activity (Figure 10).
* **Time variation**: effects were weaker in the great inflation, strongest during the great moderation and somewhat muted after 2007 (Figures 11, 12).
* **Forecasting**: the GP-VAR with SV improves on the BVAR mainly at the one-year horizon; turning off SV hurts inflation and interest-rate forecasts (Table 2).

All of these are reproduced in the [Results](results.html) page with the library's default settings.

## 10. Practical guidance

* Transform data to stationarity (growth rates, inflation rates) and let the model standardise them.
* Start with the defaults (`p=5` quarterly, `n_kappa=20`, `n_xi=50`, `sv=True`, 2000 + 2000 draws). Check `res.accept_rate` (0.2–0.9 is fine) and the trace plots.
* Order the shocked variable where identification requires; check robustness with `gpvar.ordering_robustness`.
* Use `n_draws=100`, `n_sim=5` for GIRFs; increase only for final figures.
* For larger systems (M ≥ 16 or monthly data) reduce `n_kappa`, use `sv_hessian="band"` and thin the GIRF draws.
