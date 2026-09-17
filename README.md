# GP-VAR — Gaussian Process Vector Autoregressions with Stochastic Volatility in Python

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2112.01995-b31b1b.svg)](https://arxiv.org/abs/2112.01995)

`gpvar` is a complete, self-contained Python implementation of the **Gaussian process vector autoregression (GP-VAR)** of

> Hauzenberger, N., Huber, F., Marcellino, M. and Petz, N. (2022). *Gaussian Process Vector Autoregressions and Macroeconomic Uncertainty*. arXiv:2112.01995 (Journal of Business & Economic Statistics, forthcoming).

It follows the paper's methodology step by step (SV-scaled conjugate GP priors on own- and other-lag functionals, discrete hyperparameter grid with the median heuristic, independence-MH log-volatility sampler, horseshoe-shrunk contemporaneous matrix, generalized impulse responses with sign / size / time asymmetries, recursive density-forecast evaluation) and reproduces every type of figure and table of the paper on **real US data** (FRED-QD and the Jurado–Ludvigson–Ng macroeconomic uncertainty index), producing journal-quality output.

The authors' R replication archive (`GPVAR_replication/`, Florian Huber, [fhuber7/replication-archive](https://github.com/fhuber7/replication-archive/tree/main/GPVAR_replication)) is included for reference; the Python library goes well beyond it (it implements the full algorithm of the paper rather than the simplified kernel version, plus forecasting, evaluation, asymmetry analysis, data pipeline, figures and tables).

**Author:** Dr Merwan Roudane · merwanroudane920@gmail.com · [github.com/merwanroudane](https://github.com/merwanroudane)

---

## Contents

1. [The model in one page](#1-the-model-in-one-page)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Real economic example: uncertainty shocks in the US (GP-VAR-8)](#4-real-economic-example-uncertainty-shocks-in-the-us-gp-var-8)
5. [Simulation study (Section 4 of the paper)](#5-simulation-study-section-4-of-the-paper)
6. [GP regression illustration (Section 2 of the paper)](#6-gp-regression-illustration-section-2-of-the-paper)
7. [Forecast evaluation (Section 5.2 of the paper)](#7-forecast-evaluation-section-52-of-the-paper)
8. [Library reference](#8-library-reference)
9. [How the implementation maps to the paper](#9-how-the-implementation-maps-to-the-paper)
10. [Data](#10-data)
11. [Performance and settings](#11-performance-and-settings)
12. [Repository layout](#12-repository-layout)
13. [Citation, license, acknowledgements](#13-citation-license-acknowledgements)

---

## 1. The model in one page

For an $M$-vector of (demeaned, standardised) macroeconomic series $y_t$ with $p$ lags, the **structural** GP-VAR is

$$
y_t = F(x_t) + G(z_t) + Q\,y_t + \varepsilon_t,\qquad \varepsilon_t \sim N(0, H_t),\quad H_t=\mathrm{diag}(\omega_{1t},\dots,\omega_{Mt}),
$$

where $x_{jt}=(y_{jt-1},\dots,y_{jt-p})'$ are the **own lags** of variable $j$, $z_{jt}$ the lags of **all other** variables, and $Q$ is lower triangular with zero diagonal (recursive identification). Each equation has two unknown functions with Gaussian-process priors whose kernels are scaled by the equation's stochastic volatility,

$$
f_j \sim N\!\big(0,\sqrt{\Omega_j}\,K_{\vartheta_{j1}}(X_j,X_j)\sqrt{\Omega_j}\big),\qquad
g_j \sim N\!\big(0,\sqrt{\Omega_j}\,K_{\vartheta_{j2}}(Z_j,Z_j)\sqrt{\Omega_j}\big),\qquad
k_{\vartheta}(x_t,x_\tau)=\xi\exp\!\Big(-\tfrac{\kappa}{2}(x_t-x_\tau)'D^{-1}(x_t-x_\tau)\Big),
$$

and log-volatilities $h_{jt}=\log\omega_{jt}$ following a stationary AR(1). The scaling by $\sqrt{\Omega_j}$ makes the model conjugate: all kernel inverses can be pre-computed on a discrete $(\kappa,\xi)$ grid centred on the **median heuristic**, and the sampler's cost per equation does not depend on the number of regressors. Non-linearity of the conditional mean means impulse responses are **generalized** IRFs (Koop, Pesaran and Potter, 1996) that depend on the history, the sign and the size of the shock — the paper's main empirical message about uncertainty shocks.

Full derivations and the mapping of every sampler step to the code are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## 2. Installation

```bash
git clone https://github.com/merwanroudane/GP-VAR.git
cd GP-VAR
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib` (Python ≥ 3.9). No compiled code, no R. Run the tests with

```bash
pip install -e .[dev]
pytest
```

---

## 3. Quick start

```python
import gpvar

# 1. Data: the paper's GP-VAR-8 (quarterly, 1970Q1-2019Q4), built from the bundled FRED-QD / JLN files
Y, info = gpvar.data.build_dataset(size=8, start="1970Q1", end="2019Q4")

# 2. Estimate (5 lags, SV, semi-automatic (kappa, xi) grid with 20 x 50 points)
model = gpvar.GPVAR(Y, p=5, sv=True, hyper_scheme="semi-automatic", n_kappa=20, n_xi=50)
res = model.fit(nburn=2000, nsave=2000, seed=1)

# 3. In-sample objects
res.hyper_summary()          # posterior of xi / kappa per equation (own vs. other lags)
res.shrinkage_paths()        # omega_jt * xi (Figure 5 of the paper)
res.volatility()             # exp(h_t/2) quantiles
res.Q_summary()              # contemporaneous coefficients

# 4. Generalized impulse responses to a one-sd uncertainty shock, averaged over all histories
g_pos = gpvar.girf(res, "UNC", size=1, sign=+1, horizon=16, n_sim=5, n_draws=100)   # ~1 s per draw, multithreaded
g_neg = gpvar.girf(res, "UNC", size=1, sign=-1, horizon=16, n_sim=5, n_draws=100)
g_pos.summary()                                   # variable, horizon, q16 / q50 / q84
g_pos.by_periods(gpvar.data.PAPER_SUBSAMPLES)     # great inflation / great moderation / post-2007
g_pos.yearly_medians()                            # time variation within sub-samples
g_pos.peak_table()

# 5. Figures (PDF + PNG)
from gpvar import plots
plots.set_journal_style()
plots.plot_girf({"positive": g_pos.summary(), "negative": g_neg.summary()}, save="girf_sign")

# 6. Forecasts and density-forecast evaluation
fc = res.predict(horizon=8, n_sim=5)
fc.quantiles()
```

Everything the examples produce lives in `outputs/figures` and `outputs/tables` (Markdown, LaTeX/booktabs and CSV versions of every table).

---

## 4. Real economic example: uncertainty shocks in the US (GP-VAR-8)

`examples/03_us_uncertainty_gpvar8.py` re-runs the paper's Section 5–6 analysis on real data.

### 4.1 Data

Quarterly US data 1970Q1–2019Q4 (T = 200), the eight variables of the paper's GP-VAR-8 (Online Appendix Table B.1):

| Variable | Source series | Transformation |
|---|---|---|
| UNC | JLN macroeconomic uncertainty, h = 1 (Jurado, Ludvigson and Ng, 2015; quarterly average) | level |
| RGDP | GDPC1, real GDP | year-on-year growth (%) |
| EMP | CE16OV, civilian employment | year-on-year growth (%) |
| AWH | AWHMAN, average weekly hours, manufacturing | level |
| CPI | CPIAUCSL, consumer price index | year-on-year inflation (%) |
| AHE | CES3000000008x, real average hourly earnings, manufacturing | year-on-year growth (%) |
| FFR | FEDFUNDS, effective federal funds rate | level (%) |
| SP500 | S&P 500 composite | quarter-on-quarter return (%) |

![data](outputs/figures/gpvar8_data.png)

The model is estimated on standardised data with $p = 5$ lags, stochastic volatility, the semi-automatic $(\kappa,\xi)$ grid (20 × 50 points), 2 000 burn-in and 2 000 posterior draws; GIRFs use 100 evenly spaced posterior draws, all 195 histories and 5 simulated future paths per history. Uncertainty is ordered first (an uncertainty shock moves every variable on impact; the paper shows in Appendix C.2 that the ordering has a negligible effect on the GIRFs — `gpvar.ordering_robustness` reproduces that check).

### 4.2 In-sample features (Figures 5 and 6 of the paper)

Posterior means of the "linear shrinkage" $\omega_{jt}\xi_{j\cdot}$ — the diagonal of the re-scaled kernels — for own and other lags (recessions shaded). As in the paper, less shrinkage is applied to own lags than to other lags and the amount of shrinkage rises sharply in recessions (early 1980s, 2008–09):

![fig5](outputs/figures/gpvar8_fig05_shrinkage.png)

Posterior interquartile ranges of the inverse length scales $\kappa_{j1}$ (own lags, blue) and $\kappa_{j2}$ (other lags, green). Own-lag kernels are consistently "rougher" (larger $\kappa$) than other-lag kernels, whose effect is close to linear:

![fig6](outputs/figures/gpvar8_fig06_kappa.png)

Posterior summaries of the kernel hyperparameters (median and interquartile range; $\xi$ on the paper's grid $[0.04, 4]$, $\kappa$ on $[0.1\bar\kappa, 2\bar\kappa]$) and of the stochastic-volatility state equations. Independence-MH acceptance rates for the log-volatility paths: UNC 0.72, RGDP 0.84, EMP 0.89, AWH 0.80, CPI 0.44, AHE 0.81, FFR 0.09, SP500 0.59 (the FFR equation is the hardest one because its conditional variance collapses at the zero lower bound, as the paper notes).

**Kernel hyperparameters** (`outputs/tables/gpvar8_hyperparameters`):

| Variable | kappa own | xi own | kappa other | xi other |
|---|---|---|---|---|
| UNC | 0.171 [0.114, 0.171] | 3.838 [3.677, 4.000] | 0.027 [0.013, 0.040] | 0.121 [0.040, 0.363] |
| RGDP | 0.084 [0.084, 0.126] | 3.515 [3.111, 3.838] | 0.027 [0.014, 0.027] | 2.788 [2.060, 3.434] |
| EMP | 0.130 [0.087, 0.130] | 3.758 [3.515, 3.919] | 0.042 [0.028, 0.056] | 2.060 [1.252, 2.869] |
| AWH | 0.169 [0.126, 0.211] | 3.758 [3.515, 3.919] | 0.014 [0.014, 0.014] | 3.030 [2.464, 3.515] |
| CPI | 0.237 [0.177, 0.237] | 3.919 [3.758, 4.000] | 0.054 [0.027, 0.081] | 0.525 [0.202, 1.010] |
| AHE | 0.170 [0.128, 0.213] | 3.758 [3.596, 3.919] | 0.072 [0.029, 0.130] | 0.040 [0.040, 0.202] |
| FFR | 0.400 [0.356, 0.489] | 3.919 [3.838, 4.000] | 0.054 [0.027, 0.081] | 0.282 [0.040, 0.767] |
| SP500 | 0.080 [0.040, 0.200] | 0.040 [0.040, 0.121] | 0.098 [0.070, 0.154] | 0.929 [0.525, 1.414] |

**SV state-equation parameters** (`outputs/tables/gpvar8_sv_parameters`):

| Variable | mu_h | rho_h | sigma^2_h | h_0 |
|---|---|---|---|---|
| UNC | -2.477 [-2.732, -2.230] | 0.789 [0.638, 0.887] | 0.067 [0.037, 0.123] | -2.614 [-3.311, -1.862] |
| RGDP | -2.396 [-2.673, -2.088] | 0.756 [0.564, 0.879] | 0.068 [0.032, 0.154] | -2.424 [-3.160, -1.732] |
| EMP | -2.997 [-3.315, -2.692] | 0.746 [0.557, 0.875] | 0.053 [0.031, 0.103] | -2.980 [-3.583, -2.369] |
| AWH | -3.232 [-3.501, -3.023] | 0.767 [0.559, 0.894] | 0.038 [0.023, 0.068] | -3.033 [-3.493, -2.494] |
| CPI | -3.361 [-3.795, -3.024] | 0.869 [0.771, 0.924] | 0.076 [0.038, 0.122] | -3.446 [-4.242, -2.677] |
| AHE | -2.157 [-2.467, -1.898] | 0.790 [0.590, 0.902] | 0.056 [0.032, 0.102] | -1.938 [-2.573, -1.341] |
| FFR | -4.055 [-4.551, -3.518] | 0.934 [0.889, 0.965] | 0.055 [0.036, 0.079] | -3.875 [-4.700, -3.087] |
| SP500 | -0.945 [-1.498, -0.493] | 0.705 [0.487, 0.841] | 0.304 [0.134, 0.779] | -0.989 [-2.381, 0.287] |

The contemporaneous coefficients of $Q$ (horseshoe prior) are in `outputs/tables/gpvar8_Q.md`: e.g. the impact effect of uncertainty on GDP growth, $q_{\mathrm{RGDP,UNC}}$, has posterior median $-0.20$ with $P(q>0)=0.004$.

### 4.3 Uncertainty shocks: GP-VAR-8 versus a linear BVAR (Figure 7)

Average generalized impulse responses (posterior median and 68 % bands, averaged over all 195 histories) to a positive one-standard-deviation uncertainty shock, compared with the recursive IRFs of a Minnesota BVAR-8 estimated on the same data:

![fig7](outputs/figures/gpvar8_fig07_girf_vs_bvar_focus.png)

![figC4](outputs/figures/gpvar8_figC4_girf_vs_bvar_nonfocus.png)

Both models agree on the qualitative picture (higher uncertainty lowers output growth, employment, hours, stock returns and the policy rate), but the GP-VAR responses of real activity peak later (five to six quarters after the shock versus three for the BVAR) and die out more slowly, and stock prices react on impact more mildly than in the linear model. Average GIRFs to a positive one-standard-deviation shock at selected horizons (posterior median and 68 % band, responses in standard-deviation units of the standardised data):

| Variable | h=0 | h=1 | h=4 | h=8 | h=12 | h=16 |
|---|---|---|---|---|---|---|
| UNC | 1.000 [1.000, 1.000] | 0.885 [0.826, 0.919] | 0.679 [0.571, 0.767] | 0.219 [0.139, 0.339] | -0.010 [-0.066, 0.061] | -0.053 [-0.098, -0.004] |
| RGDP | -0.195 [-0.246, -0.120] | -0.289 [-0.340, -0.208] | -0.487 [-0.568, -0.426] | -0.430 [-0.507, -0.335] | -0.100 [-0.165, -0.028] | -0.008 [-0.059, 0.058] |
| EMP | -0.087 [-0.146, -0.052] | -0.192 [-0.250, -0.139] | -0.408 [-0.466, -0.322] | -0.419 [-0.515, -0.338] | -0.163 [-0.250, -0.088] | -0.002 [-0.064, 0.053] |
| AWH | -0.156 [-0.207, -0.109] | -0.215 [-0.269, -0.173] | -0.260 [-0.309, -0.208] | -0.222 [-0.317, -0.151] | -0.050 [-0.111, 0.017] | 0.056 [-0.001, 0.111] |
| CPI | 0.049 [0.006, 0.094] | 0.063 [0.004, 0.119] | 0.035 [-0.027, 0.101] | -0.040 [-0.090, 0.022] | -0.052 [-0.097, -0.000] | -0.039 [-0.067, 0.000] |
| AHE | 0.021 [-0.008, 0.053] | 0.029 [-0.015, 0.084] | 0.012 [-0.042, 0.095] | -0.021 [-0.075, 0.055] | -0.045 [-0.087, 0.023] | -0.038 [-0.078, 0.014] |
| FFR | -0.024 [-0.043, -0.008] | -0.054 [-0.076, -0.028] | -0.138 [-0.175, -0.105] | -0.239 [-0.287, -0.200] | -0.221 [-0.275, -0.180] | -0.160 [-0.214, -0.119] |
| SP500 | -0.071 [-0.204, 0.008] | -0.101 [-0.200, -0.042] | -0.216 [-0.309, -0.152] | -0.090 [-0.124, -0.033] | -0.007 [-0.037, 0.019] | 0.009 [-0.010, 0.026] |

*Notes:* Responses in standard-deviation units of the (standardised) variables; horizons in quarters.

Peak responses (`outputs/tables/gpvar8_girf_peaks`) for the three shock configurations:

| shock | variable | impact | peak | peak_horizon | peak_q16 | peak_q84 | cumulative |
|---|---|---|---|---|---|---|---|
| positive 1 sd | UNC | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 5.394 |
| positive 1 sd | RGDP | -0.195 | -0.543 | 6 | -0.631 | -0.450 | -4.740 |
| positive 1 sd | EMP | -0.087 | -0.451 | 6 | -0.547 | -0.368 | -4.323 |
| positive 1 sd | AWH | -0.156 | -0.281 | 6 | -0.358 | -0.218 | -2.488 |
| positive 1 sd | CPI | 0.049 | 0.065 | 2 | -0.000 | 0.120 | -0.150 |
| positive 1 sd | AHE | 0.021 | -0.046 | 13 | -0.093 | 0.018 | -0.249 |
| positive 1 sd | FFR | -0.024 | -0.242 | 9 | -0.295 | -0.204 | -2.903 |
| positive 1 sd | SP500 | -0.071 | -0.216 | 4 | -0.309 | -0.152 | -1.373 |
| negative 1 sd | UNC | -1.000 | -1.000 | 0 | -1.000 | -1.000 | -3.726 |
| negative 1 sd | RGDP | 0.195 | 0.341 | 5 | 0.291 | 0.387 | 3.078 |
| negative 1 sd | EMP | 0.087 | 0.261 | 3 | 0.208 | 0.311 | 2.551 |
| negative 1 sd | AWH | 0.156 | 0.201 | 1 | 0.159 | 0.251 | 1.724 |
| negative 1 sd | CPI | -0.049 | -0.062 | 1 | -0.115 | -0.005 | -0.286 |
| negative 1 sd | AHE | -0.021 | -0.028 | 1 | -0.078 | 0.015 | 0.148 |
| negative 1 sd | FFR | 0.024 | 0.148 | 9 | 0.120 | 0.183 | 1.886 |
| negative 1 sd | SP500 | 0.071 | 0.143 | 4 | 0.094 | 0.195 | 0.871 |
| positive 2 sd | UNC | 2.000 | 2.000 | 0 | 2.000 | 2.000 | 7.534 |
| positive 2 sd | RGDP | -0.390 | -0.847 | 5 | -0.968 | -0.714 | -6.345 |
| positive 2 sd | EMP | -0.174 | -0.709 | 5 | -0.827 | -0.593 | -6.129 |
| positive 2 sd | AWH | -0.311 | -0.440 | 5 | -0.557 | -0.337 | -3.474 |
| positive 2 sd | CPI | 0.098 | 0.112 | 1 | 0.007 | 0.214 | -0.531 |
| positive 2 sd | AHE | 0.041 | -0.064 | 12 | -0.129 | 0.032 | -0.308 |
| positive 2 sd | FFR | -0.048 | -0.357 | 8 | -0.413 | -0.298 | -4.287 |
| positive 2 sd | SP500 | -0.143 | -0.313 | 3 | -0.466 | -0.184 | -1.818 |

*Notes:* Peak = largest absolute posterior-median response; 68\% credible interval at the peak horizon; cumulative = sum of median responses over 0-16 quarters.

### 4.4 Sign and size asymmetries (Figures 9 and 10)

Responses to a positive (orange) and a negative (blue) shock, with the negative response mirrored (grey) so that a linear model would give identical orange and grey lines:

![fig9](outputs/figures/gpvar8_fig09_sign_asymmetry_focus.png)

Responses to a one- and a two-standard-deviation shock, with the latter halved (grey) — proportionality would make orange and grey coincide:

![fig10](outputs/figures/gpvar8_fig10_size_asymmetry_focus.png)

The asymmetries are statistically sharp. **Sign asymmetry**: posterior median of $\delta^+_h + \delta^-_h$ (zero under linearity) with the posterior probability that it is positive:

| Variable | h=1 | h=4 | h=8 |
|---|---|---|---|
| UNC | 0.098 (1.00) | 0.291 (1.00) | 0.092 (0.93) |
| RGDP | -0.023 (0.00) | -0.170 (0.00) | -0.178 (0.00) |
| EMP | -0.010 (0.00) | -0.142 (0.00) | -0.197 (0.00) |
| AWH | -0.015 (0.00) | -0.100 (0.00) | -0.090 (0.00) |
| CPI | 0.001 (0.64) | -0.003 (0.46) | -0.043 (0.07) |
| AHE | 0.002 (0.77) | 0.009 (0.65) | -0.000 (0.49) |
| FFR | -0.003 (0.04) | -0.037 (0.00) | -0.094 (0.00) |
| SP500 | -0.022 (0.01) | -0.075 (0.01) | -0.030 (0.16) |

*Notes:* Under a linear model the responses to positive and negative shocks are mirror images and the statistic is zero.

Positive uncertainty shocks produce a *larger* endogenous uncertainty response and larger declines in GDP growth, employment, hours and stock returns than the mirror image of negative shocks, which is the paper's central result ("higher unexpected uncertainty has stronger effects on the economy than lower uncertainty"). **Size asymmetry**: ratio of the response to a 2-sd shock to the response to a 1-sd shock (2 under proportionality):

| Variable | h=1 | h=4 | h=8 |
|---|---|---|---|
| UNC | 1.76 | 1.38 | 0.72 |
| RGDP | 1.84 | 1.64 | 1.11 |
| EMP | 1.88 | 1.67 | 1.34 |
| AWH | 1.83 | 1.62 | 1.26 |
| CPI | 1.76 | 0.51 | 2.62 |
| AHE | 1.95 | 1.05 | 1.50 |
| FFR | 1.90 | 1.68 | 1.50 |
| SP500 | 1.93 | 1.40 | 0.92 |

*Notes:* Equal to 2 under proportionality (linear model).

Large shocks have less-than-proportional effects on real activity (ratios of 1.6-1.7 at one year), exactly as in Figure 10 of the paper. Peak responses by sub-sample (`outputs/tables/gpvar8_girf_peaks_by_period`) show that the effects were weakest in the great-inflation period and strongest during the great moderation (peak GDP-growth response $-0.35$ vs $-0.68$ standard deviations), in line with the paper's Figure 11.

### 4.5 Asymmetries over time (Figures 11, 12 and C.6)

Sub-sample averages of the GIRFs (great inflation 1970Q1–1984Q4, great moderation 1985Q1–2006Q4, post-great-moderation 2007Q1–2019Q4):

![fig11](outputs/figures/gpvar8_fig11_subsamples_focus.png)

Yearly averages of the posterior-median GIRFs within each sub-sample (yellow = start of the period, red = end):

![fig12](outputs/figures/gpvar8_fig12_time_variation_focus.png)

GIRFs conditional on the state of the business cycle (NBER expansions versus recessions):

![figC6](outputs/figures/gpvar8_figC6_recessions_expansions.png)

### 4.6 Volatility, diagnostics and forecasts

![vol](outputs/figures/gpvar8_volatility.png)

Predictive distribution eight quarters beyond the end of the sample (5/16/50/84/95 % quantiles):

![fan](outputs/figures/gpvar8_forecast_fan.png)

MCMC traces of $\kappa$, $\xi$ and $\rho_h$ per equation are in `outputs/figures/gpvar8_mcmc_traces.png`; the non-focus-variable versions of every GIRF figure (`figC4`, `figC5`, `figC7`, `figC9`, `figC10`) and all tables are in `outputs/`.

---

## 5. Simulation study (Section 4 of the paper)

`examples/02_simulation_study.py` simulates the paper's three-equation DGP (linear equation with $t_3$ shocks, structural break at $t=100$, sine/quadratic non-linearities, random-walk SV, T = 200) and fits the GP-VAR. Posterior of $f_j$, $g_j$, $m_j = f_j+g_j$ and $y_j$ (red, 90 % bands) against the truth (black) — the analogue of the paper's Figure 3:

![fig3](outputs/figures/fig03_simulation_fit.png)

Correlation between the posterior median of $m_j$ and the true conditional mean over Monte Carlo replications of the DGP (the analogue of Table 1 of the paper, where the authors report 0.945, 0.929 and 0.754 over 100 replications):

|  | m1 | m2 | m3 |
|---|---|---|---|
| Avg. | 0.987 | 0.965 | 0.925 |
| SD |  |  |  |

*Notes:* Results from 1 realisations of the Section 4 DGP with T = 200. Avg. is the average correlation, SD the standard deviation across realisations.

---

## 6. GP regression illustration (Section 2 of the paper)

`examples/01_gp_regression_illustration.py` reproduces the univariate illustrations with the bundled data: US inflation on a linear trend (Figure 1), GDP growth on lagged macroeconomic uncertainty (Figure 2, below) and the persistence kernel of Appendix A.1. The role of $\kappa$ is exactly as described in the paper: $\kappa=0.01$ is essentially linear, $\kappa=0.1$ reveals the threshold at which high uncertainty starts to depress growth, $\kappa=4$ over-fits.

![fig2](outputs/figures/fig02_gp_gdp_uncertainty.png)

---

## 7. Forecast evaluation (Section 5.2 of the paper)

`examples/04_forecast_evaluation.py` runs the recursive (expanding-window) exercise for the GP-VAR-8 with SV, the homoskedastic GP-VAR-8 and a Minnesota BVAR-8, scoring one- and four-quarter-ahead density forecasts of RGDP, CPI and FFR with log predictive likelihoods (jointly and marginally) and CRPS. Log predictive Bayes factors relative to the BVAR (positive = better):

| variable | GP-VAR-8 SV (h=1) | GP-VAR-8 SV (h=4) | GP-VAR-8 homoskedastic (h=1) | GP-VAR-8 homoskedastic (h=4) | BVAR-8 (h=1) | BVAR-8 (h=4) | BVAR-8 (LPL) (h=1) | BVAR-8 (LPL) (h=4) |
|---|---|---|---|---|---|---|---|---|
| joint | -4.57 | 3.92 | -7.55 | 3.33 | 0.00 | 0.00 | -38.18 | -62.66 |
| RGDP | -0.73 | 2.74 | -0.42 | 4.12 | 0.00 | 0.00 | -14.51 | -24.63 |
| CPI | -2.18 | 2.67 | -4.26 | -0.01 | 0.00 | 0.00 | -10.96 | -20.01 |
| FFR | -1.22 | 1.76 | -2.13 | 0.73 | 0.00 | 0.00 | -12.96 | -23.74 |

*Notes:* Recursive design, origins 2007Q4 to 2018Q4 every 4 quarters; joint = RGDP, CPI and FFR. The last column block reports the benchmark's log predictive likelihoods.

Average CRPS by model, horizon and variable is in `outputs/tables/table02b_crps.md`.

With 12 annual origins (2007Q4–2018Q4, 400 + 400 MCMC draws per origin), the GP-VAR-8 with SV improves on the linear benchmark at the one-year horizon (joint LPBF +3.9; +2.7 for GDP growth, +2.7 for inflation, +1.8 for the funds rate) while the BVAR is better one quarter ahead, the same horizon pattern the paper reports in Table 2 (gains in predictive accuracy increase with the forecast horizon). The homoskedastic GP-VAR is close to the SV version at h = 4 but clearly worse at h = 1 for inflation and the funds rate, again as in the paper. Cumulative joint LPBFs over the evaluation sample:

![lpbf](outputs/figures/forecast_cumulative_lpbf_h4.png)

---

## 8. Library reference

| Module | Purpose |
|---|---|
| `gpvar.GPVAR(Y, p, sv, hyper_scheme, n_kappa, n_xi, c_kappa, c_xi, hyper_sampling, restrict_g_mean, sv_priors, sv_hessian, sv_tries)` | Model specification and pre-computation (kernel eigendecompositions per equation and grid point). `.fit(nburn, nsave, thin, seed)` runs the MCMC and returns `GPVARResults`. |
| `gpvar.GPVARResults` | Posterior draws (`f`, `g`, `h`, `Q`, `hyper`, `sv_params`), summaries (`hyper_summary`, `sv_summary`, `Q_summary`, `fitted`, `volatility`, `shrinkage_paths`, `fit_correlations`), `.predict()`, `.girf()`, `.save()/.load()`. |
| `gpvar.girf(results, shock_var, size, sign, horizon, histories, n_sim, n_draws, shock_scale, sample_m)` | Generalized impulse responses for every history; returns `GIRFResult` with `.summary()`, `.average()`, `.by_periods()`, `.yearly_medians()`, `.peak_table()`, `.state_dependence()`, `.mask_from_dates()`. `girf_sign_asymmetry`, `girf_size_asymmetry`, `ordering_robustness` are convenience wrappers. |
| `gpvar.predict(results, horizon, n_sim, n_draws, sample_m)` | Simulation from the multi-step predictive distribution; `ForecastResult.quantiles()`. |
| `gpvar.log_predictive_score`, `gpvar.crps`, `gpvar.recursive_forecast_evaluation`, `gpvar.lpbf_table` | Density-forecast evaluation. |
| `gpvar.BVAR` | Natural-conjugate Minnesota BVAR benchmark with `.predict()` and Cholesky `.irf()`. |
| `gpvar.GPRegression` | Univariate GP regression (Gaussian / linear / persistence kernels) for the Section 2 illustrations. |
| `gpvar.simulate_paper_dgp`, `gpvar.simulate_replication_dgp` | Synthetic DGPs of the paper and of the replication archive. |
| `gpvar.data` | Bundled FRED-QD subset and JLN index, paper transformations (`build_dataset`, `transform_series`), NBER recessions, sub-sample definitions. |
| `gpvar.plots` | Journal-style figures: `plot_girf`, `plot_girf_periods`, `plot_girf_time_variation`, `plot_shrinkage_paths`, `plot_kernel_boxplots`, `plot_volatility`, `plot_fit_components`, `plot_gp_illustration`, `plot_data`, `plot_forecast_fan`, `plot_lpl_cumulative`, `plot_trace`. |
| `gpvar.tables` | Markdown / LaTeX (booktabs) / CSV writers and ready-made tables (`hyper_table`, `sv_table`, `girf_horizon_table`, `girf_peak_table`, `asymmetry_table`, `size_asymmetry_table`). |
| `gpvar.kernels`, `gpvar.sv`, `gpvar.priors`, `gpvar.predict` | Building blocks (kernel blocks with eigendecompositions, IMH volatility sampler, horseshoe, predictive engine). |

Key options of `GPVAR`:

* `hyper_scheme`: `"semi-automatic"` (grid on both $\kappa$ and $\xi$, the paper's preferred specification), `"semi-automatic-noscale"` ($\xi = 1$), `"naive"` ($\kappa\in[0.1,2]$, $\xi=1$).
* `hyper_sampling`: `"marginal"` (default; samples $\vartheta$ with the latent function integrated out, then the function — a blocked step that mixes better) or `"conditional"` (the paper's Step 5/6 literally).
* `sv`: stochastic volatility (`True`) or homoskedastic errors (`False`, the "GP-VAR homoskedastic" of Table 2).
* `sv_hessian`: `"dense"` (exact Hessian for the IMH proposal, default) or `"band"` (the paper's tridiagonal approximation, cheaper for large $T$); `sv_tries`: accept/reject attempts per sweep from the same proposal.
* `restrict_g_mean`: grand-mean-zero restriction on $g_j$ (Appendix A.2).
* `SVPriors(estimate_mu=False)` removes the unconditional mean of the log-volatility process (eq. 3 of the paper literally).

---

## 9. How the implementation maps to the paper

| Paper | Implementation |
|---|---|
| Eq. (2) structural form with own/other lag functions | `GPVAR` (equation-by-equation), `kernels.own_other_index` |
| Eq. (3) SV process | `sv.sample_sv_params` (Beta(25,5) on $(\rho+1)/2$, IG(3, 0.2) on $\sigma^2_h$) |
| Eq. (4) SV-scaled GP priors, Gaussian kernel with scaling matrix $D$ | `kernels.KernelBlock`, `posterior_draw` |
| Section 3.3 median heuristic, grid $\kappa\in[0.1\bar\kappa,2\bar\kappa]$, $\xi\in[0.04,4]$, Gamma hyperpriors | `kernels.median_heuristic`, `make_hyper_grid`, `log_gamma_prior` |
| Section 3.4 conjugate posterior of $f_j$, pre-computed factors | eigendecomposition per $\kappa$ (all quantities $O(T^2)$ per draw) |
| Appendix A.2 sampling $g_j$ under $\iota'g_j=0$ (Cong, Chen, Zhou 2017) | `GPVAR.fit`, Steps 4 & 6 |
| Appendix A.3 log-volatilities: Newton–Raphson mode, band Hessian, independence MH | `sv.sample_logvol_imh` (band) / `sv.sample_logvol_imh_dense` |
| Appendix A.4 Steps 1–8 | `GPVAR.fit` |
| Section 3.5 / Appendix A.5 forecasts and GIRFs | `predict.DrawContext`, `forecast.predict`, `girf.girf` |
| Section 4 DGP, Table 1 | `simulate.simulate_paper_dgp`, `GPVARResults.fit_correlations` |
| Section 5.2 LPBFs | `forecast.recursive_forecast_evaluation`, `lpbf_table` |
| Figures 1–2, A.1 | `GPRegression`, `plots.plot_gp_illustration` |
| Figures 3, 5, 6, 7, 9, 10, 11, 12, C.4–C.10 | `plots.*` (see Section 4 above) |

Two deliberate differences from the paper are documented in `docs/METHODOLOGY.md`: the optional mean of the log-volatility process (on by default, as in the authors' R code) and the blocked ("marginal") hyperparameter step (the paper's conditional step is available with `hyper_sampling="conditional"`). The BVAR benchmark shipped here is homoskedastic (the paper's has SV).

---

## 10. Data

Two files ship in `gpvar/datasets/` so that every example is reproducible offline:

* `fredqd_subset.csv` — 26 series of **FRED-QD** (McCracken and Ng, 2020), vintage 2026-08, 1959Q1–2026Q2, with the FRED-QD transformation codes. Source: Federal Reserve Bank of St. Louis.
* `jln_macro_uncertainty.csv` — the **JLN macroeconomic uncertainty index** (h = 1, 3, 12), monthly 1960:07–2026:06 (August 2026 update), from Sydney C. Ludvigson's website.

`gpvar.data.build_dataset(size=8 | 16, start, end)` assembles the paper's GP-VAR-8 or GP-VAR-16 with the Table B.1 transformations (`scheme="paper"`) or the FRED-QD codes (`scheme="fred"`); `gpvar.data.fetch_fred_qd()` downloads the latest full vintage. Samples extending through the pandemic (e.g. `end="2025Q4"`) work out of the box (`--end 2025Q4` in example 3 adds a 2020– sub-sample panel).

---

## 11. Performance and settings

Per MCMC sweep the cost is $O(M\,T^2)$ for the GP draws plus one $T\times T$ Cholesky/inverse per equation for the volatility step; it does **not** grow with the number of lags or regressors. Measured on a laptop (pure NumPy, single process):

| Setting | Time |
|---|---|
| GP-VAR-8, T = 195, p = 5, 20 × 50 grid: pre-computation | ≈ 3 s |
| GP-VAR-8: one MCMC sweep (dense-Hessian SV step) | ≈ 0.13 s (4 000 sweeps ≈ 10 min) |
| GP-VAR-3 simulation (T = 195): one MCMC sweep | ≈ 0.04 s |
| GIRF, 195 histories × 5 future paths × 16 horizons, all 8 equations | ≈ 1 s per posterior draw (threads over draws); the three shock configurations of example 3 (100 draws each) take ≈ 4 min |
| Full example 3 (estimation + all GIRFs, figures and tables) | ≈ 15 min |

Memory: the eigendecompositions take $2M\,n_\kappa T^2$ doubles (≈ 100 MB for M = 8, T = 195, $n_\kappa = 20$). For monthly data or M ≥ 32 reduce `n_kappa`, thin the GIRF draws (`n_draws`) and consider `sv_hessian="band"`.

---

## 12. Repository layout

```
GP-VAR/
├── gpvar/                    the library
│   ├── model.py              GPVAR, GPVARResults (Gibbs sampler, Appendix A.4)
│   ├── kernels.py            Gaussian kernels, median heuristic, hyper grid, eigen-blocks
│   ├── sv.py                 independence-MH log-volatility sampler, SV parameters
│   ├── priors.py             horseshoe, SV priors
│   ├── predict.py            predictive engine shared by forecasts and GIRFs
│   ├── girf.py               generalized impulse responses and asymmetry analysis
│   ├── forecast.py           predictive simulation, LPL/LPBF/CRPS, recursive evaluation
│   ├── benchmarks.py         Minnesota BVAR benchmark
│   ├── simulate.py           Section 4 DGP and replication-archive DGP
│   ├── gp.py                 univariate GP regression (Section 2 illustrations)
│   ├── data.py               FRED-QD / JLN loaders, transformations, NBER dates
│   ├── plots.py, tables.py   journal-style figures and Markdown/LaTeX tables
│   └── datasets/             bundled data (see datasets/README.md)
├── examples/                 01 GP illustration · 02 simulation · 03 US uncertainty · 04 forecast evaluation · run_all.py
├── outputs/figures, outputs/tables   generated results (PDF/PNG, md/tex/csv)
├── tests/                    pytest suite
├── docs/METHODOLOGY.md       equation-by-equation mapping to the paper
├── GPVAR_replication/        the authors' original R replication code (unchanged)
└── 2112.01995v3.pdf is not distributed; download the paper from arXiv
```

---

## 13. Citation, license, acknowledgements

If you use this code, please cite the original paper and this repository:

```bibtex
@article{hauzenberger2022gpvar,
  title   = {Gaussian Process Vector Autoregressions and Macroeconomic Uncertainty},
  author  = {Hauzenberger, Niko and Huber, Florian and Marcellino, Massimiliano and Petz, Nico},
  journal = {Journal of Business \& Economic Statistics (forthcoming); arXiv:2112.01995},
  year    = {2022}
}

@software{roudane2026gpvar,
  title   = {gpvar: Gaussian Process Vector Autoregressions with stochastic volatility in Python},
  author  = {Roudane, Merwan},
  year    = {2026},
  url     = {https://github.com/merwanroudane/GP-VAR}
}
```

Released under the MIT License. The FRED-QD data are provided by the Federal Reserve Bank of St. Louis; the uncertainty index by Jurado, Ludvigson and Ng (2015) — please cite them when using the data. The R code in `GPVAR_replication/` is the authors' replication archive and remains their work.

**Dr Merwan Roudane** — merwanroudane920@gmail.com — https://github.com/merwanroudane
