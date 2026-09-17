# gpvar — user guide and syntax reference

This guide shows, step by step, how to write code with `gpvar`: from loading data to estimation, impulse responses, forecasts, tables and figures. Every code block is runnable as-is after `pip install gpvar`. All arrays follow the convention **rows = time, columns = variables**.

Contents

1. [Installation and imports](#1-installation-and-imports)
2. [Data: loading, transforming, building a data set](#2-data)
3. [Specifying the model: `GPVAR(...)`](#3-specifying-the-model)
4. [Estimation: `.fit(...)` and the results object](#4-estimation)
5. [Reading the posterior: summaries and tables](#5-reading-the-posterior)
6. [Generalized impulse responses: `girf(...)`](#6-generalized-impulse-responses)
7. [Asymmetries: sign, size, time, business cycle](#7-asymmetries)
8. [Forecasting: `predict(...)`](#8-forecasting)
9. [Density-forecast evaluation and the BVAR benchmark](#9-forecast-evaluation)
10. [Figures](#10-figures)
11. [Tables (Markdown / LaTeX / CSV)](#11-tables)
12. [Simulated data and Monte Carlo checks](#12-simulated-data)
13. [Univariate GP regression](#13-univariate-gp-regression)
14. [Saving, loading, reproducibility, performance](#14-saving-loading-reproducibility-performance)
15. [Recipes](#15-recipes)
16. [Complete signature reference](#16-complete-signature-reference)

---

## 1. Installation and imports

```bash
pip install gpvar                # from PyPI
# or, for the development version with examples and tests:
git clone https://github.com/merwanroudane/GP-VAR.git && cd GP-VAR && pip install -e .[dev]
```

```python
import gpvar                       # the library
from gpvar import plots, tables    # figure and table helpers
import numpy as np, pandas as pd
```

The top-level namespace exposes: `GPVAR`, `GPVARResults`, `girf`, `girf_sign_asymmetry`, `girf_size_asymmetry`, `GIRFResult`, `ordering_robustness`, `predict`, `ForecastResult`, `log_predictive_score`, `crps`, `recursive_forecast_evaluation`, `lpbf_table`, `BVAR`, `GPRegression`, `simulate_paper_dgp`, `simulate_replication_dgp`, `SVPriors`, and the sub-modules `data`, `plots`, `tables`, `kernels`, `sv`.

---

## 2. Data

### 2.1 The bundled US data set (paper's GP-VAR-8 / GP-VAR-16)

```python
Y, info = gpvar.data.build_dataset(size=8, start="1970Q1", end="2019Q4")
#   size   : 8 (UNC, RGDP, EMP, AWH, CPI, AHE, FFR, SP500) or 16 (adds PCE, FPI, UNRATE, AWHG, CLAIMS, HOUST, AHEALL, M2)
#   start/end : quarters as "YYYYQn" (the sample can run to 2026Q2 with the bundled vintage)
#   order  : optional list giving the variable ordering (matters for the recursive identification)
#   uncertainty_h : 1, 3 or 12 (forecast horizon of the JLN uncertainty index; paper: 1)
#   scheme : "paper" (Table B.1 transformations) or "fred" (FRED-QD codes)
print(Y.shape)          # (200, 8) - a DataFrame with a quarterly DatetimeIndex (quarter ends)
print(info)             # mnemonic, transformation code and description of every column
```

Other data helpers:

```python
fred, tcodes = gpvar.data.load_fred_qd_subset()        # 26 FRED-QD series 1959Q1-2026Q2 (levels) + FRED transformation codes
unc = gpvar.data.load_jln_uncertainty(h=1, freq="Q")    # JLN macro uncertainty, quarterly average of the monthly index
x = gpvar.data.transform_series(fred["GDPC1"], code=2)  # code 1 none | 2 yoy growth | 3 qoq growth | 4 qoq % change
desc = gpvar.data.describe(Y)                           # mean, sd, min, max, skewness, kurtosis, AR(1)
rec = gpvar.data.recession_mask(Y.index)                # boolean array: NBER recession quarters
gpvar.data.PAPER_SUBSAMPLES                             # {"1970Q1-1984Q4 (great inflation)": ("1970-01-01","1984-12-31"), ...}
fred_full, tcodes_full = gpvar.data.fetch_fred_qd()     # download the latest full FRED-QD vintage (internet required)
```

### 2.2 Using your own data

Anything that is a `(T, M)` NumPy array or a `pandas.DataFrame` works. A DataFrame supplies the variable names and the dates automatically; with an array you can pass them explicitly.

```python
# DataFrame (recommended): column names become variable names, the index becomes the time axis
Y_own = pd.read_csv("my_data.csv", index_col=0, parse_dates=True)          # rows = time, columns = variables
model = gpvar.GPVAR(Y_own, p=4)

# NumPy array
A = np.random.default_rng(0).normal(size=(150, 4))
model = gpvar.GPVAR(A, p=2, var_names=["gdp", "infl", "rate", "stocks"],
                    dates=pd.date_range("1990-03-31", periods=150, freq="QE"))
```

Rules of thumb: transform the series to be (approximately) stationary first (growth rates, yoy inflation, levels of rates); the model standardises each column internally (`standardize=True`) and reports GIRFs/forecasts in both standardised and original units.

---

## 3. Specifying the model

```python
model = gpvar.GPVAR(
    Y,                        # (T, M) DataFrame or array
    p=5,                      # number of lags (paper: 5 for quarterly data)
    standardize=True,         # demean and scale each column (undone in forecasts; GIRFs also available in original units)
    sv=True,                  # stochastic volatility (False = homoskedastic GP-VAR)
    hyper_scheme="semi-automatic",   # "semi-automatic" | "semi-automatic-noscale" | "naive"   (Section 3.3 / Table 2)
    n_kappa=20, n_xi=50,      # grid sizes for the inverse length scale kappa and the scaling xi (paper: ~1000 points)
    c_kappa=0.1, c_xi=1.0,    # prior means of the Gamma(1/2, 1/(2c)) hyperpriors (small c_kappa = smooth functions)
    kappa_range=(0.1, 2.0),   # grid for kappa as multiples of the median heuristic kappa_bar
    xi_range=(0.04, 4.0),     # grid for xi
    hyper_sampling="marginal",   # "marginal" (blocked, default) | "conditional" (paper's Step 5/6 literally)
    restrict_g_mean=True,     # grand-mean-zero identification of the other-lag function g_j
    sv_priors=gpvar.SVPriors(rho_a=25, rho_b=5, sig2_a=3, sig2_b=0.2, mu_mean=0, mu_var=10, estimate_mu=True),
    sv_hessian="dense",       # "dense" (exact Hessian in the volatility MH step) | "band" (paper's tridiagonal approximation)
    sv_tries=3,               # accept/reject attempts per sweep from the same volatility proposal
)
```

What you get before estimation:

```python
model.T, model.M, model.p            # effective sample size (T_raw - p), number of variables, lags
model.y, model.X                     # standardised targets (T, M) and stacked lags (T, M*p): [y_{t-1}', ..., y_{t-p}']
model.own_idx[j], model.other_idx[j] # columns of X that are own lags / other lags of equation j
model.blocks_own[j].grid.kappa_bar   # median heuristic of equation j's own-lag kernel
model.blocks_own[j].grid.kappa       # the kappa grid actually used (n_kappa,)
model.mean_, model.sd_               # standardisation constants
model.precompute_time                # seconds spent on the kernel eigendecompositions
```

Three specifications of the paper's Table 2:

```python
m_semi   = gpvar.GPVAR(Y, p=5, hyper_scheme="semi-automatic")          # grid on kappa and xi (preferred)
m_noscl  = gpvar.GPVAR(Y, p=5, hyper_scheme="semi-automatic-noscale")  # grid on kappa, xi = 1
m_naive  = gpvar.GPVAR(Y, p=5, hyper_scheme="naive")                   # kappa in [0.1, 2], xi = 1
m_homo   = gpvar.GPVAR(Y, p=5, sv=False)                               # homoskedastic errors
m_eq3    = gpvar.GPVAR(Y, p=5, sv_priors=gpvar.SVPriors(estimate_mu=False))   # eq. (3) literally (no SV mean)
```

---

## 4. Estimation

```python
res = model.fit(
    nburn=2000,        # burn-in sweeps
    nsave=2000,        # retained draws
    thin=1,            # keep every thin-th sweep after burn-in
    seed=1,            # reproducible
    verbose=True,      # progress line every `print_every` sweeps with SV acceptance rates
    print_every=250,
    sv_init_sweeps=10, # first burn-in sweeps initialise the volatilities at the conditional mode
)
```

`res` is a `GPVARResults` with the posterior draws:

| attribute | shape | content |
|---|---|---|
| `res.f`, `res.g` | `(nsave, T, M)` | own-lag and other-lag GP functions $f_j$, $g_j$ |
| `res.m` | `(nsave, T, M)` | conditional mean $m_j = f_j + g_j$ (property) |
| `res.h`, `res.omega` | `(nsave, T, M)` | log-volatilities and variances $\omega_{jt} = e^{h_{jt}}$ |
| `res.Q` | `(nsave, M, M)` | contemporaneous matrix (lower triangular, zero diagonal) |
| `res.hyper` | `(nsave, M, 4)` | `xi_own, kappa_own, xi_other, kappa_other` |
| `res.hyper_idx` | `(nsave, M, 4)` | grid indices of the above |
| `res.sv_params` | `(nsave, M, 4)` | `mu, rho, sig2, h0` of the SV state equation |
| `res.accept_rate` | `(M,)` | independence-MH acceptance for the volatility paths |
| `res.runtime`, `res.config` | | seconds and the settings used |
| `res.model`, `res.y`, `res.dates`, `res.var_names` | | back-references to the model and data |

A quick estimation loop for several specifications:

```python
fits = {}
for name, m in {"SV": m_semi, "homoskedastic": m_homo, "naive grid": m_naive}.items():
    fits[name] = m.fit(nburn=1000, nsave=1000, seed=1, verbose=False)
    print(name, f"{fits[name].runtime:.0f}s", fits[name].accept_rate.round(2))
```

---

## 5. Reading the posterior

```python
res.hyper_summary()                 # DataFrame: variable, parameter, mean, q05..q95 for xi/kappa (own & other)
res.sv_summary()                    # same for mu, rho, sig2, h0
res.Q_summary()                     # every free element of Q with mean, quantiles and P(q > 0)
res.fitted(probs=(0.05, 0.5, 0.95)) # dict {"f", "g", "m"} of arrays (len(probs), T, M)
res.volatility(probs=(0.16, 0.5, 0.84))   # quantiles of exp(h/2), (3, T, M)
res.shrinkage_paths()               # {"own": (T, M), "other": (T, M)} posterior means of omega_jt * xi (Figure 5)
res.fit_correlations()              # correlation between posterior-median m_j and y_j (or a supplied truth)
res.quantiles(res.m, probs=(0.16, 0.84))  # generic helper for any (nsave, ...) array
```

Turning them into time series for your own analysis:

```python
m_med = pd.DataFrame(np.median(res.m, axis=0), index=res.dates, columns=res.var_names)   # fitted conditional mean
vol   = pd.DataFrame(res.volatility()[1], index=res.dates, columns=res.var_names)        # median volatility
kappa_own = pd.Series(np.median(res.hyper[:, :, 1], axis=0), index=res.var_names)         # median kappa per equation
```

---

## 6. Generalized impulse responses

```python
g = gpvar.girf(
    res, "UNC",              # results and the shocked variable (name or index)
    size=1.0, sign=+1,       # shock size and sign
    horizon=16,              # quarters after impact
    histories=None,          # conditioning histories t (default: all T quarters); e.g. range(100, 195)
    n_sim=5,                 # simulated future shock paths per history (Monte Carlo expectation)
    n_draws=100,             # evenly spaced posterior draws
    seed=0,
    shock_scale="unit",      # "unit": impact effect on the shocked variable = size (Figure 7 normalisation)
                             # "sd"  : shock = size x exp(h_jt/2), the time-t structural standard deviation
    sample_m=False,          # True adds the GP predictive variance of the conditional mean
    n_jobs=None,             # threads over posterior draws (default: all cores)
)
# or equivalently: g = res.girf("UNC", size=1.0, sign=+1, horizon=16)
```

`g` is a `GIRFResult`; `g.irf` has shape `(n_draws, n_hist, horizon+1, M)`.

```python
g.average()                             # (n_draws, horizon+1, M): average over histories (the paper's "average GIRF")
g.average(original_units=True)          # responses multiplied by the columns' standard deviations
g.summary()                             # long DataFrame: variable, horizon, q16, q50, q84
g.summary(probs=(0.05, 0.5, 0.95), label="positive")
g.peak_table()                          # impact, peak, peak horizon, 68% band at the peak, cumulative response
g.state_dependence()                    # dispersion of history-specific responses across quarters
g.mask_from_dates("1985-01-01", "2006-12-31")     # boolean mask over histories for any of the methods above
g.by_periods(gpvar.data.PAPER_SUBSAMPLES)         # sub-sample averages (Figure 11)
g.yearly_medians()                                # yearly averages of the posterior medians (Figure 12)
g.hist_dates, g.var_names, g.shock_var, g.size, g.sign
```

History-specific responses (state dependence) directly:

```python
med = np.median(g.irf, axis=0)                     # (n_hist, horizon+1, M)
resp_2008q4 = med[list(g.hist_dates).index(pd.Timestamp("2008-12-31"))]   # (horizon+1, M) response conditional on 2008Q4
```

---

## 7. Asymmetries

```python
# Sign asymmetry (Figure 9)
g_pos = gpvar.girf(res, "UNC", sign=+1, n_draws=100, seed=1)
g_neg = gpvar.girf(res, "UNC", sign=-1, n_draws=100, seed=2)
tables.asymmetry_table(g_pos, g_neg, horizons=(1, 4, 8))      # median of delta_pos + delta_neg and P(>0); 0 under linearity

# Size asymmetry (Figure 10)
g_2sd = gpvar.girf(res, "UNC", size=2.0, n_draws=100, seed=3)
tables.size_asymmetry_table(g_pos, g_2sd)                    # ratio 2-sd / 1-sd responses; 2 under proportionality

# Time variation (Figures 11 and 12)
per = g_pos.by_periods(gpvar.data.PAPER_SUBSAMPLES)          # long table with a "label" column
yearly = g_pos.yearly_medians()

# Business cycle (Figure C.6)
rec = gpvar.data.recession_mask(g_pos.hist_dates)
in_rec, in_exp = g_pos.summary(rec, label="recessions"), g_pos.summary(~rec, label="expansions")

# Convenience wrappers
both = gpvar.girf_sign_asymmetry(res, "UNC", n_draws=100)          # {"positive": GIRFResult, "negative": GIRFResult}
sizes = gpvar.girf_size_asymmetry(res, "UNC", sizes=(1, 2), n_draws=100)

# Robustness to the variable ordering (Appendix C.2) - re-estimates the model under random orderings
rob = gpvar.ordering_robustness(Y, fixed_first=["UNC"], shock_var="UNC", n_orderings=3,
                                fit_kwargs=dict(nburn=300, nsave=300), girf_kwargs=dict(n_draws=30, n_sim=3),
                                model_kwargs=dict(p=5))
```

---

## 8. Forecasting

```python
fc = res.predict(horizon=8, n_sim=5, n_draws=None, seed=0, sample_m=True)   # or gpvar.predict(res, ...)
fc.paths        # (n_draws * n_sim, horizon, M) in standardised units
fc.paths_raw    # same in original units
fc.quantiles(probs=(0.05, 0.16, 0.5, 0.84, 0.95))   # long DataFrame: variable, horizon, q05 ... q95 (original units)
fc.mean()       # point forecasts (horizon x M)
fc.origin_date  # last date of the estimation sample
```

Scoring a realised outcome:

```python
y_next = np.array([...])                                       # realised values one step ahead (original units), length M
lpl_joint = gpvar.log_predictive_score(fc.paths_raw[:, 0, :], y_next)                 # Gaussian approximation, joint
lpl_gdp   = gpvar.log_predictive_score(fc.paths_raw[:, 0, [1]], y_next[[1]], method="kde")   # univariate KDE
score     = gpvar.crps(fc.paths_raw[:, 0, 1], y_next[1])                              # continuous ranked probability score
```

---

## 9. Forecast evaluation

Expanding-window evaluation for any model with the `fit()` / `predict()` interface:

```python
ev_gp = gpvar.recursive_forecast_evaluation(
    Y, first_origin="2007-12-31", last_origin="2018-12-31", step=4,   # origins every 4 quarters
    horizons=(1, 4), focus=["RGDP", "CPI", "FFR"],
    model_factory=lambda Yt: gpvar.GPVAR(Yt, p=5, sv=True),
    fit_kwargs=dict(nburn=400, nsave=400, seed=1),
    predict_kwargs=dict(n_sim=4, seed=0),
    label="GP-VAR-8 SV")
ev_bvar = gpvar.recursive_forecast_evaluation(
    Y, first_origin="2007-12-31", last_origin="2018-12-31", step=4, horizons=(1, 4), focus=["RGDP", "CPI", "FFR"],
    model_factory=lambda Yt: gpvar.BVAR(Yt, p=5), fit_kwargs=dict(nsave=400, seed=1), label="BVAR-8")

tab = gpvar.lpbf_table({"GP-VAR-8 SV": ev_gp, "BVAR-8": ev_bvar}, benchmark="BVAR-8")   # LPBFs relative to the benchmark
plots.plot_lpl_cumulative({"GP-VAR-8 SV": ev_gp, "BVAR-8": ev_bvar}, horizon=4, variable="joint", benchmark="BVAR-8")
```

The Minnesota BVAR benchmark on its own:

```python
bv = gpvar.BVAR(Y, p=5, lam1=0.2, lam3=1.0, prior_mean="zero", intercept=True).fit(nsave=1000, seed=1)
bv.predict(horizon=8, n_sim=2).quantiles()
bv.irf("UNC", horizon=16, size=1.0, label="BVAR-8")     # Cholesky IRFs in the same long format as GIRF summaries
```

---

## 10. Figures

All plotting functions return a `matplotlib.figure.Figure`, accept `save="path/without/extension"` (writes `.pdf` and `.png` at 300 dpi) and `title=...`.

```python
plots.set_journal_style(font_size=9)                       # call once: serif fonts, thin axes, light grid, white background

plots.plot_data(Y, ncols=4, save="fig_data")                                        # data panel with NBER shading
plots.plot_shrinkage_paths(res, save="fig05")                                       # omega_jt * xi, own vs other (Figure 5)
plots.plot_kernel_boxplots(res, save="fig06")                                       # kappa own/other per equation (Figure 6)
plots.plot_volatility(res, ncols=4, save="fig_vol", truth=None)                     # exp(h/2) with 68% bands
plots.plot_fit_components(res, truth={"F": F, "G": G, "m": m}, save="fig03")        # f, g, m, y vs truth (Figure 3)
plots.plot_trace(res, save="fig_trace")                                             # MCMC traces of kappa, xi, rho_h

plots.plot_girf({"GP-VAR-8": g_pos.summary(), "BVAR-8": bv.irf("UNC")},            # any number of labelled summaries
                variables=["UNC", "RGDP", "EMP", "SP500"], ncols=4,
                colors={"GP-VAR-8": plots.COL_POS, "BVAR-8": plots.COL_GRAY},
                legend_title=None, ylabel="", save="fig07")
by_period = plots.combine_summaries({"positive": g_pos.by_periods(gpvar.data.PAPER_SUBSAMPLES),
                                     "negative": g_neg.by_periods(gpvar.data.PAPER_SUBSAMPLES)})
plots.plot_girf_periods(by_period, variables=["UNC", "RGDP"], save="fig11")        # rows = periods (Figure 11)
plots.plot_girf_time_variation(g_pos.yearly_medians(), gpvar.data.PAPER_SUBSAMPLES, save="fig12")   # Figure 12
plots.plot_forecast_fan(Y, fc, variables=None, n_hist=24, actual=None, save="fig_fan")
plots.plot_gp_illustration(x, y, {"kappa = 0.1": gpvar.GPRegression("gaussian", kappa=0.1)}, save="fig01")
```

Colour constants: `plots.COL_POS` (orange), `plots.COL_NEG` (blue), `plots.COL_GRAY`, `plots.COL_RED`, `plots.COL_BLUE`, `plots.COL_GREEN`, `plots.SERIES_COLORS`.

---

## 11. Tables

```python
tables.save_table(df, "outputs/tables/my_table", caption="...", label="tab:x", floatfmt="{:.3f}", notes="...")
# -> my_table.md (Markdown), my_table.tex (LaTeX booktabs, with caption/label/notes), my_table.csv
print(tables.to_markdown(df))          # string
print(tables.to_latex(df, caption="Peak responses"))

tables.hyper_table(res)                          # median [IQR] of kappa and xi, own/other, per variable
tables.sv_table(res)                             # median [5%, 95%] of mu, rho, sig2, h0
tables.girf_horizon_table(g_pos, horizons=(0, 1, 4, 8, 12, 16))     # median [16%, 84%] at selected horizons
tables.girf_peak_table({"positive": g_pos, "negative": g_neg})       # peaks for several GIRF sets
tables.asymmetry_table(g_pos, g_neg); tables.size_asymmetry_table(g_pos, g_2sd)
```

---

## 12. Simulated data

```python
sim = gpvar.simulate_paper_dgp(T=200, p=5, seed=1)      # the Section 4 DGP (break, sine/quadratic, t_3 shocks, RW-SV)
sim = gpvar.simulate_replication_dgp(T=120, M=3, p=2)   # the small tanh/sin DGP of the authors' R archive
sim.Y, sim.F, sim.G, sim.m, sim.Q, sim.omega            # data (DataFrame) and the true latent components

res_sim = gpvar.GPVAR(sim.Y, p=5).fit(nburn=1000, nsave=1000, seed=1, verbose=False)
res_sim.fit_correlations(sim.m)                         # Table 1: corr(posterior median m_j, true m_j)
plots.plot_fit_components(res_sim, truth={"F": sim.F[5:], "G": sim.G[5:], "m": sim.m[5:]})
```

---

## 13. Univariate GP regression

```python
gp = gpvar.GPRegression(kernel="gaussian", kappa=0.1, xi=1.0, sigma2=None)   # sigma2=None -> chosen by marginal likelihood
gp.fit(x[:, None], y)
mean, sd = gp.posterior()                     # at the training inputs
mean_g, sd_g = gp.posterior(xgrid[:, None])   # at new inputs
draws = gp.prior_draws(xgrid[:, None], n=3)   # draws from the GP prior
gpvar.GPRegression("linear", v=1.0)           # Bayesian linear regression as a GP (kernel X V X')
gpvar.GPRegression("persistence", r=0.01)     # local-level / random-walk kernel r B B' (Appendix A.1)
```

---

## 14. Saving, loading, reproducibility, performance

```python
res.save("outputs/fit.pkl")                       # pickles the results (includes the model and its kernel blocks)
res2 = gpvar.GPVARResults.load("outputs/fit.pkl")
np.save("girf_pos.npy", g_pos.irf)                # GIRF draws are plain arrays
```

* Reproducibility: pass `seed=` to `fit`, `girf`, `predict`; the GIRF threads use per-draw seeds, so results do not depend on the number of threads.
* Cost per MCMC sweep is $O(M T^2)$ plus one $T \times T$ Cholesky per equation; it does not depend on the number of lags. Typical: 0.13 s per sweep for $M = 8$, $T = 195$ (about 10 minutes for 4000 sweeps).
* GIRFs: about 1 s per posterior draw for all 195 histories, 5 future paths and 16 horizons (multithreaded); 100 draws per shock configuration are enough for smooth bands.
* Memory: the kernel eigendecompositions take $2 M n_\kappa T^2$ doubles (100 MB for $M=8$, $T=195$, $n_\kappa=20$). For monthly data or $M \ge 32$: reduce `n_kappa`, use `sv_hessian="band"`, thin the GIRF draws.
* Check mixing with `res.accept_rate` (volatility MH; 0.2–0.9 is typical) and `plots.plot_trace(res)`.

---

## 15. Recipes

**A different shocked variable, original units, custom horizon**

```python
g_ffr = gpvar.girf(res, "FFR", size=1.0, horizon=20, shock_scale="sd")        # a one-sd monetary shock (time-t sd)
resp = g_ffr.summary(original_units=True)                                     # responses in the data's units
```

**Sample including the pandemic**

```python
Y_long, _ = gpvar.data.build_dataset(8, "1970Q1", "2025Q4")
res_long = gpvar.GPVAR(Y_long, p=5).fit(nburn=2000, nsave=2000, seed=1)
g = res_long.girf("UNC", n_draws=100)
g.by_periods({"pre-2020": ("1970-01-01", "2019-12-31"), "2020-2025": ("2020-01-01", "2025-12-31")})
```

**GP-VAR-16**

```python
Y16, info16 = gpvar.data.build_dataset(16, "1970Q1", "2019Q4")
res16 = gpvar.GPVAR(Y16, p=5, n_kappa=15).fit(nburn=1500, nsave=1500, seed=1)
```

**Conditioning on particular histories only**

```python
idx = np.where(gpvar.data.recession_mask(res.dates))[0]                       # recession quarters
g_rec = gpvar.girf(res, "UNC", histories=idx, n_draws=100)
```

**Everything in one go** — run the four example scripts (`examples/run_all.py`) to regenerate the full set of figures and tables in `outputs/`.

---

## 16. Complete signature reference

```python
gpvar.GPVAR(Y, p=5, var_names=None, dates=None, standardize=True, sv=True,
            hyper_scheme="semi-automatic", n_kappa=20, n_xi=50, c_kappa=0.1, c_xi=1.0,
            kappa_range=(0.1, 2.0), xi_range=(0.04, 4.0), hyper_sampling="marginal",
            restrict_g_mean=True, sv_priors=None, sigma2_prior=(0.01, 0.01),
            sv_proposal_df=None, sv_hessian="dense", sv_tries=3)
GPVAR.fit(nburn=1000, nsave=1000, thin=1, seed=None, verbose=True, print_every=100, sv_init_sweeps=10) -> GPVARResults

GPVARResults.hyper_summary(probs) / sv_summary(probs) / Q_summary(probs) / fitted(probs) / volatility(probs)
GPVARResults.shrinkage_paths() / fit_correlations(truth_m=None) / quantiles(arr, probs)
GPVARResults.predict(horizon=8, n_sim=1, n_draws=None, seed=0, sample_m=True, verbose=False) -> ForecastResult
GPVARResults.girf(shock_var, **kwargs) -> GIRFResult
GPVARResults.save(path) / GPVARResults.load(path)

gpvar.girf(results, shock_var, size=1.0, sign=1, horizon=16, histories=None, n_sim=5, n_draws=100,
           seed=0, shock_scale="unit", sample_m=False, verbose=True, n_jobs=None) -> GIRFResult
GIRFResult.average(mask=None, original_units=False) / summary(mask, probs, original_units, label)
GIRFResult.by_periods(periods, probs, original_units) / yearly_medians(mask, original_units)
GIRFResult.peak_table(mask, original_units) / state_dependence(mask) / mask_from_dates(start, end)
gpvar.girf_sign_asymmetry(results, shock_var, **kw) / girf_size_asymmetry(results, shock_var, sizes=(1, 2), **kw)
gpvar.ordering_robustness(Y, fixed_first, shock_var, n_orderings=5, fit_kwargs=None, girf_kwargs=None, model_kwargs=None, seed=0)

gpvar.predict(results, horizon=8, n_sim=1, n_draws=None, seed=0, sample_m=True, verbose=False) -> ForecastResult
ForecastResult.quantiles(probs, original_units=True) / mean(original_units=True)
gpvar.log_predictive_score(draws, y_obs, method="gaussian" | "kde") / gpvar.crps(draws, y_obs)
gpvar.recursive_forecast_evaluation(Y, first_origin, last_origin=None, step=1, horizons=(1, 4), focus=None,
                                    model_factory=None, fit_kwargs=None, predict_kwargs=None,
                                    score_method="gaussian", label="GP-VAR", verbose=True) -> DataFrame
gpvar.lpbf_table(evals: dict, benchmark: str) -> DataFrame

gpvar.BVAR(Y, p=5, lam1=0.2, lam2=0.5, lam3=1.0, lam_c=100.0, prior_mean="zero", intercept=True, standardize=True)
BVAR.fit(nsave=1000, seed=None) -> BVARResults;  BVARResults.predict(horizon, n_sim, n_draws, seed) / irf(shock_var, horizon, size, n_draws, probs, label)

gpvar.GPRegression(kernel="gaussian", kappa=0.1, xi=1.0, v=1.0, r=0.01, sigma2=1.0, scale_inputs=False)
GPRegression.fit(X, y) / posterior(Xnew=None) / prior_draws(X, n=3, seed=0) / prior_sd(X)

gpvar.simulate_paper_dgp(T=200, p=5, seed=1, burn=100, break_point=None) -> SimulatedData
gpvar.simulate_replication_dgp(T=120, M=3, p=2, burn=200, seed=1) -> SimulatedData

gpvar.data.build_dataset(size=8, start="1970Q1", end="2019Q4", uncertainty_h=1, order=None, scheme="paper", fred=None)
gpvar.data.load_fred_qd_subset() / load_jln_uncertainty(h=1, freq="Q", agg="mean") / fetch_fred_qd(url)
gpvar.data.transform_series(x, code, scheme="paper") / standardize(Y) / recession_mask(dates) / describe(Y)

gpvar.plots.set_journal_style(font_size=9); plot_data; plot_shrinkage_paths; plot_kernel_boxplots; plot_volatility;
gpvar.plots.plot_fit_components; plot_trace; plot_girf; plot_girf_periods; plot_girf_time_variation; combine_summaries;
gpvar.plots.plot_forecast_fan; plot_lpl_cumulative; plot_gp_illustration
gpvar.tables.save_table; to_markdown; to_latex; hyper_table; sv_table; girf_horizon_table; girf_peak_table;
gpvar.tables.asymmetry_table; size_asymmetry_table
```
