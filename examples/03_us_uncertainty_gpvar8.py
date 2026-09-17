"""
Example 3 - The macroeconomic effects of uncertainty shocks in the US
(Sections 5 and 6 of the paper) with real data.

Data (quarterly, 1970Q1-2019Q4, T = 200): the GP-VAR-8 of Table B.1
    UNC   JLN macroeconomic uncertainty (h = 1)
    RGDP  real GDP, yoy growth                 EMP   civilian employment, yoy growth
    AWH   average weekly hours (manufacturing) CPI   CPI inflation, yoy
    AHE   real average hourly earnings, yoy    FFR   federal funds rate
    SP500 S&P 500, qoq return
Uncertainty is ordered first (recursive identification: the uncertainty shock
moves every variable on impact, as in the paper's Figure 7; use ``--order`` to
change the ordering).

Produces (outputs/figures, outputs/tables):
    data panel and descriptive statistics
    Figure 5  linear shrinkage parameters om_jt * xi          Figure 6  inverse length scales kappa
    Figure 7  average GIRFs, GP-VAR-8 vs. BVAR-8              Figure 9  sign asymmetries
    Figure 10 size asymmetries                                 Figure 11 sub-sample GIRFs
    Figure 12 yearly GIRFs within sub-samples                  Figure C.6 recessions vs. expansions
    stochastic volatility paths, MCMC traces, forecast fan chart
    tables of hyperparameters, SV parameters, Q, GIRF peaks/horizons, asymmetry statistics

Command line:  python 03_us_uncertainty_gpvar8.py [--nburn 2000] [--nsave 2000] [--girf-draws 100]
               [--girf-sims 5] [--end 2019Q4] [--quick] [--reuse-fit]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gpvar  # noqa: E402
from gpvar import plots, tables  # noqa: E402
from gpvar.data import PAPER_SUBSAMPLES, recession_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nburn", type=int, default=2000)  # paper-size chains; ~10 min for M=8, T=200
ap.add_argument("--nsave", type=int, default=2000)
ap.add_argument("--girf-draws", type=int, default=100)
ap.add_argument("--girf-sims", type=int, default=5)
ap.add_argument("--horizon", type=int, default=16)
ap.add_argument("--start", default="1970Q1")
ap.add_argument("--end", default="2019Q4")
ap.add_argument("--p", type=int, default=5)
ap.add_argument("--order", default="UNC,RGDP,EMP,AWH,CPI,AHE,FFR,SP500")
ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
ap.add_argument("--tag", default="gpvar8")
ap.add_argument("--reuse-fit", action="store_true", help="load outputs/<tag>_fit.pkl instead of re-estimating")
args = ap.parse_args()
if args.quick:
    args.nburn, args.nsave, args.girf_draws, args.girf_sims = 60, 60, 6, 2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "outputs", "figures")
TAB = os.path.join(ROOT, "outputs", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plots.set_journal_style()
tag = args.tag
focus = ["UNC", "RGDP", "EMP", "SP500"]
timings = {}

# =============================================================== 1. data
Y, info = gpvar.data.build_dataset(8, args.start, args.end, order=args.order.split(","))
nonfocus = [c for c in Y.columns if c not in focus]
print(Y.shape, Y.index[0].date(), "->", Y.index[-1].date())
plots.plot_data(Y, save=os.path.join(FIG, f"{tag}_data"), title="GP-VAR-8 data set (shaded: NBER recessions)")
tables.save_table(info.reset_index(), os.path.join(TAB, f"{tag}_data_description"), caption="Data description",
                  label="tab:data", notes="Transformations: 1 none, 2 year-on-year growth rate, 3 quarter-on-quarter growth rate. "
                  "Sources: FRED-QD (McCracken and Ng, 2020) and Jurado, Ludvigson and Ng (2015).")
tables.save_table(gpvar.data.describe(Y).reset_index().rename(columns={"index": "variable"}),
                  os.path.join(TAB, f"{tag}_descriptives"), caption="Descriptive statistics", label="tab:desc")

# =============================================================== 2. estimation
t0 = time.time()
fit_path = os.path.join(ROOT, "outputs", f"{tag}_fit.pkl")
if args.reuse_fit and os.path.exists(fit_path):
    res = gpvar.GPVARResults.load(fit_path)
    model = res.model
    print(f"loaded posterior from {fit_path} ({res.nsave} draws, runtime {res.runtime:.0f}s)")
else:
    model = gpvar.GPVAR(Y, p=args.p, sv=True, hyper_scheme="semi-automatic", n_kappa=20, n_xi=50)
    print(f"precomputation: {model.precompute_time:.1f}s; median-heuristic kappa_bar (own/other):")
    for j, nm in enumerate(model.var_names):
        print(f"   {nm:6s} {model.blocks_own[j].grid.kappa_bar:.3f} / {model.blocks_other[j].grid.kappa_bar:.3f}")
    res = model.fit(nburn=args.nburn, nsave=args.nsave, seed=2026, print_every=250)
    print(f"MCMC: {res.runtime:.0f}s, SV-IMH acceptance: {dict(zip(model.var_names, res.accept_rate.round(2)))}")
    res.save(fit_path)
timings["fit_seconds"] = res.runtime

tables.save_table(tables.hyper_table(res), os.path.join(TAB, f"{tag}_hyperparameters"),
                  caption="Kernel hyperparameters: posterior median [interquartile range]", label="tab:hyper")
tables.save_table(tables.sv_table(res), os.path.join(TAB, f"{tag}_sv_parameters"),
                  caption="Stochastic-volatility state-equation parameters: posterior median [5\\%, 95\\%]", label="tab:sv")
tables.save_table(res.Q_summary(), os.path.join(TAB, f"{tag}_Q"),
                  caption="Free elements of the contemporaneous matrix Q (horseshoe prior)", label="tab:Q")
plots.plot_shrinkage_paths(res, save=os.path.join(FIG, f"{tag}_fig05_shrinkage"),
                           title="Linear shrinkage parameters of equation-specific kernels (posterior means of $\\omega_{jt}\\xi_{j\\cdot}$)")
plots.plot_kernel_boxplots(res, save=os.path.join(FIG, f"{tag}_fig06_kappa"),
                           title="Inverse length scale parameters of equation-specific kernels (median and IQR)")
plots.plot_volatility(res, save=os.path.join(FIG, f"{tag}_volatility"),
                      title="Posterior of the structural error standard deviations $\\exp(h_{jt}/2)$ (standardised data)")
plots.plot_trace(res, save=os.path.join(FIG, f"{tag}_mcmc_traces"))

# =============================================================== 3. linear benchmark
bvar = gpvar.BVAR(Y, p=args.p).fit(nsave=max(200, args.nsave), seed=1)
bvar_irf = bvar.irf("UNC", horizon=args.horizon, label="BVAR-8")

# =============================================================== 4. GIRFs
gk = dict(horizon=args.horizon, n_sim=args.girf_sims, n_draws=args.girf_draws, shock_scale="unit")
t0 = time.time()
g_pos = gpvar.girf(res, "UNC", size=1.0, sign=+1, seed=1, **gk)
g_neg = gpvar.girf(res, "UNC", size=1.0, sign=-1, seed=2, **gk)
g_pos2 = gpvar.girf(res, "UNC", size=2.0, sign=+1, seed=3, **gk)
timings["girf_seconds"] = time.time() - t0
for name, g in [("pos1", g_pos), ("neg1", g_neg), ("pos2", g_pos2)]:
    np.save(os.path.join(ROOT, "outputs", f"{tag}_girf_{name}.npy"), g.irf)

# ---- Figure 7 (+ C.4): GP-VAR vs BVAR, average GIRF
s_gp = g_pos.summary()
plots.plot_girf({"GP-VAR-8": s_gp, "BVAR-8": bvar_irf}, variables=focus, colors={"GP-VAR-8": plots.COL_POS, "BVAR-8": plots.COL_GRAY},
                save=os.path.join(FIG, f"{tag}_fig07_girf_vs_bvar_focus"),
                title="Impulse responses of focus variables to a positive uncertainty shock: GP-VAR-8 vs. BVAR-8 (68% bands)")
plots.plot_girf({"GP-VAR-8": s_gp, "BVAR-8": bvar_irf}, variables=nonfocus, colors={"GP-VAR-8": plots.COL_POS, "BVAR-8": plots.COL_GRAY},
                save=os.path.join(FIG, f"{tag}_figC4_girf_vs_bvar_nonfocus"),
                title="Impulse responses of non-focus variables: GP-VAR-8 vs. BVAR-8")

# ---- Figure 9: sign asymmetry
s_neg = g_neg.summary()
s_negm = s_neg.copy()
for c in ("q16", "q50", "q84"):
    s_negm[c] = -s_negm[c]
s_negm[["q16", "q84"]] = s_negm[["q84", "q16"]].to_numpy()
sign_sets = {"positive": s_gp, "negative": s_neg, "negative x (-1)": s_negm}
sign_cols = {"positive": plots.COL_POS, "negative": plots.COL_NEG, "negative x (-1)": plots.COL_GRAY}
plots.plot_girf(sign_sets, variables=focus, colors=sign_cols, legend_title="Shock sign",
                save=os.path.join(FIG, f"{tag}_fig09_sign_asymmetry_focus"),
                title="Shock sign asymmetries in responses of focus variables (GP-VAR-8, average GIRFs)")
plots.plot_girf(sign_sets, variables=nonfocus, colors=sign_cols, legend_title="Shock sign",
                save=os.path.join(FIG, f"{tag}_figC9_sign_asymmetry_nonfocus"),
                title="Shock sign asymmetries in responses of non-focus variables")

# ---- Figure 10: size asymmetry
s_2 = g_pos2.summary()
s_2h = s_2.copy()
for c in ("q16", "q50", "q84"):
    s_2h[c] = s_2h[c] / 2.0
size_sets = {"1 sd": s_gp, "2 sd": s_2, "2 sd x (1/2)": s_2h}
size_cols = {"1 sd": plots.COL_POS, "2 sd": plots.COL_NEG, "2 sd x (1/2)": plots.COL_GRAY}
plots.plot_girf(size_sets, variables=focus, colors=size_cols, legend_title="Shock size",
                save=os.path.join(FIG, f"{tag}_fig10_size_asymmetry_focus"),
                title="Shock size asymmetries in responses of focus variables (GP-VAR-8, average GIRFs)")
plots.plot_girf(size_sets, variables=nonfocus, colors=size_cols, legend_title="Shock size",
                save=os.path.join(FIG, f"{tag}_figC10_size_asymmetry_nonfocus"),
                title="Shock size asymmetries in responses of non-focus variables")

# ---- Figure 11: sub-samples (positive and negative)
periods = {k.split(" ")[0]: v for k, v in PAPER_SUBSAMPLES.items()}
if pd.Timestamp(Y.index[-1]) > pd.Timestamp("2020-06-30"):
    periods["2020Q1-end"] = ("2020-01-01", str(Y.index[-1].date()))
bp = plots.combine_summaries({"positive": g_pos.by_periods(periods), "negative": g_neg.by_periods(periods)})
plots.plot_girf_periods(bp, variables=focus, save=os.path.join(FIG, f"{tag}_fig11_subsamples_focus"),
                        title="Impulse responses of focus variables across sub-sample periods")
plots.plot_girf_periods(bp, variables=nonfocus, save=os.path.join(FIG, f"{tag}_figC5_subsamples_nonfocus"),
                        title="Impulse responses of non-focus variables across sub-sample periods")

# ---- Figure 12: time variation within sub-samples
yearly = g_pos.yearly_medians()
plots.plot_girf_time_variation(yearly, periods, variables=focus, save=os.path.join(FIG, f"{tag}_fig12_time_variation_focus"),
                               title="Period-specific impulse responses (yearly averages of posterior medians)")
plots.plot_girf_time_variation(yearly, periods, variables=nonfocus, save=os.path.join(FIG, f"{tag}_figC7_time_variation_nonfocus"),
                               title="Period-specific impulse responses of non-focus variables")

# ---- Figure C.6: recessions vs expansions
rec = recession_mask(g_pos.hist_dates)
rec_sets = plots.combine_summaries({"positive": pd.concat([g_pos.summary(~rec, label="Expansions"), g_pos.summary(rec, label="NBER recessions")]),
                                    "negative": pd.concat([g_neg.summary(~rec, label="Expansions"), g_neg.summary(rec, label="NBER recessions")])})
plots.plot_girf_periods(rec_sets, variables=focus, save=os.path.join(FIG, f"{tag}_figC6_recessions_expansions"),
                        title="Impulse responses of focus variables in expansions and recessions")

# ---- tables
tables.save_table(tables.girf_horizon_table(g_pos), os.path.join(TAB, f"{tag}_girf_positive_horizons"),
                  caption="Average GIRFs to a positive one-standard-deviation uncertainty shock: posterior median [16\\%, 84\\%]",
                  label="tab:girfpos", notes="Responses in standard-deviation units of the (standardised) variables; horizons in quarters.")
tables.save_table(tables.girf_horizon_table(g_neg), os.path.join(TAB, f"{tag}_girf_negative_horizons"),
                  caption="Average GIRFs to a negative one-standard-deviation uncertainty shock", label="tab:girfneg")
tables.save_table(tables.girf_peak_table({"positive 1 sd": g_pos, "negative 1 sd": g_neg, "positive 2 sd": g_pos2}),
                  os.path.join(TAB, f"{tag}_girf_peaks"), caption="Peak responses to uncertainty shocks", label="tab:peaks",
                  notes="Peak = largest absolute posterior-median response; 68\\% credible interval at the peak horizon; cumulative = sum of median responses over 0-16 quarters.")
tables.save_table(tables.asymmetry_table(g_pos, g_neg), os.path.join(TAB, f"{tag}_sign_asymmetry"),
                  caption="Sign asymmetry: posterior median of $\\delta^{+}_h + \\delta^{-}_h$ (probability $>0$ in parentheses)",
                  label="tab:sign", notes="Under a linear model the responses to positive and negative shocks are mirror images and the statistic is zero.")
tables.save_table(tables.size_asymmetry_table(g_pos, g_pos2), os.path.join(TAB, f"{tag}_size_asymmetry"),
                  caption="Size asymmetry: ratio of the response to a 2-sd shock to the response to a 1-sd shock", label="tab:size",
                  notes="Equal to 2 under proportionality (linear model).")
per_tab = []
for lab, (s0, e0) in periods.items():
    t = g_pos.peak_table(g_pos.mask_from_dates(s0, e0)); t.insert(0, "period", lab); per_tab.append(t)
tables.save_table(pd.concat(per_tab, ignore_index=True), os.path.join(TAB, f"{tag}_girf_peaks_by_period"),
                  caption="Peak responses to a positive uncertainty shock by sub-sample", label="tab:peakper")
tables.save_table(g_pos.state_dependence()[lambda d: d.horizon.isin([1, 4, 8])], os.path.join(TAB, f"{tag}_state_dependence"),
                  caption="State dependence: dispersion of history-specific GIRFs (posterior medians) across quarters", label="tab:state")

# =============================================================== 5. forecasts from the end of the sample
fc = res.predict(horizon=8, n_sim=5, n_draws=min(400, args.nsave), seed=7)
plots.plot_forecast_fan(Y, fc, save=os.path.join(FIG, f"{tag}_forecast_fan"),
                        title=f"Predictive distribution {fc.origin_date.date()} + 1..8 quarters (GP-VAR-8)")
tables.save_table(fc.quantiles().query("horizon in [1, 4, 8]"), os.path.join(TAB, f"{tag}_forecast_quantiles"),
                  caption="Predictive quantiles from the end of the sample", label="tab:fc")

timings["total_seconds"] = time.time() - t0 + res.runtime
with open(os.path.join(ROOT, "outputs", f"{tag}_run_info.json"), "w") as fh:
    json.dump(dict(args=vars(args), timings=timings, accept_rate=dict(zip(model.var_names, res.accept_rate.round(3).tolist())),
                   T=model.T, M=model.M), fh, indent=2)
print("Example 3 done.", json.dumps(timings))
