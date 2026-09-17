"""
Example 4 - Recursive density-forecast evaluation (Section 5.2 of the paper).

Expanding-window exercise for the GP-VAR-8 with SV, the homoskedastic
GP-VAR-8 and a Minnesota BVAR-8 benchmark.  For every forecast origin the
models are re-estimated, predictive draws for h = 1 and h = 4 quarters are
simulated and log predictive likelihoods (LPLs) of the realised values are
computed for the focus variables RGDP, CPI and FFR (jointly and marginally).
The table reports log predictive Bayes factors (LPBFs) relative to the BVAR
(positive = better than the benchmark), in the spirit of Table 2.

The paper uses 80 quarterly origins (2000Q1-2019Q4) and several thousand
MCMC draws per origin; the defaults here use fewer origins and draws so that
the exercise finishes in well under an hour on a laptop.

Command line: python 04_forecast_evaluation.py [--first 2007Q4] [--last 2018Q4] [--step 4] [--nburn 400] [--nsave 400]
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gpvar  # noqa: E402
from gpvar import plots, tables  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--first", default="2007Q4")
ap.add_argument("--last", default="2018Q4")
ap.add_argument("--step", type=int, default=4)
ap.add_argument("--nburn", type=int, default=400)
ap.add_argument("--nsave", type=int, default=400)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.nburn, args.nsave, args.step = 30, 30, 20

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "outputs", "figures")
TAB = os.path.join(ROOT, "outputs", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plots.set_journal_style()

Y, _ = gpvar.data.build_dataset(8, "1970Q1", "2019Q4")
focus = ["RGDP", "CPI", "FFR"]
first = pd.Period(args.first, "Q").to_timestamp(how="end").normalize()
last = pd.Period(args.last, "Q").to_timestamp(how="end").normalize()
common = dict(first_origin=first, last_origin=last, step=args.step, horizons=(1, 4), focus=focus,
              predict_kwargs=dict(n_sim=4, seed=0))

t0 = time.time()
evals = {}
evals["GP-VAR-8 SV"] = gpvar.recursive_forecast_evaluation(
    Y, model_factory=lambda Yt: gpvar.GPVAR(Yt, p=5, sv=True), fit_kwargs=dict(nburn=args.nburn, nsave=args.nsave, seed=1),
    label="GP-VAR-8 SV", **common)
evals["GP-VAR-8 homoskedastic"] = gpvar.recursive_forecast_evaluation(
    Y, model_factory=lambda Yt: gpvar.GPVAR(Yt, p=5, sv=False), fit_kwargs=dict(nburn=args.nburn, nsave=args.nsave, seed=1),
    label="GP-VAR-8 homoskedastic", **common)
evals["BVAR-8"] = gpvar.recursive_forecast_evaluation(
    Y, model_factory=lambda Yt: gpvar.BVAR(Yt, p=5), fit_kwargs=dict(nsave=max(args.nsave, 200), seed=1),
    label="BVAR-8", **{**common, "predict_kwargs": dict(n_sim=4, seed=0)})
print(f"evaluation finished in {time.time() - t0:.0f}s")

allev = pd.concat(evals.values(), ignore_index=True)
allev.to_csv(os.path.join(TAB, "forecast_evaluation_raw.csv"), index=False)

tab = gpvar.lpbf_table(evals, benchmark="BVAR-8")
wide = tab.pivot(index="variable", columns="horizon", values=[c for c in tab.columns if c not in ("horizon", "variable")])
wide.columns = [f"{m} (h={h})" for m, h in wide.columns]
wide = wide.reindex(["joint"] + focus)
tables.save_table(wide.reset_index(), os.path.join(TAB, "table02_lpbf"),
                  caption="Log predictive Bayes factors relative to the BVAR-8 (positive = better than the benchmark)",
                  label="tab:lpbf", floatfmt="{:.2f}",
                  notes=f"Recursive design, origins {args.first} to {args.last} every {args.step} quarters; joint = RGDP, CPI and FFR. "
                  "The last column block reports the benchmark's log predictive likelihoods.")
print(tables.to_markdown(wide.reset_index(), floatfmt="{:.2f}"))

crps_tab = allev[allev.variable != "joint"].groupby(["model", "horizon", "variable"])["crps"].mean().unstack("model").reset_index()
tables.save_table(crps_tab, os.path.join(TAB, "table02b_crps"), caption="Average CRPS (lower is better)", label="tab:crps")

for h in (1, 4):
    plots.plot_lpl_cumulative(evals, horizon=h, variable="joint", benchmark="BVAR-8",
                              save=os.path.join(FIG, f"forecast_cumulative_lpbf_h{h}"),
                              title=f"Cumulative joint LPBF vs. BVAR-8, h = {h}")
print("Example 4 done.")
