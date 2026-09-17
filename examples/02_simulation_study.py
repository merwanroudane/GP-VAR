"""
Example 2 - Illustration using synthetic data (Section 4 of the paper).

* Simulates the highly non-linear three-equation DGP (structural break,
  sine/quadratic non-linearities, t_3 shocks in equation 1, random-walk SV).
* Fits the GP-VAR with SV and reproduces Figure 3 (posterior of f_j, g_j,
  m_j = f_j + g_j and y_j against the truth), the stochastic-volatility paths,
  and Table 1 (correlation between the posterior median of m_j and the true
  conditional mean over R Monte Carlo replications).

Command line:  python 02_simulation_study.py [--reps R] [--nburn B] [--nsave S]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gpvar  # noqa: E402
from gpvar import plots, tables  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=10, help="Monte Carlo replications for Table 1 (paper: 100)")
ap.add_argument("--nburn", type=int, default=1000)
ap.add_argument("--nsave", type=int, default=1000)
ap.add_argument("--T", type=int, default=200)
args = ap.parse_args()

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "outputs", "figures")
TAB = os.path.join(ROOT, "outputs", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plots.set_journal_style()

# ------------------------------------------------------------------ single realisation (Figure 3)
sim = gpvar.simulate_paper_dgp(T=args.T, p=5, seed=1)
t0 = time.time()
model = gpvar.GPVAR(sim.Y, p=5, standardize=True)
res = model.fit(nburn=args.nburn, nsave=args.nsave, seed=1, print_every=500)
print(f"fit: {res.runtime:.0f}s; SV acceptance {res.accept_rate.round(2)}")

# the model works with standardised data: put the truth on the same scale (affine per column)
sd, mu = model.sd_, model.mean_
p = model.p
truth = {"F": (sim.F[p:]) / sd, "G": (sim.G[p:]) / sd, "m": (sim.m[p:] - mu) / sd}
# f and g are only identified up to a common constant; for the plot centre both like the model does
truth["G"] = truth["G"] - truth["G"].mean(axis=0)
truth["F"] = truth["m"] - truth["G"]
plots.plot_fit_components(res, truth=truth, save=os.path.join(FIG, "fig03_simulation_fit"),
                          title=r"Posterior of $f_j$, $g_j$, $m_j=f_j+g_j$ and $y_j$ versus the true DGP components")
plots.plot_volatility(res, truth=np.sqrt(sim.omega[p:]) / sd, save=os.path.join(FIG, "fig03b_simulation_volatility"),
                      title="Posterior of the structural error standard deviations versus the truth (standardised scale)", ncols=3)

corr1 = res.fit_correlations(sim.m)
print(corr1)

# ------------------------------------------------------------------ Table 1: Monte Carlo
rows = []
for r in range(args.reps):
    s = gpvar.simulate_paper_dgp(T=args.T, p=5, seed=100 + r)
    rr = gpvar.GPVAR(s.Y, p=5).fit(nburn=max(200, args.nburn // 4), nsave=max(200, args.nsave // 4),
                                  seed=r, verbose=False)
    c = rr.fit_correlations(s.m)["corr"].to_numpy()
    rows.append(c)
    print(f"replication {r + 1}/{args.reps}: corr = {np.round(c, 3)}")
C = np.array(rows)
tab1 = pd.DataFrame({"": ["Avg.", "SD"], "m1": [C[:, 0].mean(), C[:, 0].std(ddof=1)],
                     "m2": [C[:, 1].mean(), C[:, 1].std(ddof=1)], "m3": [C[:, 2].mean(), C[:, 2].std(ddof=1)]})
tables.save_table(tab1, os.path.join(TAB, "table01_simulation_correlations"),
                  caption="Correlations between the posterior median of $m_j = f_j + g_j$ and the actual realisation",
                  label="tab:sim", notes=f"Results from {args.reps} realisations of the Section 4 DGP with T = {args.T}. "
                  "Avg. is the average correlation, SD the standard deviation across realisations.")
print(tables.to_markdown(tab1))
res.save(os.path.join(ROOT, "outputs", "simulation_fit.pkl"))
print("Example 2 done.")
