"""
Example 1 - Gaussian process regression on US data (Section 2 of the paper).

Reproduces the logic of Figures 1, 2 and A.1:
  * Figure 1: quarterly US CPI inflation (yoy) on a linear time trend, 2005Q1-2015Q4,
    for the linear kernel and the Gaussian kernel with kappa in {0.01, 0.1, 4}.
  * Figure 2: US real GDP growth (yoy) on the first lag of the JLN macroeconomic
    uncertainty index, same sample.
  * Figure A.1: inflation 1970Q1-2019Q4 under the linear persistence kernel r B B'
    for r in {0.001, 0.01, 0.1}.

Outputs are written to ``outputs/figures``.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gpvar  # noqa: E402
from gpvar import plots  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures")
os.makedirs(OUT, exist_ok=True)
plots.set_journal_style()

fred, _ = gpvar.data.load_fred_qd_subset()
unc = gpvar.data.load_jln_uncertainty(h=1)
infl = gpvar.data.transform_series(fred["CPIAUCSL"], 2)
gdp = gpvar.data.transform_series(fred["GDPC1"], 2)

# ---------------------------------------------------------------- Figure 1
s = slice("2005Q1", "2015Q4")
y1 = infl.loc[s].to_numpy()
x1 = np.arange(1, y1.size + 1, dtype=float)
sig2 = float(np.var(y1)) * 0.2
models = {
    "Linear kernel: XX'": gpvar.GPRegression("linear", v=0.02, sigma2=sig2),
    r"Gaussian kernel: $\kappa=0.01$": gpvar.GPRegression("gaussian", kappa=0.01, xi=float(np.var(y1)), sigma2=sig2),
    r"Gaussian kernel: $\kappa=0.1$": gpvar.GPRegression("gaussian", kappa=0.1, xi=float(np.var(y1)), sigma2=sig2),
    r"Gaussian kernel: $\kappa=4$": gpvar.GPRegression("gaussian", kappa=4.0, xi=float(np.var(y1)), sigma2=sig2),
}
plots.plot_gp_illustration(x1, y1, models, xlabel="Linear time trend (quarters since 2005Q1)",
                           title=r"Effect of $\kappa$ on the prior of $f$ and the posterior $f|y$: inflation and a linear time trend",
                           save=os.path.join(OUT, "fig01_gp_inflation_trend"), ylim=(-5, 10))

# ---------------------------------------------------------------- Figure 2
u = unc.shift(1).loc[s]
g = gdp.loc[s]
x2 = ((u - u.mean()) / u.std()).to_numpy()     # standardised lagged uncertainty
y2 = g.to_numpy()
sig2 = float(np.var(y2)) * 0.2
models = {
    "Linear kernel: XX'": gpvar.GPRegression("linear", v=2.0, sigma2=sig2),
    r"Gaussian kernel: $\kappa=0.01$": gpvar.GPRegression("gaussian", kappa=0.01, xi=float(np.var(y2)) * 4, sigma2=sig2),
    r"Gaussian kernel: $\kappa=0.1$": gpvar.GPRegression("gaussian", kappa=0.1, xi=float(np.var(y2)) * 4, sigma2=sig2),
    r"Gaussian kernel: $\kappa=4$": gpvar.GPRegression("gaussian", kappa=4.0, xi=float(np.var(y2)) * 4, sigma2=sig2),
}
plots.plot_gp_illustration(x2, y2, models, xlabel="Macroeconomic uncertainty indicator (lagged, standardised)",
                           title=r"GDP growth and the first lag of macroeconomic uncertainty",
                           save=os.path.join(OUT, "fig02_gp_gdp_uncertainty"), ylim=(-6, 8))

# ---------------------------------------------------------------- Figure A.1
s2 = slice("1970Q1", "2019Q4")
y3 = infl.loc[s2].to_numpy()
x3 = np.arange(1, y3.size + 1, dtype=float)
sig2 = float(np.var(y3)) * 0.1
models = {f"Linear persistence kernel: r = {r}": gpvar.GPRegression("persistence", r=r, sigma2=sig2) for r in (0.001, 0.01, 0.1)}
plots.plot_gp_illustration(x3, y3, models, xlabel="Quarters since 1970Q1",
                           title="Effect of r on the prior of f and the posterior f|y under the persistence kernel (inflation)",
                           save=os.path.join(OUT, "figA1_gp_persistence_kernel"), ylim=(-5, 15))
print("Example 1 done ->", OUT)
