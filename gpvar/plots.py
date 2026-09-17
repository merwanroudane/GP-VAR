"""
Publication-quality figures in the style of the paper (matplotlib only).

All functions return the ``Figure`` and accept ``save=<path without extension>``
to write both PDF (vector) and PNG (300 dpi).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import NBER_RECESSIONS

# paper-like palette
COL_POS = "#F2A93B"      # orange (positive shock / GP-VAR)
COL_NEG = "#6FA8DC"      # blue (negative shock)
COL_GRAY = "#7F7F7F"     # gray (benchmark / mirrored)
COL_RED = "#C0392B"
COL_BLUE = "#1F4E79"
COL_GREEN = "#2E8B57"
SERIES_COLORS = ["#C0392B", "#2E8B57", "#1F4E79", "#F2A93B", "#8E44AD", "#16A085", "#7F8C8D", "#D35400",
                 "#2C3E50", "#27AE60", "#E74C3C", "#3498DB", "#F39C12", "#1ABC9C", "#9B59B6", "#34495E"]


def set_journal_style(font_size: int = 9) -> None:
    """Clean, journal-like matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size - 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


def _save(fig, save: Optional[str]):
    if save:
        fig.savefig(save + ".pdf")
        fig.savefig(save + ".png", dpi=300)


def _panel_header(ax, text):
    ax.set_title(text, fontsize=mpl.rcParams["axes.titlesize"], pad=4,
                 bbox=dict(boxstyle="square,pad=0.25", facecolor="#E5E5E5", edgecolor="none"))


def _xaxis(results):
    """Time axis for in-sample plots: dates if available, otherwise 1..T."""
    d = results.dates
    if d is not None and isinstance(pd.Index(d), pd.DatetimeIndex):
        return pd.DatetimeIndex(d), True
    if d is not None and isinstance(pd.Index(d), pd.PeriodIndex):
        return pd.PeriodIndex(d).to_timestamp(how="end"), True
    return np.arange(1, results.T + 1), False


def shade_recessions(ax, dates, color="#D9D9D9"):
    for s, e in NBER_RECESSIONS:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if e < dates.min() or s > dates.max():
            continue
        ax.axvspan(max(s, dates.min()), min(e, dates.max()), color=color, alpha=0.7, lw=0, zorder=0)


# --------------------------------------------------------------------------- #
# GIRF plots
# --------------------------------------------------------------------------- #
def plot_girf(summaries: Dict[str, pd.DataFrame], variables: Optional[Sequence[str]] = None, ncols: int = 4,
              colors: Optional[Dict[str, str]] = None, title: Optional[str] = None, ylabel: str = "",
              save: Optional[str] = None, legend_title: Optional[str] = None, figsize=None,
              linestyles: Optional[Dict[str, str]] = None):
    """
    Faceted GIRF plot.  ``summaries`` maps a legend label to the long DataFrame
    returned by :meth:`GIRFResult.summary` (columns variable, horizon, q16, q50, q84).
    """
    first = next(iter(summaries.values()))
    variables = list(first["variable"].unique()) if variables is None else list(variables)
    n = len(variables)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (2.4 * ncols + 0.4, 1.9 * nrows + 0.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    default_colors = [COL_POS, COL_NEG, COL_GRAY, COL_RED, COL_GREEN, COL_BLUE]
    colors = colors or {k: default_colors[i % len(default_colors)] for i, k in enumerate(summaries)}
    linestyles = linestyles or {}
    qcols = [c for c in first.columns if c.startswith("q")]
    lo, mid, hi = qcols[0], qcols[len(qcols) // 2], qcols[-1]
    for i, v in enumerate(variables):
        ax = axes[i // ncols, i % ncols]
        for lab, df in summaries.items():
            d = df[df["variable"] == v]
            c = colors[lab]
            ax.fill_between(d["horizon"], d[lo], d[hi], color=c, alpha=0.35, lw=0)
            ax.plot(d["horizon"], d[mid], color=c, lw=1.2, label=lab, ls=linestyles.get(lab, "-"))
        ax.axhline(0, color="black", lw=0.6, ls=":")
        _panel_header(ax, v)
        ax.set_xlim(0, d["horizon"].max())
        ax.set_xticks(np.arange(0, d["horizon"].max() + 1, 4))
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if title:
        fig.suptitle(title, y=1.02)
    if len(summaries) > 1:
        if legend_title:
            labels = [f"{legend_title}: {l}" if i == 0 else l for i, l in enumerate(labels)]
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=len(summaries), bbox_to_anchor=(0.5, 0.0))
    else:
        fig.tight_layout()
    _save(fig, save)
    return fig


def plot_girf_periods(by_period: pd.DataFrame, variables: Optional[Sequence[str]] = None,
                      shock_labels=("positive", "negative"), save: Optional[str] = None, title: Optional[str] = None):
    """
    Rows = sub-sample periods, columns = variables (Figure 11).  ``by_period``
    must contain a column ``label`` (period) and a column ``shock`` in
    ``shock_labels`` (use :func:`combine_summaries`).
    """
    periods = list(by_period["label"].unique())
    variables = list(by_period["variable"].unique()) if variables is None else list(variables)
    nr, nc = len(periods), len(variables)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc + 0.4, 1.8 * nr + 0.7), squeeze=False)
    cols = {shock_labels[0]: COL_POS, shock_labels[1]: COL_NEG}
    for r, per in enumerate(periods):
        for c, v in enumerate(variables):
            ax = axes[r, c]
            for sh in shock_labels:
                d = by_period[(by_period["label"] == per) & (by_period["variable"] == v) & (by_period["shock"] == sh)]
                if d.empty:
                    continue
                ax.fill_between(d["horizon"], d["q16"], d["q84"], color=cols[sh], alpha=0.35, lw=0)
                ax.plot(d["horizon"], d["q50"], color=cols[sh], lw=1.1, label=sh)
            ax.axhline(0, color="black", lw=0.6, ls=":")
            if r == 0:
                _panel_header(ax, v)
            if c == 0:
                ax.set_ylabel(per, fontsize=mpl.rcParams["axes.labelsize"] - 1)
            ax.set_xticks(np.arange(0, d["horizon"].max() + 1, 4))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, [f"Shock sign: {labels[0]}"] + labels[1:], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    _save(fig, save)
    return fig


def plot_girf_time_variation(yearly: pd.DataFrame, periods: Dict[str, tuple], variables: Optional[Sequence[str]] = None,
                             save: Optional[str] = None, title: Optional[str] = None):
    """
    Yearly-averaged posterior-median GIRFs coloured from yellow (start of the
    period) to red (end of the period) - Figure 12 of the paper.
    """
    variables = list(yearly["variable"].unique()) if variables is None else list(variables)
    nr, nc = len(periods), len(variables)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc + 0.4, 1.8 * nr + 0.7), squeeze=False)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("yr", ["#FFE600", "#FF8C00", "#C00000"])
    for r, (per, (s, e)) in enumerate(periods.items()):
        y0, y1 = pd.Timestamp(s).year, pd.Timestamp(e).year
        sub = yearly[(yearly["year"] >= y0) & (yearly["year"] <= y1)]
        yrs = np.sort(sub["year"].unique())
        for c, v in enumerate(variables):
            ax = axes[r, c]
            for yr in yrs:
                d = sub[(sub["year"] == yr) & (sub["variable"] == v)]
                col = cmap((yr - y0) / max(y1 - y0, 1))
                ax.plot(d["horizon"], d["median"], color=col, lw=0.9)
            ax.axhline(0, color="black", lw=0.6, ls=":")
            if r == 0:
                _panel_header(ax, v)
            if c == 0:
                ax.set_ylabel(per, fontsize=mpl.rcParams["axes.labelsize"] - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.03, pad=0.06, aspect=40)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Start of period", "End of period"])
    if title:
        fig.suptitle(title, y=1.01)
    _save(fig, save)
    return fig


def combine_summaries(named: Dict[str, pd.DataFrame], key: str = "shock") -> pd.DataFrame:
    """Stack summaries with an identifying column."""
    out = []
    for k, df in named.items():
        d = df.copy()
        d[key] = k
        out.append(d)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# In-sample features
# --------------------------------------------------------------------------- #
def plot_shrinkage_paths(results, save: Optional[str] = None, title: Optional[str] = None):
    """Figure 5: posterior means of om_jt * xi (own and other lags) with recession shading."""
    sp = results.shrinkage_paths()
    dates, is_dates = _xaxis(results)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True)
    for ax, key, lab in zip(axes, ["own", "other"], ["Own lags", "Other lags"]):
        if is_dates:
            shade_recessions(ax, dates)
        for j, nm in enumerate(results.var_names):
            ax.plot(dates, sp[key][:, j], color=SERIES_COLORS[j % len(SERIES_COLORS)], lw=1.0,
                    ls=["-", "--", "-.", ":"][(j // len(SERIES_COLORS)) % 4], label=nm)
        _panel_header(ax, lab)
        ax.set_ylabel(r"$\omega_{jt}\,\xi_{j\cdot}$")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Variable")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_kernel_boxplots(results, save: Optional[str] = None, title: Optional[str] = None):
    """Figure 6: posterior IQR/median of kappa (inverse length scale) for own and other lags."""
    M = results.M
    fig, axes = plt.subplots(1, M, figsize=(1.35 * M + 0.5, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        for k, (col, key, x) in enumerate([(COL_BLUE, 1, 0.3), (COL_GREEN, 3, 0.7)]):
            d = results.hyper[:, j, key]
            q25, q50, q75 = np.quantile(d, [0.25, 0.5, 0.75])
            ax.add_patch(mpl.patches.Rectangle((x - 0.15, q25), 0.3, max(q75 - q25, 1e-4), facecolor=col, alpha=0.8, lw=0))
            ax.hlines(q50, x - 0.15, x + 0.15, color="black", lw=1.2)
        ax.set_xlim(0, 1); ax.set_xticks([])
        _panel_header(ax, results.var_names[j])
        ax.grid(True, axis="y")
    axes[0].set_ylabel(r"$\kappa$")
    h = [mpl.patches.Patch(color=COL_BLUE, label="Own"), mpl.patches.Patch(color=COL_GREEN, label="Other")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    if title:
        fig.suptitle(title, y=1.03)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_volatility(results, save: Optional[str] = None, title: Optional[str] = None, ncols: int = 4,
                    truth: Optional[np.ndarray] = None):
    """Posterior median and 68% band of the structural error standard deviations exp(h/2)."""
    q = results.volatility()
    M = results.M
    ncols = min(ncols, M)
    nrows = int(np.ceil(M / ncols))
    dates, is_dates = _xaxis(results)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols + 0.4, 1.9 * nrows + 0.5), squeeze=False)
    for j in range(M):
        ax = axes[j // ncols, j % ncols]
        if is_dates:
            shade_recessions(ax, dates)
        ax.fill_between(dates, q[0, :, j], q[2, :, j], color=COL_RED, alpha=0.25, lw=0)
        ax.plot(dates, q[1, :, j], color=COL_RED, lw=1.0, label="Posterior median")
        if truth is not None:
            ax.plot(dates, truth[:, j], color="black", lw=0.9, ls="--", label="True")
        _panel_header(ax, results.var_names[j])
    for k in range(M, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    if truth is not None:
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_fit_components(results, truth: Optional[Dict[str, np.ndarray]] = None, save: Optional[str] = None,
                        title: Optional[str] = None, probs=(0.05, 0.5, 0.95)):
    """
    Figure 3: posterior of f_j, g_j, m_j (= f_j + g_j) and y_j (rows = equations,
    columns = components).  ``truth`` may contain "F", "G", "m" arrays.
    """
    fq = results.fitted(probs)
    M, T = results.M, results.T
    x, _ = _xaxis(results)
    comps = [("f", r"$f$"), ("g", r"$g$"), ("m", r"$m$"), ("y", r"$y$")]
    fig, axes = plt.subplots(M, 4, figsize=(9.2, 1.75 * M + 0.6), squeeze=False)
    for j in range(M):
        for c, (key, lab) in enumerate(comps):
            ax = axes[j, c]
            if key == "y":
                yq = results.fitted(probs)["m"]
                om = np.quantile(results.omega, 0.5, axis=0)
                lo = yq[0, :, j] - 1.64 * np.sqrt(om[:, j]); hi = yq[2, :, j] + 1.64 * np.sqrt(om[:, j])
                ax.fill_between(x, lo, hi, color=COL_RED, alpha=0.2, lw=0)
                ax.plot(x, yq[1, :, j], color=COL_RED, lw=0.9, label="Posterior")
                ax.plot(x, results.y[:, j], color="black", lw=0.8, label="True")
            else:
                q = fq[key]
                ax.fill_between(x, q[0, :, j], q[2, :, j], color=COL_RED, alpha=0.25, lw=0)
                ax.plot(x, q[1, :, j], color=COL_RED, lw=0.9, label="Posterior")
                tk = {"f": "F", "g": "G", "m": "m"}[key]
                if truth is not None and tk in truth:
                    ax.plot(x, truth[tk][:, j], color="black", lw=0.8, label="True")
            ax.set_title(f"{lab}$_{{{j + 1}}}$", loc="left", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_gp_illustration(x, y, models: Dict[str, "GPRegression"], xgrid=None, save: Optional[str] = None,
                         title: Optional[str] = None, xlabel: str = "", n_prior_draws: int = 3, seed: int = 1,
                         ylim=None):
    """
    Figures 1/2: for each kernel/hyperparameter setting, the left panel shows
    the prior (90% band + draws) and the right panel the posterior (90% band + median).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    xgrid = np.linspace(x.min(), x.max(), 200) if xgrid is None else np.asarray(xgrid)
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(7.2, 1.55 * n + 0.6), squeeze=False)
    for i, (lab, gp) in enumerate(models.items()):
        gp.fit(x[:, None], y)
        sd = gp.prior_sd(xgrid[:, None]) if gp.kernel != "persistence" else gp.prior_sd(x[:, None])
        xg = xgrid if gp.kernel != "persistence" else x
        axl, axr = axes[i]
        axl.fill_between(xg, -1.645 * sd, 1.645 * sd, color=COL_RED, alpha=0.25, lw=0)
        for d in gp.prior_draws(xg[:, None], n_prior_draws, seed=seed + i):
            axl.plot(xg, d, color=COL_RED, lw=0.8, ls="--")
        axl.plot(x, y, "k.", ms=2.5)
        mean, psd = gp.posterior(None if gp.kernel == "persistence" else xgrid[:, None])
        axr.fill_between(xg, mean - 1.645 * psd, mean + 1.645 * psd, color=COL_RED, alpha=0.3, lw=0)
        axr.plot(xg, mean, color=COL_RED, lw=1.2)
        axr.plot(x, y, "k.", ms=2.5)
        axl.set_title(f"({'abcdefgh'[i]}) {lab}", loc="left", fontsize=9)
        if ylim is not None:
            axl.set_ylim(*ylim); axr.set_ylim(*ylim)
        if i == 0:
            axl.set_title(r"$f$", loc="center"); axr.set_title(r"$f\,|\,y$", loc="center")
    axes[-1, 0].set_xlabel(xlabel); axes[-1, 1].set_xlabel(xlabel)
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_data(Y: pd.DataFrame, ncols: int = 4, save: Optional[str] = None, title: Optional[str] = None,
              units: Optional[Dict[str, str]] = None):
    """Time-series panel of the data with NBER recession shading."""
    M = Y.shape[1]
    ncols = min(ncols, M)
    nrows = int(np.ceil(M / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols + 0.4, 1.8 * nrows + 0.5), squeeze=False)
    for j, c in enumerate(Y.columns):
        ax = axes[j // ncols, j % ncols]
        shade_recessions(ax, pd.DatetimeIndex(Y.index))
        ax.plot(Y.index, Y[c], color=COL_BLUE, lw=0.9)
        _panel_header(ax, c)
        if units and c in units:
            ax.set_ylabel(units[c], fontsize=7)
    for k in range(M, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_forecast_fan(Y: pd.DataFrame, fc, variables: Optional[Sequence[str]] = None, n_hist: int = 24,
                      save: Optional[str] = None, title: Optional[str] = None, actual: Optional[pd.DataFrame] = None):
    """Fan chart of the predictive distribution (5/16/50/84/95 percent quantiles)."""
    q = fc.quantiles(probs=(0.05, 0.16, 0.5, 0.84, 0.95))
    variables = list(fc.var_names) if variables is None else list(variables)
    n = len(variables)
    ncols = min(4, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols + 0.4, 1.9 * nrows + 0.5), squeeze=False)
    freq = pd.infer_freq(Y.index) or "QE"
    fdates = pd.date_range(Y.index[-1], periods=fc.horizon + 1, freq=freq)[1:]
    for i, v in enumerate(variables):
        ax = axes[i // ncols, i % ncols]
        ax.plot(Y.index[-n_hist:], Y[v].iloc[-n_hist:], color="black", lw=0.9)
        d = q[q["variable"] == v]
        ax.fill_between(fdates, d["q05"], d["q95"], color=COL_BLUE, alpha=0.2, lw=0)
        ax.fill_between(fdates, d["q16"], d["q84"], color=COL_BLUE, alpha=0.35, lw=0)
        ax.plot(fdates, d["q50"], color=COL_BLUE, lw=1.1)
        if actual is not None and v in actual:
            a = actual[v].reindex(fdates)
            ax.plot(fdates, a, color=COL_RED, lw=0.9, ls="--", marker="o", ms=2)
        ax.axvline(Y.index[-1], color=COL_GRAY, lw=0.6, ls=":")
        _panel_header(ax, v)
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_lpl_cumulative(evals: Dict[str, pd.DataFrame], horizon: int, variable: str = "joint",
                        benchmark: Optional[str] = None, save: Optional[str] = None, title: Optional[str] = None):
    """Cumulative log predictive likelihood (relative to a benchmark if given) over the evaluation sample."""
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    base = None
    if benchmark is not None:
        b = evals[benchmark]
        base = b[(b["horizon"] == horizon) & (b["variable"] == variable)].set_index("target")["lpl"]
    for i, (lab, df) in enumerate(evals.items()):
        d = df[(df["horizon"] == horizon) & (df["variable"] == variable)].set_index("target")["lpl"]
        if base is not None:
            if lab == benchmark:
                continue
            d = d - base.reindex(d.index)
        ax.plot(d.index, d.cumsum(), lw=1.2, label=lab, color=SERIES_COLORS[i % len(SERIES_COLORS)])
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("Cumulative LPBF" if base is not None else "Cumulative LPL")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_trace(results, save: Optional[str] = None):
    """MCMC diagnostics: traces of the kernel hyperparameters and SV persistence per equation."""
    M = results.M
    fig, axes = plt.subplots(M, 3, figsize=(8, 1.4 * M + 0.5), squeeze=False)
    for j in range(M):
        axes[j, 0].plot(results.hyper[:, j, 1], lw=0.5, color=COL_BLUE); axes[j, 0].plot(results.hyper[:, j, 3], lw=0.5, color=COL_GREEN)
        axes[j, 1].plot(results.hyper[:, j, 0], lw=0.5, color=COL_BLUE); axes[j, 1].plot(results.hyper[:, j, 2], lw=0.5, color=COL_GREEN)
        axes[j, 2].plot(results.sv_params[:, j, 1], lw=0.5, color=COL_RED)
        axes[j, 0].set_ylabel(results.var_names[j], fontsize=7)
    axes[0, 0].set_title(r"$\kappa$ (own blue / other green)"); axes[0, 1].set_title(r"$\xi$"); axes[0, 2].set_title(r"$\rho_h$")
    fig.tight_layout()
    _save(fig, save)
    return fig
