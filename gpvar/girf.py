"""
Generalized impulse response functions (Koop, Pesaran and Potter, 1996) for the
GP-VAR - Section 3.5 and Online Appendix A.5 of the paper.

For a posterior draw and a history ``t`` the response at horizon ``k`` to a
structural shock of size ``varsigma`` in equation ``j`` is

    delta_{k,t} = E[ y_{t+k} | I_t, eps_jt = varsigma ] - E[ y_{t+k} | I_t ],

where both expectations are simulated with *common* future structural shocks
and log-volatility innovations (only the shock of interest differs).  At
impact the shock moves ``y_t`` by ``varsigma * q_j`` with ``q_j`` the j-th
column of ``(I - Q)^{-1}``.  Because the conditional mean is non-linear the
responses depend on the history (state dependence), the sign and the size of
the shock; averaging over ``t`` gives the "average GIRF" reported in the
paper's Figure 7, sub-sample averages give Figure 11, yearly medians Figure 12.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .predict import DrawContext, select_draws


@dataclass
class GIRFResult:
    """Container for GIRF draws with shape (n_draws, n_hist, horizon+1, M)."""
    irf: np.ndarray
    hist_idx: np.ndarray
    hist_dates: Optional[pd.Index]
    var_names: Sequence[str]
    shock_var: str
    size: float
    sign: int
    shock_scale: str
    sd: np.ndarray            # column standard deviations used for standardisation

    @property
    def horizon(self) -> int:
        return self.irf.shape[2] - 1

    # ------------------------------------------------------------- selection
    def mask_from_dates(self, start=None, end=None) -> np.ndarray:
        if self.hist_dates is None:
            raise ValueError("no dates attached to the GIRF result")
        m = np.ones(len(self.hist_dates), dtype=bool)
        if start is not None:
            m &= self.hist_dates >= pd.Timestamp(start)
        if end is not None:
            m &= self.hist_dates <= pd.Timestamp(end)
        return m

    # -------------------------------------------------------------- summaries
    def average(self, mask: Optional[np.ndarray] = None, original_units: bool = False) -> np.ndarray:
        """Average over histories -> (n_draws, horizon+1, M)."""
        arr = self.irf if mask is None else self.irf[:, mask]
        out = arr.mean(axis=1)
        if original_units:
            out = out * self.sd[None, None, :]
        return out

    def summary(self, mask: Optional[np.ndarray] = None, probs=(0.16, 0.5, 0.84),
                original_units: bool = False, label: Optional[str] = None) -> pd.DataFrame:
        """Long DataFrame: variable, horizon, one column per quantile."""
        avg = self.average(mask, original_units)
        q = np.quantile(avg, probs, axis=0)      # (len(probs), H+1, M)
        rows = []
        for j, nm in enumerate(self.var_names):
            for k in range(self.horizon + 1):
                rows.append([nm, k] + [q[i, k, j] for i in range(len(probs))])
        cols = ["variable", "horizon"] + [f"q{int(round(p * 100)):02d}" for p in probs]
        df = pd.DataFrame(rows, columns=cols)
        if label is not None:
            df.insert(0, "label", label)
        return df

    def by_periods(self, periods: Dict[str, Tuple[str, str]], probs=(0.16, 0.5, 0.84),
                   original_units: bool = False) -> pd.DataFrame:
        """Sub-sample averages (e.g. Figure 11): ``periods = {"1970Q1-1984Q4": ("1970-01-01", "1984-12-31")}``."""
        dfs = []
        for lab, (s, e) in periods.items():
            m = self.mask_from_dates(s, e)
            if m.sum() == 0:
                continue
            dfs.append(self.summary(m, probs, original_units, label=lab))
        return pd.concat(dfs, ignore_index=True)

    def yearly_medians(self, mask: Optional[np.ndarray] = None, original_units: bool = False) -> pd.DataFrame:
        """Posterior median of the yearly-averaged GIRFs (Figure 12 of the paper)."""
        if self.hist_dates is None:
            raise ValueError("dates required")
        years = np.asarray(self.hist_dates.year)
        sel = np.ones(len(years), dtype=bool) if mask is None else mask
        rows = []
        for yr in np.unique(years[sel]):
            m = sel & (years == yr)
            med = np.median(self.average(m, original_units), axis=0)    # (H+1, M)
            for j, nm in enumerate(self.var_names):
                for k in range(self.horizon + 1):
                    rows.append([int(yr), nm, k, med[k, j]])
        return pd.DataFrame(rows, columns=["year", "variable", "horizon", "median"])

    def peak_table(self, mask: Optional[np.ndarray] = None, original_units: bool = False) -> pd.DataFrame:
        """Peak response, horizon of the peak and cumulative response (posterior median)."""
        avg = self.average(mask, original_units)
        med = np.median(avg, axis=0)
        lo, hi = np.quantile(avg, [0.16, 0.84], axis=0)
        rows = []
        for j, nm in enumerate(self.var_names):
            kpk = int(np.argmax(np.abs(med[:, j])))
            rows.append([nm, med[0, j], med[kpk, j], kpk, lo[kpk, j], hi[kpk, j], float(np.sum(med[:, j]))])
        return pd.DataFrame(rows, columns=["variable", "impact", "peak", "peak_horizon", "peak_q16", "peak_q84", "cumulative"])

    def state_dependence(self, mask: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Cross-history dispersion of the (posterior-median) responses at each horizon."""
        med = np.median(self.irf if mask is None else self.irf[:, mask], axis=0)   # (n_hist, H+1, M)
        rows = []
        for j, nm in enumerate(self.var_names):
            for k in range(self.horizon + 1):
                v = med[:, k, j]
                rows.append([nm, k, v.mean(), v.std(), v.min(), v.max()])
        return pd.DataFrame(rows, columns=["variable", "horizon", "mean", "sd_across_histories", "min", "max"])


def _girf_one_draw(results, d, j_s, hist_idx, n_sim, horizon, size, sign, shock_scale, sample_m, seed):
    """GIRF array (n_hist, horizon+1, M) for one posterior draw."""
    mdl = results.model
    M = mdl.M
    nh = hist_idx.size
    rng = np.random.default_rng(seed)
    ctx = DrawContext(results, d, sample_m=sample_m)
    B = nh * n_sim
    rep = np.repeat(np.arange(nh), n_sim)
    out = np.empty((nh, horizon + 1, M))
    sb = ctx.states(hist_idx)[rep]                     # baseline states (B, Mp)
    if shock_scale == "unit":
        vs = np.full(B, float(size) * sign)
    elif shock_scale == "sd":
        vs = float(size) * sign * np.exp(0.5 * ctx.h[hist_idx[rep], j_s])
    else:
        raise ValueError("shock_scale must be 'unit' or 'sd'")
    delta0 = vs[:, None] * ctx.Qinv[:, j_s][None, :]
    ss = sb.copy()
    ss[:, :M] += delta0
    out[:, 0, :] = delta0.reshape(nh, n_sim, M).mean(axis=1)
    h_cur = ctx.h[hist_idx[rep], :]
    for k in range(1, horizon + 1):
        h_cur = ctx.next_logvol(h_cur, rng.standard_normal((B, M)))
        sig_new = np.exp(0.5 * h_cur)
        eps = sig_new * rng.standard_normal((B, M))
        z_m = rng.standard_normal((B, M)) if sample_m else None
        y_both = ctx.step(np.vstack([sb, ss]), np.vstack([sig_new, sig_new]), np.vstack([eps, eps]),
                          None if z_m is None else np.vstack([z_m, z_m]))
        yb, ys = y_both[:B], y_both[B:]
        out[:, k, :] = (ys - yb).reshape(nh, n_sim, M).mean(axis=1)
        sb = ctx.roll(sb, yb, M)
        ss = ctx.roll(ss, ys, M)
    return out


def girf(results, shock_var, size: float = 1.0, sign: int = 1, horizon: int = 16,
         histories: Optional[Sequence[int]] = None, n_sim: int = 5, n_draws: Optional[int] = 100,
         seed: Optional[int] = 0, shock_scale: str = "unit", sample_m: bool = False,
         verbose: bool = True, n_jobs: Optional[int] = None) -> GIRFResult:
    """
    Compute generalized impulse responses.

    Parameters
    ----------
    results : GPVARResults
    shock_var : str or int
        Variable whose structural shock is perturbed.
    size, sign : float, int
        Shock size in standard deviations (see ``shock_scale``) and sign.
    horizon : int
        Number of periods after impact.
    histories : sequence of int, optional
        Indices ``t`` (into the estimation sample) used as conditioning
        histories; default: all.
    n_sim : int
        Number of simulated future shock paths per history and draw (the
        expectation in the GIRF definition is approximated by their average).
    n_draws : int, optional
        Number of (evenly spaced) posterior draws to use.
    shock_scale : {"unit", "sd"}
        "unit": the structural shock equals ``size`` on the (standardised)
        scale of the data, so that the impact response of ``shock_var`` is
        exactly ``size`` (the normalisation of Figure 7).  "sd": the shock
        equals ``size`` times the time-``t`` posterior standard deviation
        ``exp(h_jt/2)`` of the structural innovation.
    sample_m : bool
        Draw the conditional mean from its GP predictive (adds the GP
        predictive variance) instead of using its predictive mean.
    n_jobs : int, optional
        Posterior draws are processed in parallel threads (NumPy releases the
        GIL in the kernel evaluations); default: number of CPU cores.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    mdl = results.model
    M, T = mdl.M, mdl.T
    if isinstance(shock_var, str):
        j_s = mdl.var_names.index(shock_var)
    else:
        j_s = int(shock_var)
        shock_var = mdl.var_names[j_s]
    hist_idx = np.arange(T) if histories is None else np.asarray(histories, dtype=int)
    draws = select_draws(results.nsave, n_draws)
    seeds = np.random.default_rng(seed).integers(0, 2 ** 31, size=draws.size)
    n_jobs = n_jobs or max(1, os.cpu_count() or 1)
    out = np.empty((draws.size, hist_idx.size, horizon + 1, M))
    t0 = time.time()
    done = [0]

    def work(i):
        out[i] = _girf_one_draw(results, draws[i], j_s, hist_idx, n_sim, horizon, size, sign, shock_scale,
                                sample_m, int(seeds[i]))
        done[0] += 1
        if verbose and (done[0] % max(1, draws.size // 10) == 0 or done[0] == draws.size):
            print(f"[GIRF {shock_var} size={size:+.1f}x{sign:+d}] {done[0]}/{draws.size} draws  ({time.time() - t0:.0f}s)",
                  flush=True)

    if n_jobs > 1 and draws.size > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(work, range(draws.size)))
    else:
        for i in range(draws.size):
            work(i)
    dates = None if mdl.dates is None else mdl.dates[hist_idx]
    return GIRFResult(irf=out, hist_idx=hist_idx, hist_dates=dates, var_names=mdl.var_names,
                      shock_var=shock_var, size=size, sign=sign, shock_scale=shock_scale, sd=mdl.sd_)


def girf_sign_asymmetry(results, shock_var, **kw) -> Dict[str, GIRFResult]:
    """Positive and negative one-standard-deviation shocks (Figure 9)."""
    return {"positive": girf(results, shock_var, sign=+1, **kw),
            "negative": girf(results, shock_var, sign=-1, **kw)}


def girf_size_asymmetry(results, shock_var, sizes=(1.0, 2.0), **kw) -> Dict[float, GIRFResult]:
    """Shocks of different sizes (Figure 10)."""
    return {s: girf(results, shock_var, size=s, **kw) for s in sizes}


def ordering_robustness(Y: pd.DataFrame, fixed_first: Sequence[str], shock_var: str, n_orderings: int = 5,
                        fit_kwargs: Optional[dict] = None, girf_kwargs: Optional[dict] = None,
                        model_kwargs: Optional[dict] = None, seed: int = 0) -> pd.DataFrame:
    """
    Re-estimate the model under randomly permuted orderings of the variables
    not in ``fixed_first`` and report the correlation of the average-GIRF
    posterior medians across orderings (Online Appendix C.2).
    """
    from .model import GPVAR
    rng = np.random.default_rng(seed)
    rest = [c for c in Y.columns if c not in fixed_first]
    fit_kwargs = fit_kwargs or {}
    girf_kwargs = girf_kwargs or {}
    model_kwargs = model_kwargs or {}
    meds = []
    for i in range(n_orderings):
        order = list(fixed_first) + list(rng.permutation(rest))
        res = GPVAR(Y[order], **model_kwargs).fit(verbose=False, **fit_kwargs)
        gr = girf(res, shock_var, verbose=False, **girf_kwargs)
        med = np.median(gr.average(), axis=0)          # (H+1, M) in the permuted order
        meds.append(pd.DataFrame(med, columns=order)[list(Y.columns)].to_numpy())
    rows = []
    for j, nm in enumerate(Y.columns):
        C = np.corrcoef(np.stack([m[:, j] for m in meds]))
        rows.append([nm, C[np.triu_indices(n_orderings, 1)].mean(), C[np.triu_indices(n_orderings, 1)].min()])
    return pd.DataFrame(rows, columns=["variable", "mean_corr", "min_corr"])
