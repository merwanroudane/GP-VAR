"""
Forecasting and density-forecast evaluation (Section 3.5 and Section 5.2).

* :func:`predict` simulates from the multi-step predictive distribution
  p(y_{T+h} | I_T) draw by draw (conditional mean from the GP predictive,
  structural shocks from N(0, H_{T+h}) with the log-volatilities simulated
  forward).
* :func:`log_predictive_score` evaluates log predictive likelihoods (LPLs) of
  realised values under the simulated predictive distribution (Gaussian or
  kernel-density approximation).  Differences of LPLs between two models are
  log predictive Bayes factors (LPBFs) - the metric of Table 2 in the paper.
* :func:`recursive_forecast_evaluation` implements the recursive
  (expanding-window) design of Section 5.2.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, multivariate_normal, norm

from .predict import DrawContext, select_draws


@dataclass
class ForecastResult:
    paths: np.ndarray                 # (n_paths, horizon, M) standardised units
    paths_raw: np.ndarray             # (n_paths, horizon, M) original units
    var_names: Sequence[str]
    origin_date: Optional[pd.Timestamp]
    freq: Optional[str] = None

    @property
    def horizon(self) -> int:
        return self.paths.shape[1]

    def quantiles(self, probs=(0.05, 0.16, 0.5, 0.84, 0.95), original_units: bool = True) -> pd.DataFrame:
        arr = self.paths_raw if original_units else self.paths
        q = np.quantile(arr, probs, axis=0)   # (len(probs), H, M)
        rows = []
        for j, nm in enumerate(self.var_names):
            for k in range(self.horizon):
                rows.append([nm, k + 1] + [q[i, k, j] for i in range(len(probs))])
        cols = ["variable", "horizon"] + [f"q{int(round(p * 100)):02d}" for p in probs]
        return pd.DataFrame(rows, columns=cols)

    def mean(self, original_units: bool = True) -> pd.DataFrame:
        arr = self.paths_raw if original_units else self.paths
        return pd.DataFrame(arr.mean(axis=0), columns=self.var_names, index=np.arange(1, self.horizon + 1))


def predict(results, horizon: int = 8, n_sim: int = 1, n_draws: Optional[int] = None,
            seed: Optional[int] = 0, sample_m: bool = True, verbose: bool = False) -> ForecastResult:
    """Simulate ``horizon``-step-ahead predictive paths from the end of the sample."""
    mdl = results.model
    M, T = mdl.M, mdl.T
    draws = select_draws(results.nsave, n_draws)
    rng = np.random.default_rng(seed)
    paths = np.empty((draws.size * n_sim, horizon, M))
    t0 = time.time()
    for di, d in enumerate(draws):
        ctx = DrawContext(results, d, sample_m=sample_m)
        st = np.repeat(ctx.state_at(T - 1)[None, :], n_sim, axis=0)
        h_cur = np.repeat(ctx.h[T - 1][None, :], n_sim, axis=0)
        for k in range(horizon):
            h_cur = ctx.next_logvol(h_cur, rng.standard_normal((n_sim, M)))
            sig = np.exp(0.5 * h_cur)
            eps = sig * rng.standard_normal((n_sim, M))
            z_m = rng.standard_normal((n_sim, M)) if sample_m else None
            y_new = ctx.step(st, sig, eps, z_m)
            paths[di * n_sim:(di + 1) * n_sim, k, :] = y_new
            st = ctx.roll(st, y_new, M)
        if verbose and (di + 1) % max(1, draws.size // 5) == 0:
            print(f"[predict] draw {di + 1}/{draws.size} ({time.time() - t0:.0f}s)")
    raw = paths * mdl.sd_[None, None, :] + mdl.mean_[None, None, :]
    origin = None if mdl.dates is None else mdl.dates[-1]
    return ForecastResult(paths=paths, paths_raw=raw, var_names=mdl.var_names, origin_date=origin)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def log_predictive_score(draws: np.ndarray, y_obs: np.ndarray, method: str = "gaussian") -> float:
    """
    Log predictive likelihood of ``y_obs`` (k,) under the simulated predictive
    draws (N, k).  ``method="gaussian"`` fits a (multivariate) normal to the
    draws; ``method="kde"`` uses a Gaussian kernel density (univariate only).
    """
    draws = np.atleast_2d(draws)
    y_obs = np.atleast_1d(y_obs)
    if draws.shape[1] != y_obs.size:
        draws = draws.T
    k = y_obs.size
    if method == "gaussian":
        mu = draws.mean(axis=0)
        if k == 1:
            return float(norm.logpdf(y_obs[0], mu[0], max(draws[:, 0].std(ddof=1), 1e-8)))
        cov = np.cov(draws, rowvar=False) + 1e-8 * np.eye(k)
        return float(multivariate_normal.logpdf(y_obs, mu, cov))
    if method == "kde":
        if k != 1:
            raise ValueError("kde scoring is univariate")
        return float(np.log(max(gaussian_kde(draws[:, 0])(y_obs[0])[0], 1e-300)))
    raise ValueError("method must be 'gaussian' or 'kde'")


def crps(draws: np.ndarray, y_obs: float) -> float:
    """Continuous ranked probability score from an ensemble (lower is better)."""
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    n = x.size
    term1 = np.mean(np.abs(x - y_obs))
    # E|X - X'| via the sorted-sample identity
    term2 = 2.0 * np.sum(x * (np.arange(1, n + 1) - 0.5 * (n + 1))) / (n * n)
    return float(term1 - 0.5 * term2)


# --------------------------------------------------------------------------- #
# Recursive evaluation
# --------------------------------------------------------------------------- #
def recursive_forecast_evaluation(Y: pd.DataFrame, first_origin, last_origin=None, step: int = 1,
                                  horizons: Sequence[int] = (1, 4), focus: Optional[Sequence[str]] = None,
                                  model_factory: Optional[Callable[[pd.DataFrame], object]] = None,
                                  fit_kwargs: Optional[dict] = None, predict_kwargs: Optional[dict] = None,
                                  score_method: str = "gaussian", label: str = "GP-VAR",
                                  verbose: bool = True) -> pd.DataFrame:
    """
    Expanding-window density-forecast evaluation.

    ``model_factory(Y_train)`` must return an object with ``fit(**fit_kwargs)``
    returning a results object accepted by :func:`predict`; for the GP-VAR use
    ``lambda Y: GPVAR(Y, p=5)``.  A benchmark model can be evaluated with the
    same function by supplying a different factory (see ``gpvar.benchmarks``).

    Returns a long DataFrame with one row per (origin, horizon, variable) plus
    the joint LPL over the focus variables (variable = "joint").
    """
    from .model import GPVAR
    fit_kwargs = fit_kwargs or {}
    predict_kwargs = predict_kwargs or {}
    model_factory = model_factory or (lambda Yt: GPVAR(Yt))
    focus = list(Y.columns) if focus is None else list(focus)
    idx = Y.index
    i0 = idx.get_loc(pd.Timestamp(first_origin)) if not isinstance(first_origin, int) else first_origin
    i1 = len(idx) - max(horizons) - 1 if last_origin is None else (
        idx.get_loc(pd.Timestamp(last_origin)) if not isinstance(last_origin, int) else last_origin)
    rows = []
    t0 = time.time()
    origins = list(range(i0, i1 + 1, step))
    for n, io in enumerate(origins):
        Ytr = Y.iloc[:io + 1]
        mdl = model_factory(Ytr)
        res = mdl.fit(verbose=False, **fit_kwargs)
        if hasattr(res, "predict"):
            fc = res.predict(horizon=max(horizons), **predict_kwargs)
        else:
            fc = predict(res, horizon=max(horizons), **predict_kwargs)
        for hzn in horizons:
            if io + hzn >= len(idx):
                continue
            y_act = Y.iloc[io + hzn]
            cols = [Y.columns.get_loc(c) for c in focus]
            dr = fc.paths_raw[:, hzn - 1, :]
            # joint
            lpl_joint = log_predictive_score(dr[:, cols], y_act.iloc[cols].to_numpy(), "gaussian")
            rows.append([label, idx[io], idx[io + hzn], hzn, "joint", lpl_joint, np.nan])
            for c in focus:
                jc = Y.columns.get_loc(c)
                lpl = log_predictive_score(dr[:, [jc]], np.array([y_act[c]]), score_method)
                rows.append([label, idx[io], idx[io + hzn], hzn, c, lpl, crps(dr[:, jc], y_act[c])])
        if verbose:
            print(f"[recursive {label}] origin {n + 1}/{len(origins)} ({idx[io].date()})  {time.time() - t0:.0f}s")
    return pd.DataFrame(rows, columns=["model", "origin", "target", "horizon", "variable", "lpl", "crps"])


def lpbf_table(evals: Dict[str, pd.DataFrame], benchmark: str) -> pd.DataFrame:
    """
    Sum LPLs over the evaluation sample and express them relative to a
    benchmark model (log predictive Bayes factors, as in Table 2).
    """
    parts = []
    for name, df in evals.items():
        s = df.groupby(["horizon", "variable"])["lpl"].sum().rename(name)
        parts.append(s)
    tab = pd.concat(parts, axis=1)
    bench = tab[benchmark]
    out = tab.sub(bench, axis=0)
    out[benchmark + " (LPL)"] = bench
    return out.reset_index()
