"""
Benchmark model: Bayesian VAR with a Minnesota-type natural-conjugate
(normal-inverse-Wishart) prior and homoskedastic errors.

    y_t = c + sum_l A_l y_{t-l} + u_t,   u_t ~ N(0, Sigma)
    vec(B) | Sigma ~ N(vec(B0), Sigma (x) V0),   Sigma ~ IW(S0, nu0)

The posterior is available in closed form (Koop, 2003, Ch. 4; Karlsson,
2013), so draws are cheap.  The model serves as the linear benchmark for the
log predictive Bayes factors of Section 5.2.  (The paper's benchmark also has
stochastic volatility; this simpler variant is provided so that the whole
forecast-evaluation pipeline runs in pure Python without extra dependencies.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import invwishart

from .forecast import ForecastResult
from .kernels import lag_matrix
from .predict import select_draws


@dataclass
class BVARResults:
    model: "BVAR"
    B: np.ndarray        # (nsave, K, M)
    Sigma: np.ndarray    # (nsave, M, M)

    @property
    def nsave(self) -> int:
        return self.B.shape[0]

    def predict(self, horizon: int = 8, n_sim: int = 1, n_draws: Optional[int] = None,
                seed: Optional[int] = 0, **_) -> ForecastResult:
        mdl = self.model
        rng = np.random.default_rng(seed)
        draws = select_draws(self.nsave, n_draws)
        M, p = mdl.M, mdl.p
        paths = np.empty((draws.size * n_sim, horizon, M))
        Y = mdl.Y
        state0 = np.concatenate([Y[-1 - k] for k in range(p)])
        for di, d in enumerate(draws):
            B, S = self.B[d], self.Sigma[d]
            L = np.linalg.cholesky(S)
            for s in range(n_sim):
                st = state0.copy()
                for k in range(horizon):
                    x = np.concatenate([[1.0], st]) if mdl.intercept else st
                    y_new = x @ B + L @ rng.standard_normal(M)
                    paths[di * n_sim + s, k] = y_new
                    st = np.concatenate([y_new, st[:-M]]) if p > 1 else y_new
        raw = paths * mdl.sd_[None, None, :] + mdl.mean_[None, None, :]
        origin = None if mdl.dates is None else mdl.dates[-1]
        return ForecastResult(paths=paths, paths_raw=raw, var_names=mdl.var_names, origin_date=origin)

    def irf(self, shock_var, horizon: int = 16, size: float = 1.0, n_draws: Optional[int] = None,
            probs=(0.16, 0.5, 0.84), label: str = "BVAR") -> pd.DataFrame:
        """
        Recursive (Cholesky) impulse responses to a shock in ``shock_var`` whose
        impact effect on ``shock_var`` equals ``size`` (same normalisation as the
        GP-VAR GIRFs with ``shock_scale="unit"``).  Returns the long summary
        format used by :func:`gpvar.plots.plot_girf`.
        """
        mdl = self.model
        M, p = mdl.M, mdl.p
        j = mdl.var_names.index(shock_var) if isinstance(shock_var, str) else int(shock_var)
        draws = select_draws(self.nsave, n_draws)
        out = np.empty((draws.size, horizon + 1, M))
        off = 1 if mdl.intercept else 0
        for di, d in enumerate(draws):
            B, S = self.B[d], self.Sigma[d]
            A = [B[off + l * M: off + (l + 1) * M].T for l in range(p)]      # A_l (M x M)
            L = np.linalg.cholesky(S)
            imp = L[:, j] / L[j, j] * size
            resp = np.zeros((horizon + 1, M))
            resp[0] = imp
            for k in range(1, horizon + 1):
                acc = np.zeros(M)
                for l in range(1, min(k, p) + 1):
                    acc += A[l - 1] @ resp[k - l]
                resp[k] = acc
            out[di] = resp
        q = np.quantile(out, probs, axis=0)
        rows = []
        for v, nm in enumerate(mdl.var_names):
            for k in range(horizon + 1):
                rows.append([nm, k] + [q[i, k, v] for i in range(len(probs))])
        cols = ["variable", "horizon"] + [f"q{int(round(pp * 100)):02d}" for pp in probs]
        df = pd.DataFrame(rows, columns=cols)
        df.insert(0, "label", label)
        return df


class BVAR:
    """Minnesota natural-conjugate BVAR (homoskedastic)."""

    def __init__(self, Y, p: int = 5, lam1: float = 0.2, lam2: float = 0.5, lam3: float = 1.0,
                 lam_c: float = 100.0, prior_mean: str = "zero", intercept: bool = True,
                 standardize: bool = True, var_names: Optional[Sequence[str]] = None):
        if isinstance(Y, pd.DataFrame):
            var_names = list(Y.columns) if var_names is None else list(var_names)
            self.dates = Y.index[p:]
            Y = Y.to_numpy(dtype=float)
        else:
            self.dates = None
        Y = np.asarray(Y, dtype=float)
        self.Traw, self.M = Y.shape
        self.p, self.intercept = p, intercept
        self.var_names = var_names or [f"y{j + 1}" for j in range(self.M)]
        self.mean_ = Y.mean(0) if standardize else np.zeros(self.M)
        self.sd_ = Y.std(0, ddof=1) if standardize else np.ones(self.M)
        self.Y = (Y - self.mean_) / self.sd_
        self.y, X = lag_matrix(self.Y, p)
        self.X = np.hstack([np.ones((self.y.shape[0], 1)), X]) if intercept else X
        self.T, self.K = self.X.shape
        # AR(1) residual variances for the Minnesota scaling
        s2 = np.empty(self.M)
        for j in range(self.M):
            yy, xx = self.Y[1:, j], np.column_stack([np.ones(self.Traw - 1), self.Y[:-1, j]])
            b = np.linalg.lstsq(xx, yy, rcond=None)[0]
            s2[j] = np.var(yy - xx @ b, ddof=2)
        self.s2 = s2
        # prior
        B0 = np.zeros((self.K, self.M))
        v0 = np.empty(self.K)
        pos = 0
        if intercept:
            v0[0] = lam_c ** 2
            pos = 1
        for l in range(1, p + 1):
            for j in range(self.M):
                # same prior variance across equations (natural conjugate restriction):
                v0[pos] = (lam1 ** 2) / (l ** (2 * lam3)) / s2[j]
                if prior_mean == "rw" and l == 1:
                    B0[pos, j] = 1.0
                pos += 1
        self.B0, self.V0inv = B0, np.diag(1.0 / v0)
        self.S0 = np.diag(s2)
        self.nu0 = self.M + 2

    def fit(self, nsave: int = 1000, seed: Optional[int] = None, verbose: bool = False, **_) -> BVARResults:
        rng = np.random.default_rng(seed)
        X, y = self.X, self.y
        V1 = np.linalg.inv(self.V0inv + X.T @ X)
        B1 = V1 @ (self.V0inv @ self.B0 + X.T @ y)
        S1 = self.S0 + y.T @ y + self.B0.T @ self.V0inv @ self.B0 - B1.T @ np.linalg.inv(V1) @ B1
        S1 = (S1 + S1.T) / 2
        nu1 = self.nu0 + self.T
        L = np.linalg.cholesky(V1)
        Bs = np.empty((nsave, self.K, self.M)); Ss = np.empty((nsave, self.M, self.M))
        for i in range(nsave):
            S = invwishart.rvs(df=nu1, scale=S1, random_state=rng)
            S = np.atleast_2d(S)
            Cs = np.linalg.cholesky(S)
            Bs[i] = B1 + L @ rng.standard_normal((self.K, self.M)) @ Cs.T
            Ss[i] = S
        return BVARResults(model=self, B=Bs, Sigma=Ss)
