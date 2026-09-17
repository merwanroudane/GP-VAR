"""
Predictive engine shared by forecasting and generalized impulse responses
(Section 3.5 and Online Appendix A.5).

For a posterior draw the one-step-ahead predictive distribution of the
conditional mean of equation ``j`` at a new input ``w = (x_new, z_new)`` is

    m_j,new ~ N( sig_new * k*_j(w, W_j) alpha_j ,  sig_new^2 [ xi1 + xi2 - k*_j S_j^{-1} k*_j' ] ),

    alpha_j = S_j^{-1} Om_j^{-1/2} ( Y_j - sum_{k<j} q_jk Y_k ),   S_j = xi1 K1 + xi2 K2 + I_T,
    k*_j    = xi1 k1(x_new, X_j) + xi2 k2(z_new, Z_j),

where ``sig_new = exp(h_new / 2)`` is the (simulated) volatility at the new
period.  The vector of endogenous variables then follows from the structural
form:  y_new = (I - Q)^{-1} ( m_new + eps_new ),  eps_new ~ N(0, diag(sig_new^2)).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .kernels import cross_sq_dists, gaussian_kernel


class DrawContext:
    """All draw-specific quantities needed to simulate the GP-VAR forward."""

    def __init__(self, results, d: int, sample_m: bool = False, dtype=np.float32):
        mdl = results.model
        self.mdl = mdl
        self.M, self.p, self.T = mdl.M, mdl.p, mdl.T
        self.sample_m = sample_m
        # kernel evaluations for new inputs are done in single precision by
        # default (about twice as fast, error ~1e-7 on kernel values)
        self.dtype = dtype
        self.Xo = [mdl.blocks_own[j].X.astype(dtype) for j in range(self.M)]
        self.Xt = [mdl.blocks_other[j].X.astype(dtype) for j in range(self.M)]
        self.Do = [mdl.blocks_own[j].D.astype(dtype) for j in range(self.M)]
        self.Dt = [mdl.blocks_other[j].D.astype(dtype) for j in range(self.M)]
        self.Q = results.Q[d]
        self.Qinv = np.linalg.inv(np.eye(self.M) - self.Q)      # lower triangular, unit diagonal
        self.hyper = results.hyper[d]                            # (M, 4)
        self.h = results.h[d]                                    # (T, M)
        self.sv = results.sv_params[d]                           # (M, 4): mu, rho, sig2, h0
        self.alpha = []
        self.chol = []
        y = mdl.y
        for j in range(self.M):
            xi1, k1, xi2, k2 = self.hyper[j]
            ik1 = results.hyper_idx[d, j, 0]
            ik2 = results.hyper_idx[d, j, 2]
            S = mdl.blocks_own[j].kernel(ik1, xi1) + mdl.blocks_other[j].kernel(ik2, xi2)
            S[np.diag_indices_from(S)] += 1.0
            cf = cho_factor(S, lower=True)
            ytil = y[:, j] - (y[:, :j] @ self.Q[j, :j] if j > 0 else 0.0)
            self.alpha.append(cho_solve(cf, ytil * np.exp(-0.5 * self.h[:, j])))
            self.chol.append(cf[0] if sample_m else None)

    # ------------------------------------------------------------------ state
    def state_at(self, t: int) -> np.ndarray:
        """
        Regressor vector available after observing ``y_t`` (index into the
        estimation sample): ``[y_t', y_{t-1}', ..., y_{t-p+1}']``.
        """
        Y = self.mdl.Y
        rows = [Y[self.p + t - k] for k in range(self.p)]
        return np.concatenate(rows)

    def states(self, t_idx) -> np.ndarray:
        return np.vstack([self.state_at(t) for t in t_idx])

    # ------------------------------------------------------------ one step
    def cond_mean(self, states: np.ndarray, sig_new: np.ndarray, z_m: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Conditional means ``m_new`` (B, M) for a batch of states (B, Mp) and
        new volatilities ``sig_new`` (B, M).  If ``sample_m`` a draw from the
        GP predictive is returned (using ``z_m`` (B, M) standard normals).
        """
        B = states.shape[0]
        m = np.empty((B, self.M))
        st = states.astype(self.dtype, copy=False)
        for j in range(self.M):
            xi1, k1, xi2, k2 = self.hyper[j]
            xo = st[:, self.mdl.own_idx[j]]
            xt = st[:, self.mdl.other_idx[j]]
            Kstar = gaussian_kernel(cross_sq_dists(xo, self.Xo[j], self.Do[j]), self.dtype(k1), self.dtype(xi1))
            Kstar += gaussian_kernel(cross_sq_dists(xt, self.Xt[j], self.Dt[j]), self.dtype(k2), self.dtype(xi2))
            mean = (Kstar @ self.alpha[j].astype(self.dtype)).astype(np.float64)
            if self.sample_m:
                w = solve_triangular(self.chol[j], Kstar.T, lower=True)
                var = np.maximum(xi1 + xi2 - np.sum(w * w, axis=0), 0.0)
                z = z_m[:, j] if z_m is not None else np.random.standard_normal(B)
                mean = mean + np.sqrt(var) * z
            m[:, j] = sig_new[:, j] * mean
        return m

    def step(self, states: np.ndarray, sig_new: np.ndarray, eps: np.ndarray,
             z_m: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw ``y_new`` (B, M) given states, volatilities and structural shocks."""
        m = self.cond_mean(states, sig_new, z_m)
        return (m + eps) @ self.Qinv.T

    @staticmethod
    def roll(states: np.ndarray, y_new: np.ndarray, M: int) -> np.ndarray:
        """Shift the lag stack and insert ``y_new`` in front."""
        return np.hstack([y_new, states[:, :-M]]) if states.shape[1] > M else y_new.copy()

    def next_logvol(self, h_cur: np.ndarray, z: np.ndarray) -> np.ndarray:
        mu, rho, sig2 = self.sv[:, 0], self.sv[:, 1], self.sv[:, 2]
        return mu + rho * (h_cur - mu) + np.sqrt(sig2) * z


def select_draws(nsave: int, n_draws: Optional[int]) -> np.ndarray:
    if n_draws is None or n_draws >= nsave:
        return np.arange(nsave)
    return np.unique(np.linspace(0, nsave - 1, n_draws).round().astype(int))
