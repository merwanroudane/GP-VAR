"""
Synthetic data generating processes.

* :func:`simulate_paper_dgp` - the highly non-linear three-equation DGP of
  Section 4 of the paper (structural break in equation 2, sine/quadratic
  non-linearities in equation 3, t_3 shocks in equation 1, random-walk SV).
* :func:`simulate_replication_dgp` - the small tanh/sin VAR of the authors'
  replication archive (``simulate_data.R``).
Both return the realised latent components so that the fit can be compared
to the truth (Figure 3 / Table 1 of the paper).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimulatedData:
    Y: pd.DataFrame            # (T, M)
    F: np.ndarray              # (T, M) own-lag functions
    G: np.ndarray              # (T, M) other-lag functions
    m: np.ndarray              # (T, M) conditional means F + G
    Q: np.ndarray              # (M, M)
    omega: np.ndarray          # (T, M) structural error variances
    p: int
    extra: dict


def simulate_paper_dgp(T: int = 200, p: int = 5, seed: Optional[int] = 1, burn: int = 100,
                       break_point: Optional[int] = None) -> SimulatedData:
    """
    DGP of Section 4 (M = 3):

        F(x_t) = ( sum_k phi_11k y_{1,t-k} ;
                   sum_k phi_22k y_{2,t-k} I(t<=100) + phi_221 y_{2,t-1} I(t>100) ;
                   1/12 sin(pi/2 y_{3,t-1} y_{3,t-2}) + 1/3 (y_{3,t-3}-1)^2 + 1/12 y_{3,t-4} + 1/12 y_{3,t-5} )
        G(z_t) = ( 0 ;
                   0 x I(t<=100) + sum_k sum_{j in {1,3}} phi_2jk y_{j,t-k} I(t>100) ;
                   1/18 sin(pi/2 y_{1,t-1} y_{2,t-1}) + 2/9 (y_{1,t-2}-1)^2 + 1/18 y_{1,t-3} + 1/18 y_{2,t-5} )

    phi_111 = 0.8, phi_221 = 0.65, phi_ijk ~ N(0, (0.3/k)^2) for i != j, q_jk ~ N(0, 0.1^2),
    omega_jt = lambda_jt * exp(htil_jt), htil random walk with sd 0.01 and
    htil_0 = log(0.01); lambda_1t ~ IG(3/2, 3/2) (t_3 errors), lambda_2 = lambda_3 = 1.
    """
    rng = np.random.default_rng(seed)
    M = 3
    p = max(p, 5)
    bp = T // 2 if break_point is None else break_point
    phi_own = np.zeros((M, p))
    phi_own[0, 0] = 0.8
    phi_own[1, 0] = 0.65
    phi_cross = rng.normal(0.0, 1.0, size=(M, M, p)) * (0.3 / np.arange(1, p + 1))[None, None, :]
    for i in range(M):
        phi_cross[i, i, :] = 0.0
    Q = np.zeros((M, M))
    Q[np.tril_indices(M, -1)] = rng.normal(0.0, 0.1, size=M * (M - 1) // 2)
    Qinv = np.linalg.inv(np.eye(M) - Q)

    Tt = T + burn
    htil = np.empty((Tt, M))
    htil[0] = np.log(0.01)
    for t in range(1, Tt):
        htil[t] = htil[t - 1] + 0.01 * rng.standard_normal(M)
    lam = np.ones((Tt, M))
    lam[:, 0] = 1.0 / rng.gamma(1.5, 1.0 / 1.5, size=Tt)
    omega = lam * np.exp(htil)

    Y = np.zeros((Tt, M))
    F = np.zeros((Tt, M))
    G = np.zeros((Tt, M))
    Y[:p] = rng.normal(0, 0.1, size=(p, M))
    for t in range(p, Tt):
        tt = t - burn + 1            # calendar time index 1..T after burn-in
        after = tt > bp
        lags = Y[t - p:t][::-1]        # (p, M): lags[k-1] = y_{t-k}
        # equation 1: linear own lags
        F[t, 0] = np.sum(phi_own[0] * lags[:, 0])
        G[t, 0] = 0.0
        # equation 2: break in own lags, other lags active after the break
        if not after:
            F[t, 1] = np.sum(phi_own[1] * lags[:, 1])
            G[t, 1] = 0.0
        else:
            F[t, 1] = phi_own[1, 0] * lags[0, 1]
            G[t, 1] = np.sum(phi_cross[1, 0, :] * lags[:, 0]) + np.sum(phi_cross[1, 2, :] * lags[:, 2])
        # equation 3: non-linear
        F[t, 2] = (np.sin(np.pi / 2 * lags[0, 2] * lags[1, 2]) / 12.0 + (lags[2, 2] - 1.0) ** 2 / 3.0
                   + lags[3, 2] / 12.0 + lags[4, 2] / 12.0)
        G[t, 2] = (np.sin(np.pi / 2 * lags[0, 0] * lags[0, 1]) / 18.0 + 2.0 / 9.0 * (lags[1, 0] - 1.0) ** 2
                   + lags[2, 0] / 18.0 + lags[4, 1] / 18.0)
        eps = np.sqrt(omega[t]) * rng.standard_normal(M)
        Y[t] = Qinv @ (F[t] + G[t] + eps)
    sl = slice(burn, Tt)
    idx = pd.RangeIndex(1, T + 1, name="t")
    Ydf = pd.DataFrame(Y[sl], columns=["y1", "y2", "y3"], index=idx)
    return SimulatedData(Y=Ydf, F=F[sl], G=G[sl], m=F[sl] + G[sl], Q=Q, omega=omega[sl], p=p,
                         extra=dict(phi_own=phi_own, phi_cross=phi_cross, break_point=bp, lam=lam[sl]))


def simulate_replication_dgp(T: int = 120, M: int = 3, p: int = 2, burn: int = 200,
                             seed: Optional[int] = 1) -> SimulatedData:
    """
    Small VAR with tanh/sin non-linearities and mild SV (replication archive):

        y_t = sum_l A_l y_{t-l} + 0.2 tanh(2 y_{t-1}) + 0.1 sin(y_{t-1, shifted}) + L u_t,
        u_jt ~ N(0, exp(h_jt)),  h_jt = 0.95 h_{j,t-1} + 0.15 e_t.
    """
    rng = np.random.default_rng(seed)
    Tt = T + burn
    A = []
    for l in range(1, p + 1):
        a = rng.normal(0, 0.05, size=(M, M))
        np.fill_diagonal(a, 0.5 / l)
        A.append(a)
    h = np.zeros((Tt, M))
    for t in range(1, Tt):
        h[t] = 0.95 * h[t - 1] + rng.normal(0, 0.15, size=M)
    sig = np.exp(h / 2)
    L = np.eye(M)
    L[np.tril_indices(M, -1)] = rng.normal(0, 0.3, size=M * (M - 1) // 2)
    Y = np.zeros((Tt, M)); F = np.zeros((Tt, M)); G = np.zeros((Tt, M))
    shift = np.r_[1:M, 0]
    for t in range(p, Tt):
        lin = sum(A[l] @ Y[t - l - 1] for l in range(p))
        own = np.array([A[l][j, j] * Y[t - l - 1, j] for l in range(p) for j in range(M)]).reshape(p, M).sum(0)
        nl_own = 0.2 * np.tanh(2 * Y[t - 1])
        nl_other = 0.1 * np.sin(Y[t - 1, shift])
        F[t] = own + nl_own
        G[t] = (lin - own) + nl_other
        Y[t] = F[t] + G[t] + L @ (sig[t] * rng.standard_normal(M))
    sl = slice(burn, Tt)
    Ydf = pd.DataFrame(Y[sl], columns=[f"y{j + 1}" for j in range(M)], index=pd.RangeIndex(1, T + 1, name="t"))
    Q = np.eye(M) - np.linalg.inv(L)      # structural form y = Q y + ... + eps with eps = diag part
    return SimulatedData(Y=Ydf, F=F[sl], G=G[sl], m=F[sl] + G[sl], Q=Q, omega=sig[sl] ** 2, p=p,
                         extra=dict(A=A, L=L))
