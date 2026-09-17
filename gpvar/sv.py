"""
Stochastic volatility for the GP-VAR (Section 3.4 and Online Appendix A.3).

The log-volatilities ``h_j = (h_{j1}, ..., h_{jT})'`` of equation ``j`` are
sampled *marginally of the latent GP functions* ``f_j`` and ``g_j``.  After
integrating them out the equation reads

    Ytil_j = sqrt(Om_j) W_j eps,     W_j W_j' = xi1 K1 + xi2 K2 + I_T  =: S_j,

so that  Ytil_j | h_j ~ N(0, sqrt(Om_j) S_j sqrt(Om_j)).  The full conditional
of ``h_j`` is not of a known form; following Chan (2017) we use an independence
Metropolis-Hastings step whose Gaussian proposal is centred on the mode of the
log conditional posterior (found by Newton-Raphson) with covariance equal to the
inverse of the negative Hessian.  As in the paper, the Hessian of the
log-likelihood is band-approximated (tridiagonal) so that all operations are
O(T) once ``S_j^{-1}`` is available.

Log-likelihood (A.6), with  Yhat = Ytil * exp(-h/2):

    log p(Ytil | h) = -T/2 log 2pi - 1/2 sum_t h_t - 1/2 log|S| - 1/2 Yhat' S^{-1} Yhat
    gradient       = -1/2 * 1 + 1/2 (S^{-1} Yhat) o Yhat
    neg. Hessian   =  1/4 diag(Yhat) S^{-1} diag(Yhat) + 1/4 diag( (S^{-1} Yhat) o Yhat )

Prior (A.4)-(A.5) with ``e_t = h_t - mu - rho (h_{t-1} - mu)``, ``h_0`` given:

    log p(h | theta) = const - 1/(2 sig2) e'e,
    gradient         = -(1/sig2) D' e,
    neg. Hessian     =  (1/sig2) D'D   (tridiagonal).

The state-equation parameters (rho, sig2, mu, h_0) are drawn from standard
conditionals (Step 7 of Appendix A.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky_banded, solve_banded, solveh_banded

from .priors import SVPriors, rinvgamma


# --------------------------------------------------------------------------- #
# Banded helpers (upper storage for scipy: ab[0, 1:] = super-diagonal, ab[1] = diag)
# --------------------------------------------------------------------------- #
def _band_matvec(diag: np.ndarray, off: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(tridiagonal symmetric matrix) @ x, with ``off`` the super-diagonal (T-1)."""
    y = diag * x
    y[:-1] += off * x[1:]
    y[1:] += off * x[:-1]
    return y


def _band_quad(diag: np.ndarray, off: np.ndarray, x: np.ndarray) -> float:
    return float(x @ _band_matvec(diag, off, x))


def _band_chol(diag: np.ndarray, off: np.ndarray):
    """Upper banded Cholesky factor of a symmetric tridiagonal PD matrix."""
    T = diag.size
    ab = np.zeros((2, T))
    ab[0, 1:] = off
    ab[1, :] = diag
    return cholesky_banded(ab, lower=False)      # (2, T) upper form


def _band_solve(diag: np.ndarray, off: np.ndarray, b: np.ndarray) -> np.ndarray:
    T = diag.size
    ab = np.zeros((2, T))
    ab[0, 1:] = off
    ab[1, :] = diag
    return solveh_banded(ab, b, lower=False)


def _band_logdet_from_chol(c: np.ndarray) -> float:
    return 2.0 * float(np.sum(np.log(c[1, :])))


def _draw_from_precision(c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw N(0, Lambda^{-1}) with Lambda = U'U, ``c`` the upper banded factor U."""
    T = c.shape[1]
    z = rng.standard_normal(T)
    # solve U x = z  (upper bidiagonal)
    return solve_banded((0, 1), c, z)


# --------------------------------------------------------------------------- #
# The independence MH step for one equation
# --------------------------------------------------------------------------- #
@dataclass
class SVParams:
    mu: float
    rho: float
    sig2: float
    h0: float


def _prior_parts(h: np.ndarray, prm: SVParams):
    """Residuals e = D h - mtil, gradient and tridiagonal negative Hessian of the log prior."""
    T = h.size
    hm = h - prm.mu
    prev = np.empty(T)
    prev[0] = prm.h0 - prm.mu
    prev[1:] = hm[:-1]
    e = hm - prm.rho * prev
    # D'e
    De = e.copy()
    De[:-1] -= prm.rho * e[1:]
    grad = -De / prm.sig2
    diag = np.full(T, (1.0 + prm.rho ** 2) / prm.sig2)
    diag[-1] = 1.0 / prm.sig2
    off = np.full(T - 1, -prm.rho / prm.sig2)
    logp = -0.5 * float(e @ e) / prm.sig2
    return logp, grad, diag, off


def _lik_parts(h: np.ndarray, Ytil: np.ndarray, Sinv: np.ndarray, Sinv_diag: np.ndarray,
               Sinv_off: np.ndarray, logdetS: float):
    Yhat = Ytil * np.exp(-0.5 * h)
    v = Sinv @ Yhat
    loglik = -0.5 * float(np.sum(h)) - 0.5 * logdetS - 0.5 * float(Yhat @ v)
    grad = -0.5 + 0.5 * v * Yhat
    diag = 0.25 * Yhat * Yhat * Sinv_diag + 0.25 * v * Yhat
    off = 0.25 * Yhat[:-1] * Sinv_off * Yhat[1:]
    return loglik, grad, diag, off


def _log_post(h, Ytil, Sinv, logdetS, prm):
    Yhat = Ytil * np.exp(-0.5 * h)
    v = Sinv @ Yhat
    ll = -0.5 * float(np.sum(h)) - 0.5 * logdetS - 0.5 * float(Yhat @ v)
    hm = h - prm.mu
    prev = np.concatenate(([prm.h0 - prm.mu], hm[:-1]))
    e = hm - prm.rho * prev
    return ll - 0.5 * float(e @ e) / prm.sig2


def sample_logvol_imh(h: np.ndarray, Ytil: np.ndarray, Sinv: np.ndarray, logdetS: float,
                      prm: SVParams, rng: np.random.Generator,
                      max_iter: int = 50, tol: float = 1e-3, h_floor: float = -20.0,
                      force_accept: bool = False, proposal_df: float = None, n_tries: int = 1):
    """
    One independence-MH update of the whole log-volatility path of one
    equation (Online Appendix A.3).  Returns ``(h_new, accepted)``.

    ``force_accept=True`` returns the proposal unconditionally; this is used
    only during the first burn-in sweeps to initialise the chain at (a draw
    around) the conditional mode - an independence sampler started far from
    the mode would otherwise reject for a very long time.

    ``proposal_df`` (optional) replaces the Gaussian proposal by a multivariate
    Student-t with that many degrees of freedom (same location/scale); heavier
    tails make the independence sampler more robust when the Gaussian
    approximation is too tight (e.g. interest rates at the zero lower bound).
    """
    T = h.size
    Sinv_diag = np.diag(Sinv).copy()
    Sinv_off = np.diag(Sinv, 1).copy()

    # ---- Newton-Raphson for the mode of the log conditional posterior ------
    hh = h.copy()
    lp_old = _log_post(hh, Ytil, Sinv, logdetS, prm)
    for _ in range(max_iter):
        ll, gl, dl, ol = _lik_parts(hh, Ytil, Sinv, Sinv_diag, Sinv_off, logdetS)
        lpp, gp_, dp, op = _prior_parts(hh, prm)
        grad = gl + gp_
        diag = dl + dp
        off = ol + op
        try:
            step = _band_solve(diag, off, grad)
        except np.linalg.LinAlgError:
            step = grad / np.maximum(diag, 1e-8)
        # damped Newton: shrink the step if the objective does not improve
        scale = 1.0
        for _try in range(8):
            cand = hh + scale * step
            lp_new = _log_post(cand, Ytil, Sinv, logdetS, prm)
            if np.isfinite(lp_new) and lp_new >= lp_old - 1e-10:
                break
            scale *= 0.5
        hh = cand
        converged = np.max(np.abs(scale * step)) < tol
        lp_old = lp_new
        if converged:
            break

    # ---- Gaussian proposal N(h_hat, R_hat^{-1}) ---------------------------
    _, _, dl, ol = _lik_parts(hh, Ytil, Sinv, Sinv_diag, Sinv_off, logdetS)
    _, _, dp, op = _prior_parts(hh, prm)
    diag = dl + dp
    off = ol + op
    try:
        c = _band_chol(diag, off)
    except np.linalg.LinAlgError:
        diag = np.maximum(dl, 0.0) + dp
        off = op
        c = _band_chol(diag, off)

    def _lq(x):
        q = _band_quad(diag, off, x - hh)
        return -0.5 * q if proposal_df is None else -0.5 * (proposal_df + T) * np.log1p(q / proposal_df)

    def _draw():
        z = _draw_from_precision(c, rng)
        if proposal_df is not None:
            z = z / np.sqrt(rng.chisquare(proposal_df) / proposal_df)
        return np.maximum(hh + z, h_floor)

    if force_accept:
        return _draw(), True

    # ---- MH acceptance; the proposal does not depend on the current state, so
    # several accept/reject steps can share the (expensive) mode computation ----
    cur = h.copy()
    lp_curr = _log_post(cur, Ytil, Sinv, logdetS, prm)
    lq_curr = _lq(cur)
    accepted = False
    for _ in range(max(1, n_tries)):
        prop = _draw()
        lp_prop = _log_post(prop, Ytil, Sinv, logdetS, prm)
        lq_prop = _lq(prop)
        if np.log(rng.uniform()) < (lp_prop - lp_curr) - (lq_prop - lq_curr):
            cur, lp_curr, lq_curr, accepted = prop, lp_prop, lq_prop, True
    return cur, accepted


def sample_logvol_imh_dense(h: np.ndarray, Ytil: np.ndarray, Sinv: np.ndarray, logdetS: float,
                            prm: SVParams, rng: np.random.Generator, max_iter: int = 50, tol: float = 1e-3,
                            h_floor: float = -20.0, force_accept: bool = False, n_tries: int = 1):
    """
    Same independence-MH step as :func:`sample_logvol_imh` but with the *exact*
    (dense) negative Hessian of the log-likelihood,

        R_L = 1/4 diag(Yhat) S^{-1} diag(Yhat) + 1/4 diag((S^{-1} Yhat) o Yhat),

    instead of its tridiagonal band.  Costs O(T^3) per Newton iteration (a few
    milliseconds for T ~ 200) and yields a proposal that is closer to the
    target when S^{-1} is far from banded (e.g. very smooth kernels).
    """
    from scipy.linalg import cho_factor, cho_solve, solve_triangular
    T = h.size

    def parts(hv):
        Yhat = Ytil * np.exp(-0.5 * hv)
        v = Sinv @ Yhat
        lpp, gp_, dp, op = _prior_parts(hv, prm)
        grad = (-0.5 + 0.5 * v * Yhat) + gp_
        R = 0.25 * (Yhat[:, None] * Sinv * Yhat[None, :])
        R[np.diag_indices(T)] += 0.25 * v * Yhat + dp
        R[np.arange(T - 1), np.arange(1, T)] += op
        R[np.arange(1, T), np.arange(T - 1)] += op
        return grad, R

    hh = h.copy()
    lp_old = _log_post(hh, Ytil, Sinv, logdetS, prm)
    for _ in range(max_iter):
        grad, R = parts(hh)
        try:
            step = cho_solve(cho_factor(R), grad)
        except np.linalg.LinAlgError:
            step = grad / np.maximum(np.diag(R), 1e-8)
        scale = 1.0
        for _try in range(8):
            cand = hh + scale * step
            lp_new = _log_post(cand, Ytil, Sinv, logdetS, prm)
            if np.isfinite(lp_new) and lp_new >= lp_old - 1e-10:
                break
            scale *= 0.5
        hh = cand
        lp_old = lp_new
        if np.max(np.abs(scale * step)) < tol:
            break
    grad, R = parts(hh)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        R[np.diag_indices(T)] += 1e-6
        L = np.linalg.cholesky(R)
    def _draw():
        return np.maximum(hh + solve_triangular(L.T, rng.standard_normal(T), lower=False), h_floor)

    def _lq(x):
        d = x - hh
        return -0.5 * float(d @ R @ d)

    if force_accept:
        return _draw(), True
    cur = h.copy()
    lp_curr = _log_post(cur, Ytil, Sinv, logdetS, prm)
    lq_curr = _lq(cur)
    accepted = False
    for _ in range(max(1, n_tries)):
        prop = _draw()
        lp_prop = _log_post(prop, Ytil, Sinv, logdetS, prm)
        lq_prop = _lq(prop)
        if np.log(rng.uniform()) < (lp_prop - lp_curr) - (lq_prop - lq_curr):
            cur, lp_curr, lq_curr, accepted = prop, lp_prop, lq_prop, True
    return cur, accepted


# --------------------------------------------------------------------------- #
# State-equation parameters (Step 7)
# --------------------------------------------------------------------------- #
def sample_sv_params(h: np.ndarray, prm: SVParams, priors: SVPriors, rng: np.random.Generator,
                     rho_prop_scale: float = 1.0) -> SVParams:
    """
    Draw (sig2, rho, mu, h0) given the log-volatility path.

    * sig2 | h, rho, mu, h0 ~ IG(a + (T+1)/2, b + [e'e + (1-rho^2)(h0-mu)^2]/2)
    * rho: MH with the OLS-type Gaussian conditional as proposal and the Beta
      prior x stationary initial-condition term in the acceptance ratio.
    * mu | .  ~ N  (conjugate, uses h_1..h_T conditional on h_0)
    * h0 | .  ~ N( mu + rho (h_1 - mu), sig2 )   (see derivation in docs)
    """
    T = h.size
    mu, rho, sig2, h0 = prm.mu, prm.rho, prm.sig2, prm.h0
    hm = h - mu
    prev = np.concatenate(([h0 - mu], hm[:-1]))

    # --- sig2 ----------------------------------------------------------------
    e = hm - rho * prev
    ss = float(e @ e) + (1.0 - rho ** 2) * (h0 - mu) ** 2
    sig2 = float(rinvgamma(rng, priors.sig2_a + (T + 1) / 2.0, priors.sig2_b + ss / 2.0))

    # --- rho (MH, proposal from the conditional Gaussian of the AR regression)
    sxx = float(prev @ prev)
    if sxx > 0:
        rho_hat = float(prev @ hm) / sxx
        rho_sd = np.sqrt(sig2 / sxx) * rho_prop_scale
        rho_prop = rho_hat + rho_sd * rng.standard_normal()
        if -1.0 < rho_prop < 1.0:
            def _extra(r):
                # prior + stationary initial condition term (not in the Gaussian proposal)
                return priors.log_prior_rho(r) + 0.5 * np.log1p(-r ** 2) - (1.0 - r ** 2) * (h0 - mu) ** 2 / (2.0 * sig2)
            log_alpha = _extra(rho_prop) - _extra(rho)
            if np.log(rng.uniform()) < log_alpha:
                rho = float(rho_prop)

    # --- mu ------------------------------------------------------------------
    if priors.estimate_mu:
        # h_t - rho h_{t-1} = (1-rho) mu + nu_t
        prev_h = np.concatenate(([h0], h[:-1]))
        z = h - rho * prev_h
        x = (1.0 - rho)
        prec = T * x * x / sig2 + 1.0 / priors.mu_var
        mean = (x * float(np.sum(z)) / sig2 + priors.mu_mean / priors.mu_var) / prec
        mu = float(mean + rng.standard_normal() / np.sqrt(prec))
    else:
        mu = 0.0

    # --- h0 ------------------------------------------------------------------
    h0 = float(mu + rho * (h[0] - mu) + np.sqrt(sig2) * rng.standard_normal())

    return SVParams(mu=mu, rho=rho, sig2=sig2, h0=h0)


def simulate_logvol_forward(h_last: np.ndarray, prm_mu: np.ndarray, prm_rho: np.ndarray,
                            prm_sig2: np.ndarray, horizon: int, rng: np.random.Generator,
                            shocks: np.ndarray = None) -> np.ndarray:
    """
    Simulate the AR(1) log-volatility ``horizon`` steps ahead.

    ``h_last`` (B, M); parameters (M,); returns (horizon, B, M).  ``shocks``
    (horizon, B, M) can be supplied to use common random numbers.
    """
    B, M = h_last.shape
    if shocks is None:
        shocks = rng.standard_normal((horizon, B, M))
    out = np.empty((horizon, B, M))
    h = h_last.copy()
    sd = np.sqrt(prm_sig2)
    for k in range(horizon):
        h = prm_mu + prm_rho * (h - prm_mu) + sd * shocks[k]
        out[k] = h
    return out
