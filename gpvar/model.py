"""
The Gaussian process vector autoregression (GP-VAR) of

    Hauzenberger, N., Huber, F., Marcellino, M. and Petz, N. (2022),
    "Gaussian Process Vector Autoregressions and Macroeconomic Uncertainty",
    arXiv:2112.01995 (Journal of Business & Economic Statistics, forthcoming).

Structural form (eq. 2):

    y_t = F(x_t) + G(z_t) + Q y_t + eps_t,   eps_t ~ N(0, H_t),  H_t = diag(om_1t, ..., om_Mt)

with ``x_jt`` the own lags of variable ``j``, ``z_jt`` the lags of all other
variables, ``Q`` lower triangular with zero diagonal, and ``log om_jt`` following
the AR(1) stochastic-volatility process (3).  Equation ``j`` in full-data form:

    Y_j = f_j + g_j + sum_{k<j} q_jk Y_k + eps_j,   eps_j ~ N(0, Om_j),
    f_j ~ N(0, sqrt(Om_j) K_{th_j1}(X_j, X_j) sqrt(Om_j)),
    g_j ~ N(0, sqrt(Om_j) K_{th_j2}(Z_j, Z_j) sqrt(Om_j)).

Posterior simulation follows Appendix A.4 (steps 1-8) equation by equation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from .kernels import KernelBlock, lag_matrix, own_other_index
from .priors import HorseshoeState, SVPriors, rinvgamma
from .sv import SVParams, sample_logvol_imh, sample_logvol_imh_dense, sample_sv_params

HYPER_NAMES = ["xi_own", "kappa_own", "xi_other", "kappa_other"]
SV_NAMES = ["mu", "rho", "sig2", "h0"]


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #
@dataclass
class GPVARResults:
    """Posterior draws and metadata returned by :meth:`GPVAR.fit`."""
    model: "GPVAR"
    f: np.ndarray            # (nsave, T, M)   own-lag GP functional
    g: np.ndarray            # (nsave, T, M)   other-lag GP functional
    h: np.ndarray            # (nsave, T, M)   log-volatilities
    Q: np.ndarray            # (nsave, M, M)   contemporaneous matrix (lower, zero diag)
    hyper: np.ndarray        # (nsave, M, 4)   xi_own, kappa_own, xi_other, kappa_other
    hyper_idx: np.ndarray    # (nsave, M, 4)   grid indices (ik1, ix1, ik2, ix2)
    sv_params: np.ndarray    # (nsave, M, 4)   mu, rho, sig2, h0
    accept_rate: np.ndarray  # (M,)            IMH acceptance for the log-volatilities
    runtime: float
    config: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def nsave(self) -> int:
        return self.f.shape[0]

    @property
    def T(self) -> int:
        return self.f.shape[1]

    @property
    def M(self) -> int:
        return self.f.shape[2]

    @property
    def var_names(self) -> List[str]:
        return self.model.var_names

    @property
    def dates(self):
        return self.model.dates

    @property
    def y(self) -> np.ndarray:
        return self.model.y

    @property
    def m(self) -> np.ndarray:
        """Conditional mean draws ``m_j = f_j + g_j`` (nsave, T, M)."""
        return self.f + self.g

    @property
    def omega(self) -> np.ndarray:
        """Error variances ``om_jt = exp(h_jt)`` (nsave, T, M)."""
        return np.exp(self.h)

    # ---------------------------------------------------------------- summaries
    def quantiles(self, arr: np.ndarray, probs=(0.05, 0.16, 0.5, 0.84, 0.95)) -> np.ndarray:
        return np.quantile(arr, probs, axis=0)

    def fitted(self, probs=(0.05, 0.5, 0.95)) -> Dict[str, np.ndarray]:
        """Posterior quantiles of f, g, m (each (len(probs), T, M))."""
        return {"f": self.quantiles(self.f, probs), "g": self.quantiles(self.g, probs),
                "m": self.quantiles(self.m, probs)}

    def volatility(self, probs=(0.16, 0.5, 0.84)) -> np.ndarray:
        """Posterior quantiles of the standard deviations exp(h/2), (len(probs), T, M)."""
        return self.quantiles(np.exp(0.5 * self.h), probs)

    def shrinkage_paths(self) -> Dict[str, np.ndarray]:
        """
        Posterior means of ``om_jt * xi_j1`` (own) and ``om_jt * xi_j2`` (other):
        the diagonal elements of the re-scaled kernels (Figure 5 of the paper).
        """
        om = self.omega
        own = np.mean(om * self.hyper[:, None, :, 0], axis=0)
        other = np.mean(om * self.hyper[:, None, :, 2], axis=0)
        return {"own": own, "other": other}

    def hyper_summary(self, probs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        """Posterior quantiles of the kernel hyperparameters per equation."""
        rows = []
        for j, nm in enumerate(self.var_names):
            for k, hn in enumerate(HYPER_NAMES):
                q = np.quantile(self.hyper[:, j, k], probs)
                rows.append([nm, hn, float(np.mean(self.hyper[:, j, k]))] + list(q))
        cols = ["variable", "parameter", "mean"] + [f"q{int(p * 100):02d}" for p in probs]
        return pd.DataFrame(rows, columns=cols)

    def sv_summary(self, probs=(0.05, 0.5, 0.95)) -> pd.DataFrame:
        rows = []
        for j, nm in enumerate(self.var_names):
            for k, sn in enumerate(SV_NAMES):
                q = np.quantile(self.sv_params[:, j, k], probs)
                rows.append([nm, sn, float(np.mean(self.sv_params[:, j, k]))] + list(q))
        cols = ["variable", "parameter", "mean"] + [f"q{int(p * 100):02d}" for p in probs]
        return pd.DataFrame(rows, columns=cols)

    def Q_summary(self, probs=(0.16, 0.5, 0.84)) -> pd.DataFrame:
        rows = []
        for j in range(1, self.M):
            for k in range(j):
                d = self.Q[:, j, k]
                q = np.quantile(d, probs)
                rows.append([f"q[{self.var_names[j]},{self.var_names[k]}]", float(np.mean(d))] + list(q)
                            + [float(np.mean(d > 0))])
        cols = ["element", "mean"] + [f"q{int(p * 100):02d}" for p in probs] + ["P(>0)"]
        return pd.DataFrame(rows, columns=cols)

    def fit_correlations(self, truth_m: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Correlation between the posterior median of ``m_j`` and either the
        observed ``y_j`` or a supplied true mean function (Table 1 of the paper).
        """
        med = np.median(self.m, axis=0)
        target = self.y if truth_m is None else np.asarray(truth_m, dtype=float)
        if target.shape[0] == self.T + self.model.p:      # full-sample truth: drop the initial lags
            target = target[self.model.p:]
        c = [np.corrcoef(med[:, j], target[:, j])[0, 1] for j in range(self.M)]
        return pd.DataFrame({"variable": self.var_names, "corr": c})

    # ------------------------------------------------------------- predictive
    def predict(self, horizon: int = 8, n_sim: int = 1, n_draws: Optional[int] = None,
                seed: Optional[int] = 0, sample_m: bool = True, verbose: bool = False):
        """Simulate from the predictive distribution (see :func:`gpvar.forecast.predict`)."""
        from .forecast import predict
        return predict(self, horizon=horizon, n_sim=n_sim, n_draws=n_draws, seed=seed,
                       sample_m=sample_m, verbose=verbose)

    def girf(self, shock_var, **kwargs):
        """Generalized impulse responses (see :func:`gpvar.girf.girf`)."""
        from .girf import girf
        return girf(self, shock_var, **kwargs)

    # ------------------------------------------------------------------ persist
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "GPVARResults":
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GPVAR:
    """
    Gaussian process VAR with stochastic volatility.

    Parameters
    ----------
    Y : array (T_raw, M) or DataFrame
        Endogenous variables (rows = time).  A DataFrame supplies ``var_names``
        and ``dates`` automatically.
    p : int
        Number of lags (the paper uses 5 for quarterly US data).
    standardize : bool
        Demean and scale each column to unit variance before estimation
        (the paper works with demeaned/standardised data; footnote 5).
    sv : bool
        Stochastic volatility (True) or homoskedastic errors (False).
    hyper_scheme : {"semi-automatic", "semi-automatic-noscale", "naive"}
        Grid for (kappa, xi) - see Section 3.3 / 5.2.
    n_kappa, n_xi : int
        Grid sizes (the paper uses roughly 1000 combinations).
    c_kappa, c_xi : float
        Prior means of the Gamma(1/2, 1/(2c)) hyperpriors.
    hyper_sampling : {"marginal", "conditional"}
        "conditional" reproduces the paper's Step 5/6 literally (sample the
        hyperparameters from p(theta | f)).  "marginal" (default) draws them
        from p(theta | y, g, ...) with f integrated out and then redraws f -
        a blocked version of the same Gibbs sampler that mixes better and is
        numerically stable when K is near-singular.
    restrict_g_mean : bool
        Impose the grand-mean-zero identification restriction on g_j
        (Section 3.4, Appendix A.2).
    sv_priors : SVPriors
        Priors of the log-volatility state equation.
    sv_proposal_df : float, optional
        Degrees of freedom of a Student-t proposal for the log-volatility
        independence-MH step (None = Gaussian proposal as in the paper).
    sv_hessian : {"band", "dense"}
        Use the tridiagonal band approximation of the likelihood Hessian
        (paper, Appendix A.3; O(T) per Newton step) or the exact dense Hessian
        (O(T^3); default, it yields higher acceptance rates for T of a few hundred).
    sv_tries : int
        Number of accept/reject attempts per sweep from the same independence
        proposal (the proposal does not depend on the current state, so extra
        attempts are cheap and raise the effective acceptance rate).
    """

    def __init__(self, Y, p: int = 5, var_names: Optional[Sequence[str]] = None, dates=None,
                 standardize: bool = True, sv: bool = True,
                 hyper_scheme: str = "semi-automatic", n_kappa: int = 20, n_xi: int = 50,
                 c_kappa: float = 0.1, c_xi: float = 1.0,
                 kappa_range=(0.1, 2.0), xi_range=(0.04, 4.0),
                 hyper_sampling: str = "marginal", restrict_g_mean: bool = True,
                 sv_priors: Optional[SVPriors] = None, sigma2_prior=(0.01, 0.01),
                 sv_proposal_df: Optional[float] = None, sv_hessian: str = "dense", sv_tries: int = 3):
        if isinstance(Y, pd.DataFrame):
            var_names = list(Y.columns) if var_names is None else list(var_names)
            dates = Y.index if dates is None else dates
            Y = Y.to_numpy(dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("Y must be two-dimensional (T x M)")
        self.Traw, self.M = Y.shape
        self.p = int(p)
        self.var_names = list(var_names) if var_names is not None else [f"y{j + 1}" for j in range(self.M)]
        self.dates_raw = None if dates is None else pd.Index(dates)
        self.standardize = standardize
        self.mean_ = Y.mean(axis=0) if standardize else np.zeros(self.M)
        self.sd_ = Y.std(axis=0, ddof=1) if standardize else np.ones(self.M)
        self.Y_raw = Y
        self.Y = (Y - self.mean_) / self.sd_
        self.sv = sv
        self.hyper_sampling = hyper_sampling
        self.restrict_g_mean = restrict_g_mean
        self.sv_priors = sv_priors or SVPriors()
        self.sigma2_prior = sigma2_prior
        self.sv_proposal_df = sv_proposal_df
        self.sv_hessian = sv_hessian
        self.sv_tries = int(sv_tries)
        self.grid_kwargs = dict(scheme=hyper_scheme, n_kappa=n_kappa, n_xi=n_xi,
                                kappa_range=kappa_range, xi_range=xi_range,
                                c_kappa=c_kappa, c_xi=c_xi)

        # design matrices
        self.y, self.X = lag_matrix(self.Y, self.p)
        self.T = self.y.shape[0]
        self.dates = None if self.dates_raw is None else self.dates_raw[self.p:]

        # kernel blocks (own / other) per equation
        self.own_idx, self.other_idx = [], []
        self.blocks_own: List[KernelBlock] = []
        self.blocks_other: List[KernelBlock] = []
        t0 = time.time()
        for j in range(self.M):
            own, other = own_other_index(self.M, self.p, j)
            self.own_idx.append(own)
            self.other_idx.append(other)
            self.blocks_own.append(KernelBlock.build(self.X[:, own], self.grid_kwargs))
            self.blocks_other.append(KernelBlock.build(self.X[:, other], self.grid_kwargs))
        self.precompute_time = time.time() - t0

    # ---------------------------------------------------------------- helpers
    def _sample_grid(self, logpost: np.ndarray, rng: np.random.Generator):
        lp = logpost - logpost.max()
        pr = np.exp(lp)
        pr /= pr.sum()
        idx = rng.choice(pr.size, p=pr.ravel())
        return np.unravel_index(idx, logpost.shape)

    def _S_matrix(self, j: int, ik1: int, xi1: float, ik2: int, xi2: float) -> np.ndarray:
        S = self.blocks_own[j].kernel(ik1, xi1)
        S += self.blocks_other[j].kernel(ik2, xi2)
        S[np.diag_indices_from(S)] += 1.0
        return S

    # -------------------------------------------------------------------- fit
    def fit(self, nburn: int = 1000, nsave: int = 1000, thin: int = 1, seed: Optional[int] = None,
            verbose: bool = True, print_every: int = 100, sv_init_sweeps: int = 10) -> GPVARResults:
        """
        Run the MCMC sampler of Appendix A.4 and return :class:`GPVARResults`.

        During the first ``sv_init_sweeps`` burn-in sweeps the log-volatility
        proposal is accepted unconditionally (initialisation at the conditional
        mode); afterwards the exact independence-MH acceptance step is used.
        """
        rng = np.random.default_rng(seed)
        T, M, y = self.T, self.M, self.y
        ntot = nburn + nsave * thin
        conditional = self.hyper_sampling.lower() == "conditional"

        # ---- initial values ------------------------------------------------
        f = np.zeros((T, M))
        g = np.zeros((T, M))
        # OLS residual variance for the initial volatility level
        beta_ols = np.linalg.lstsq(self.X, y, rcond=None)[0]
        resid = y - self.X @ beta_ols
        s2 = np.maximum(np.var(resid, axis=0, ddof=1), 1e-4)
        h = np.tile(np.log(s2), (T, 1))
        Q = np.zeros((M, M))
        svp = [SVParams(mu=float(np.log(s2[j])), rho=0.95, sig2=0.05, h0=float(np.log(s2[j]))) for j in range(M)]
        n_free = M * (M - 1) // 2
        hs = HorseshoeState.init(max(n_free, 1))
        # grid indices: start at the prior median-heuristic-ish centre
        gi = np.zeros((M, 4), dtype=int)
        for j in range(M):
            gi[j, 0] = self.blocks_own[j].grid.n_kappa // 2
            gi[j, 1] = int(np.argmin(np.abs(self.blocks_own[j].grid.xi - 1.0)))
            gi[j, 2] = self.blocks_other[j].grid.n_kappa // 2
            gi[j, 3] = int(np.argmin(np.abs(self.blocks_other[j].grid.xi - 1.0)))

        # ---- storage -------------------------------------------------------
        f_st = np.empty((nsave, T, M)); g_st = np.empty((nsave, T, M)); h_st = np.empty((nsave, T, M))
        Q_st = np.empty((nsave, M, M)); hy_st = np.empty((nsave, M, 4)); hi_st = np.empty((nsave, M, 4), dtype=int)
        sv_st = np.empty((nsave, M, 4))
        accept = np.zeros(M); n_sv_steps = 0
        a_sig, b_sig = self.sigma2_prior

        # row slices of the stacked free elements of Q (row-major lower triangle)
        row_slices = []
        pos = 0
        for j in range(M):
            row_slices.append(slice(pos, pos + j))
            pos += j

        t0 = time.time()
        isave = 0
        for irep in range(ntot):
            for j in range(M):
                bo, bt = self.blocks_own[j], self.blocks_other[j]
                ik1, ix1, ik2, ix2 = gi[j]
                xi1, xi2 = bo.grid.xi[ix1], bt.grid.xi[ix2]
                sqrt_om = np.exp(0.5 * h[:, j])

                # ---- Step 1: contemporaneous coefficients q_j. | f, g, Om ----
                if j > 0:
                    W = y[:, :j]
                    r = y[:, j] - f[:, j] - g[:, j]
                    Wt = W / sqrt_om[:, None]
                    rt = r / sqrt_om
                    pv = hs.prior_var[row_slices[j]]
                    Vinv = Wt.T @ Wt + np.diag(1.0 / pv)
                    try:
                        cf = cho_factor(Vinv)
                        mean = cho_solve(cf, Wt.T @ rt)
                        L = np.linalg.cholesky(cho_solve(cf, np.eye(j)))
                    except np.linalg.LinAlgError:
                        V = np.linalg.pinv(Vinv)
                        mean = V @ (Wt.T @ rt)
                        L = np.linalg.cholesky(V + 1e-10 * np.eye(j))
                    Q[j, :j] = mean + L @ rng.standard_normal(j)
                ytil = y[:, j] - (y[:, :j] @ Q[j, :j] if j > 0 else 0.0)

                # ---- Step 2: volatilities marginally of f and g ---------------
                S = self._S_matrix(j, ik1, xi1, ik2, xi2)
                cf = cho_factor(S, lower=True)
                logdetS = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
                if self.sv:
                    Sinv = cho_solve(cf, np.eye(T))
                    # initialisation sweeps + stuck-guard (burn-in only): see sample_logvol_imh
                    force = irep < sv_init_sweeps or (irep < nburn and accept[j] == 0 and irep % 25 == 24)
                    if self.sv_hessian == "dense":
                        h[:, j], acc = sample_logvol_imh_dense(h[:, j], ytil, Sinv, logdetS, svp[j], rng,
                                                               force_accept=force, n_tries=self.sv_tries)
                    else:
                        h[:, j], acc = sample_logvol_imh(h[:, j], ytil, Sinv, logdetS, svp[j], rng, force_accept=force,
                                                         proposal_df=self.sv_proposal_df, n_tries=self.sv_tries)
                    if irep >= sv_init_sweeps:
                        accept[j] += acc
                else:
                    quad = float(ytil @ cho_solve(cf, ytil))
                    s2j = float(rinvgamma(rng, a_sig + T / 2.0, b_sig + quad / 2.0))
                    h[:, j] = np.log(s2j)
                sqrt_om = np.exp(0.5 * h[:, j])

                # ---- Steps 3 & 5: f_j and theta_j1 -----------------------------
                r1 = (ytil - g[:, j]) / sqrt_om
                if not conditional:
                    lp = bo.log_marginal_grid(r1) + bo.grid.log_prior
                    ik1, ix1 = self._sample_grid(lp, rng)
                    xi1 = bo.grid.xi[ix1]
                f[:, j] = bo.posterior_draw(ik1, xi1, r1, sqrt_om, rng)
                if conditional:
                    lp = bo.log_conditional_grid(f[:, j] / sqrt_om) + bo.grid.log_prior
                    ik1, ix1 = self._sample_grid(lp, rng)
                    xi1 = bo.grid.xi[ix1]

                # ---- Steps 4 & 6: g_j and theta_j2 -----------------------------
                r2 = (ytil - f[:, j]) / sqrt_om
                if not conditional:
                    lp = bt.log_marginal_grid(r2) + bt.grid.log_prior
                    ik2, ix2 = self._sample_grid(lp, rng)
                    xi2 = bt.grid.xi[ix2]
                gstar = bt.posterior_draw(ik2, xi2, r2, sqrt_om, rng)
                if self.restrict_g_mean:
                    # Cong, Chen and Zhou (2017): g = g* - V G'(G V G')^{-1} G g*, G = iota'/T
                    U = bt.U[ik2]
                    d = xi2 * bt.lam[ik2] / (1.0 + xi2 * bt.lam[ik2])
                    a = sqrt_om * (U @ (d * (U.T @ sqrt_om)))       # V_g iota
                    gstar = gstar - a * (np.sum(gstar) / np.sum(a))
                g[:, j] = gstar
                if conditional:
                    lp = bt.log_conditional_grid(g[:, j] / sqrt_om) + bt.grid.log_prior
                    ik2, ix2 = self._sample_grid(lp, rng)
                    xi2 = bt.grid.xi[ix2]
                gi[j] = (ik1, ix1, ik2, ix2)

                # ---- Step 7: SV state-equation parameters ---------------------
                if self.sv:
                    svp[j] = sample_sv_params(h[:, j], svp[j], self.sv_priors, rng)

            if irep >= sv_init_sweeps:
                n_sv_steps += 1

            # ---- Step 8: horseshoe hyperparameters of Q ------------------------
            if n_free > 0:
                qvec = np.concatenate([Q[j, :j] for j in range(1, M)])
                hs.update(qvec, rng)

            # ---- storage ---------------------------------------------------------
            if irep >= nburn and (irep - nburn) % thin == 0:
                f_st[isave] = f; g_st[isave] = g; h_st[isave] = h; Q_st[isave] = Q
                for j in range(M):
                    ik1, ix1, ik2, ix2 = gi[j]
                    hy_st[isave, j] = (self.blocks_own[j].grid.xi[ix1], self.blocks_own[j].grid.kappa[ik1],
                                       self.blocks_other[j].grid.xi[ix2], self.blocks_other[j].grid.kappa[ik2])
                    hi_st[isave, j] = gi[j]
                    sv_st[isave, j] = (svp[j].mu, svp[j].rho, svp[j].sig2, svp[j].h0)
                isave += 1

            if verbose and (irep + 1) % print_every == 0:
                el = time.time() - t0
                acc_txt = ", ".join(f"{a / n_sv_steps:.2f}" for a in accept) if self.sv else "n/a"
                print(f"[GP-VAR] iter {irep + 1}/{ntot}  ({el:.0f}s, {el / (irep + 1) * 1000:.0f} ms/iter)"
                      f"  SV-IMH acceptance: {acc_txt}", flush=True)

        runtime = time.time() - t0
        return GPVARResults(model=self, f=f_st, g=g_st, h=h_st, Q=Q_st, hyper=hy_st, hyper_idx=hi_st,
                            sv_params=sv_st, accept_rate=accept / max(n_sv_steps, 1), runtime=runtime,
                            config=dict(nburn=nburn, nsave=nsave, thin=thin, seed=seed, p=self.p, sv=self.sv,
                                        hyper_sampling=self.hyper_sampling, **self.grid_kwargs))
