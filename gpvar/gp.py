"""
Univariate Gaussian process regression (Section 2 of the paper).

Used for the illustrative Figures 1, 2 and A.1: prior draws and the posterior
of ``f`` under

* the Gaussian kernel  k(x, x') = xi exp(-kappa/2 ||x - x'||^2),
* the linear kernel    K = X V X'   (a Bayesian linear regression),
* the persistence kernel K = r B B' with B lower triangular of ones
  (an unobserved-components / local-level model, Appendix A.1).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .kernels import cross_sq_dists, gaussian_kernel, scaled_sq_dists


class GPRegression:
    """
    y = f(x) + e,  f ~ GP(0, k),  e ~ N(0, sigma2).

    Parameters
    ----------
    kernel : {"gaussian", "linear", "persistence"}
    kappa, xi : float          Gaussian-kernel hyperparameters
    v : float                  prior variance of the linear-kernel coefficients
    r : float                  jump variance of the persistence kernel
    sigma2 : float or None     error variance; if None it is estimated by
                               maximising the log marginal likelihood on a grid
    """

    def __init__(self, kernel: str = "gaussian", kappa: float = 0.1, xi: float = 1.0, v: float = 1.0,
                 r: float = 0.01, sigma2: Optional[float] = 1.0, scale_inputs: bool = False):
        self.kernel = kernel
        self.kappa, self.xi, self.v, self.r = kappa, xi, v, r
        self.sigma2 = sigma2
        self.scale_inputs = scale_inputs

    # ------------------------------------------------------------------ kernels
    def _K(self, X: np.ndarray, Xnew: Optional[np.ndarray] = None) -> np.ndarray:
        if self.kernel == "gaussian":
            D = np.var(X, axis=0, ddof=1) if self.scale_inputs else None
            D2 = scaled_sq_dists(X, D) if Xnew is None else cross_sq_dists(Xnew, X, D)
            return gaussian_kernel(D2, self.kappa, self.xi)
        if self.kernel == "linear":
            Xa = np.hstack([np.ones((X.shape[0], 1)), X])
            Xb = Xa if Xnew is None else np.hstack([np.ones((Xnew.shape[0], 1)), Xnew])
            return self.v * (Xb @ Xa.T)
        if self.kernel == "persistence":
            n = X.shape[0]
            B = np.tril(np.ones((n, n)))
            return self.r * (B @ B.T)
        raise ValueError(self.kernel)

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray, sigma2_grid: Optional[np.ndarray] = None) -> "GPRegression":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] != y.size:
            X = X.T
        self.X_, self.y_ = X, np.asarray(y, dtype=float)
        K = self._K(X)
        if self.sigma2 is None:
            grid = np.logspace(-3, 1, 41) if sigma2_grid is None else sigma2_grid
            best, bestll = None, -np.inf
            for s2 in grid:
                ll = self._log_marginal(K, s2)
                if ll > bestll:
                    best, bestll = s2, ll
            self.sigma2 = float(best)
        self.K_ = K
        A = K + self.sigma2 * np.eye(K.shape[0])
        self.alpha_ = np.linalg.solve(A, self.y_)
        self.Ainv_ = np.linalg.inv(A)
        return self

    def _log_marginal(self, K, s2):
        A = K + s2 * np.eye(K.shape[0])
        sign, logdet = np.linalg.slogdet(A)
        return -0.5 * (self.y_ @ np.linalg.solve(A, self.y_)) - 0.5 * logdet

    # --------------------------------------------------------------- inference
    def prior_draws(self, X: np.ndarray, n: int = 3, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T
        K = self._K(X) + 1e-8 * np.eye(X.shape[0])
        return rng.multivariate_normal(np.zeros(X.shape[0]), K, size=n, method="eigh")

    def prior_sd(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T
        return np.sqrt(np.diag(self._K(X)))

    def posterior(self, Xnew: Optional[np.ndarray] = None):
        """Posterior mean and pointwise standard deviation of f at ``Xnew`` (default: training inputs)."""
        if Xnew is None:
            Ks = self.K_
            kss = np.diag(self.K_)
        else:
            Xnew = np.atleast_2d(np.asarray(Xnew, dtype=float))
            if Xnew.shape[0] == 1 and Xnew.shape[1] > 1:
                Xnew = Xnew.T
            Ks = self._K(self.X_, Xnew)
            kss = np.diag(self._K(Xnew)) if self.kernel != "persistence" else np.diag(self.K_)
        mean = Ks @ self.alpha_
        var = kss - np.einsum("ij,jk,ik->i", Ks, self.Ainv_, Ks)
        return mean, np.sqrt(np.maximum(var, 0.0))
