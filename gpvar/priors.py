"""
Prior distributions and their Gibbs updates.

* Horseshoe prior (Carvalho, Polson and Scott, 2010) on the free elements of the
  contemporaneous matrix Q, updated with the inverse-Gamma scale-mixture
  representation of Makalic and Schmidt (2015) - only inverse-Gamma draws.
* Inverse-Gamma helpers (parameterised by shape ``a`` and rate ``b``).
* Beta prior on the transformed persistence parameter (rho + 1)/2 ~ B(25, 5) and
  the inverse-Gamma prior on the state-innovation variance of the log-volatility
  (mean 0.1, variance 0.01, i.e. IG(3, 0.2)) of Section 3.2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rinvgamma(rng: np.random.Generator, shape: float, rate, size=None) -> np.ndarray:
    """Inverse-Gamma(shape, rate) draws: ``1 / Gamma(shape, scale = 1/rate)``."""
    return 1.0 / rng.gamma(shape, 1.0 / np.asarray(rate, dtype=float), size=size)


@dataclass
class HorseshoeState:
    """Local scales ``lam``, global scale ``tau`` and auxiliaries ``nu``, ``zeta``."""
    lam: np.ndarray
    nu: np.ndarray
    tau: float = 1.0
    zeta: float = 1.0

    @classmethod
    def init(cls, k: int) -> "HorseshoeState":
        return cls(lam=np.ones(k), nu=np.ones(k), tau=1.0, zeta=1.0)

    @property
    def prior_var(self) -> np.ndarray:
        """Prior variances ``lam_i * tau`` implied by the current scales."""
        return self.lam * self.tau

    def update(self, beta: np.ndarray, rng: np.random.Generator, max_var: float = 10.0,
               min_var: float = 1e-8) -> None:
        """
        One Gibbs sweep of Makalic and Schmidt (2015):

            tau  | .  ~ IG( (k+1)/2 , 1/zeta + sum_i beta_i^2 / (2 lam_i) )
            lam_i| .  ~ IG( 1 , 1/nu_i + beta_i^2 / (2 tau) )
            nu_i | .  ~ IG( 1 , 1 + 1/lam_i )
            zeta | .  ~ IG( 1 , 1 + 1/tau )
        """
        k = beta.size
        self.tau = float(rinvgamma(rng, (k + 1) / 2.0, 1.0 / self.zeta + np.sum(beta ** 2 / self.lam) / 2.0))
        self.lam = rinvgamma(rng, 1.0, 1.0 / self.nu + beta ** 2 / (2.0 * self.tau), size=k)
        self.nu = rinvgamma(rng, 1.0, 1.0 + 1.0 / self.lam, size=k)
        self.zeta = float(rinvgamma(rng, 1.0, 1.0 + 1.0 / self.tau))
        # numerical guards (same spirit as the replication archive)
        self.lam = np.clip(self.lam, min_var / max(self.tau, 1e-12), max_var / max(self.tau, 1e-12))


@dataclass
class SVPriors:
    """
    Priors for the stochastic-volatility state equation (3):

        h_t = mu + rho (h_{t-1} - mu) + nu_t,   nu_t ~ N(0, sig2),
        h_0 ~ N(mu, sig2 / (1 - rho^2)).

    * (rho + 1)/2 ~ Beta(rho_a, rho_b)              (paper: Beta(25, 5))
    * sig2        ~ IG(sig2_a, sig2_b)              (paper: mean 0.1, var 0.01 -> IG(3, 0.2))
    * mu          ~ N(mu_mean, mu_var)              (only if ``estimate_mu``; the
                                                     paper's eq. (3) fixes mu = 0)
    """
    rho_a: float = 25.0
    rho_b: float = 5.0
    sig2_a: float = 3.0
    sig2_b: float = 0.2
    mu_mean: float = 0.0
    mu_var: float = 10.0
    estimate_mu: bool = True

    def log_prior_rho(self, rho: float) -> float:
        if not (-1.0 < rho < 1.0):
            return -np.inf
        x = (rho + 1.0) / 2.0
        return (self.rho_a - 1.0) * np.log(x) + (self.rho_b - 1.0) * np.log1p(-x)


def ig_from_mean_var(mean: float, var: float):
    """Shape/rate of an inverse-Gamma with the given mean and variance."""
    a = mean ** 2 / var + 2.0
    b = mean * (a - 1.0)
    return a, b
