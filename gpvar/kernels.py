"""
Kernel machinery for the GP-VAR.

Implements the equation-specific Gaussian (squared-exponential) kernels of
Hauzenberger, Huber, Marcellino and Petz (2022, Section 3.2)

    k(x_t, x_tau) = xi * exp( -kappa/2 * (x_t - x_tau)' D^{-1} (x_t - x_tau) ),

where ``D`` is the diagonal matrix of empirical variances of the columns of the
input matrix, ``xi`` is the linear scaling ("signal variance") and ``kappa`` the
inverse length scale.  The median heuristic of Chaudhuri et al. (2017) and the
discrete (kappa, xi) grid of Section 3.3 are also implemented here.

Because ``K_{xi,kappa} = xi * K_kappa``, only the ``kappa`` dimension of the
grid requires distinct matrices.  For every ``kappa`` on the grid we store the
eigendecomposition ``K_kappa = U diag(lam) U'``; all posterior quantities that
depend on ``(xi, kappa)`` are then O(T^2) instead of O(T^3):

    (xi K + I)^{-1}              = U diag( 1 / (1 + xi lam) ) U'
    xi K (xi K + I)^{-1}         = U diag( xi lam / (1 + xi lam) ) U'
    xi K - xi K (xi K + I)^{-1} xi K = U diag( xi lam / (1 + xi lam) ) U'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Squared distances and kernel evaluation
# --------------------------------------------------------------------------- #
def column_scaling(X: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Diagonal scaling ``D`` = empirical variances of the columns of ``X``."""
    v = np.var(X, axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
    return np.maximum(v, floor)


def scaled_sq_dists(X: np.ndarray, D: Optional[np.ndarray] = None) -> np.ndarray:
    """Matrix of ``(x_t - x_tau)' D^{-1} (x_t - x_tau)`` for all pairs (T x T)."""
    if D is None:
        D = np.ones(X.shape[1])
    Xs = X / np.sqrt(D)
    sq = np.sum(Xs * Xs, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return D2


def cross_sq_dists(Xnew: np.ndarray, X: np.ndarray, D: Optional[np.ndarray] = None) -> np.ndarray:
    """Scaled squared distances between rows of ``Xnew`` (B x k) and ``X`` (T x k)."""
    if D is None:
        D = np.ones(X.shape[1])
    s = np.sqrt(D)
    A = Xnew / s
    B = X / s
    D2 = np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(D2, 0.0, out=D2)
    return D2


def gaussian_kernel(D2: np.ndarray, kappa: float, xi: float = 1.0) -> np.ndarray:
    """``xi * exp(-kappa/2 * D2)`` for a matrix of scaled squared distances."""
    return xi * np.exp(-0.5 * kappa * D2)


def median_heuristic(D2: np.ndarray) -> float:
    """
    Median heuristic for the inverse length scale (Section 3.3):

        kappa_bar = median_{t != tau} ( 1 / ||x_t - x_tau|| ).

    Zero distances (identical rows) are excluded.
    """
    iu = np.triu_indices_from(D2, k=1)
    d = np.sqrt(D2[iu])
    d = d[d > 1e-12]
    if d.size == 0:
        return 1.0
    return float(np.median(1.0 / d))


# --------------------------------------------------------------------------- #
# Hyperparameter grid
# --------------------------------------------------------------------------- #
def log_gamma_prior(x: np.ndarray, c: float) -> np.ndarray:
    """
    Log density (up to a constant) of the Gamma(1/2, rate = 1/(2c)) prior used
    for xi and kappa in Section 3.3.  ``c`` is the prior mean; small ``c``
    pushes the hyperparameter towards zero.
    """
    return -0.5 * np.log(x) - x / (2.0 * c)


@dataclass
class HyperGrid:
    """Two-dimensional discrete grid for (kappa, xi) with its prior weights."""
    kappa: np.ndarray            # (n_kappa,)
    xi: np.ndarray               # (n_xi,)
    log_prior: np.ndarray        # (n_kappa, n_xi), normalised on the grid
    kappa_bar: float

    @property
    def n_kappa(self) -> int:
        return self.kappa.size

    @property
    def n_xi(self) -> int:
        return self.xi.size


def make_hyper_grid(kappa_bar: float,
                    scheme: str = "semi-automatic",
                    n_kappa: int = 20,
                    n_xi: int = 50,
                    kappa_range: Tuple[float, float] = (0.1, 2.0),
                    xi_range: Tuple[float, float] = (0.04, 4.0),
                    c_kappa: float = 0.1,
                    c_xi: float = 1.0,
                    naive_kappa_range: Tuple[float, float] = (0.1, 2.0)) -> HyperGrid:
    """
    Build the (kappa, xi) grid of Section 3.3 / Section 5.2.

    scheme
        ``"semi-automatic"``          kappa in [0.1 kb, 2 kb], xi in [0.04, 4]
        ``"semi-automatic-noscale"``  kappa in [0.1 kb, 2 kb], xi = 1
        ``"naive"``                   kappa in [0.1, 2],       xi = 1
    where ``kb`` is the median heuristic.
    """
    scheme = scheme.lower()
    if scheme == "semi-automatic":
        kappa = np.linspace(kappa_range[0] * kappa_bar, kappa_range[1] * kappa_bar, n_kappa)
        xi = np.linspace(xi_range[0], xi_range[1], n_xi)
    elif scheme in ("semi-automatic-noscale", "semi-automatic w/o linear scaling"):
        kappa = np.linspace(kappa_range[0] * kappa_bar, kappa_range[1] * kappa_bar, n_kappa)
        xi = np.array([1.0])
    elif scheme == "naive":
        kappa = np.linspace(naive_kappa_range[0], naive_kappa_range[1], n_kappa)
        xi = np.array([1.0])
    else:
        raise ValueError(f"unknown hyperparameter scheme '{scheme}'")

    lp = log_gamma_prior(kappa, c_kappa)[:, None] + log_gamma_prior(xi, c_xi)[None, :]
    lp = lp - np.logaddexp.reduce(lp.ravel())
    return HyperGrid(kappa=kappa, xi=xi, log_prior=lp, kappa_bar=kappa_bar)


# --------------------------------------------------------------------------- #
# Pre-computed kernel block for one (equation, kernel-type) pair
# --------------------------------------------------------------------------- #
@dataclass
class KernelBlock:
    """
    Everything the sampler needs for one kernel (own or other lags) of one
    equation: scaled inputs, the T x T scaled distance matrix, the hyper grid
    and the eigendecompositions ``K_kappa = U diag(lam) U'`` for every kappa.
    """
    X: np.ndarray                 # (T, k) raw inputs
    D: np.ndarray                 # (k,)   column variances (scaling)
    D2: np.ndarray                # (T, T) scaled squared distances
    grid: HyperGrid
    U: np.ndarray = field(repr=False)     # (n_kappa, T, T)
    lam: np.ndarray = field(repr=False)   # (n_kappa, T)   eigenvalues >= 0
    jitter: float = 1e-8

    # ---- construction -----------------------------------------------------
    @classmethod
    def build(cls, X: np.ndarray, grid_kwargs: dict, D: Optional[np.ndarray] = None) -> "KernelBlock":
        X = np.asarray(X, dtype=float)
        if D is None:
            D = column_scaling(X)
        D2 = scaled_sq_dists(X, D)
        kb = median_heuristic(D2)
        grid = make_hyper_grid(kb, **grid_kwargs)
        T = X.shape[0]
        U = np.empty((grid.n_kappa, T, T))
        lam = np.empty((grid.n_kappa, T))
        for i, k in enumerate(grid.kappa):
            K = gaussian_kernel(D2, k)
            w, V = np.linalg.eigh(K)
            lam[i] = np.maximum(w, 0.0)
            U[i] = V
        return cls(X=X, D=D, D2=D2, grid=grid, U=U, lam=lam)

    # ---- helpers ----------------------------------------------------------
    def kernel(self, ik: int, xi: float = 1.0) -> np.ndarray:
        """Dense T x T kernel for grid index ``ik`` and scaling ``xi``."""
        return gaussian_kernel(self.D2, self.grid.kappa[ik], xi)

    def cross_kernel(self, Xnew: np.ndarray, kappa: float, xi: float = 1.0) -> np.ndarray:
        """B x T kernel between new inputs and the training inputs."""
        return gaussian_kernel(cross_sq_dists(Xnew, self.X, self.D), kappa, xi)

    def log_marginal_grid(self, r_tilde: np.ndarray) -> np.ndarray:
        """
        Log N(r_tilde; 0, xi K_kappa + I_T) for every grid point (n_kappa, n_xi),
        with ``r_tilde`` already scaled by Omega^{-1/2}.  Used to sample the
        hyperparameters marginally of the latent function (blocked step).
        """
        c = np.einsum("ktj,t->kj", self.U, r_tilde)            # U' r  (n_kappa, T)
        A = 1.0 + self.grid.xi[None, :, None] * self.lam[:, None, :]   # (n_kappa, n_xi, T)
        return -0.5 * np.sum(np.log(A), axis=2) - 0.5 * np.sum((c * c)[:, None, :] / A, axis=2)

    def log_conditional_grid(self, f_tilde: np.ndarray) -> np.ndarray:
        """
        Log N(f_tilde; 0, xi K_kappa) for every grid point, i.e. the exact
        conditional posterior ordinate of Section 3.4 (with a small jitter
        added to the eigenvalues to keep K invertible).
        """
        c = np.einsum("ktj,t->kj", self.U, f_tilde)
        lam = self.lam + self.jitter
        A = self.grid.xi[None, :, None] * lam[:, None, :]
        return -0.5 * np.sum(np.log(A), axis=2) - 0.5 * np.sum((c * c)[:, None, :] / A, axis=2)

    def posterior_draw(self, ik: int, xi: float, r_tilde: np.ndarray, sqrt_omega: np.ndarray,
                       rng: np.random.Generator, sample: bool = True) -> np.ndarray:
        """
        Draw the latent function from its conjugate conditional posterior
        (Section 3.4):

            f | .  ~ N( sqrt(Om) U d U' Om^{-1/2} r ,  sqrt(Om) U diag(d) U' sqrt(Om) ),
            d = xi lam / (1 + xi lam).
        """
        U = self.U[ik]
        d = xi * self.lam[ik] / (1.0 + xi * self.lam[ik])
        c = U.T @ r_tilde
        mean = U @ (d * c)
        if sample:
            z = rng.standard_normal(U.shape[0])
            mean = mean + U @ (np.sqrt(d) * z)
        return sqrt_omega * mean

    def posterior_mean_var_diag(self, ik: int, xi: float, r_tilde: np.ndarray, sqrt_omega: np.ndarray):
        """Posterior mean and pointwise variance of the latent function."""
        U = self.U[ik]
        d = xi * self.lam[ik] / (1.0 + xi * self.lam[ik])
        c = U.T @ r_tilde
        mean = sqrt_omega * (U @ (d * c))
        var = (sqrt_omega ** 2) * np.einsum("tj,j,tj->t", U, d, U)
        return mean, var


# --------------------------------------------------------------------------- #
# Lag matrices
# --------------------------------------------------------------------------- #
def lag_matrix(Y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ``y`` (T x M) and the stacked lag matrix ``X`` (T x Mp) with the
    ordering [y_{t-1}', y_{t-2}', ..., y_{t-p}'] so that the own lags of
    variable ``j`` sit at columns ``j + M*k``, k = 0..p-1.
    """
    Y = np.asarray(Y, dtype=float)
    Traw, M = Y.shape
    X = np.hstack([Y[p - k - 1:Traw - k - 1] for k in range(p)])
    return Y[p:], X


def own_other_index(M: int, p: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """Column indices of the own lags of variable ``j`` and of all other lags."""
    own = j + M * np.arange(p)
    other = np.setdiff1d(np.arange(M * p), own)
    return own, other
