"""Unit and integration tests for the gpvar package (run with ``pytest``)."""
import numpy as np
import pandas as pd
import pytest

import gpvar
from gpvar.kernels import (KernelBlock, cross_sq_dists, gaussian_kernel, lag_matrix, make_hyper_grid,
                           median_heuristic, own_other_index, scaled_sq_dists)
from gpvar.priors import HorseshoeState, SVPriors, ig_from_mean_var
from gpvar.sv import SVParams, sample_logvol_imh, sample_sv_params


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
def test_scaled_sq_dists_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 3))
    D = np.var(X, axis=0, ddof=1)
    D2 = scaled_sq_dists(X, D)
    brute = np.array([[np.sum((X[i] - X[j]) ** 2 / D) for j in range(15)] for i in range(15)])
    assert np.allclose(D2, brute, atol=1e-10)
    C = cross_sq_dists(X[:4], X, D)
    assert np.allclose(C, brute[:4], atol=1e-10)


def test_lag_matrix_and_indices():
    Y = np.arange(20, dtype=float).reshape(10, 2)
    y, X = lag_matrix(Y, 3)
    assert y.shape == (7, 2) and X.shape == (7, 6)
    assert np.allclose(X[0], np.concatenate([Y[2], Y[1], Y[0]]))
    own, other = own_other_index(2, 3, 1)
    assert list(own) == [1, 3, 5] and list(other) == [0, 2, 4]


def test_kernel_block_eigen_reconstruction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    blk = KernelBlock.build(X, dict(scheme="semi-automatic", n_kappa=4, n_xi=3))
    for i, k in enumerate(blk.grid.kappa):
        K = gaussian_kernel(blk.D2, k)
        Krec = (blk.U[i] * blk.lam[i]) @ blk.U[i].T
        assert np.allclose(K, Krec, atol=1e-8)
    assert blk.grid.kappa[0] == pytest.approx(0.1 * blk.grid.kappa_bar)
    assert blk.grid.kappa[-1] == pytest.approx(2.0 * blk.grid.kappa_bar)
    assert np.isclose(np.logaddexp.reduce(blk.grid.log_prior.ravel()), 0.0)


def test_marginal_grid_matches_dense_gaussian():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 2))
    blk = KernelBlock.build(X, dict(scheme="semi-automatic", n_kappa=3, n_xi=2))
    r = rng.normal(size=25)
    lm = blk.log_marginal_grid(r)
    for i, k in enumerate(blk.grid.kappa):
        for jx, xi in enumerate(blk.grid.xi):
            S = xi * gaussian_kernel(blk.D2, k) + np.eye(25)
            sign, ld = np.linalg.slogdet(S)
            ref = -0.5 * ld - 0.5 * r @ np.linalg.solve(S, r)
            assert np.isclose(lm[i, jx], ref, atol=1e-6)


def test_posterior_draw_moments():
    """Posterior mean of f must equal sqrt(Om) xi K (xi K + I)^{-1} Om^{-1/2} r."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 2))
    blk = KernelBlock.build(X, dict(scheme="semi-automatic", n_kappa=2, n_xi=2))
    r = rng.normal(size=20)
    som = np.exp(0.3 * rng.normal(size=20))
    xi = blk.grid.xi[1]
    mean = blk.posterior_draw(1, xi, r / som, som, rng, sample=False)
    K = xi * gaussian_kernel(blk.D2, blk.grid.kappa[1])
    ref = som * (K @ np.linalg.solve(K + np.eye(20), r / som))
    assert np.allclose(mean, ref, atol=1e-8)


def test_hyper_grid_schemes():
    g = make_hyper_grid(0.5, scheme="naive", n_kappa=5)
    assert g.xi.size == 1 and g.kappa[0] == 0.1 and g.kappa[-1] == 2.0
    g = make_hyper_grid(0.5, scheme="semi-automatic-noscale", n_kappa=5)
    assert g.xi.size == 1 and np.isclose(g.kappa[0], 0.05)


def test_median_heuristic_positive():
    rng = np.random.default_rng(4)
    D2 = scaled_sq_dists(rng.normal(size=(40, 3)))
    assert median_heuristic(D2) > 0


# --------------------------------------------------------------------------- #
# priors / SV
# --------------------------------------------------------------------------- #
def test_ig_from_mean_var():
    a, b = ig_from_mean_var(0.1, 0.01)
    assert np.isclose(a, 3.0) and np.isclose(b, 0.2)


def test_horseshoe_update_runs():
    rng = np.random.default_rng(5)
    hs = HorseshoeState.init(6)
    for _ in range(20):
        hs.update(rng.normal(scale=0.1, size=6), rng)
    assert np.all(hs.prior_var > 0)


def test_sv_imh_and_params():
    rng = np.random.default_rng(6)
    T = 200
    h_true = 2.0 * np.sin(np.linspace(0, 2 * np.pi, T))       # pronounced volatility cycle
    Ytil = np.exp(0.5 * h_true) * rng.normal(size=T)
    Sinv = np.eye(T)
    prm = SVParams(mu=0.0, rho=0.9, sig2=0.05, h0=0.0)
    h = np.zeros(T)
    acc = 0
    hbar = np.zeros(T)
    for it in range(150):
        h, a = sample_logvol_imh(h, Ytil, Sinv, 0.0, prm, rng)
        acc += a
        prm = sample_sv_params(h, prm, SVPriors(), rng)
        if it >= 50:
            hbar += h / 100
    assert acc > 10
    assert -1 < prm.rho < 1 and prm.sig2 > 0
    assert np.corrcoef(hbar, h_true)[0, 1] > 0.7


# --------------------------------------------------------------------------- #
# model, GIRF, forecast
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_fit():
    sim = gpvar.simulate_replication_dgp(T=70, M=3, p=2, seed=42)
    m = gpvar.GPVAR(sim.Y, p=2, n_kappa=6, n_xi=8)
    res = m.fit(nburn=40, nsave=40, seed=1, verbose=False)
    return sim, res


def test_fit_shapes_and_sanity(small_fit):
    sim, res = small_fit
    assert res.f.shape == (40, 68, 3) and res.Q.shape == (40, 3, 3)
    assert np.all(np.triu(res.Q[0]) == 0)             # lower triangular, zero diagonal
    assert np.all(np.isfinite(res.h))
    assert np.all(res.accept_rate >= 0)
    if res.model.restrict_g_mean:
        assert np.allclose(res.g.mean(axis=1), 0.0, atol=1e-8)   # grand-mean-zero restriction
    corr = res.fit_correlations(sim.m)["corr"]
    assert corr.min() > 0.5


def test_girf_basic(small_fit):
    sim, res = small_fit
    g = gpvar.girf(res, "y1", horizon=4, n_sim=2, n_draws=5, verbose=False)
    assert g.irf.shape == (5, 68, 5, 3)
    assert np.allclose(g.irf[:, :, 0, 0], 1.0)          # unit normalisation on impact
    s = g.summary()
    assert set(s.columns) == {"variable", "horizon", "q16", "q50", "q84"}
    neg = gpvar.girf(res, "y1", sign=-1, horizon=4, n_sim=2, n_draws=5, verbose=False)
    assert np.allclose(neg.irf[:, :, 0, 0], -1.0)
    pt = g.peak_table()
    assert len(pt) == 3


def test_girf_sd_scale(small_fit):
    sim, res = small_fit
    g = gpvar.girf(res, "y2", horizon=2, n_sim=1, n_draws=3, shock_scale="sd", verbose=False)
    assert np.all(g.irf[:, :, 0, 1] > 0)


def test_predict_and_scores(small_fit):
    sim, res = small_fit
    fc = res.predict(horizon=3, n_sim=2, n_draws=5)
    assert fc.paths.shape == (10, 3, 3)
    lpl = gpvar.log_predictive_score(fc.paths_raw[:, 0, :], sim.Y.iloc[-1].to_numpy())
    assert np.isfinite(lpl)
    c = gpvar.crps(fc.paths_raw[:, 0, 0], float(sim.Y.iloc[-1, 0]))
    assert c >= 0


def test_homoskedastic_and_conditional():
    sim = gpvar.simulate_replication_dgp(T=50, M=2, p=1, seed=7)
    r1 = gpvar.GPVAR(sim.Y, p=1, sv=False, n_kappa=4, n_xi=4).fit(nburn=10, nsave=10, verbose=False)
    assert np.allclose(r1.h[:, 0, :], r1.h[:, -1, :])
    r2 = gpvar.GPVAR(sim.Y, p=1, hyper_sampling="conditional", n_kappa=4, n_xi=4).fit(nburn=10, nsave=10, verbose=False)
    assert r2.hyper.shape == (10, 2, 4)


def test_bvar_benchmark():
    sim = gpvar.simulate_replication_dgp(T=80, M=3, p=2, seed=8)
    res = gpvar.BVAR(sim.Y, p=2).fit(nsave=30, seed=1)
    fc = res.predict(horizon=2, n_draws=10)
    assert fc.paths.shape == (10, 2, 3)
    irf = res.irf("y1", horizon=3, n_draws=10)
    assert np.isclose(irf[(irf.variable == "y1") & (irf.horizon == 0)]["q50"].iloc[0], 1.0)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def test_build_dataset():
    Y, info = gpvar.data.build_dataset(8, "1970Q1", "2019Q4")
    assert Y.shape == (200, 8)
    assert list(Y.columns) == ["UNC", "RGDP", "EMP", "AWH", "CPI", "AHE", "FFR", "SP500"]
    assert not Y.isna().any().any()
    assert gpvar.data.recession_mask(Y.index).sum() > 20


def test_gp_regression():
    rng = np.random.default_rng(9)
    x = np.linspace(0, 10, 40)
    y = np.sin(x) + 0.2 * rng.normal(size=40)
    gp = gpvar.GPRegression("gaussian", kappa=1.0, xi=1.0, sigma2=0.05).fit(x[:, None], y)
    mean, sd = gp.posterior()
    assert np.corrcoef(mean, np.sin(x))[0, 1] > 0.95
    lin = gpvar.GPRegression("linear", v=1.0, sigma2=None).fit(x[:, None], y)
    assert lin.sigma2 > 0
