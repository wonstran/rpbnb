import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from rpbnb_jax.margins import (log_dnbinom2, log_dpois, nb2_cdf_triple,
                               pois_cdf_triple)


def test_log_dnbinom2_matches_scipy():
    # TMB's dnbinom2(y, mu, mu + m*mu^2) has size r = 1/m, prob = r/(r + mu).
    mu, m = 3.7, 0.45
    r = 1.0 / m
    y = np.arange(0, 40)
    want = nbinom.logpmf(y, n=r, p=r / (r + mu))
    got = np.asarray(log_dnbinom2(jnp.asarray(y, dtype=jnp.float64), mu, m))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_log_dpois_matches_scipy():
    mu = 2.25
    y = np.arange(0, 40)
    want = poisson.logpmf(y, mu)
    got = np.asarray(log_dpois(jnp.asarray(y, dtype=jnp.float64), mu))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_nb2_cdf_triple_matches_scipy():
    mu, m = 2.4, 0.6
    r = 1.0 / m
    kmax = 30
    y = np.array([0, 1, 5, 12, 30])
    lc, lcm, lp = nb2_cdf_triple(jnp.asarray(y), mu, r, kmax)
    p = r / (r + mu)
    np.testing.assert_allclose(np.exp(lc), nbinom.cdf(y, r, p), rtol=1e-11)
    np.testing.assert_allclose(np.exp(lp), nbinom.pmf(y, r, p), rtol=1e-11)
    # y = 0 leaves log_cdf_ym1 at log P(Y = 0) by convention; callers gate
    # on the observed count. src/rpbnb_tmb.cpp:157.
    np.testing.assert_allclose(np.exp(lcm[1:]), nbinom.cdf(y[1:] - 1, r, p),
                               rtol=1e-11)
    np.testing.assert_allclose(np.exp(lcm[0]), nbinom.pmf(0, r, p), rtol=1e-11)


def test_pois_cdf_triple_matches_scipy():
    mu, kmax = 3.1, 30
    y = np.array([0, 1, 5, 12, 30])
    lc, lcm, lp = pois_cdf_triple(jnp.asarray(y), mu, kmax)
    np.testing.assert_allclose(np.exp(lc), poisson.cdf(y, mu), rtol=1e-11)
    np.testing.assert_allclose(np.exp(lp), poisson.pmf(y, mu), rtol=1e-11)


def test_deep_tail_mass_survives_where_linear_space_underflows():
    # The failure nb2_cdf_pair()'s header records: a linear-space seed
    # underflows and zeroes the whole recursion. src/rpbnb_tmb.cpp:128-135.
    mu = r = 2000.0
    lc, lcm, lp = nb2_cdf_triple(jnp.asarray([2000]), mu, r, 2000)
    assert np.isfinite(float(lp[0]))
    np.testing.assert_allclose(np.exp(float(lp[0])),
                               nbinom.pmf(2000, r, r / (r + mu)), rtol=1e-8)


def test_gradient_is_finite_at_a_saturated_cdf():
    def f(mu):
        lc, lcm, lp = nb2_cdf_triple(jnp.asarray([0, 70, 266]), mu, 2.0, 266)
        return jnp.sum(lp)

    g = jax.grad(f)(1.0)
    assert np.isfinite(float(g))


# --- gradients ------------------------------------------------------------
#
# The masked logsumexp is the hazard this block exists for. Measured
# 2026-08-23 in this JAX (0.11.1): logsumexp() has no `initial=` argument at
# all, and `where=` alone is neither better nor worse than selecting the
# masked entries down to a finite -1e30 -- both return the same value and the
# same gradient in every regime below, and both return NaN if the grid itself
# carries an input-dependent -inf. Keeping -inf out of the grid (the mu/r
# floors in margins.py) is what actually works; the mask choice is cosmetic.
#
# Four regimes: y = 0, an interior y, y = kmax, and mu far from y in each
# direction.
_GRAD_REGIMES = [
    ("y=0", np.array([0]), 30, 3.1),
    ("interior", np.array([5]), 30, 3.1),
    ("y=kmax", np.array([30]), 30, 3.1),
    ("mu far below y", np.array([200]), 200, 1e-8),
    ("mu far above y", np.array([0]), 200, 1e6),
]
_SLOTS = [(0, "log_cdf_y"), (1, "log_cdf_ym1"), (2, "log_pmf_y")]


@pytest.mark.parametrize("slot,slot_name", _SLOTS)
@pytest.mark.parametrize("label,y,kmax,mu0", _GRAD_REGIMES)
def test_pois_cdf_triple_gradient_is_finite(label, y, kmax, mu0, slot,
                                            slot_name):
    def f(mu):
        return jnp.sum(pois_cdf_triple(jnp.asarray(y), mu, kmax)[slot])

    g = float(jax.grad(f)(mu0))
    assert np.isfinite(g), f"{label} / {slot_name}: d/dmu = {g}"


@pytest.mark.parametrize("slot,slot_name", _SLOTS)
@pytest.mark.parametrize("label,y,kmax,mu0", _GRAD_REGIMES)
def test_nb2_cdf_triple_gradient_is_finite_in_mu_and_r(label, y, kmax, mu0,
                                                       slot, slot_name):
    yj = jnp.asarray(y)
    g_mu = float(jax.grad(
        lambda mu: jnp.sum(nb2_cdf_triple(yj, mu, 2.0, kmax)[slot]))(mu0))
    g_r = float(jax.grad(
        lambda r: jnp.sum(nb2_cdf_triple(yj, mu0, r, kmax)[slot]))(2.0))
    assert np.isfinite(g_mu), f"{label} / {slot_name}: d/dmu = {g_mu}"
    assert np.isfinite(g_r), f"{label} / {slot_name}: d/dr = {g_r}"


def test_pois_log_pmf_gradient_matches_the_analytic_score():
    # Finite is not enough -- it must also be right. d/dmu log P(Y = y) is
    # y/mu - 1.
    for mu0, y in [(3.1, 5), (1e-8, 200), (1e6, 0)]:
        g = float(jax.grad(
            lambda mu: pois_cdf_triple(jnp.asarray([y]), mu, 200)[2][0])(mu0))
        np.testing.assert_allclose(g, y / mu0 - 1.0, rtol=1e-10)


def test_nb2_log_pmf_gradient_matches_a_central_difference():
    y, r, kmax, mu0, h = 12, 2.0, 30, 4.0, 1e-6

    def lp(mu, r_):
        return float(nb2_cdf_triple(jnp.asarray([y]), mu, r_, kmax)[2][0])

    g_mu = float(jax.grad(
        lambda mu: nb2_cdf_triple(jnp.asarray([y]), mu, r, kmax)[2][0])(mu0))
    g_r = float(jax.grad(
        lambda r_: nb2_cdf_triple(jnp.asarray([y]), mu0, r_, kmax)[2][0])(r))
    np.testing.assert_allclose(
        g_mu, (lp(mu0 + h, r) - lp(mu0 - h, r)) / (2 * h), rtol=1e-6)
    np.testing.assert_allclose(
        g_r, (lp(mu0, r + h) - lp(mu0, r - h)) / (2 * h), rtol=1e-6)


def test_log_cdf_gradient_matches_a_central_difference():
    y, kmax, mu0, h = 5, 30, 3.1, 1e-6

    def lc(mu, slot):
        return float(pois_cdf_triple(jnp.asarray([y]), mu, kmax)[slot][0])

    for slot in (0, 1):
        g = float(jax.grad(
            lambda mu: pois_cdf_triple(jnp.asarray([y]), mu, kmax)[slot][0]
        )(mu0))
        np.testing.assert_allclose(
            g, (lc(mu0 + h, slot) - lc(mu0 - h, slot)) / (2 * h), rtol=1e-6)


# --- shapes ---------------------------------------------------------------

def test_triples_broadcast_over_the_n_by_R_grid_the_copula_path_builds():
    # The shape Task 4 hits first: mu is (n, R) because every draw has its
    # own linear predictor, y is (n, 1) because the count is shared across
    # draws. Trailing-axis broadcasting alone does NOT line the mask up here.
    n, R, kmax = 4, 3, 12
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 4.0, size=(n, R))
    y = np.array([[0], [3], [7], [12]])
    y_full = np.broadcast_to(y, (n, R))

    lc, lcm, lp = pois_cdf_triple(jnp.asarray(y), jnp.asarray(mu), kmax)
    assert lc.shape == lcm.shape == lp.shape == (n, R)
    np.testing.assert_allclose(np.exp(lp), poisson.pmf(y_full, mu), rtol=1e-11)
    np.testing.assert_allclose(np.exp(lc), poisson.cdf(y_full, mu), rtol=1e-11)
    np.testing.assert_allclose(np.exp(lcm[1:]),
                               poisson.cdf(y_full[1:] - 1, mu[1:]), rtol=1e-11)
    np.testing.assert_allclose(np.exp(lcm[0]), poisson.pmf(0, mu[0]),
                               rtol=1e-11)

    r = 1.6
    lc, lcm, lp = nb2_cdf_triple(jnp.asarray(y), jnp.asarray(mu), r, kmax)
    assert lc.shape == lcm.shape == lp.shape == (n, R)
    p = r / (r + mu)
    np.testing.assert_allclose(np.exp(lp), nbinom.pmf(y_full, r, p), rtol=1e-11)
    np.testing.assert_allclose(np.exp(lc), nbinom.cdf(y_full, r, p), rtol=1e-11)


def test_rank_three_gradient_is_finite():
    n, R, kmax = 4, 3, 12
    y = jnp.asarray(np.array([[0], [3], [7], [12]]))

    def f(scale):
        mu = scale * jnp.ones((n, R))
        lc, lcm, lp = nb2_cdf_triple(y, mu, 1.6, kmax)
        return jnp.sum(lc) + jnp.sum(lcm) + jnp.sum(lp)

    assert np.isfinite(float(jax.grad(f)(2.0)))


def test_mu_may_be_a_python_float_a_scalar_array_or_an_array():
    y, kmax = jnp.asarray([0, 4]), 10
    want = pois_cdf_triple(y, 2.5, kmax)
    for alt in (jnp.asarray(2.5), np.float64(2.5), jnp.full((2,), 2.5)):
        got = pois_cdf_triple(y, alt, kmax)
        for a, b in zip(want, got):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0)
    # r likewise.
    want = nb2_cdf_triple(y, 2.5, 1.6, kmax)
    for alt in (jnp.asarray(1.6), np.float64(1.6)):
        got = nb2_cdf_triple(y, 2.5, alt, kmax)
        for a, b in zip(want, got):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0)


# --- the kmax guard -------------------------------------------------------

def test_a_count_above_kmax_is_rejected_rather_than_silently_misread():
    with pytest.raises(ValueError, match="below max"):
        nb2_cdf_triple(jnp.asarray([0, 31]), 2.4, 1.6, 30)
    with pytest.raises(ValueError, match="below max"):
        pois_cdf_triple(jnp.asarray([31]), 3.1, 30)


def test_a_bad_kmax_or_negative_count_is_rejected():
    with pytest.raises(ValueError, match="kmax must be"):
        pois_cdf_triple(jnp.asarray([0]), 3.1, -1)
    with pytest.raises(ValueError, match="non-negative"):
        pois_cdf_triple(jnp.asarray([-1]), 3.1, 10)


def test_kmax_must_be_static():
    # kmax sets an array dimension, so a tracer must not reach it.
    with pytest.raises(Exception):
        jax.jit(lambda k: pois_cdf_triple(jnp.asarray([2]), 3.1, k))(10)


def test_a_traced_count_above_the_grid_gives_nan_not_a_plausible_number():
    # Under jit the concrete y check cannot run, so the gather must fail
    # loudly instead of returning the mass at some other index.
    f = jax.jit(lambda y, mu: pois_cdf_triple(y, mu, 10)[2])
    assert np.isnan(float(f(jnp.asarray([11]), 3.1)[0]))
    assert np.isfinite(float(f(jnp.asarray([4]), 3.1)[0]))
