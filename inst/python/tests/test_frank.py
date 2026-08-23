import jax
import jax.numpy as jnp
import numpy as np

from rpbnb_jax.frank import frank_log_cell_prob


def _frank_C(u, v, th):
    return -np.log1p(np.expm1(-th * u) * np.expm1(-th * v)
                     / np.expm1(-th)) / th


def test_matches_second_difference_where_that_is_still_accurate():
    th = 2.5
    a, am = 0.62, 0.41
    b, bm = 0.55, 0.30
    want = (_frank_C(a, b, th) - _frank_C(am, b, th)
            - _frank_C(a, bm, th) + _frank_C(am, bm, th))
    got = float(jnp.exp(frank_log_cell_prob(
        a, am, jnp.log(a - am), b, bm, jnp.log(b - bm), th)))
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_near_independence_branch_is_continuous():
    a, am, b, bm = 0.62, 0.41, 0.55, 0.30
    lpa, lpb = jnp.log(a - am), jnp.log(b - bm)
    lo = float(frank_log_cell_prob(a, am, lpa, b, bm, lpb, 9e-6))
    hi = float(frank_log_cell_prob(a, am, lpa, b, bm, lpb, 1.1e-5))
    np.testing.assert_allclose(lo, hi, rtol=1e-6)


def test_deep_tail_returns_a_finite_log_not_a_floor():
    # The truck observation from src/rpbnb_tmb.cpp:258-264: a cell whose
    # probability is 1e-300. The log form must not clip it to -690.776.
    val = float(frank_log_cell_prob(1.0, 1.0 - 1e-300, -687.8,
                                    0.6, 0.4, jnp.log(0.2), 5.0))
    assert np.isfinite(val)
    assert val < -600.0


def test_gradient_is_finite_in_the_deep_tail():
    def f(th):
        return frank_log_cell_prob(1.0, 1.0 - 1e-300, -687.8,
                                   0.6, 0.4, jnp.log(0.2), th)
    assert np.isfinite(float(jax.grad(f)(5.0)))


def test_gradient_is_finite_where_the_unselected_expansion_is_singular():
    # Rule 2, measured 2026-08-23 rather than assumed. The near-independence
    # expansion's log1p argument is th*(1-a-am)*(1-b-bm)/2, which reaches
    # exactly -1 for th = +-2 against saturated corners. log1p(-1) is -inf,
    # and its VJP is g / (1 + x) = 0 / 0 = NaN -- so the branch that is NOT
    # selected poisoned the gradient of the one that was. Both cases below
    # returned nan before the argument was sanitised.
    def a_saturated(th):  # (1-a-am) = -1, (1-b-bm) = +1, arg = -th/2
        return frank_log_cell_prob(1.0, 1.0, -1e-300, 0.0, 0.0, -1e-300, th)

    def both_saturated(th):  # (1-a-am) = (1-b-bm) = -1, arg = +th/2
        return frank_log_cell_prob(1.0, 1.0, -700.0, 1.0, 1.0, -700.0, th)

    assert np.isfinite(float(jax.grad(a_saturated)(2.0)))
    assert np.isfinite(float(jax.grad(both_saturated)(-2.0)))


def test_gradient_is_finite_across_the_whole_theta_range():
    # The link caps |theta| at FRANK_THETA_MAX = 35, and both signs reach
    # different branches of the kernel: th > 0 takes pos_branch, th < 0 takes
    # the three-regime neg_branch, and |th| < 1e-5 takes the expansion.
    def f(th):
        return frank_log_cell_prob(0.62, 0.41, jnp.log(0.21),
                                   0.55, 0.30, jnp.log(0.25), th)
    for th in (-35.0, -2.0, -1e-5, -9e-6, 0.0, 9e-6, 1e-5, 2.0, 35.0):
        assert np.isfinite(float(f(th))), th
        assert np.isfinite(float(jax.grad(f)(th))), th
