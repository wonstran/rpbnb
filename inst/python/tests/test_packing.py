"""Layout arity, the check that stands between an R scalar and a finite,
wrong objective."""

import jax.numpy as jnp
import numpy as np
import pytest

from rpbnb_jax.packing import Layout


def _layout(n_free=9):
    """The shape the parity fixture builds: k1=k2=2, q1=q2=1, so 9 slots."""
    return Layout(2, 2, 1, 1, np.zeros(9), np.arange(n_free))


def test_full_length_free_vector_unpacks():
    lay = _layout()
    p = lay.unpack(jnp.arange(9, dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(p["beta1"]), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(p["beta2"]), [2.0, 3.0])
    np.testing.assert_allclose(np.asarray(p["log_sd1"]), [4.0])
    np.testing.assert_allclose(np.asarray(p["log_sd2"]), [5.0])
    assert float(p["log_m1"]) == 6.0
    assert float(p["log_m2"]) == 7.0
    assert float(p["z_dep"]) == 8.0


def test_length_one_free_vector_raises_rather_than_broadcasting():
    # The silent case: .at[idx].set() would broadcast the scalar across all
    # nine free coordinates and return a finite objective at the wrong point.
    lay = _layout()
    with pytest.raises(ValueError, match=r"shape \(9,\), got \(1,\)"):
        lay.unpack(jnp.asarray([0.5]))


def test_short_free_vector_raises():
    lay = _layout()
    with pytest.raises(ValueError, match=r"shape \(9,\), got \(3,\)"):
        lay.unpack(jnp.asarray([0.5, 0.6, 0.7]))


def test_scalar_free_vector_raises():
    lay = _layout()
    with pytest.raises(ValueError, match=r"shape \(9,\), got \(\)"):
        lay.unpack(jnp.asarray(0.5))


def test_pinned_coordinates_keep_their_template_value():
    # free_idx omits slot 8 (z_dep), the family < 0 case.
    lay = Layout(2, 2, 1, 1, np.full(9, -7.0), np.arange(8))
    p = lay.unpack(jnp.zeros(8))
    assert float(p["z_dep"]) == -7.0
    assert lay.n_free == 8


def test_template_length_is_checked_against_the_dimensions():
    with pytest.raises(ValueError, match="k1\\+k2\\+q1\\+q2\\+3"):
        Layout(2, 2, 1, 1, np.zeros(8), np.arange(8))
