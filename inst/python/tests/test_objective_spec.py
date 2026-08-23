"""Construction-time coercion of the .build_tmb_data() list.

These rules, not the arithmetic, are what has actually broken: reticulate
collapses a length-1 R vector to a bare Python scalar, so every single
random-coefficient model hands `rand_idx1` over as an int rather than a
sequence.
"""

import numpy as np
import pytest

from rpbnb_jax import FAM_FAMOYE, FAM_FRANK, FAM_INDEP
from rpbnb_jax.objective import _spec_from_data, build_objective
from rpbnb_jax.packing import Layout


def _data(**over):
    """A 3-observation, 4-draw, one-random-coefficient model.

    Defaults spell the index/code fields the way reticulate delivers them for
    q = 1: bare scalars, not length-1 sequences.
    """
    d = dict(
        family=FAM_INDEP, pois1=0, pois2=0,
        Y1=np.array([0.0, 1.0, 2.0]), Y2=np.array([1.0, 0.0, 3.0]),
        X1=np.ones((3, 2)), X2=np.ones((3, 2)),
        Z1=np.full((4, 1), 0.5), Z2=np.full((4, 1), 0.25),
        rand_idx1=1, rand_idx2=1, dist1=0, dist2=0, sign1=1, sign2=1,
        lamLo=-0.9, lamHi=0.9, est_method=0,
    )
    d.update(over)
    return d


def test_scalar_index_and_code_fields_become_length_one_tuples():
    spec = _spec_from_data(_data())
    assert spec.margin1.rand_idx == (1,)
    assert spec.margin1.dist == (0,)
    assert spec.margin1.sign == (1,)
    assert spec.margin2.rand_idx == (1,)


def test_sequence_index_fields_survive_unchanged():
    spec = _spec_from_data(_data(
        rand_idx1=np.array([0, 1]), dist1=np.array([0, 3]),
        sign1=np.array([1, -1]), Z1=np.full((4, 2), 0.5)))
    assert spec.margin1.rand_idx == (0, 1)
    assert spec.margin1.dist == (0, 3)
    assert spec.margin1.sign == (1, -1)


def test_n_draws_is_Z_rows_when_any_coefficient_is_random():
    assert _spec_from_data(_data()).n_draws == 4


def test_n_draws_collapses_to_one_with_no_random_coefficients():
    # src/rpbnb_tmb.cpp:996 -- R is 1 regardless of how many rows Z has.
    spec = _spec_from_data(_data(
        rand_idx1=np.array([], dtype=int), rand_idx2=np.array([], dtype=int),
        dist1=np.array([], dtype=int), dist2=np.array([], dtype=int),
        sign1=np.array([], dtype=int), sign2=np.array([], dtype=int),
        Z1=np.zeros((4, 0)), Z2=np.zeros((4, 0))))
    assert spec.margin1.rand_idx == ()
    assert spec.n_draws == 1


def test_pois_flags_become_bools():
    spec = _spec_from_data(_data(pois1=1, pois2=0))
    assert spec.margin1.is_pois is True
    assert spec.margin2.is_pois is False


def test_one_dimensional_Z_raises_rather_than_transposing_the_draw_axis():
    with pytest.raises(ValueError, match="must be a 2-D"):
        _spec_from_data(_data(Z1=np.full(4, 0.5)))


def test_laplace_est_method_is_refused():
    with pytest.raises(NotImplementedError, match="est_method = 0"):
        _spec_from_data(_data(est_method=1))


def test_unsupported_family_raises_at_construction_not_at_first_call():
    lay = Layout(2, 2, 1, 1, np.zeros(9), np.arange(9))
    with pytest.raises(NotImplementedError, match="lands in a later task"):
        build_objective(_data(family=FAM_FRANK), lay)


def test_supported_families_build():
    lay = Layout(2, 2, 1, 1, np.zeros(9), np.arange(9))
    for family in (FAM_INDEP, FAM_FAMOYE):
        v, g = build_objective(_data(family=family), lay)(np.zeros(9))
        assert np.isfinite(v)
        assert g.shape == (9,)


def test_obs_chunk_is_validated_even_though_this_path_ignores_it():
    lay = Layout(2, 2, 1, 1, np.zeros(9), np.arange(9))
    with pytest.raises(ValueError, match="obs_chunk must be positive"):
        build_objective(_data(), lay, obs_chunk=0)
    with pytest.raises(ValueError):
        build_objective(_data(), lay, obs_chunk="banana")


def test_length_one_free_vector_is_refused_through_the_built_objective():
    # The Critical path end to end: R hands a scalar, np.atleast_1d makes it
    # (1,), Layout.unpack refuses it instead of broadcasting.
    lay = Layout(2, 2, 1, 1, np.zeros(9), np.arange(9))
    fg = build_objective(_data(), lay)
    with pytest.raises(ValueError, match=r"shape \(9,\), got \(1,\)"):
        fg(0.5)
