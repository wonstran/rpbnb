import jax.numpy as jnp
import rpbnb_jax  # noqa: F401  (imported for its x64 side effect)


def test_x64_is_enabled():
    assert jnp.zeros(1).dtype == jnp.float64


def test_grad_is_float64():
    import jax
    g = jax.grad(lambda x: jnp.sum(x ** 2))(jnp.ones(3))
    assert g.dtype == jnp.float64


def test_dynamic_range_reaches_the_1e300_floor():
    # The design's floors live at 1e-300; under float32 this collapses to 0.
    assert float(jnp.asarray(1e-300)) > 0.0


def test_all_names_exist():
    assert set(rpbnb_jax.__all__) <= set(dir(rpbnb_jax))
