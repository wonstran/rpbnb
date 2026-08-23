import jax.numpy as jnp
import rpbnb_jax  # noqa: F401  (imported for its x64 side effect)


def test_x64_is_enabled():
    assert jnp.zeros(1).dtype == jnp.float64


def test_grad_is_float64():
    import jax
    g = jax.grad(lambda x: jnp.sum(x ** 2))(jnp.ones(3))
    assert g.dtype == jnp.float64
