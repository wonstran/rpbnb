import jax.numpy as jnp
import numpy as np
from scipy.stats import nbinom, poisson

from rpbnb_jax.margins import log_dnbinom2, log_dpois


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
