"""Marginal count distributions.

Closed-form log masses for the Famoye/independence path, and (from Task 3
onward) the log-space CDF triple the copula families need. Mirrors
src/rpbnb_tmb.cpp.
"""

import jax.numpy as jnp
from jax.scipy.special import gammaln


def log_dnbinom2(y, mu, m):
    """log dnbinom2(y, mu, mu + m*mu^2), i.e. NB2 with size r = 1/m.

    TMB reaches the same quantity by a longer route: dnbinom2 forwards to
    dnbinom_robust with log_var_minus_mu = log(m*mu^2), which sets
    logit_p = -log(m*mu) and size = exp(log_mu + logit_p) = 1/m. dnbinom_logit
    then opens its accumulator at `ans = size * log_p` -- that is where the
    `r * log_p` below comes from -- and, for y != 0, adds
    -lbeta(size, y+1) - log(size+y) + y*log(1-p).

    The identity -lbeta(r, y+1) - log(r+y) = lgamma(y+r) - lgamma(r) -
    lgamma(y+1) turns those three into the three gammaln terms below, and
    log(1-p) = log_q. All five terms accounted for; the two forms differ only
    in rounding, at the 1e-16 relative level.
    """
    r = 1.0 / m
    log_p = jnp.log(r) - jnp.log(r + mu)   # log(r / (r + mu))
    log_q = jnp.log(mu) - jnp.log(r + mu)  # log(mu / (r + mu))
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * log_p + y * log_q)


def log_dpois(y, mu):
    """log dpois(y, mu). Same three terms as TMB's dpois (lgamma.hpp:144).

    TMB writes them as -lambda + x*log(lambda) - lgamma(x+1); the first two
    are swapped here. That is a reordering of one commutative addition, not a
    reassociation, so the result is bit-identical rather than merely equal.
    """
    return y * jnp.log(mu) - mu - gammaln(y + 1.0)
