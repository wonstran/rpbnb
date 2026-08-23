"""Marginal count distributions.

Closed-form log masses for the Famoye/independence path, and the log-space
CDF triple the copula families need. Mirrors src/rpbnb_tmb.cpp.
"""

import numpy as np

import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

# Floor applied to mu and r before any log(). Not cosmetic: log(0) is -inf,
# and ks * log_q then evaluates 0 * -inf = NaN at k = 0, poisoning the VALUE
# and not merely the gradient. Measured 2026-08-23: masking a -inf out of the
# logsumexp does not rescue it -- jnp.where and logsumexp(where=) both return
# a NaN gradient when the masked entry's -inf carries a dependence on the
# differentiated input. Only keeping the -inf out of the graph works.
_POS_FLOOR = 1e-300


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


def _static_kmax(kmax, y):
    """Validate kmax as a static Python int and check y against it.

    kmax sets an array dimension, so it cannot be a tracer; int() raises on
    one, which is the enforcement. The y check is the important half: an
    out-of-range count would otherwise read the grid at some other index and
    return a plausible wrong number, the failure mode this whole port exists
    to avoid. It runs only when y is concrete (it always is at
    objective-construction time); _triple_from_grid's mode="fill" is the
    backstop under trace.
    """
    kmax = int(kmax)
    if kmax < 0:
        raise ValueError(f"kmax must be >= 0, got {kmax}")
    try:
        y_max = int(np.max(np.asarray(y)))
        y_min = int(np.min(np.asarray(y)))
    except Exception:
        return kmax  # traced y; the caller guarantees the bound
    if y_min < 0:
        raise ValueError(f"y must be non-negative, got min {y_min}")
    if y_max > kmax:
        raise ValueError(
            f"kmax = {kmax} is below max(y) = {y_max}; the grid would not "
            "reach the observed count")
    return kmax


def _triple_from_grid(log_pmf_all, y, kmax):
    """Reduce a (..., kmax+1) log-mass grid to the (cdf_y, cdf_ym1, pmf_y)
    triple TMB's *_cdf_pair() returns, in log space.

    These three log values are the complete interface. Frank and Gaussian
    want linear CDFs and a linear mass, Clayton wants the logs directly;
    all four are an exp() away, so nothing further belongs in the return.

    Convention matches src/rpbnb_tmb.cpp:157 -- for y = 0 the "cdf_ym1" slot
    holds log P(Y = 0), not -inf; every caller gates on the observed count.
    Task 5's Clayton depends on it being finite, because under vmap every
    branch is evaluated for every observation.

    The batch shapes of y and of the grid are broadcast explicitly rather
    than left to trailing-axis broadcasting. Measured 2026-08-23 on the
    (n, R, kmax+1) grid the copula path actually builds, against y of shape
    (n, 1): `y[..., None]` is (n, 1, 1), so `ks <= y[..., None]` collapses to
    an (n, kmax+1) mask and logsumexp raises "Incompatible shapes for
    broadcasting: shapes=[(4, 7), (4, 3, 7), ()]". The draft form was tested
    only at rank 1, where the bug is invisible.
    """
    ks = jnp.arange(kmax + 1)
    y_arr = jnp.asarray(y)
    if not jnp.issubdtype(y_arr.dtype, jnp.integer):
        # Counts arrive from R as doubles; round rather than truncate so a
        # value stored as 4.999999 indexes 5.
        y_arr = jnp.round(y_arr).astype(jnp.int32)
    batch = jnp.broadcast_shapes(y_arr.shape, log_pmf_all.shape[:-1])
    grid = jnp.broadcast_to(log_pmf_all, batch + (kmax + 1,))
    idx = jnp.broadcast_to(y_arr[..., None], batch + (1,))

    log_cdf_y = logsumexp(grid, axis=-1, where=(ks <= idx))
    # y = 0 would leave this mask all-false. Measured: that yields a -inf
    # VALUE with a clean 0.0 gradient -- so the hazard is the value, not the
    # gradient, and Clayton's log_ratio() differencing that -inf against
    # log_pmf_y is the Inf - Inf the C++ header records at :196-197.
    # Selecting k = 0 alone gives the slot log P(Y = 0), per :157.
    mask_m1 = jnp.where(idx == 0, ks == 0, ks <= idx - 1)
    log_cdf_ym1 = logsumexp(grid, axis=-1, where=mask_m1)
    # mode="fill" makes a count above the grid return NaN rather than the
    # mass at some in-range index. Explicit because it must not depend on a
    # gather default.
    log_pmf_y = jnp.take_along_axis(grid, idx, axis=-1, mode="fill",
                                    fill_value=jnp.nan)[..., 0]
    return log_cdf_y, log_cdf_ym1, log_pmf_y


def nb2_cdf_triple(y, mu, r, kmax):
    """Log-space (log_cdf_y, log_cdf_ym1, log_pmf_y) for NB2.

    Vectorized replacement for nb2_cdf_pair() (src/rpbnb_tmb.cpp:160).

    `r` is the SIZE, not the dispersion: r = 1/m, where m is what
    log_dnbinom2() above takes. The C++ likewise passes r1/r2, not m.

    The per-k increment log(r + j - 1) - log(j) depends only on r, a scalar
    shared by every observation, so the cumsum runs once rather than once per
    observation. An array-valued r is accepted and gives a per-element cumsum,
    at (batch, kmax) memory -- correct, but not the intended use.

    Never seeded in linear space: P(Y = 0) = p^r underflows to exact 0 around
    mu = r = 2000, and a linear recursion multiplies that zero forward through
    every later term. src/rpbnb_tmb.cpp:128-135.
    """
    kmax = _static_kmax(kmax, y)
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _POS_FLOOR)
    r = jnp.maximum(jnp.asarray(r, dtype=jnp.float64), _POS_FLOOR)
    ks = jnp.arange(kmax + 1, dtype=jnp.float64)
    log_p = jnp.log(r) - jnp.log(r + mu)
    log_q = jnp.log(mu) - jnp.log(r + mu)
    js = jnp.arange(1, kmax + 1, dtype=jnp.float64)
    inc = jnp.log(r[..., None] + js - 1.0) - jnp.log(js)
    cum = jnp.concatenate(
        [jnp.zeros(inc.shape[:-1] + (1,), dtype=jnp.float64),
         jnp.cumsum(inc, axis=-1)], axis=-1)
    log_pmf_all = (r * log_p)[..., None] + ks * log_q[..., None] + cum
    return _triple_from_grid(log_pmf_all, y, kmax)


def pois_cdf_triple(y, mu, kmax):
    """Poisson analogue of nb2_cdf_triple (src/rpbnb_tmb.cpp:208).

    Replaces logging a linear-space ppois()/dpois(), which underflows both
    the CDF and the mass to an exact 0.0 when a random-coefficient draw
    pushes mu far from the observed count; Clayton then differences two
    -Inf logs and every free parameter's gradient goes NaN.
    src/rpbnb_tmb.cpp:189-202.
    """
    kmax = _static_kmax(kmax, y)
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _POS_FLOOR)
    ks = jnp.arange(kmax + 1, dtype=jnp.float64)
    cum = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.float64),
        jnp.cumsum(-jnp.log(jnp.arange(1, kmax + 1, dtype=jnp.float64))),
    ])
    log_pmf_all = (-mu)[..., None] + ks * jnp.log(mu)[..., None] + cum
    return _triple_from_grid(log_pmf_all, y, kmax)
