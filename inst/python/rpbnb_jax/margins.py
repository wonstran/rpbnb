"""Marginal count distributions.

Closed-form log masses for the Famoye/independence path, and the log-space
CDF triple the copula families need. Mirrors src/rpbnb_tmb.cpp.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

# np.asarray() raises one of these on a tracer, which is how the concrete-y
# guard below detects that it cannot run. Caught by name rather than as a bare
# Exception: the broad form would also swallow, say, a ragged y and disable
# the guard silently. TracerArrayConversionError is NOT a subclass of
# ConcretizationTypeError in JAX 0.11.1, so both are named.
_TRACER_ERRORS = (jax.errors.TracerArrayConversionError,
                  jax.errors.ConcretizationTypeError)

# Floor applied to mu and r before any log(). Not cosmetic: log(0) is -inf,
# and ks * log_q then evaluates 0 * -inf = NaN at k = 0, poisoning the VALUE
# and not merely the gradient. Measured 2026-08-23: masking a -inf out of the
# logsumexp does not rescue it -- jnp.where and logsumexp(where=) both return
# a NaN gradient when the masked entry's -inf carries a dependence on the
# differentiated input. Only keeping the -inf out of the graph works.
_POS_FLOOR = 1e-300

# Integrality tolerance for a count arriving as a double. Matches the 1e-8 in
# .check_counts() (R/data_prep.R:22-27), which is the upstream gate.
_COUNT_TOL = 1e-8


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


def _count_index(y, kmax):
    """Counts as grid indices, plus kmax as a static Python int.

    ONE conversion, returned and used for both the bound check and the
    gather. An earlier split -- int(np.max(y)) here, jnp.round(y) at the
    gather -- disagreed at a non-integral y, and the disagreement was silent:
    y = 30.7 against kmax = 30 truncated to 30 and passed the check, then
    rounded to 31 and indexed off the grid, so log_pmf_y came back NaN while
    log_cdf_y and log_cdf_ym1 stayed finite. On the low side int(-0.5) == 0
    passed the check and was then read as y = 0, returning log P(Y = 0) --
    a plausible wrong number. A NaN there takes every free parameter's
    gradient with it through the sum-over-observations reduction, which is
    the failure src/rpbnb_tmb.cpp:196-197 records.

    Rounding rather than rejecting a whole-number float is deliberate, and
    the integrality tolerance below mirrors .check_counts()
    (R/data_prep.R:22-27), which rejects a non-integer count to 1e-8 and
    returns as.integer(round(y)). The production path therefore hands over an
    R integer vector; doubles arrive only from the parity fixture, which
    passes as.numeric(Y1). Enforcing the same contract here is what closes
    the gap -- rounding ALONE does not, because round() is half-to-even, so
    y = -0.5 becomes -0.0 and would be read as a legitimate y = 0.

    kmax sets an array dimension, so it cannot be a tracer; int() raises on
    one, which is the enforcement.
    """
    kmax = int(kmax)
    if kmax < 0:
        raise ValueError(f"kmax must be >= 0, got {kmax}")
    y_arr = jnp.asarray(y)
    idx = y_arr if jnp.issubdtype(y_arr.dtype, jnp.integer) else \
        jnp.round(y_arr).astype(jnp.int32)
    try:
        concrete = np.asarray(y_arr)
    except _TRACER_ERRORS:
        # Traced y; the caller guarantees the bound and _triple_from_grid's
        # mode="fill" is the backstop. Concrete at object-construction time.
        return idx, kmax
    if not concrete.size:
        return idx, kmax
    if np.issubdtype(concrete.dtype, np.floating):
        off = np.abs(concrete - np.round(concrete))
        if np.any(off > _COUNT_TOL):
            bad = concrete[off > _COUNT_TOL]
            raise ValueError(
                f"y must be whole numbers to {_COUNT_TOL:g}; got e.g. "
                f"{bad.flat[0]!r}")
    rounded = np.round(concrete).astype(np.int64)
    y_min, y_max = int(rounded.min()), int(rounded.max())
    if y_min < 0:
        raise ValueError(f"y must be non-negative, got min {y_min}")
    if y_max > kmax:
        raise ValueError(
            f"kmax = {kmax} is below max(y) = {y_max}; the grid would "
            "not reach the observed count")
    return idx, kmax


def _triple_from_grid(log_pmf_all, y_idx, kmax):
    """Reduce a (..., kmax+1) log-mass grid to the (cdf_y, cdf_ym1, pmf_y)
    triple TMB's *_cdf_pair() returns, in log space. y_idx comes from
    _count_index(); this function does no conversion of its own.

    These three log values are the complete interface. Frank and Gaussian
    want the linear CDFs and a linear mass, Clayton wants the logs directly.
    Each of those is an exp() away, so nothing further belongs in the return.

    Convention matches src/rpbnb_tmb.cpp:157 -- for y = 0 the "cdf_ym1" slot
    holds log P(Y = 0), not -inf; every caller gates on the observed count.
    Task 5's Clayton depends on it being finite, because under vmap every
    branch is evaluated for every observation.

    The batch shapes are broadcast explicitly rather than left to
    trailing-axis broadcasting. Measured 2026-08-23, and narrower than it
    first looked: against the (n, R, kmax+1) grid the copula path builds,
    plain broadcasting ALREADY WORKS at the (n, 1) y the objective passes.
    It fails only for a RANK-1 y of shape (n,), where `y[..., None]` is
    (n, 1), the mask collapses to (n, kmax+1), and logsumexp raises
    "Incompatible shapes for broadcasting: shapes=[(4, 7), (4, 3, 7), ()]".
    So the explicit form is not a fix to the real path; it buys a clearer
    error on the (n,) case -- jnp.broadcast_shapes reports
    "shapes=[(4,), (4, 3)]" naming the two inputs, before any reduction.
    """
    ks = jnp.arange(kmax + 1)
    batch = jnp.broadcast_shapes(y_idx.shape, log_pmf_all.shape[:-1])
    grid = jnp.broadcast_to(log_pmf_all, batch + (kmax + 1,))
    # Padded to the grid's RANK with leading 1s, not widened to `batch`.
    # Measured on the truck shape (n=3487, R=300, kmax=266): widening doubles
    # peak temporaries, 8524 MiB against 4270 MiB, for bit-identical results.
    # XLA does not elide it.
    idx = y_idx.reshape((1,) * (len(batch) - y_idx.ndim) + y_idx.shape + (1,))

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

    The per-k increment log(r + j - 1) - log(j) depends only on r, so the
    cumsum runs once per evaluation rather than once per observation. r must
    therefore be a SCALAR, and is rejected otherwise: dispersion is a
    model-level parameter with no per-observation meaning here, and an (n, R)
    r would silently turn the (kmax,) increment into (n, R, kmax) -- 266
    doubles against 2.2 GB on the truck workload, an OOM that only appears at
    full scale.

    Never seeded in linear space: P(Y = 0) = p^r underflows to exact 0 around
    mu = r = 2000, and a linear recursion multiplies that zero forward through
    every later term. src/rpbnb_tmb.cpp:128-135.
    """
    idx, kmax = _count_index(y, kmax)
    if jnp.ndim(r) != 0:
        raise ValueError(
            f"r must be a scalar size (1/m), got shape {jnp.shape(r)}; a "
            "per-observation r would allocate a (batch, kmax) cumsum")
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _POS_FLOOR)
    r = jnp.maximum(jnp.asarray(r, dtype=jnp.float64), _POS_FLOOR)
    ks = jnp.arange(kmax + 1, dtype=jnp.float64)
    log_p = jnp.log(r) - jnp.log(r + mu)
    log_q = jnp.log(mu) - jnp.log(r + mu)
    js = jnp.arange(1, kmax + 1, dtype=jnp.float64)
    inc = jnp.log(r + js - 1.0) - jnp.log(js)
    cum = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(inc)])
    log_pmf_all = (r * log_p)[..., None] + ks * log_q[..., None] + cum
    return _triple_from_grid(log_pmf_all, idx, kmax)


def pois_cdf_triple(y, mu, kmax):
    """Poisson analogue of nb2_cdf_triple (src/rpbnb_tmb.cpp:208).

    Replaces logging a linear-space ppois()/dpois(), which underflows both
    the CDF and the mass to an exact 0.0 when a random-coefficient draw
    pushes mu far from the observed count; Clayton then differences two
    -Inf logs and every free parameter's gradient goes NaN.
    src/rpbnb_tmb.cpp:189-202.
    """
    idx, kmax = _count_index(y, kmax)
    mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), _POS_FLOOR)
    ks = jnp.arange(kmax + 1, dtype=jnp.float64)
    cum = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.float64),
        jnp.cumsum(-jnp.log(jnp.arange(1, kmax + 1, dtype=jnp.float64))),
    ])
    log_pmf_all = (-mu)[..., None] + ks * jnp.log(mu)[..., None] + cum
    return _triple_from_grid(log_pmf_all, idx, kmax)
