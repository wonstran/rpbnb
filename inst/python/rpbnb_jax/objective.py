"""The SML negative log-likelihood, mirroring
objective_function::operator() in src/rpbnb_tmb.cpp:951 with est_method = 0.

Only the closed-form-margin families live here so far: independence
(FAM_INDEP) and Famoye/Sarmanov (FAM_FAMOYE). The copula families need the
count CDF triples and their own cell-probability kernels, and are added in
later tasks; until then they raise rather than silently doing something else.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp, ndtri

from . import (DIST_LOGNORMAL, DIST_NORMAL, DIST_TRIANGULAR, DIST_UNIFORM,
               ETA_CEILING, FAM_FAMOYE, FAM_INDEP)
from .margins import log_dnbinom2, log_dpois

# Famoye's loop-invariant d = 1 - exp(-1) (src/rpbnb_tmb.cpp:1089).
# A plain Python float, not a jnp scalar: a module-scope jnp call would
# initialise a device at import time and would be bound to whatever dtype
# config happened to be live then.
#
# math.exp(-1.0) and the C++ exp(Type(-1.0)) are the same libm call on the
# same double. Checked, rather than assumed: this constant compares identical
# to R's own 1 - exp(-1) at 0.63212055882855767, and the R binary and the
# compiled template share the toolchain's libm.
FAMOYE_D = 1.0 - math.exp(-1.0)


def _int_list(v):
    """Coerce a possibly-scalar index/code vector to a list of Python ints.

    reticulate hands a length-1 R vector to Python as a bare scalar, so
    data$rand_idx1 for a single random coefficient arrives as `1`, not `[1]`.
    Iterating that raises TypeError. np.atleast_1d is the normalisation.
    """
    return [int(x) for x in np.atleast_1d(v)]


def _u_to_base(u, dist_code):
    """Inverse CDF of the mixing distribution. dist_code is a static int.

    Mirrors u_to_base() (src/rpbnb_tmb.cpp:1063).
    """
    if dist_code in (DIST_NORMAL, DIST_LOGNORMAL):
        return ndtri(u)
    if dist_code == DIST_TRIANGULAR:
        return jnp.where(u < 0.5, -1.0 + jnp.sqrt(2.0 * u),
                         1.0 - jnp.sqrt(2.0 * (1.0 - u)))
    return u  # DIST_UNIFORM


def _compute_dev(b, s, base, dist_code, sign_code):
    """Deviation added to the linear predictor. dist_code/sign_code static.

    Mirrors compute_dev() (src/rpbnb_tmb.cpp:1077).
    """
    if dist_code == DIST_NORMAL:
        return s * base
    if dist_code == DIST_LOGNORMAL:
        return sign_code * jnp.exp(b + s * base) - b
    if dist_code == DIST_UNIFORM:
        return s * (2.0 * base - 1.0)
    return s * base  # DIST_TRIANGULAR


def _deviations(beta, log_sd, rand_idx, Z, dist, sign, n_draws):
    """(R, q) matrix of per-draw deviations. Mirrors src/rpbnb_tmb.cpp:1114."""
    q = len(rand_idx)
    if q == 0:
        return jnp.zeros((n_draws, 0), dtype=jnp.float64)
    sd = jnp.exp(jnp.clip(log_sd, -20.0, 20.0))
    cols = []
    for j in range(q):  # q is small and dist/sign are static, so unroll
        base = _u_to_base(Z[:, j], dist[j])
        cols.append(_compute_dev(beta[rand_idx[j]], sd[j], base,
                                 dist[j], sign[j]))
    return jnp.stack(cols, axis=1)


def _eta(X, beta, rand_idx, dev, n_draws):
    """(n, R) linear predictors. Mirrors src/rpbnb_tmb.cpp:1159-1183.

    The per-coefficient accumulation is left as an explicit loop rather than
    collapsed into X[:, cols] @ dev.T so that the summation order matches the
    `eta1 += X1(i, col) * d` of the C++ term for term.
    """
    xb = X @ beta                                        # (n,)
    eta = jnp.broadcast_to(xb[:, None], (X.shape[0], n_draws))
    for j in range(len(rand_idx)):
        col = rand_idx[j]
        eta = eta + X[:, col][:, None] * dev[:, j][None, :]
    return eta


def _eta_floor(log_m_clamped, is_pois):
    """nb2_eta_floor(), src/rpbnb_tmb.cpp:1024-1031.

    dnbinom2 evaluates log(var - mu) = log(m*mu^2); that increment is lost to
    rounding once log(m) + log(mu) falls under about -36, so the floor on eta
    has to move with the estimated m.
    """
    if is_pois:
        return -35.0
    return -35.0 - jnp.minimum(log_m_clamped, 0.0)


def _log_margin(y, mu, m, is_pois):
    """src/rpbnb_tmb.cpp:1191-1196 / :1226-1231."""
    if is_pois:
        return log_dpois(y, mu)
    return log_dnbinom2(y, mu, m)


def _famoye_c(mu, m, is_pois):
    """src/rpbnb_tmb.cpp:1198-1203.

    stable_log1p() in the C++ is a Taylor shim for CppAD's missing log1p;
    jnp.log1p is the real thing, with correct derivatives (plan, "JAX builtins
    replace the stability shims").
    """
    if is_pois:
        return jnp.exp(-FAMOYE_D * mu)
    return jnp.exp(-jnp.log1p(FAMOYE_D * m * mu) / m)


def build_objective(data, layout, obs_chunk=256):
    """Return f(free_vec) -> (nll, grad) with grad as a numpy array.

    `data` is the list .build_tmb_data() produces, handed over by reticulate.

    obs_chunk is accepted and IGNORED here. It exists for interface stability
    with the copula path, which is the only one that materialises the
    (n_chunk, R, KMAX+1) count grid that chunking is for; the closed-form
    margins on this path never build it. See the plan's "Memory, and why
    obs_chunk exists".
    """
    del obs_chunk  # documented no-op on the closed-form-margin path

    family = int(data["family"])
    pois1 = bool(int(data["pois1"]))
    pois2 = bool(int(data["pois2"]))
    Y1 = jnp.asarray(data["Y1"], dtype=jnp.float64)
    Y2 = jnp.asarray(data["Y2"], dtype=jnp.float64)
    X1 = jnp.asarray(data["X1"], dtype=jnp.float64)
    X2 = jnp.asarray(data["X2"], dtype=jnp.float64)
    Z1 = jnp.asarray(np.atleast_2d(data["Z1"]), dtype=jnp.float64)
    Z2 = jnp.asarray(np.atleast_2d(data["Z2"]), dtype=jnp.float64)
    rand_idx1 = _int_list(data["rand_idx1"])
    rand_idx2 = _int_list(data["rand_idx2"])
    dist1 = _int_list(data["dist1"])
    dist2 = _int_list(data["dist2"])
    sign1 = _int_list(data["sign1"])
    sign2 = _int_list(data["sign2"])
    lamLo = float(data["lamLo"])
    lamHi = float(data["lamHi"])
    est_method = int(data.get("est_method", 0))
    if est_method != 0:
        raise NotImplementedError(
            "the JAX engine implements est_method = 0 (SML) only; "
            "the Laplace path stays with TMB")
    # R = Z1.rows() when there is at least one random coefficient, else 1
    # (src/rpbnb_tmb.cpp:996).
    n_draws = int(Z1.shape[0]) if (rand_idx1 or rand_idx2) else 1

    def nll(free_vec):
        p = layout.unpack(free_vec)
        log_m1 = jnp.clip(p["log_m1"], -20.0, 20.0)
        log_m2 = jnp.clip(p["log_m2"], -20.0, 20.0)
        m1 = jnp.exp(log_m1)
        m2 = jnp.exp(log_m2)

        dev1 = _deviations(p["beta1"], p["log_sd1"], rand_idx1, Z1,
                           dist1, sign1, n_draws)
        dev2 = _deviations(p["beta2"], p["log_sd2"], rand_idx2, Z2,
                           dist2, sign2, n_draws)
        eta1 = _eta(X1, p["beta1"], rand_idx1, dev1, n_draws)
        eta2 = _eta(X2, p["beta2"], rand_idx2, dev2, n_draws)
        mu1 = jnp.exp(jnp.clip(eta1, _eta_floor(log_m1, pois1), ETA_CEILING))
        mu2 = jnp.exp(jnp.clip(eta2, _eta_floor(log_m2, pois2), ETA_CEILING))

        y1 = Y1[:, None]
        y2 = Y2[:, None]
        lm1 = _log_margin(y1, mu1, m1, pois1)
        lm2 = _log_margin(y2, mu2, m2, pois2)

        if family == FAM_INDEP:
            log_draw = lm1 + lm2
        elif family == FAM_FAMOYE:
            # invlogit(), spelled the way TMB spells it (convenience.hpp:114).
            sig = 1.0 / (1.0 + jnp.exp(-p["z_dep"]))
            eps = 1e-6
            lam = lamLo + (lamHi - lamLo) * (eps + (1.0 - 2.0 * eps) * sig)
            c1 = _famoye_c(mu1, m1, pois1)
            c2 = _famoye_c(mu2, m2, pois2)
            dep = 1.0 + lam * (jnp.exp(-y1) - c1) * (jnp.exp(-y2) - c2)
            bad = dep <= 0.0
            # Double-where: log() must never see the non-positive value, even
            # though that branch is discarded, or jax.grad pushes NaN back
            # through it. src/rpbnb_tmb.cpp:1217-1224.
            #
            # The 1e10 term is a value-only barrier, exactly as in the C++:
            # `bad` is a step, so it contributes zero gradient on both sides.
            # A fit that lands on it is invalid, not converged.
            safe_dep = jnp.where(bad, 1e-300, dep)
            log_draw = (lm1 + lm2 + jnp.log(safe_dep)
                        - jnp.where(bad, 1e10, 0.0))
        else:
            raise NotImplementedError(f"family {family} lands in a later task")

        # log-sum-exp over draws minus log R. src/rpbnb_tmb.cpp:1317-1327.
        return -jnp.sum(logsumexp(log_draw, axis=1) - jnp.log(float(n_draws)))

    jitted = jax.jit(jax.value_and_grad(nll))

    def value_and_grad(free_vec):
        # Converted here rather than in R: reticulate has no py_to_r for a
        # jax.Array, and would hand back an opaque object reference.
        v, g = jitted(jnp.asarray(np.atleast_1d(free_vec),
                                  dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    value_and_grad.jitted = jitted
    return value_and_grad
