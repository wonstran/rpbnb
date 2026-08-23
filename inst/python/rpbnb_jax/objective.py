"""The SML negative log-likelihood, mirroring
objective_function::operator() in src/rpbnb_tmb.cpp:951 with est_method = 0.

Three families live here so far: independence (FAM_INDEP), Famoye/Sarmanov
(FAM_FAMOYE), and Frank (FAM_FRANK). Clayton and Gaussian need their own
cell-probability kernels and are added in later tasks.

Shape of the file, and why it is this shape at two families rather than at
five: the C++ carries its family dispatch as one branch chain inside the
observation loop, which is free there because the loop body is untaped
control flow. Here the equivalent chain lives inside a traced closure, and
the plan's remaining tasks add three more branches to it, each with its own
corner machinery. So the dispatch is a table of small adapters over a uniform
signature, `nll` stays a fixed dozen lines, and a new family is one entry in
`_LOG_DRAW`, one entry in `_dependence`, and its own kernel module.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp, ndtri

from . import (DIST_LOGNORMAL, DIST_NORMAL, DIST_TRIANGULAR, DIST_UNIFORM,
               ETA_CEILING, FAM_FAMOYE, FAM_FRANK, FAM_INDEP, FRANK_THETA_MAX)
from .frank import frank_log_cell_prob
from .margins import (_count_index, log_dnbinom2, log_dpois, nb2_cdf_triple,
                      pois_cdf_triple)

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


# --------------------------------------------------------------------------
# Construction-time data coercion
# --------------------------------------------------------------------------

class MarginSpec(NamedTuple):
    """One margin's data, coerced once at construction time.

    Y      (n,)    float counts, as the closed-form log masses want them
    y_int  (n,)    the same counts as grid INDICES, for the CDF triples
    X      (n, k)  design matrix
    Z      (R, q)  uniform draws, one column per random coefficient
    rand_idx       0-based columns of X carrying a random coefficient
    dist           DIST_* code per random coefficient
    sign           +-1 per random coefficient (read only by the lognormal form)
    is_pois        exact-Poisson margin rather than NB2

    Y and y_int are both stored because the two paths want different types
    and the counts are DATA: converting per evaluation would put the
    integrality check (margins.py:106-112) inside the traced closure, where
    it cannot run at all. The C++ makes the same split at
    src/rpbnb_tmb.cpp:1100-1110 -- Y1_int/Y2_int are cast once, outside the
    observation loop, "so this conversion costs nothing on the tape".
    """
    Y: jnp.ndarray
    y_int: jnp.ndarray
    X: jnp.ndarray
    Z: jnp.ndarray
    rand_idx: tuple
    dist: tuple
    sign: tuple
    is_pois: bool


class Spec(NamedTuple):
    """Everything build_objective() reads out of `data` before tracing.

    kmax is the count grid's top index, a static Python int: it sets an array
    dimension, so it can never become a traced quantity.
    """
    family: int
    margin1: MarginSpec
    margin2: MarginSpec
    lamLo: float
    lamHi: float
    n_draws: int
    kmax: int


def _int_tuple(v):
    """Coerce a possibly-scalar index/code vector to a tuple of Python ints.

    reticulate hands a length-1 R vector to Python as a bare scalar, so
    data$rand_idx1 for a single random coefficient arrives as `1`, not `[1]`.
    Iterating that raises TypeError. np.atleast_1d is the normalisation, and
    it is the coercion rather than the arithmetic that has already broken
    once, which is why _spec_from_data() is separately testable.
    """
    return tuple(int(x) for x in np.atleast_1d(v))


def _counts_as_indices(Y):
    """Validated integer counts, computed once at construction.

    Delegates to margins._count_index(), which is the single place the
    integrality (to 1e-8) and non-negativity contract lives; duplicating it
    here is what the count-index hole was. That function also checks the
    count against the grid top, which is not known yet at this point, so it
    is handed the counts' own maximum -- the real kmax is enforced by
    nb2_cdf_triple()/pois_cdf_triple() when the objective is traced, and a
    caller-supplied kmax below max(y) raises there rather than silently
    indexing off the grid.
    """
    concrete = np.asarray(Y)
    if not concrete.size:
        self_bound = 0
    else:
        # np.round mirrors _count_index's own rounding, so this bound cannot
        # reject a count the validation there would have accepted.
        self_bound = max(int(np.max(np.round(concrete))), 0)
    idx, _ = _count_index(Y, self_bound)
    return idx


def _margin_spec_from_data(data, suffix):
    """Build one MarginSpec from the `.build_tmb_data()` list."""
    Z = jnp.asarray(data["Z" + suffix], dtype=jnp.float64)
    if Z.ndim != 2:
        # Deliberately a hard error and not np.atleast_2d(). A 1-D length-R
        # draw vector would promote to shape (1, R), silently transposing the
        # draw dimension into the coefficient dimension and averaging over
        # one draw instead of R.
        raise ValueError(
            "Z{} must be a 2-D (R, q) matrix, got {} dimension(s) with "
            "shape {}".format(suffix, Z.ndim, tuple(Z.shape)))
    Y = jnp.asarray(data["Y" + suffix], dtype=jnp.float64)
    return MarginSpec(
        Y=Y,
        y_int=_counts_as_indices(Y),
        X=jnp.asarray(data["X" + suffix], dtype=jnp.float64),
        Z=Z,
        rand_idx=_int_tuple(data["rand_idx" + suffix]),
        dist=_int_tuple(data["dist" + suffix]),
        sign=_int_tuple(data["sign" + suffix]),
        is_pois=bool(int(data["pois" + suffix])),
    )


def _spec_from_data(data, kmax=None):
    """Coerce the `.build_tmb_data()` list into a Spec.

    Separated from build_objective() so the reticulate coercion rules -- the
    length-1 scalar collapse above all -- can be tested without constructing
    an objective.

    `kmax` defaults to max(max(Y1), max(Y2)), the smallest grid that reaches
    every observed count. It is exposed so a caller can widen it; narrowing
    it below an observed count is refused by the CDF triples.
    """
    est_method = int(data.get("est_method", 0))
    if est_method != 0:
        raise NotImplementedError(
            "the JAX engine implements est_method = 0 (SML) only; "
            "the Laplace path stays with TMB")

    margin1 = _margin_spec_from_data(data, "1")
    margin2 = _margin_spec_from_data(data, "2")
    # R = Z1.rows() when there is at least one random coefficient, else 1
    # (src/rpbnb_tmb.cpp:996).
    has_random = bool(margin1.rand_idx or margin2.rand_idx)
    n_draws = int(margin1.Z.shape[0]) if has_random else 1
    y_max = max(int(np.max(np.asarray(margin1.y_int))),
                int(np.max(np.asarray(margin2.y_int))))
    if kmax is None:
        kmax = y_max
    # int() rather than a bare assignment: kmax sets an array dimension, and
    # reticulate hands an R numeric over as a float, which jnp.arange would
    # accept and then size the grid from a float.
    kmax = int(kmax)
    # _count_index() checks this too, and cannot get the chance: measured
    # 2026-08-23, the counts reach nb2_cdf_triple() as TRACERS, because every
    # jnp op inside a jit-traced closure is staged out even when its operands
    # are constants. _count_index() then takes its tracer branch and skips
    # the bound. So a kmax below an observed count would fall through to
    # _triple_from_grid's mode="fill" backstop, whose NaN takes every free
    # parameter's gradient with it through the sum-over-observations
    # reduction (src/rpbnb_tmb.cpp:196-202). Checked here, on the concrete
    # data, where it is a construction error rather than a silent NaN.
    if kmax < y_max:
        raise ValueError(
            f"kmax = {kmax} is below max(y) = {y_max}; the count grid would "
            "not reach the observed count")
    return Spec(
        family=int(data["family"]),
        margin1=margin1,
        margin2=margin2,
        lamLo=float(data["lamLo"]),
        lamHi=float(data["lamHi"]),
        n_draws=n_draws,
        kmax=kmax,
    )


# --------------------------------------------------------------------------
# Per-draw margin evaluation
# --------------------------------------------------------------------------

class Margin(NamedTuple):
    """One margin evaluated at the current parameters.

    y      (n, 1)  counts, broadcast against the draw axis
    y_int  (n, 1)  the same counts as grid indices, for the CDF triples
    mu     (n, R)  clamped conditional means
    m      scalar  NB2 dispersion (ignored when is_pois)
    """
    y: jnp.ndarray
    y_int: jnp.ndarray
    mu: jnp.ndarray
    m: jnp.ndarray
    is_pois: bool


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


def _deviations(beta, log_sd, ms, n_draws):
    """(R, q) matrix of per-draw deviations. Mirrors src/rpbnb_tmb.cpp:1114."""
    q = len(ms.rand_idx)
    if q == 0:
        return jnp.zeros((n_draws, 0), dtype=jnp.float64)
    sd = jnp.exp(jnp.clip(log_sd, -20.0, 20.0))
    cols = []
    for j in range(q):  # q is small and dist/sign are static, so unroll
        base = _u_to_base(ms.Z[:, j], ms.dist[j])
        cols.append(_compute_dev(beta[ms.rand_idx[j]], sd[j], base,
                                 ms.dist[j], ms.sign[j]))
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
    has to move with the estimated m. A Poisson margin has no such increment
    and keeps the bare -35.
    """
    if is_pois:
        return -35.0
    return -35.0 - jnp.minimum(log_m_clamped, 0.0)


def _margin_at(ms, beta, log_sd, log_m_clamped, n_draws):
    """Evaluate one margin at the current parameters. src/rpbnb_tmb.cpp:1185."""
    dev = _deviations(beta, log_sd, ms, n_draws)
    eta = _eta(ms.X, beta, ms.rand_idx, dev, n_draws)
    mu = jnp.exp(jnp.clip(eta, _eta_floor(log_m_clamped, ms.is_pois),
                          ETA_CEILING))
    return Margin(y=ms.Y[:, None], y_int=ms.y_int[:, None], mu=mu,
                  m=jnp.exp(log_m_clamped), is_pois=ms.is_pois)


def _log_margin(mar):
    """log P(Y = y) for one margin. src/rpbnb_tmb.cpp:1191-1196 / :1226-1231."""
    if mar.is_pois:
        return log_dpois(mar.y, mar.mu)
    return log_dnbinom2(mar.y, mar.mu, mar.m)


def _margin_triple(mar, kmax):
    """Log-space (cdf_y, cdf_ym1, pmf_y) for one margin, over the count grid.

    Selects the exact-Poisson form off the same STATIC flag the C++ uses
    (src/rpbnb_tmb.cpp:1244-1253), so an NB2 margin never puts the Poisson
    recursion on the graph and vice versa.

    `1.0 / mar.m` is the NB2 SIZE r, and it is a scalar because log_m is one
    model-level parameter: nb2_cdf_triple() enforces that, since a
    per-observation r would turn its (kmax,) increment into (n, R, kmax) --
    266 doubles against 2.2 GB on the truck workload.
    """
    if mar.is_pois:
        return pois_cdf_triple(mar.y_int, mar.mu, kmax)
    return nb2_cdf_triple(mar.y_int, mar.mu, 1.0 / mar.m, kmax)


def _linear_corners(mar, kmax):
    """(F(y), F(y - 1), log P(Y = y)) for one margin, corners in LINEAR space.

    The C++'s two accumulators DISAGREE at y = 0, deliberately, and this
    function is that disagreement. nb2_cdf_pair()/pois_cdf_pair() open with
    `cdf_ym1 = Type(0)` and overwrite it only `if (y > 0)`
    (src/rpbnb_tmb.cpp:169/176 and :216/223), so the LINEAR corner is F(-1) =
    0 -- the true value. log_cdf_ym1 is instead left at log P(Y = 0), which
    is a convention for Clayton's benefit alone and is documented as
    unreadable at y = 0 (:157-159). Frank and Gaussian take the LINEAR
    corners, so exp(log_cdf_ym1) is the wrong quantity for every zero count.

    Measured 2026-08-23 on the 60-observation parity fixture, whose counts
    are Poisson with mean about 1.5 and so roughly a fifth zero: reading
    exp(log_cdf_ym1) anyway cost 0.36 nats at z_dep = 0.1 and 6.0 nats at
    z_dep = -1.5, and put the z_dep score at +3.39 against TMB's -0.14 --
    the wrong sign, not a tolerance question.

    F(-1) = 0 identically, so selecting the constant is the right derivative
    too: no score flows from a corner that cannot move. The mask is read off
    the observed counts, which is where the C++ takes y1_zero/y2_zero from
    and for the reason recorded at :443-453 -- a CDF that has underflowed to
    zero is a different thing from a count that is zero.
    """
    log_cdf, log_cdf_m1, log_pmf = _margin_triple(mar, kmax)
    return (jnp.exp(log_cdf),
            jnp.where(mar.y_int == 0, 0.0, jnp.exp(log_cdf_m1)),
            log_pmf)


# --------------------------------------------------------------------------
# Family dispatch
# --------------------------------------------------------------------------

def _dependence(z_dep, family, lamLo, lamHi):
    """Dependence parameter on its natural scale.

    The one place a family's link lives, mirroring the single pre-loop block
    at src/rpbnb_tmb.cpp:1043-1056. Gaussian's tanh and Clayton's clamped exp
    land here with their families in later tasks.
    """
    if family == FAM_INDEP:
        # The C++ leaves lam/theta/rho at 0 for FAM_INDEP and never reads
        # them; z_dep is pinned by R in that case anyway. Returned as a zero
        # rather than None so every adapter has the same signature.
        return jnp.zeros_like(z_dep)
    if family == FAM_FAMOYE:
        # invlogit(), spelled the way TMB spells it (convenience.hpp:114).
        sig = 1.0 / (1.0 + jnp.exp(-z_dep))
        eps = 1e-6
        return lamLo + (lamHi - lamLo) * (eps + (1.0 - 2.0 * eps) * sig)
    if family == FAM_FRANK:
        # A smooth bounded link keeps exp(-theta * u) finite while retaining
        # theta = 0 and unit derivative at independence
        # (src/rpbnb_tmb.cpp:1048-1051). theta = 0 is a removable singularity
        # of the copula, which is why R starts the fit at z_dep = 0.1 rather
        # than 0 (R/fit_rpbnb_tmb.R:358-363): at exactly 0 the dependence
        # score vanishes from TMB's recorded tape. Nothing here needs that
        # offset -- tanh is smooth through 0 and frank_log_cell_prob() has
        # its own |th| < 1e-5 expansion -- but the fits this engine is
        # compared against do start there.
        return FRANK_THETA_MAX * jnp.tanh(z_dep / FRANK_THETA_MAX)
    # Reachable only if a family is added to _LOG_DRAW without a link here.
    raise NotImplementedError(f"no dependence link for family {family}")


# Families _dependence() has a link for. Kept beside it so the two cannot
# drift, and checked at construction alongside _LOG_DRAW: without it, a
# family wired into _LOG_DRAW but missing a link here would raise on the
# first fn() call -- i.e. from inside stats::nlminb() -- which is the exact
# timing the family check was hoisted out of.
_DEPENDENCE_FAMILIES = frozenset({FAM_INDEP, FAM_FAMOYE, FAM_FRANK})


def _famoye_c(mu, m, is_pois):
    """src/rpbnb_tmb.cpp:1198-1203.

    stable_log1p() in the C++ is a Taylor shim for CppAD's missing log1p;
    jnp.log1p is the real thing, with correct derivatives (plan, "JAX builtins
    replace the stability shims").
    """
    if is_pois:
        return jnp.exp(-FAMOYE_D * mu)
    return jnp.exp(-jnp.log1p(FAMOYE_D * m * mu) / m)


def _log_draw_indep(mar1, mar2, dep, spec):
    """(n, R) log joint mass under independence. src/rpbnb_tmb.cpp:1225-1232.

    `dep` and `spec` are unused; they are part of the uniform adapter
    signature the dispatch table calls through.
    """
    return _log_margin(mar1) + _log_margin(mar2)


def _log_draw_famoye(mar1, mar2, lam, spec):
    """(n, R) log joint mass under Famoye/Sarmanov. src/rpbnb_tmb.cpp:1190-1224.

    `spec` is unused; lamLo/lamHi are already folded into `lam` by
    _dependence().
    """
    c1 = _famoye_c(mar1.mu, mar1.m, mar1.is_pois)
    c2 = _famoye_c(mar2.mu, mar2.m, mar2.is_pois)
    dep = 1.0 + lam * (jnp.exp(-mar1.y) - c1) * (jnp.exp(-mar2.y) - c2)
    bad = dep <= 0.0
    # Double-where: log() must never see the non-positive value, even though
    # that branch is discarded, or jax.grad pushes NaN back through it.
    # src/rpbnb_tmb.cpp:1217-1224.
    #
    # The 1e10 term is a value-only barrier, exactly as in the C++: `bad` is a
    # step, so it contributes zero gradient on both sides. A fit that lands on
    # it is invalid, not converged.
    safe_dep = jnp.where(bad, 1e-300, dep)
    return (_log_margin(mar1) + _log_margin(mar2) + jnp.log(safe_dep)
            - jnp.where(bad, 1e10, 0.0))


def _log_draw_frank(mar1, mar2, theta, spec):
    """(n, R) log cell probability under Frank. src/rpbnb_tmb.cpp:1242-1272.

    Frank is the one family that returns a LOG cell probability, and so the
    one family that does NOT pass through the caller's 1e-300 floor
    (:1260-1272). That floor is not underflow hygiene here: the truck data's
    `m1` boundary refit produces a genuine cell probability of 1.03e-300,
    three ulp above it, and clipping replaces the likelihood with the
    constant 690.776. Clayton and Gaussian return linear probabilities and
    will need the floor when they land; do not hoist one over this branch.

    The corners are LINEAR and the masses are LOGS, which is the split the
    kernel's header exists to explain: the masses are what must not pass
    through linear space, the corners are already O(1). See _linear_corners()
    for why exp(log_cdf_ym1) is not the F(y - 1) this wants.
    """
    a1, a1m, lpmf1 = _linear_corners(mar1, spec.kmax)
    b1, b1m, lpmf2 = _linear_corners(mar2, spec.kmax)
    return frank_log_cell_prob(a1, a1m, lpmf1, b1, b1m, lpmf2, theta)


# family code -> (mar1, mar2, dep, spec) -> (n, R) log joint mass.
# Clayton and Gaussian are added here by Tasks 5-6.
_LOG_DRAW = {
    FAM_INDEP: _log_draw_indep,
    FAM_FAMOYE: _log_draw_famoye,
    FAM_FRANK: _log_draw_frank,
}


# --------------------------------------------------------------------------
# Objective construction
# --------------------------------------------------------------------------

def build_objective(data, layout, kmax=None, obs_chunk=256):
    """Return f(free_vec) -> (nll, grad) with grad as a numpy array.

    `data` is the list .build_tmb_data() produces, handed over by reticulate.

    kmax is the top of the count grid the copula families sum over, and
    defaults to max(max(Y1), max(Y2)) -- the smallest grid that reaches every
    observed count. Coerced to a static Python int, because it sizes an array.

    obs_chunk is validated but STILL unused. Frank is the first family to
    materialise the (n, R, kmax+1) count grid that chunking exists to bound,
    so the interface is no longer merely notional -- but the fixtures are
    small (60 x 16 x 7 here) and the truck workload's 2.2 GB only appears at
    full scale. Wiring it into a jax.lax.map belongs with the benchmark that
    first runs out of memory. Validated rather than discarded because nothing
    lints inst/python, so a bad value would otherwise be accepted in silence.
    """
    obs_chunk = int(obs_chunk)
    if obs_chunk <= 0:
        raise ValueError(f"obs_chunk must be positive, got {obs_chunk}")

    spec = _spec_from_data(data, kmax)
    # Both halves are checked, because a family needs an entry in each: an
    # adapter in _LOG_DRAW and a link in _dependence(). At construction, not
    # on the first fn() call -- which would surface from inside
    # stats::nlminb(). Same timing as the est_method check above.
    if spec.family not in _LOG_DRAW or spec.family not in _DEPENDENCE_FAMILIES:
        supported = sorted(set(_LOG_DRAW) & _DEPENDENCE_FAMILIES)
        raise NotImplementedError(
            f"family {spec.family} lands in a later task; this build "
            f"supports {supported}")
    log_draw_fn = _LOG_DRAW[spec.family]

    def nll(free_vec):
        p = layout.unpack(free_vec)
        mar1 = _margin_at(spec.margin1, p["beta1"], p["log_sd1"],
                          jnp.clip(p["log_m1"], -20.0, 20.0), spec.n_draws)
        mar2 = _margin_at(spec.margin2, p["beta2"], p["log_sd2"],
                          jnp.clip(p["log_m2"], -20.0, 20.0), spec.n_draws)
        dep = _dependence(p["z_dep"], spec.family, spec.lamLo, spec.lamHi)
        log_draw = log_draw_fn(mar1, mar2, dep, spec)
        # log-sum-exp over draws minus log R. src/rpbnb_tmb.cpp:1317-1327.
        return -jnp.sum(logsumexp(log_draw, axis=1)
                        - jnp.log(float(spec.n_draws)))

    jitted = jax.jit(jax.value_and_grad(nll))

    def value_and_grad(free_vec):
        # Converted here rather than in R: reticulate has no py_to_r for a
        # jax.Array, and would hand back an opaque object reference.
        v, g = jitted(jnp.asarray(np.atleast_1d(free_vec),
                                  dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    return value_and_grad
