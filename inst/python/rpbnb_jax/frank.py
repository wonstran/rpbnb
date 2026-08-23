"""Frank's log cell probability.

Line-for-line port of frank_log_cell_prob() (src/rpbnb_tmb.cpp:305-380). Read
that function's header (:231-304) before changing anything here: the
telescoped form, the log-space masses, and the identity used for M each
repair a specific measured failure of the naive second difference.

Three of those failures, in the header's own words:

* The naive second difference C(a,b) - C(a',b) - C(a,b') + C(a',b') returns
  pure rounding noise from about y = 26 and exactly zero above about y = 40
  at the truck fit's own starting values, where the true probabilities are
  1e-13 and 1e-20 (:246-252).
* Returning the LOG, with the marginal masses arriving as logs, is a separate
  repair. The truck data's `m1` boundary refit carries a genuine cell
  probability of 1.03e-300 at observation 2230 -- three ulp above the caller's
  1e-300 floor -- and clipping it replaced the likelihood with the constant
  690.776, whose spurious curvature (-12,181 at h = 1e-2 against a true
  0.1928) cost Laplace its positive-definite inner Hessian (:254-285).
* M is computed from an identity rather than as 1 + A(u)B(v)/D, because each
  factor is exp(-th * C) -- 6.3e-16 at the FRANK_THETA_MAX cap, under three
  ulp of 1 -- and recovering that from two O(1) quantities costs 5.6%
  relative error there (:287-299).

stable_expm1/stable_log1p in the C++ (:81-97) are Taylor shims for primitives
CppAD lacks; jnp.expm1/jnp.log1p are the real thing, with correct
derivatives.

Every jnp.maximum(..., 1e-300) below is one of the C++'s vmax() floors, and
each is load-bearing for the same reason there and here: both branches of a
CondExp/where are evaluated, so the unselected log() is handed its argument
regardless. Flooring the ARGUMENT is what keeps the unselected branch finite;
masking the result afterwards does not (plan, "The real NaN source is an
input-dependent -inf, not the mask").
"""

import jax.numpy as jnp

# The near-independence half-width. |th| below this selects the expansion at
# the bottom of this file; safe_th replaces th by this magnitude everywhere
# else, so the regular branch never divides by a th that has reached zero.
# src/rpbnb_tmb.cpp:308-309.
_TH_EPS = 1e-5

# Floor on every argument that reaches a log() on either side of a select.
_POS_FLOOR = 1e-300


def frank_log_cell_prob(a, am, log_pmf_a, b, bm, log_pmf_b, th):
    """log P((a', a] x (b', b]) under Frank's copula with parameter `th`.

    a/am are F1(y) and F1(y - 1) in LINEAR space, log_pmf_a is log P(Y1 = y);
    likewise b/bm/log_pmf_b for the second margin. `th` is the dependence
    parameter after the bounded link, so |th| <= FRANK_THETA_MAX = 35.

    Returns the LOG of the cell probability. The caller must not floor it --
    see the module docstring and src/rpbnb_tmb.cpp:1260-1272.
    """
    signed_eps = jnp.where(th >= 0.0, _TH_EPS, -_TH_EPS)
    safe_th = jnp.where(jnp.abs(th) < _TH_EPS, signed_eps, th)
    abs_th = jnp.abs(safe_th)
    log_abs_th = jnp.log(abs_th)

    def log_abs_delta(um, log_pmf):
        """log |A(u) - A(u')| = -th u' + log |expm1(-th * pmf)|.

        The mass enters as its LOGARITHM. x = |th| * pmf underflows to an
        exact 0 for a mass below about 1e-309; log x = log|th| + log_pmf never
        does, and for x under the switch log|expm1| and log x agree to under
        half an ulp anyway. src/rpbnb_tmb.cpp:314-327.
        """
        log_x = log_abs_th + log_pmf
        x = jnp.exp(log_x)
        x_safe = jnp.maximum(x, _POS_FLOOR)
        exact = jnp.where(safe_th > 0.0,
                          jnp.log(-jnp.expm1(-x_safe)),
                          jnp.log(jnp.expm1(x_safe)))
        return -safe_th * um + jnp.where(x < 1e-8, log_x, exact)

    def log_M(u, v):
        """log(1 + A(u) B(v) / D) by the identity in the header (:287-299).

        Both numerator terms carry the sign of the denominator for either
        sign of th, so the quotient is positive BY CONSTRUCTION.
        src/rpbnb_tmb.cpp:330-335.
        """
        t1 = jnp.exp(-safe_th * u) * (-jnp.expm1(-safe_th * v))
        t2 = jnp.exp(-safe_th * v) * (-jnp.expm1(-safe_th * (1.0 - v)))
        return jnp.log((t1 + t2) / (-jnp.expm1(-safe_th)))

    # log |dA * dB / (D * M)|. The quotient's SIGN is -sign(th) throughout:
    # dA and dB each carry -sign(th), so their product is positive, M is
    # positive, and D = expm1(-th) carries -sign(th). src/rpbnb_tmb.cpp:337-341.
    L = (log_abs_delta(am, log_pmf_a) + log_abs_delta(bm, log_pmf_b)
         - jnp.log(jnp.abs(jnp.expm1(-safe_th)))
         - log_M(am, b) - log_M(a, bm))

    # th > 0: ratio = -exp(L), p = -log1p(-exp(L)) / th, and L < 0 is what
    # keeps the cell probability finite. For L below the switch,
    # -log1p(-exp(L)) is exp(L) to well under an ulp, so log p is L itself.
    # L runs to -1e15 here (a Poisson mass against mu at the eta ceiling), so
    # the unselected log() would otherwise be handed exp(L) = 0 and return
    # -inf; the floor is what keeps it finite. src/rpbnb_tmb.cpp:343-358.
    L_neg = jnp.minimum(L, -1e-15)
    pos_branch = jnp.where(
        L_neg < -30.0,
        L_neg,
        jnp.log(jnp.maximum(-jnp.log1p(-jnp.exp(L_neg)), _POS_FLOOR)))

    # th < 0: ratio = +exp(L), p = log1p(exp(L)) / |th|. Three regimes, since
    # L runs from far below zero (an ordinary tail cell) to about +35 (both
    # margins saturated against a strongly negative theta), where log1p(exp(L))
    # is L and exp(L) alone would be the only thing at risk of overflowing.
    # src/rpbnb_tmb.cpp:359-368.
    L_cap = jnp.minimum(L, 30.0)
    neg_branch = jnp.where(
        L > 30.0,
        jnp.log(jnp.maximum(L, _POS_FLOOR)),
        jnp.where(L < -30.0,
                  L,
                  jnp.log(jnp.maximum(jnp.log1p(jnp.exp(L_cap)),
                                      _POS_FLOOR))))

    regular = jnp.where(safe_th > 0.0, pos_branch, neg_branch) - log_abs_th

    # Second difference of the near-independence expansion the naive form
    # used, C(u,v) ~ u v + th u v (1-u) (1-v) / 2, which telescopes to this in
    # closed form and so carries no cancellation either. |th| < 1e-5 bounds
    # the log1p argument by 1e-5 in magnitude on the branch that is SELECTED,
    # so it cannot approach -1 there. src/rpbnb_tmb.cpp:372-377.
    #
    # OFF that branch it can equal -1 exactly, and that is the one place this
    # port needs a double-`where` the C++ does not spell out. Both corner
    # factors are bounded by 1 in magnitude, so the argument covers
    # [-|th|/2, |th|/2] and hits -1 at |th| = 2 whenever a margin's CDF pair
    # has saturated (1 - a - am = -1 at a = am = 1, an ordinary deep-tail
    # cell). log1p(-1) is -inf and its VJP is g / (1 + x) = 0 / 0 = NaN, so
    # the UNSELECTED expansion returned a NaN gradient for the `regular`
    # branch. Measured 2026-08-23 at th = +-2; the un-sanitised form gave nan
    # where the value itself was a perfectly ordinary -1399.16.
    #
    # Sanitised with the same predicate that selects, per Rule 2: on the
    # selected branch the argument is untouched, and off it log1p(0) = 0
    # differentiates to a clean zero.
    near_arg = th * (1.0 - a - am) * (1.0 - b - bm) / 2.0
    near = jnp.abs(th) < _TH_EPS
    near_independence = (log_pmf_a + log_pmf_b
                         + jnp.log1p(jnp.where(near, near_arg, 0.0)))

    return jnp.where(near, near_independence, regular)
