// src/rpbnb_tmb.cpp — TMB template for the rpbnb TMB engine, plus the
// package's SINGLE R_init_rpbnb (at the bottom of this file).
//
// TMB.hpp has no include guard and (without WITH_LIBTMB) DEFINES non-inline
// extern "C" functions.  It must appear in EXACTLY ONE translation unit of
// this DLL.  Never include <Rcpp.h> here, and never include <TMB.hpp> in
// RcppExports.cpp, halton_parallel.cpp, or copula_parallel.cpp.
//
// TMB_LIB_INIT is deliberately NOT defined: that macro makes tmb_core.hpp emit
// its own R_init_ plus a calldef table, which would collide with the Rcpp side.
// Leaving it undefined still exposes TMB_CALLDEFS and TMB_CCALLABLES for the
// custom init below.  Nothing injects it for a LinkingTo: TMB package --
// -DTMB_LIB_INIT=... comes only from TMB::compile(), which is for standalone
// models, and $(TMB_CXXFLAGS) is an undefined make variable.
#include <TMB.hpp>
#include <R_ext/Rdynload.h>
#include <cmath>

// Family codes
#define FAM_INDEP     -1
#define FAM_FAMOYE     0
#define FAM_FRANK      1
#define FAM_GAUSSIAN   2
#define FAM_CLAYTON    3

// Distribution codes
#define DIST_NORMAL     0
#define DIST_LOGNORMAL  1
#define DIST_UNIFORM    2
#define DIST_TRIANGULAR 3

// Ceiling of the bounded Frank link.  exp(-theta * u) must stay finite, so
// theta is squashed into (-FRANK_THETA_MAX, FRANK_THETA_MAX).  This caps
// attainable Frank dependence at Kendall's tau of about 0.891; R reports a
// boundary warning when an estimate is pinned there.  Must match
// FRANK_THETA_MAX in R/utilities.R.
#define FRANK_THETA_MAX 35.0

// 16-point Gauss-Legendre, positive half of the symmetric node set. Applied
// per panel by gaussian_cell_prob(), which splits its interval into five.
static const double gauss_x16[8] = {
  0.09501250983763769, 0.28160355077925908, 0.45801677765722731,
  0.61787624440264388, 0.75540440835500322, 0.86563120238783220,
  0.94457502307323249, 0.98940093499165027
};
static const double gauss_w16[8] = {
  0.18945061045506834, 0.18260341504492425, 0.16915651939500245,
  0.14959598881657588, 0.12462897125553447, 0.09515851168249304,
  0.06225352393864833, 0.02715245941175411
};
// Panel geometry, in units of the conditional SD divided by |rho| -- the width
// of the transition the integrand makes at each edge. GAUSS_EDGE brackets each
// edge, and is measured rather than guessed: 4 with 16 nodes holds the
// worst-case relative error to 7e-7 out to |rho| = 0.99999.
//
// There was a companion GAUSS_WINDOW that also CLIPPED the interval to the
// region where the integrand is not negligible. That is deleted rather than
// retuned; see gaussian_cell_prob() for why no finite value of it is safe.
#define GAUSS_EDGE 4.0

// Cut points on the STANDARD NORMAL's own scale, merged into the panel layout
// alongside the edge brackets.  The brackets resolve the transition; nothing
// resolved phi(z) itself, so at |rho| >= 0.99 -- where the brackets collapse to
// near-zero width and leave two very wide outer panels -- a 16-point rule was
// stepping across phi's peak.  Over the truck cell grid that cost 15.3 nats at
// rho = 0.9977, 15.4 at 0.999 and 36.6 at -0.999; with these it is 0.94, 1.28
// and 3.11.
//
// Two points, not three.  A cut at z = 0 as well changes NONE of those numbers
// to two decimals -- 16 nodes already resolve phi across [-2, 2] -- and each
// cut is a panel, which the timings show costs a flat 20% of the Gaussian
// Laplace path.  Subdividing the outer panels instead moves nothing either.
// The one variant that does help is {-3, -1, 1, 3}, worth a few nats more at
// mu = (10.5, 1.32) for a fourth panel; not taken, because at that size the
// two independent references disagree by about a nat themselves.
#define GAUSS_PHI_N 2
static const double gauss_phi_cuts[GAUSS_PHI_N] = {-2.0, 2.0};

// CppAD in the supported TMB toolchain does not overload std::expm1/log1p.
// These series-backed equivalents retain precision and AD derivatives near 0.
template<class Type>
Type stable_expm1(Type x) {
  Type x2 = x * x;
  Type series = x + x2 / Type(2) + x2 * x / Type(6) +
    x2 * x2 / Type(24);
  Type regular = exp(x) - Type(1);
  return CppAD::CondExpLt(fabs(x), Type(1e-4), series, regular);
}

template<class Type>
Type stable_log1p(Type x) {
  Type x2 = x * x;
  Type series = x - x2 / Type(2) + x2 * x / Type(3) -
    x2 * x2 / Type(4);
  Type regular = log(Type(1) + x);
  return CppAD::CondExpLt(fabs(x), Type(1e-4), series, regular);
}

// NB2 distribution function by direct summation of the mass function, writing
// F(y), F(y - 1) and P(Y = y) from one recursion because the discrete-copula
// likelihood needs all three.
//
// P(Y = y) is returned in its own right rather than left to the caller as
// F(y) - F(y - 1). In the far tail both CDFs have saturated at 1 and their
// difference is exactly zero in double precision, while the recursion still
// holds the mass to full relative precision.
//
// The textbook route is F(y) = pbeta(r / (r + mu), r, y + 1), and that is what
// this template used until the Laplace estimator exposed it.  TMB's pbeta()
// wraps TOMS 708, whose branches truncate a shape parameter to its integer part
// and reassign the remainder as a constant.  The VALUE stays accurate; the
// derivatives taken through those branches do not.  Two failures follow:
//
//   * the first derivative with respect to shape1 collapses to exactly zero at
//     some ordinary arguments -- including r = 2, which is where this package's
//     own default dispersion start (m = 0.5) puts it -- so the score for
//     log_m1/log_m2 was silently wrong there;
//   * third derivatives are NaN over wide regions of ordinary parameter values,
//     and the Laplace outer gradient differentiates the joint negative
//     log-likelihood three times.  Every copula family therefore failed under
//     method = "laplace" with "inner newton optimization failed during gradient
//     calculation", one or two nlminb steps in.
//
// Summing the mass function is exact and differentiable to every order.  Cost
// is O(y) per margin per evaluation, against one atomic call; on the truck
// workload that is about 12 extra terms per observation.
//
// Seeding the recursion with the linear-space P(Y = 0) is unsound: that term
// underflows to exact 0 whenever r * log(p) drops below about -745 (e.g.
// mu = r = 2000, an ordinary low-dispersion/high-mean region -- p^r =
// 0.5^2000), and once the seed is zero every later term stays zero too,
// because each step only ever multiplies the previous one. The recursion
// then reports P(Y = y) = 0 even where the true mass near the mode is
// perfectly representable (0.0063 at mu = r = y = 2000), handing the
// optimizer a flat, wrong objective instead of the real likelihood.
//
// Tracking the same recursion in log space avoids this: log P(Y = 0) =
// r * log(p) is an ordinary finite double even when P(Y = 0) itself
// underflows, and each step adds a log increment that is well-conditioned
// wherever the true mass is well-conditioned, regardless of how small the
// k = 0 term was. The running cdf is accumulated with a log-sum-exp so it
// never has to pass through the linear-space representation of a term
// until the final result -- which only underflows to 0 when the true
// probability truly is negligible.
template<class Type>
Type log_add_exp(Type a, Type b) {
  Type hi = CppAD::CondExpGt(a, b, a, b);
  Type lo = CppAD::CondExpGt(a, b, b, a);
  return hi + stable_log1p(exp(lo - hi));
}

// The log-space accumulators are returned alongside the linear ones because
// Clayton needs them. Its cell probability is built from a^-theta, and with
// theta reaching 4.85e8 that quantity only stays finite in logs; exp()ing the
// CDF first and taking the log again would also lose every cell whose CDF has
// underflowed, which is exactly the regime this recursion exists to keep.
// log_cdf_ym1 is the log of P(Y <= y - 1) for y > 0 and is left at log P(Y = 0)
// for y = 0, where the caller must not use it -- there is no representable log
// of zero to return, and every caller selects that case on the observed count.
template<class Type>
void nb2_cdf_pair(int y, Type mu, Type r,
                  Type &cdf_y, Type &cdf_ym1, Type &pmf_y,
                  Type &log_cdf_y, Type &log_cdf_ym1, Type &log_pmf_y) {
  Type log_p = log(r) - log(r + mu);
  Type log_q = log(mu) - log(r + mu);
  Type log_term = r * log_p;  // log P(Y = 0)
  Type log_cum = log_term;
  Type lcm1 = log_term;
  cdf_ym1 = Type(0);
  for (int k = 1; k <= y; k++) {
    lcm1 = log_cum;
    // log P(Y = k) = log P(Y = k - 1) + log(r + k - 1) - log(k) + log(q)
    log_term += log(r + Type(k - 1)) - log(Type(k)) + log_q;
    log_cum = log_add_exp(log_cum, log_term);
  }
  if (y > 0) cdf_ym1 = exp(lcm1);
  cdf_y = exp(log_cum);
  pmf_y = exp(log_term);
  log_cdf_y = log_cum;
  log_cdf_ym1 = lcm1;
  log_pmf_y = log_term;
}

// Poisson analog of nb2_cdf_pair() above, for the exact m = 0 branch
// (poisson_1/poisson_2 = TRUE): same log-space log_add_exp recursion, same
// return convention (log_cdf_ym1 left at log P(Y = 0) for y = 0, unread by
// every caller, which all gate on the observed count instead).
//
// This replaces computing ppois()/dpois() in LINEAR space and logging the
// result afterward, which was the previous implementation and is exactly the
// failure nb2_cdf_pair() above was written to avoid: for mu far enough from y
// -- a random-coefficient draw that pushes mu to, say, 1e12 against a small
// observed count, ordinary here since most counts in count data are small --
// ppois()/dpois() underflow to an exact linear-space 0.0 before log() ever
// runs, so BOTH the CDF-at-(y-1) and the PMF-at-y can come back as -Inf at
// once. clayton_cell_prob()'s log_ratio() then computes their difference
// (log_pmf_u - log_um), an Inf - Inf subtraction, which is NaN by
// construction and poisons the whole taped objective (every free parameter's
// gradient becomes NaN, not just the ones near that one observation, because
// NaN propagates through the sum-of-observations reduction whatever it
// touches). Observed on the truck data's `m1` boundary LR test under a
// Kimeldorf copula: the very first outer gradient evaluation came back NaN.
// Accumulating in log space the same way nb2_cdf_pair() does means the
// linear-space cdf_y/cdf_ym1/pmf_y returned here (exp() of the log
// accumulators) are never the ones a NaN could come from -- they underflow to
// an honest 0.0 without ever standing in for a genuine -Inf log the way a
// direct ppois()/dpois() call would.
template<class Type>
void pois_cdf_pair(int y, Type mu,
                   Type &cdf_y, Type &cdf_ym1, Type &pmf_y,
                   Type &log_cdf_y, Type &log_cdf_ym1, Type &log_pmf_y) {
  Type log_mu = log(mu);
  Type log_term = -mu;  // log P(Y = 0)
  Type log_cum = log_term;
  Type lcm1 = log_term;
  cdf_ym1 = Type(0);
  for (int k = 1; k <= y; k++) {
    lcm1 = log_cum;
    // log P(Y = k) = log P(Y = k - 1) + log(mu) - log(k)
    log_term += log_mu - log(Type(k));
    log_cum = log_add_exp(log_cum, log_term);
  }
  if (y > 0) cdf_ym1 = exp(lcm1);
  cdf_y = exp(log_cum);
  pmf_y = exp(log_term);
  log_cdf_y = log_cum;
  log_cdf_ym1 = lcm1;
  log_pmf_y = log_term;
}

// Frank's joint LOG probability of the cell (a', a] x (b', b], evaluated as ONE
// log1p rather than as the second difference
// C(a,b) - C(a',b) - C(a,b') + C(a',b').
//
// With C(u,v) = -log1p(A(u) B(v) / D) / th, A(u) = expm1(-th u) and
// D = expm1(-th), the four logarithms telescope exactly:
//
//   p     = -log1p( dA * dB / (D * M) ) / th
//   dA    = A(a) - A(a') = exp(-th a') * expm1(-th * pmf_a)
//   M     = (1 + A(a') B(b) / D) * (1 + A(a) B(b') / D)
//
// so every cancelling difference becomes an expm1/log1p of a small argument,
// and dA is built from the marginal mass pmf_a directly instead of from two
// CDFs that have both saturated at 1.
//
// The naive second difference is unusable in the tail. At the truck fit's own
// starting values -- all slopes zero, so mu = 1 against counts running to 266
// -- it returns pure rounding noise for counts from about 26 and exactly zero
// above about 40, where the true probabilities are 1e-13 and 1e-20. Under SML
// that noise only corrupts the objective; under Laplace it puts negative
// curvature into the inner Hessian (161 of 27,896 latent rows on the truck
// data), and TMB's inner Newton cannot take even its first step.
//
// RETURNING THE LOG, AND TAKING THE MARGINAL MASSES AS LOGS, is what the two
// paragraphs above do NOT buy on their own, and it is a separate failure.  The
// telescoped form is cancellation-free but still LINEAR: it forms dA from
// pmf_a, and the caller floors the result at 1e-300 before logging it.  A cell
// probability of 1e-300 is not an underflow artefact on this data -- it is an
// ordinary observation.  The `m1` boundary LR test refits the truck model with
// margin 1 forced Poisson, and observation 2230 (y = 125 against mu = 0.193)
// then carries log P(Y1 = 125) = -687.8, giving a cell probability of 1.03e-300
// -- three ulp above the floor.  One step of TMB's inner Newton moves mu enough
// to cross it, and on the far side the objective is not the likelihood but the
// constant -log(1e-300) = 690.776.
//
// That clip is a kink, and Laplace differentiates the joint twice.  A centred
// second difference of -log p in log(mu1) across it returns -12,181 (h = 1e-2)
// and -94,836 (h = 1e-3) where the true curvature is mu1 = 0.193: the inner
// Hessian picks up negative curvature of order 1e5 on that one latent row, is
// no longer positive definite, and TMB reports
//
//   Not improving much - will try early exit...PD hess?: FALSE
//   Error in newton(...): Newton drop out: Too many failed attempts.
//   Error in ff(x, order = 1): inner newton optimization failed during
//     gradient calculation
//
// followed by a NaN outer gradient, an nlminb "false convergence (8)", and an
// NA row for `m1` in the boundary-test table.  Accumulating in log space -- the
// masses arrive as log_pmf_a/log_pmf_b, |dA| is formed as -th*a' + log|expm1|
// with the log mass carried through, and the return value never passes through
// the linear representation of the cell -- keeps the true curvature (0.1928 at
// every step size) and the floor never binds.  This is the same argument
// pois_cdf_pair() and clayton_cell_prob()'s log_ratio() already make one level
// down; the caller's 1e-300 floor was re-imposing at the top exactly what they
// remove underneath.
//
// M IS COMPUTED FROM AN IDENTITY, not as written above.  Each factor is
// exp(-th * C(.,.)), which for saturated corners is exp(-th): 2.5e-9 at the
// theta = 19.8 this test reaches, and 6.3e-16 at the FRANK_THETA_MAX cap of 35.
// Forming it as 1 + A B / D recovers that from two O(1) quantities -- 5.6%
// relative error at the cap, where 6.3e-16 is under three ulp of 1.  Writing
// p = exp(-th u), q = exp(-th v) and expanding D + A(u) B(v) gives
//
//   1 + A(u) B(v) / D = [ p * (1 - q) + q * (1 - exp(-th (1 - v))) ]
//                       / (1 - exp(-th))
//
// whose two numerator terms carry the sign of the denominator for either sign
// of th, so the quotient is positive BY CONSTRUCTION and every term is accurate
// to full relative precision.
//
// This is Frank-specific: it relies on C being a log of a bilinear form in
// A(u) and B(v). The Gaussian and Clayton branches below still return a linear
// cell probability and keep the caller's 1e-300 floor, so a Laplace fit whose
// true cell probabilities reach 1e-300 remains exposed there.
template<class Type>
Type frank_log_cell_prob(Type a, Type am, Type log_pmf_a,
                         Type b, Type bm, Type log_pmf_b, Type th) {
  Type signed_eps = CppAD::CondExpGe(th, Type(0), Type(1e-5), Type(-1e-5));
  Type safe_th = CppAD::CondExpLt(fabs(th), Type(1e-5), signed_eps, th);
  Type abs_th = fabs(safe_th);
  Type log_abs_th = log(abs_th);
  auto vmax = [](Type u, Type v) { return CppAD::CondExpGt(u, v, u, v); };

  // log |A(u) - A(u')| = -th u' + log |expm1(-th * pmf)|, with the mass
  // entering as its LOGARITHM.  x = |th| * pmf underflows to an exact 0 for a
  // mass below about 1e-309; log x = log|th| + log_pmf never does, and for x
  // under the switch log|expm1| and log x agree to under half an ulp anyway.
  // Both CondExp branches are evaluated, so the exact branch gets a floored
  // argument even where it is not the one selected.
  auto log_abs_delta = [&](Type um, Type log_pmf) -> Type {
    Type log_x = log_abs_th + log_pmf;
    Type x = exp(log_x);
    Type x_safe = vmax(x, Type(1e-300));
    Type exact = CppAD::CondExpGt(safe_th, Type(0),
                                  log(-stable_expm1(-x_safe)),
                                  log(stable_expm1(x_safe)));
    return -safe_th * um + CppAD::CondExpLt(x, Type(1e-8), log_x, exact);
  };

  // log(1 + A(u) B(v) / D) by the identity in the header.
  auto log_M = [&](Type u, Type v) -> Type {
    Type t1 = exp(-safe_th * u) * (-stable_expm1(-safe_th * v));
    Type t2 = exp(-safe_th * v) * (-stable_expm1(-safe_th * (Type(1) - v)));
    return log((t1 + t2) / (-stable_expm1(-safe_th)));
  };

  // log |dA * dB / (D * M)|.  The quotient's SIGN is -sign(th) throughout:
  // dA and dB each carry -sign(th), so their product is positive, M is
  // positive, and D = expm1(-th) carries -sign(th).
  Type L = log_abs_delta(am, log_pmf_a) + log_abs_delta(bm, log_pmf_b) -
    log(fabs(stable_expm1(-safe_th))) - log_M(am, b) - log_M(a, bm);

  // th > 0: ratio = -exp(L), p = -log1p(-exp(L)) / th, and L < 0 is what
  // keeps the cell probability finite -- the same admissibility the linear
  // form enforced by clamping ratio at -1 + 1e-15.  For L below the switch,
  // -log1p(-exp(L)) is exp(L) to well under an ulp, so log p is L itself and
  // exp(L) is never formed at a magnitude that underflows the log.
  //
  // Both CondExp branches are evaluated whichever one is selected, and L runs
  // to -1e15 here (a Poisson mass against mu at the eta ceiling), so the
  // unselected log() would otherwise be handed exp(L) = 0 and return -Inf.
  // Flooring its argument keeps every value on the tape finite.
  Type L_neg = CppAD::CondExpGt(L, Type(-1e-15), Type(-1e-15), L);
  Type pos_branch = CppAD::CondExpLt(
    L_neg, Type(-30),
    L_neg,
    log(vmax(-stable_log1p(-exp(L_neg)), Type(1e-300)))
  );
  // th < 0: ratio = +exp(L), p = log1p(exp(L)) / |th|.  Three regimes, since
  // L runs from far below zero (an ordinary tail cell) to about +35 (both
  // margins saturated against a strongly negative theta), where log1p(exp(L))
  // is L and exp(L) alone would be the only thing at risk of overflowing.
  Type L_cap = CppAD::CondExpGt(L, Type(30), Type(30), L);
  Type neg_branch = CppAD::CondExpGt(
    L, Type(30), log(vmax(L, Type(1e-300))),
    CppAD::CondExpLt(L, Type(-30), L,
                     log(vmax(stable_log1p(exp(L_cap)), Type(1e-300))))
  );
  Type regular = CppAD::CondExpGt(safe_th, Type(0), pos_branch, neg_branch) -
    log_abs_th;

  // Second difference of the near-independence expansion the naive form used,
  // C(u,v) ~ u v + th u v (1-u) (1-v) / 2, which telescopes to this in closed
  // form and so carries no cancellation either.  |th| < 1e-5 bounds the log1p
  // argument by 1e-5 in magnitude, so it cannot approach -1.
  Type near_independence = log_pmf_a + log_pmf_b +
    stable_log1p(th * (Type(1) - a - am) * (Type(1) - b - bm) / Type(2));

  return CppAD::CondExpLt(fabs(th), Type(1e-5), near_independence, regular);
}

// Clayton's joint probability of the cell (a', a] x (b', b], rearranged so
// that no step subtracts two nearly-equal numbers.
//
// Clayton has the same tail-cancellation defect the naive second difference
// gave Frank, and it is not hypothetical.  On the truck data at the fit's own
// starting values -- all slopes zero, so mu = 1 against counts running to 266
// -- C(a,b) - C(a',b) - C(a,b') + C(a',b') returns a NON-POSITIVE probability
// for 243 of 3,487 observations and a strictly NEGATIVE one for 11, where the
// true cell probabilities run down to 1e-136.  Under SML those only corrupt
// the objective (each clamped cell contributes the 1e-300 floor, about 691
// nats).  Under Laplace the negative ones put negative curvature into the
// inner Hessian, and TMB's inner Newton cannot take even its first step:
// method = "laplace" with copula("kimeldorf") failed outright with "inner
// newton optimization failed during gradient calculation".
//
// Write C(u,v) = s^k with s = 1 + A(u) + A(v), A(u) = u^-th - 1 and
// k = -1/th.  Factoring out the corner s00 = 1 + A(a) + A(b) and setting
// x = dA / s00, y = dB / s00 with dA = A(a') - A(a) >= 0 leaves
//
//   p  = s00^k * [ exp(u2) * expm1(u1 - u2) + ex * expm1(u1) ]
//   ex = expm1(k * log1p(x))
//   u1 = k * log1p(y / (1 + x))
//   u2 = k * log1p(y)
//   u1 - u2 = k * log1p(-x*y / ((1 + x)(1 + y)))
//
// The last identity is what removes the cancellation.  The naive difference
// has to recover an O(xy) second-order term by subtracting four O(1)
// quantities; here that term is written in closed form.  Because k < 0 and
// x, y >= 0, every factor above is sign-determined and both bracket terms are
// positive, so p is positive BY CONSTRUCTION rather than by clamping -- which
// is exactly what the inner Newton needs.
//
// x and y are carried as LOGARITHMS throughout.  th = exp(z_dep) with z_dep
// clamped to [-20, 20], so th reaches 4.85e8 and a^-th = exp(-th log a)
// overflows a double for any a bounded away from 1 -- log x, by contrast, is
// an ordinary number near 1e8.  Every quantity the result needs is a function
// of log1p(x), log1p(y) and log(1 + x + y), each of which follows from log x
// and log y by log-sum-exp, so x and y themselves are never formed.
//
// An earlier version instead capped x and y at 1e15, on the reasoning that
// (1 + x)^k is indistinguishable from x^k beyond that point.  That reasoning
// was wrong, and the error was not small.  The cap is applied BEFORE the
// result raises the ratio to the power k = -1/th, and k * log x is materially
// different from k * log(1e15) whenever th is large: capping loses
// |k| * (log x - 34.5) in the exponent.  For the symmetric cell y1 = y2 = 1 at
// mu = 1, m = 0.5, the exact probability at z_dep = 5 is 0.29077 and the
// capped form returned 0.15036.  Worse, Clayton tends to the comonotonic
// copula as th grows, so this cell must approach P(Y = 1) = 0.29630; the cap
// instead drove it to 2.4e-10 at z_dep = 20, reversing the likelihood's
// behaviour across the whole strong-dependence region that R/inference.R
// reports as interior.
//
// The one place the cap did real work was keeping 1 - xy/((1+x)(1+y)) off the
// rounding floor.  That is no longer needed either, because the quantity has
// an exact closed form with no subtraction at all:
//
//   1 - xy/((1+x)(1+y)) = (1 + x + y) / ((1+x)(1+y))
//
// so log1p(-xy/((1+x)(1+y))) = log(1+x+y) - log1p(x) - log1p(y), which is what
// the code computes.
//
// The y = 0 branches are separate because A(0) is infinite: there the cell is
// bounded by the axis and the second difference degenerates to a first
// difference.  They are selected from the OBSERVED COUNTS, passed in as
// y1_zero/y2_zero.  They previously tested asDouble(am) == 0.0, which is a
// different thing in two ways: asDouble() resolves when the tape is built, so
// the branch was frozen at whatever the starting values implied; and am is a
// parameter-dependent CDF that underflows to exactly zero for positive counts
// in ordinary regions (P(Y <= 0) at mu = r = 2000 is 0.5^2000).  A tape built
// at such a start kept the axis formula permanently, and the same parameter
// vector then scored differently depending on where its tape had been made --
// 2.256 against 1.592 nats on a single observation.
template<class Type>
Type clayton_cell_prob(Type log_a, Type log_am, Type log_pmf_a,
                       Type log_b, Type log_bm, Type log_pmf_b,
                       Type th, bool y1_zero, bool y2_zero) {
  auto vmax = [](Type u, Type v) { return CppAD::CondExpGt(u, v, u, v); };
  Type k = Type(-1.0) / th;

  // log s00, s00 = a^-th + b^-th - 1 >= 1 (a, b <= 1 keeps both terms >= 1).
  Type La = -th * log_a;
  Type Lb = -th * log_b;
  Type M = vmax(La, Lb);
  Type log_s00 = M + log(exp(La - M) + exp(Lb - M) - exp(-M));
  Type C00 = exp(k * log_s00);  // C(a, b)

  // log x for x = (u'^-th - u^-th) / s00 > 0.
  //
  // The ratio log(u/u') is built from the MARGINAL MASS, not from
  // log_u - log_um.  That difference is the whole reason nb2_cdf_pair()
  // returns a mass at all: once the CDF saturates, log F(y) and log F(y - 1)
  // are equal to the last bit and their difference is exactly zero, which
  // sends log() to -infinity and every derivative through it to NaN.  On the
  // truck margins at mu = 1 that happens for all 197 counts from y = 70 up,
  // covering 50 of the 3,487 observations, and the difference is already 32%
  // wrong at y = 69 before it collapses.  The mass route stays finite and
  // accurate there: at y = 266 it gives 1.45e-125 where the difference gives
  // 0.  This is the same cancellation the CDF-difference form of this
  // function was written to remove, and computing the ratio from two log CDFs
  // reintroduced it in log space.
  //
  // dA = u'^-th - u^-th = u'^-th * (1 - exp(-th * log(u/u'))), so with
  // LR = log(u/u') = log1p(pmf_u / u') the bracket is -expm1(-th * LR).
  auto log_ratio = [&](Type log_um, Type log_pmf_u) -> Type {
    Type t = log_pmf_u - log_um;
    Type LR = log_add_exp(Type(0), t);        // log(u / u') > 0
    // log LR without forming LR when the mass is far below the CDF: there
    // LR -> exp(t), so log LR -> t.  Keeps the small-S branch below finite
    // even if LR itself underflows.
    Type log_LR = CppAD::CondExpLt(t, Type(-30), t, log(vmax(LR, Type(1e-300))));
    Type S = th * LR;
    Type log_S = log(th) + log_LR;
    // Both CondExp branches are evaluated, so the exact branch gets a floored
    // argument; for S below the switch, -expm1(-S)/S differs from 1 by less
    // than half an ulp, so log_S is the accurate form anyway.
    Type S_safe = vmax(S, Type(1e-300));
    Type log_bracket = CppAD::CondExpLt(
      S, Type(1e-8), log_S, log(-stable_expm1(-S_safe))
    );
    return -th * log_um + log_bracket - log_s00;
  };

  if (y1_zero && y2_zero) return C00;

  if (y2_zero) {                      // cell bounded by the y2 axis
    Type L1x = log_add_exp(Type(0), log_ratio(log_am, log_pmf_a));
    return -C00 * stable_expm1(k * L1x);
  }
  Type log_y = log_ratio(log_bm, log_pmf_b);
  Type L1y = log_add_exp(Type(0), log_y);
  if (y1_zero) return -C00 * stable_expm1(k * L1y);

  Type log_x = log_ratio(log_am, log_pmf_a);
  Type L1x = log_add_exp(Type(0), log_x);

  // u1 = k * log1p(y / (1 + x)) and du = k * log1p(-w),
  // w = xy / ((1+x)(1+y)).  Both are written so that neither end of the range
  // cancels, which needs more care than it looks.
  //
  // Writing u1 as k * (log(1+x+y) - log1p(x)) is exact on paper and useless in
  // arithmetic: for x >> y both logs are O(x) and their difference is O(y), so
  // y is lost entirely.  The log-sum-exp form below never forms that
  // difference.
  //
  // du is the term that carries the cell.  Expanding the bracket for small
  // x, y gives (1+x+y)^k - (1+x)^k - (1+y)^k + 1 = k(k-1)xy + O(3), and the
  // two terms of the return are -k*xy and k^2*xy -- the SAME order.  Dropping
  // either one is not a rounding error but a factor of k/(k-1).  Computing
  // du as log(1+x+y) - log1p(x) - log1p(y) does exactly that: all three are
  // O(x+y) and their difference is O(xy), so it underflows to zero and the
  // cell comes back as C00*k^2*xy instead of C00*k(k-1)*xy -- a factor of 3.7
  // at theta = e, and it gets worse as theta falls.
  //
  // So the small-w side is computed from log1p(-w) directly, with log w built
  // additively (no cancellation), and only the large-w side -- where x and y
  // are big and the three logs are well separated -- uses the difference.
  Type u1 = k * log_add_exp(Type(0), log_y - L1x);
  Type u2 = k * L1y;
  Type log_w = log_x + log_y - L1x - L1y;      // log of xy/((1+x)(1+y)) < 0
  Type Lsum = log_add_exp(L1x, log_y);         // log(1 + x + y)
  // Both CondExp branches are evaluated, so the log1p branch gets an argument
  // floored away from -1 even when it is not the one selected.
  Type w_safe = exp(CppAD::CondExpGt(log_w, Type(-0.7), Type(-0.7), log_w));
  Type du = k * CppAD::CondExpLt(log_w, Type(-0.7),
                                 stable_log1p(-w_safe),
                                 Lsum - L1x - L1y);
  Type ex = stable_expm1(k * L1x);
  return C00 * (exp(u2) * stable_expm1(du) + ex * stable_expm1(u1));
}

// Gaussian's joint probability of the cell (a', a] x (b', b], evaluated as ONE
// strip integral instead of the second difference of four corner CDFs.
//
// Gaussian was the last family still taking the naive second difference, and
// it cancels there exactly as Frank and Clayton did.  On the truck data at the
// fit's own starting values (mu = 1 against counts running to 266) the
// four-corner form returns a non-positive cell probability for 457 to 600 of
// the 3,487 observations depending on rho, of which 212 to 355 are strictly
// NEGATIVE -- the negative-curvature source that stops TMB's inner Newton
// under method = "laplace".  The strip integral below leaves 245 floored
// cells and no negative ones, which lowers the objective at those starting
// values from about 334,000-432,000 to 195,737-203,352 over
// z_dep in {-0.55, 0.2, 0.7}.
//
// Conditioning the bivariate normal on the first margin gives
//
//   P = int_{q(a')}^{q(a)} phi(z) [ Phi((q(b) - rho z)/s)
//                                 - Phi((q(b') - rho z)/s) ] dz
//
// with s = sqrt(1 - rho^2).  The integrand is a product of non-negative
// factors (q(b) >= q(b') makes the bracket non-negative) and the limits are
// ordered, so P is non-negative BY CONSTRUCTION rather than by clamping.
//
// Integrating in z rather than in the probability variable matters: the same
// rule applied to int_{a'}^{a} ... dt has to evaluate qnorm() near the ends of
// the interval, where it is singular, and Gauss-Legendre then converges only
// as O(1/n) -- 1.5e-4 relative error at 20 points, against 1.5e-10 for the
// form below.  The probability-space version buys an exactly-known interval
// width (the marginal mass) at the cost of that singularity; it is not worth
// the trade here.
//
// The inner difference is taken on whichever tail is not saturated, since
// Phi(A) - Phi(B) cancels when both arguments are large and positive -- and
// they are: the truck data's second margin reaches counts of 47 against
// mu = 1, putting b at 1 - 1e-22.  The branch is applied to the ARGUMENTS
// rather than to the two pnorm() results: CppAD::CondExp evaluates both of its
// value branches, so selecting after the fact would put four pnorm() calls on
// the tape per node instead of two.
//
// A single fixed rule across the whole quantile interval is not enough, and
// this is the one part of the function that is not a matter of taste.  The
// integrand is a smoothed indicator of [q(b')/rho, q(b)/rho] whose edges have
// width s/|rho|.  As |rho| -> 1 that width collapses while the interval does
// not, and the nodes simply step over the plateau.  At rho = 0.9999 on an
// ordinary observation -- y = (0, 17), mu = (0.193, 17.6), a cell whose true
// probability is 0.0401, nowhere near any rounding floor -- one 20-point rule
// over [-7.94, 1.06] returns 1.7e-25 and loses 53.8 log-likelihood units on
// that observation alone.  Making the integrand non-negative had not made it
// accurate.  This is inside the supported domain: R/inference.R reports
// rho = 0.9999 as an interior estimate, not a boundary.
//
// So the interval is split at +/- GAUSS_EDGE around each edge centre, giving
// five panels each smooth on its own scale.  The cut points are forced
// monotone by a running maximum, which both keeps every panel width
// non-negative (a negative width would subtract mass) and lets panels
// collapse to nothing when the plateau is narrower than the brackets.  The
// union of the panels is exactly [q(a'), q(a)] regardless of how many
// collapse.
//
// What the panels must NOT do is shorten that interval.  An earlier version
// also clipped it to [clo - 10 e, chi + 10 e], the region where the bracket is
// not negligible, on the reasoning that the integrand is worthless outside.
// It is not worthless, it is merely small, and the clip did not make it small
// -- it made it ZERO, because a clipped-away interval left every panel with a
// width the monotone pass then floored at nothing.  The cells whose quantile
// interval falls outside that region are the DISCORDANT ones: y1 far into its
// right tail while y2 sits near its median, under strong positive rho.  Rare,
// not impossible, and the truck data has hundreds -- 74 to 266 of its 439
// distinct cells once |rho| passes 0.9, and 1 even at rho = -0.5, with true
// probabilities running down to 1e-30.  Each was returned as 0, floored by the
// caller to 1e-300, and so contributed 690.78 nats instead of 69: over the
// truck cell grid that is 617 nats of pure quadrature error at rho = -0.5,
// 43,298 at rho = 0.9 and 119,733 at rho = -0.9.
//
// The gradient mattered more than the value.  A clamp is flat, so those
// observations contributed exactly zero score.  Under SML the draw average
// dilutes that; under method = "laplace" there is one evaluation point per
// observation, so such an observation is a constant in the joint objective --
// it informs neither the inner Newton nor the outer score, the Hessian at the
// optimum loses rank, and sdreport() returns NaN standard errors for the
// affected parameters.  That was the whole of the "Gaussian + Laplace gives
// NA" report.
//
// No larger window fixes this, which is why the constant is gone rather than
// raised: the clip bites whenever the interval and the transition region are
// disjoint, and widening the region only moves which cells qualify.  Keeping
// the full interval costs nothing at moderate dependence -- the outer panels
// are then wide but carry a monotone integrand with no edge in them, and the
// worst-case relative error over the truck cell grid is 8e-10 out to
// |rho| = 0.9, against 1.0 (total loss) for the clipped form.
//
// It was not free at the top of the range, which is what gauss_phi_cuts is
// for: with only the edge brackets, |rho| >= 0.99 leaves two very wide outer
// panels resolved on the transition scale but not on phi's own, and the truck
// grid lost 1.6 nats at rho = 0.99, 15.3 at 0.9977, 15.4 at 0.999 and 36.6 at
// -0.999.  That is not a corner of the domain: a truck subset fits at
// rho = 0.9977.  See the constant for the measured effect.
//
// One residual was left here deliberately.  At rho = 0.9999 with mu = 1 the
// grid lost about 5 nats, and no panel layout moved it -- 7, 10 and 12 panels
// all gave the same number, so it was the margins reaching the old 1e-15
// quantile clamp, not the quadrature.  That clamp is gone; see the closing
// paragraph and gauss_corner_quantiles().
//
// Dividing by rho needs |rho| bounded away from zero.  The floor only affects
// where the cut points land, and at |rho| = 1e-6 they land 1e6 quantiles away
// from any interval, so every one of them clamps to an endpoint and a single
// panel spans [q(a'), q(a)] -- which is right, because at rho = 0 the bracket
// is constant in z and one panel resolves it exactly.
//
// The limitation this comment used to close on -- that a saturated marginal
// CDF put q(a) and q(a') on the same point and returned 0, "which would take a
// separately accumulated survival function" to fix -- is fixed, and by exactly
// that means.  The corners now reach this function already computed from
// whichever tail is representable, so a saturated CDF no longer collapses the
// interval; see gauss_corner_quantiles() at the end of this file, which also
// records what the collapse was costing (it is what stalled the truck data's
// m1 boundary refit at nlminb "false convergence (8)").  This function is
// unchanged by that work: it still receives four quantiles and a rho, and
// still assumes only that they are ordered.
template<class Type>
Type gaussian_cell_prob(Type qa, Type qam, Type qb, Type qbm, Type rho) {
  Type sig2 = Type(1.0) - rho * rho;
  sig2 = CppAD::CondExpLt(sig2, Type(1e-12), Type(1e-12), sig2);
  Type sig = sqrt(sig2);

  auto vmax = [](Type u, Type v) { return CppAD::CondExpGt(u, v, u, v); };
  auto vmin = [](Type u, Type v) { return CppAD::CondExpLt(u, v, u, v); };

  // Signed rho with |rho| floored, so the edge centres below stay finite.
  Type rho_abs = vmax(rho, -rho);
  Type rho_faf = vmax(rho_abs, Type(1e-6));
  Type rho_sgn = CppAD::CondExpLt(rho, Type(0), -rho_faf, rho_faf);

  Type edge = sig / rho_faf;               // transition width, in z
  Type c1 = qb / rho_sgn;
  Type c2 = qbm / rho_sgn;
  Type clo = vmin(c1, c2);
  Type chi = vmax(c1, c2);

  // The whole quantile interval, never a sub-interval of it.  vmax() only
  // guards the caller's ordering; it is not a truncation.
  Type lo = qam;
  Type hi = vmax(qa, qam);

  // Interior cut points: the four edge brackets, plus GAUSS_PHI_CUTS on phi's
  // own scale.  These have to be SORTED before the monotone pass below, not
  // merely interleaved in some fixed order: that pass raises each cut to the
  // running maximum, so an out-of-order fixed cut would drag an edge bracket
  // with it and destroy the placement the brackets exist to provide.  A
  // bubble-sort network run unconditionally sorts branch-free on the tape --
  // 21 comparators against the 384 transcendental calls the panels cost, so
  // the sort does not show up in the profile.
  const int NCUT = 4 + GAUSS_PHI_N;
  Type v[NCUT];
  v[0] = clo - Type(GAUSS_EDGE) * edge;
  v[1] = clo + Type(GAUSS_EDGE) * edge;
  v[2] = chi - Type(GAUSS_EDGE) * edge;
  v[3] = chi + Type(GAUSS_EDGE) * edge;
  for (int j = 0; j < GAUSS_PHI_N; j++) v[4 + j] = Type(gauss_phi_cuts[j]);
  for (int i = 1; i < NCUT; i++) {
    for (int j = i; j > 0; j--) {
      Type a = v[j - 1], b = v[j];
      v[j - 1] = vmin(a, b);
      v[j] = vmax(a, b);
    }
  }

  // Cut points clamped into [lo, hi] and then forced non-decreasing.
  Type cuts[NCUT + 2];
  cuts[0] = lo;
  for (int j = 0; j < NCUT; j++) cuts[j + 1] = v[j];
  cuts[NCUT + 1] = hi;
  for (int j = 1; j < NCUT + 2; j++) {
    cuts[j] = vmin(vmax(cuts[j], lo), hi);
    cuts[j] = vmax(cuts[j], cuts[j - 1]);
  }

  Type total = Type(0);
  for (int p = 0; p < NCUT + 1; p++) {
    Type half = (cuts[p + 1] - cuts[p]) / Type(2);
    Type mid = (cuts[p + 1] + cuts[p]) / Type(2);
    Type acc = Type(0);
    for (int i = 0; i < 8; i++) {
      for (int side = -1; side <= 1; side += 2) {
        Type z = mid + half * Type(side) * Type(gauss_x16[i]);
        Type A = (qb - rho * z) / sig;
        Type B = (qbm - rho * z) / sig;
        Type flip = A + B;
        Type Au = CppAD::CondExpGt(flip, Type(0), -B, A);
        Type Bu = CppAD::CondExpGt(flip, Type(0), -A, B);
        acc += Type(gauss_w16[i]) *
          dnorm(z, Type(0), Type(1), false) * (pnorm(Au) - pnorm(Bu));
      }
    }
    total += half * acc;
  }
  return total;
}

// Checkpointed form of gaussian_cell_prob().  Inlined, the kernel above puts
// ~1,500 operations on the tape per observation-draw -- (NCUT+1) panels x 16
// nodes, each with two pnorm() calls and a dnorm() -- which is ~90% of the
// gaussian SML tape and is what priced 300 draws on the truck data at ~44 GiB
// of retained tape and a ~177 GiB construction peak.  REGISTER_ATOMIC turns
// the kernel into an atomic symbol that costs ONE tape operation per call
// (O(n+m) memory per occurrence, independent of the kernel's flops): the
// kernel is taped once -- to third-order nesting under CPPAD_FRAMEWORK, so
// the Laplace path's nested sweeps still differentiate through it -- and
// those inner tapes are replayed on demand, with per-thread copies for the
// parallel regions.
//
// Replaying a tape recorded at one input on other inputs is only valid
// because the kernel is branch-free: every branch above is a CppAD::CondExp
// and both loop bounds are compile-time constants, so the recorded graph is
// identical at every input.  Do NOT add a value-dependent if() or a
// data-dependent loop to gaussian_cell_prob() without removing this wrapper.
//
// Under CPPAD_FRAMEWORK the inner tapes are built on the FIRST call with
// double arguments, and taping of the parallel regions runs concurrently, so
// the objective triggers that initialization explicitly inside an OpenMP
// critical section before any AD call can race on it (see the FAM_GAUSSIAN
// block ahead of the accumulation loop).
template<class Type>
vector<Type> gauss_cell_vec(vector<Type> x) {
  vector<Type> y(1);
  y[0] = gaussian_cell_prob(x[0], x[1], x[2], x[3], x[4]);
  return y;
}
REGISTER_ATOMIC(gauss_cell_vec)

// Normal quantiles of one margin's two cell corners, each computed from
// whichever tail is still representable in double precision.
//
// Gaussian is the only family that has to pass its margins through qnorm(),
// which is singular at 0 and at 1, so it is the only one whose cell can be
// destroyed by the marginal CDF saturating.  The previous form clamped the CDF
// to [1e-15, 1 - 1e-15] and took qnorm() of that, which collapses the cell to
// ZERO WIDTH -- q(a) == q(a') exactly -- whenever both corners land past the
// same clamp.  The caller then floors the probability at 1e-300, and worse, a
// clamp is a CondExp step, so such a cell contributes exactly zero gradient.
//
// That is not a corner case.  On the truck data's m1 boundary refit (margin 1
// pinned Poisson against counts running to 242) it hit 2.69% of
// observation-draw cells and 16.4% of observations -- against 0.05% and 0.4%
// with the same margin left NB2 -- and it is what stalled nlminb at "false
// convergence (8)".  Along the b1 coefficients the objective swung 1,285 nats
// over steps of 2e-4 while the AD gradient, blind to every clamped cell,
// disagreed with a central finite difference by ~100% (51 against 5,094; -479
// against -1.6e6) -- next to exact agreement for the parameters that do not
// move mu1.  PORT's code 8 says precisely this: converging to a noncritical
// point, gradients possibly wrong or the function discontinuous.
//
// This is the "separately accumulated survival function" the old comment on
// gaussian_cell_prob() called for, and the observation that makes it cheap is
// that 1e-15 was never the real limit.  A probability near ZERO is
// representable to ~1e-308; only a probability near ONE loses its information,
// because a double's spacing at 1 is 1.1e-16.  So each corner stays exact as
// long as it is expressed through the SMALL quantity:
//
//   F(y) <= 1/2 :  q = +qnorm(F)      F is itself small and exact
//   F(y) >  1/2 :  q = -qnorm(S)      S = P(Y > y), never formed as 1 - F
//
// and in the upper tail the second corner follows from
//
//   S(y-1) = P(Y > y-1) = S(y) + P(Y = y)
//
// an ADDITION of two positive quantities, so it cannot cancel -- unlike the
// 1 - F it replaces.  S comes from the margin's exact upper-tail special
// function, both of which TMB already carries as differentiable atomics:
//
//   Poisson : P(Y > y) = pgamma(mu, y+1)
//   NB2     : P(Y > y) = pbeta(mu/(mu+r), y+1, r)
//
// Validated against the only invariant that pins it down without an external
// reference -- the strip must carry exactly the marginal mass,
// Phi(q(y)) - Phi(q(y-1)) == P(Y = y) -- to a worst relative error of 4.2e-13
// over cells whose pmf spans 1.4e-296 to 0.37, where 35 of those same 55 cells
// collapsed outright under the old clamp.  Where the two forms disagree
// (F > 1 - 1e-11) that round trip also says which is right: 1.2e-14 relative
// error here against 9.3e-4 for qnorm(F).  Where the old form was sound they
// agree to 4e-10 over 5,000 random cells.
//
// The branch is applied to the ARGUMENT and a SIGN rather than to two finished
// quantiles, for the same reason gaussian_cell_prob() does it with its pnorm
// calls: CppAD::CondExp evaluates both of its value branches, so choosing
// afterwards would put four qnorm() calls on the tape per margin instead of
// two.
//
// y == 0 is settled off the tape.  Its lower corner is a true -infinity (no
// mass below zero), and in the upper-tail branch S(-1) is exactly 1, so
// qnorm() would hand the quadrature +Inf and it would then form Inf - Inf.
// The sentinel is qnorm(1e-300) = -37.0471, which is what the lower branch
// computes there anyway, so the two branches agree at y = 0.
//
// Residual limitation, 293 decades further out than the one it replaces: if
// the SMALL tail itself underflows -- S < 1e-300 upper, F < 1e-300 lower --
// both corners floor to the sentinel and the cell collapses as before.  The
// lower-tail half of that is in principle recoverable, since nb2_cdf_pair()
// and pois_cdf_pair() already return an accurate log F; it needs a log-scale
// qnorm, which TMB does not provide (its qnorm atomic has no log_p argument).
// The truck data's largest count is 242, where the survival is 4e-228 --
// nowhere near this floor, against 2.69% of cells reaching the old one.
// Both thresholds below are deliberately conservative, and the first version of
// this function got both wrong in the same way -- by rerouting cells that were
// never broken.  Recorded because the failure was silent in the values and
// showed up only as a changed OPTIMUM:
//
// SWITCHING ON F > 1/2 rather than on F near 1.  qnorm(F) is perfectly accurate
// until F approaches 1; the survival is needed only once F saturates.  On the
// convergence-polish fixture (counts around mu = 1.5) the halfway switch routed
// 67.6% of cells onto the survival path while 0.0% of them needed it, which put
// the strip's absolute position on pbeta/pgamma while its width still came from
// the CDF accumulator's pmf.  The values agree to ~1e-16 -- pbeta is exact even
// at r = 1e9, so this was not an accuracy failure -- but the two are no longer
// the SAME arithmetic, and on Poisson-generated data the free dispersion then
// stopped collapsing (m1 0.014, m2 0.029, against < 1e-3 before and a true
// value of 0) and the polished gradient came back non-finite.  So the switch is
// at 1 - 1e-10: below it nothing changes at all.
//
// SENDING y == 0 TO qnorm(1e-300) = -37.05.  Its lower corner is a true
// -infinity, so -37.05 is strictly closer to right than the -7.94 it replaced,
// and the mass between them (1e-300 against 1e-15) is immaterial to the cell.
// It is not immaterial to the QUADRATURE: gaussian_cell_prob() spends a fixed
// node budget on [q(a'), q(a)], so this made that interval about five times
// wider for every zero count -- and zeros are most of a count sample -- for no
// gain in mass.  y == 0 therefore keeps the old sentinel.  Nothing is lost by
// that: the case is structural (there is no mass below zero), not a saturated
// CDF, so it is not what this function exists to repair.
//
// Note the asymmetry is self-correcting for y > 0, which is why only y == 0
// needs the special case: if F(y-1) is deep enough to floor, F(y) is deep too,
// so both corners move together and the interval stays narrow.
//
// `use_sf` restricts the whole mechanism to POISSON margins, and it is a plain
// bool -- pois1/pois2 are fixed by the caller's arguments, not by parameters --
// so for an NB2 margin the survival call disappears from the tape entirely and
// that path is bit-identical to what it was before this function existed.
//
// That restriction is not conservatism for its own sake; it is the third thing
// the first version of this function got wrong, and the one that cost the most
// to find.  The NB2 survival is pbeta(mu/(mu+r), y+1, r), and r = 1/m runs away
// exactly when the data wants a Poisson margin: the convergence-polish fixture
// drives m to ~1e-6, so r ~ 1e6, and log_m's own clamp allows r up to
// exp(20) = 4.85e8.  pbeta's VALUE is exact even at r = 1e9 (checked against
// pnbinom: relative error 0), so this is not the accuracy failure it looks
// like -- but its derivative with respect to r is ~1e-13 there, and because
// CppAD::CondExp evaluates both of its branches, that call sat on the tape for
// every cell whether or not the survival was selected.  With it present the
// polished fit came back with a NON-FINITE max|gradient| and the free
// dispersion stopped collapsing on Poisson-generated data (m1 0.018, m2 0.034
// against < 1e-3 before, true value 0); narrowing the switch from F > 1/2 to
// F > 1 - 1e-10, so that essentially no cell selected it, changed those numbers
// hardly at all -- which is what identified the mere PRESENCE of the call,
// rather than its selection, as the cause.
//
// The Poisson margin needs no such guard: pgamma(mu, y+1) carries one
// parameter, no shape that can run away, and it is the case the dispersion
// boundary tests actually pin.  An NB2 margin keeps the old 1e-15 clamp and so
// keeps the old limitation with it -- measured at 0.05% of observation-draw
// cells and 0.4% of observations on the truck data, against the 2.69% and 16.4%
// that a Poisson-pinned margin reached.
template<class Type>
void gauss_corner_quantiles(Type F_y, Type F_ym, Type pmf, Type sf,
                            bool y_zero, bool use_sf, Type &q_y, Type &q_ym) {
  const Type TINY(1e-300);
  const Type ZERO_Q(-7.941345);       // qnorm(1e-15), the historical floor
  const Type NEAR_ONE(1.0 - 1e-10);   // only past here has F lost its tail
  const Type NEAR_ONE_P(1.0 - 1e-16); // keeps qnorm's argument below 1

  auto vmax = [](Type u, Type v) { return CppAD::CondExpGt(u, v, u, v); };
  auto vmin = [](Type u, Type v) { return CppAD::CondExpLt(u, v, u, v); };

  if (!use_sf) {
    // Exactly the historical path, clamp and all.
    auto clamped = [&](Type p) {
      return qnorm(vmin(vmax(p, Type(1e-15)), Type(1.0 - 1e-15)));
    };
    q_y  = clamped(F_y);
    q_ym = clamped(F_ym);
    return;
  }

  // Upper corner is the larger probability in BOTH branches (F_y > F_ym, and
  // S(y-1) = S(y) + pmf > S(y)), so the sign flip preserves q_y > q_ym.
  // sf + pmf is S(y-1) and cannot exceed 1 in exact arithmetic; it is capped
  // anyway because sf and pmf reach here from different computations and
  // qnorm() of anything at or above 1 is +Inf.
  Type p_y  = CppAD::CondExpGt(F_y, NEAR_ONE, vmax(sf, TINY),
                                              vmax(F_y,  TINY));
  Type p_ym = CppAD::CondExpGt(F_y, NEAR_ONE, vmin(vmax(sf + pmf, TINY),
                                                   NEAR_ONE_P),
                                              vmax(F_ym, TINY));
  Type sgn  = CppAD::CondExpGt(F_y, NEAR_ONE, Type(-1), Type(1));

  q_y  = sgn * qnorm(p_y);
  q_ym = y_zero ? ZERO_Q : sgn * qnorm(p_ym);
}

template<class Type>
Type objective_function<Type>::operator() () {

  // ---- Data ----
  DATA_VECTOR(Y1);
  DATA_VECTOR(Y2);
  DATA_MATRIX(X1);
  DATA_MATRIX(X2);
  DATA_IVECTOR(rand_idx1);
  DATA_IVECTOR(rand_idx2);
  DATA_MATRIX(Z1);
  DATA_MATRIX(Z2);
  DATA_IVECTOR(dist1);
  DATA_IVECTOR(dist2);
  DATA_IVECTOR(sign1);
  DATA_IVECTOR(sign2);
  DATA_INTEGER(family);
  DATA_INTEGER(pois1);
  DATA_INTEGER(pois2);
  DATA_SCALAR(lamLo);
  DATA_SCALAR(lamHi);
  // 0 = simulated maximum likelihood (Halton draws)
  // 1 = Laplace approximation (latent u1/u2 integrated by TMB)
  DATA_INTEGER(est_method);

  // ---- Parameters ----
  PARAMETER_VECTOR(beta1);
  PARAMETER_VECTOR(beta2);
  PARAMETER_VECTOR(log_sd1);
  PARAMETER_VECTOR(log_sd2);
  PARAMETER(log_m1);
  PARAMETER(log_m2);
  PARAMETER(z_dep);
  // Latent standard normals, one row per observation. Under est_method == 0
  // these are map-fixed at zero in R and never read; under est_method == 1
  // they are TMB random effects.
  PARAMETER_MATRIX(u1);
  PARAMETER_MATRIX(u2);

  // ---- Dimensions ----
  int n = Y1.size();
  int k1 = X1.cols(); (void)k1;
  int k2 = X2.cols(); (void)k2;
  int q1 = rand_idx1.size();
  int q2 = rand_idx2.size();
  int R = (q1 + q2 > 0) ? Z1.rows() : 1;
  // Laplace evaluates the conditional density once at the current latent
  // values; there is no draw dimension to average over.
  if (est_method == 1) R = 1;
  int openmp_compiled = 0;
#ifdef _OPENMP
  openmp_compiled = 1;
#endif
  REPORT(openmp_compiled);

  // ---- Natural-scale parameters ----
  auto clamp_ad = [](Type x, Type lo, Type hi) -> Type {
    x = CppAD::CondExpLt(x, lo, lo, x);
    return CppAD::CondExpGt(x, hi, hi, x);
  };
  Type log_m1_c = clamp_ad(log_m1, Type(-20.0), Type(20.0));
  Type log_m2_c = clamp_ad(log_m2, Type(-20.0), Type(20.0));
  Type m1 = exp(log_m1_c);
  Type m2 = exp(log_m2_c);
  Type r1 = 1.0 / m1;
  Type r2 = 1.0 / m2;

  // dnbinom2(y, mu, mu + m*mu*mu) evaluates log(var - mu).  The variance
  // increment m*mu*mu is lost to rounding once it falls below ulp(mu), i.e.
  // once log(m) + log(mu) drops under about -36.04 = log(2^-52).  Clamping
  // the linear predictor alone cannot enforce that, because m is estimated:
  // the floor has to move with m.  Keep -35 as the ceiling on the floor so
  // that over-dispersed fits are unaffected.
  auto nb2_eta_floor = [&](Type log_m_clamped) -> Type {
    Type negative_log_m = CppAD::CondExpLt(
      log_m_clamped, Type(0), log_m_clamped, Type(0)
    );
    return Type(-35.0) - negative_log_m;
  };
  Type eta_floor1 = pois1 ? Type(-35.0) : nb2_eta_floor(log_m1_c);
  Type eta_floor2 = pois2 ? Type(-35.0) : nb2_eta_floor(log_m2_c);
  vector<Type> sd1(log_sd1.size()), sd2(log_sd2.size());
  for (int j = 0; j < log_sd1.size(); j++)
    sd1(j) = exp(clamp_ad(log_sd1(j), Type(-20.0), Type(20.0)));
  for (int j = 0; j < log_sd2.size(); j++)
    sd2(j) = exp(clamp_ad(log_sd2(j), Type(-20.0), Type(20.0)));

  // Linear predictors (fixed part)
  vector<Type> xb1 = X1 * beta1;
  vector<Type> xb2 = X2 * beta2;

  // Dependence transform (Famoye: logistic map; Copula: identity/tanh/exp)
  Type eps = 1e-6;
  Type lam = Type(0), theta = Type(0), rho = Type(0);
  if (family == FAM_FAMOYE) {
    Type sig = invlogit(z_dep);  // logistic(0,1) = 1/(1+exp(-x))
    lam = lamLo + (lamHi - lamLo) * (eps + (1.0 - 2.0 * eps) * sig);
  } else if (family == FAM_FRANK) {
    // A smooth bounded link prevents exponential overflow while retaining
    // theta = 0 and unit derivative at independence.
    theta = Type(FRANK_THETA_MAX) * tanh(z_dep / Type(FRANK_THETA_MAX));
  } else if (family == FAM_GAUSSIAN) {
    rho = tanh(z_dep);
  } else if (family == FAM_CLAYTON) {
    theta = exp(clamp_ad(z_dep, Type(-20.0), Type(20.0)));
  }

  // ---- Random coefficient transforms (per draw) ----
  // Returns the deviation (eta + dev = eta with random perturbation)
  // Using the TMB Type so AD flows through everything

  // Helper functions inside operator()():
  auto u_to_base = [](Type u, int dist_code) -> Type {
    if (dist_code == DIST_NORMAL || dist_code == DIST_LOGNORMAL)
      return qnorm(u);  // TMB's qnorm for Type
    // DIST_UNIFORM: base = u (identity)
    // DIST_TRIANGULAR: symmetric triangular on [-1, 1]
    if (dist_code == DIST_TRIANGULAR) {
      // tri_icdf: if u < 0.5 then -1+sqrt(2u) else 1-sqrt(2(1-u))
      Type two_u = 2.0 * u;
      return CppAD::CondExpLt(u, Type(0.5), Type(-1.0) + sqrt(two_u),
                              Type(1.0) - sqrt(2.0 * (1.0 - u)));
    }
    return u;  // uniform
  };

  auto compute_dev = [](Type b, Type s, Type base, int dist_code, int sign_code) -> Type {
    if (dist_code == DIST_NORMAL)
      return s * base;
    if (dist_code == DIST_LOGNORMAL)
      return Type(sign_code) * exp(b + s * base) - b;
    if (dist_code == DIST_UNIFORM)
      return s * (2.0 * base - 1.0);
    // triangular
    return s * base;
  };

  // Famoye constant d = 1 - exp(-1) (loop-invariant)
  Type famoye_d = Type(1.0) - exp(Type(-1.0));

  // Pre-compute exp(-y) for Famoye path
  vector<Type> ey1(n), ey2(n);
  if (family == FAM_FAMOYE || family == FAM_INDEP) {
    for (int i = 0; i < n; i++) {
      ey1(i) = exp(-Y1(i));
      ey2(i) = exp(-Y2(i));
    }
  }

  // Integer responses for the copula margins: nb2_cdf_pair() sums the mass
  // function up to y, so the loop bound must be an int rather than a Type.
  // Y1/Y2 are data and R has already checked they are whole and non-negative,
  // so this conversion costs nothing on the tape.
  vector<int> Y1_int(n), Y2_int(n);
  if (family >= FAM_FRANK) {
    for (int i = 0; i < n; i++) {
      Y1_int(i) = (int)asDouble(Y1(i));
      Y2_int(i) = (int)asDouble(Y2(i));
    }
  }

  // Precompute parameter-dependent deviations once per simulation draw.  The
  // matrices are read-only while observation contributions are accumulated.
  matrix<Type> dev1(R, q1), dev2(R, q2);
  if (est_method != 1) {
    for (int r = 0; r < R; r++) {
      for (int j = 0; j < q1; j++) {
        Type base = u_to_base(Z1(r, j), dist1(j));
        int col = rand_idx1(j);
        dev1(r, j) = compute_dev(beta1(col), sd1(j), base,
                                 dist1(j), sign1(j));
      }
      for (int j = 0; j < q2; j++) {
        Type base = u_to_base(Z2(r, j), dist2(j));
        int col = rand_idx2(j);
        dev2(r, j) = compute_dev(beta2(col), sd2(j), base,
                                 dist2(j), sign2(j));
      }
    }
  }

  // One-time construction of the checkpointed gaussian kernel's inner tapes
  // (see gauss_cell_vec above).  The double call is what builds them; the
  // evaluation point is arbitrary because the kernel is branch-free.  It must
  // happen before any AD call to gauss_cell_vec(), and region tapes are built
  // concurrently, so the guard is a critical section rather than an
  // assumption about which thread tapes first.
  if (family == FAM_GAUSSIAN) {
#ifdef _OPENMP
#pragma omp critical(gauss_cell_vec_init)
#endif
    {
      vector<double> qinit(5);
      qinit(0) = -0.5; qinit(1) = -1.0; qinit(2) = -0.5; qinit(3) = -1.0;
      qinit(4) = 0.1;
      gauss_cell_vec(qinit);
    }
  }

  // TMB partitions these independent observation contributions across its
  // configured OpenMP regions while keeping each draw reduction local.
  parallel_accumulator<Type> nll(this);
  const Type logR = log(Type(R));

  for (int i = 0; i < n; i++) {
    vector<Type> log_draw(R);

    for (int r = 0; r < R; r++) {
      Type eta1 = xb1(i);
      Type eta2 = xb2(i);
      for (int j = 0; j < q1; j++) {
        int col = rand_idx1(j);
        // u_to_base() is deliberately NOT applied to the latent: it is a
        // per-distribution inverse CDF, not a general uniform-to-normal map,
        // and skipping it is only valid because u_to_base == qnorm for the
        // normal/lognormal distributions the Laplace path is restricted to.
        // fit_rpbnb_tmb() enforces that restriction on the R side by
        // rejecting method = "laplace" with uniform/triangular coefficients.
        Type d = (est_method == 1)
          ? compute_dev(beta1(col), sd1(j), u1(i, j), dist1(j), sign1(j))
          : dev1(r, j);
        eta1 += X1(i, col) * d;
      }
      for (int j = 0; j < q2; j++) {
        int col = rand_idx2(j);
        // Same restriction as the u1 loop above: u_to_base() is skipped
        // because the Laplace path only allows normal/lognormal
        // coefficients, for which u_to_base == qnorm.
        Type d = (est_method == 1)
          ? compute_dev(beta2(col), sd2(j), u2(i, j), dist2(j), sign2(j))
          : dev2(r, j);
        eta2 += X2(i, col) * d;
      }

      Type mu1 = exp(clamp_ad(eta1, eta_floor1,
                              Type(34.538776394910684)));
      Type mu2 = exp(clamp_ad(eta2, eta_floor2,
                              Type(34.538776394910684)));

      if (family == FAM_FAMOYE) {
        Type lnb1 = pois1 ? dpois(Y1(i), mu1, true)
                          : dnbinom2(Y1(i), mu1,
                                     mu1 + m1 * mu1 * mu1, true);
        Type lnb2 = pois2 ? dpois(Y2(i), mu2, true)
                          : dnbinom2(Y2(i), mu2,
                                     mu2 + m2 * mu2 * mu2, true);

        Type c1 = pois1
          ? exp(-famoye_d * mu1)
          : exp(-stable_log1p(famoye_d * m1 * mu1) / m1);
        Type c2 = pois2
          ? exp(-famoye_d * mu2)
          : exp(-stable_log1p(famoye_d * m2 * mu2) / m2);
        Type dep = Type(1.0) + lam * (ey1(i) - c1) * (ey2(i) - c2);
        // The Sarmanov factor can go non-positive because lamLo/lamHi are
        // frozen at the starting values rather than recomputed at the current
        // mu.  Penalising makes that visible in the objective instead of
        // hiding it behind a probability clamp.
        //
        // This is a value-only barrier, NOT a constraint: CondExpLe is a step,
        // so the penalty term contributes exactly zero gradient on both sides.
        // A gradient-driven optimizer is repelled only by the function value,
        // and cannot be steered out of the invalid region by the score.  The
        // real fix is parameter-dependent bounds evaluated here on the tape;
        // until then, treat a fit that lands on the penalty as invalid rather
        // than as a converged optimum.
        Type invalid_dep = CppAD::CondExpLe(
          dep, Type(0), Type(1), Type(0)
        );
        Type safe_dep = CppAD::CondExpLe(
          dep, Type(0), Type(1e-300), dep
        );
        log_draw(r) = lnb1 + lnb2 + log(safe_dep) -
          invalid_dep * Type(1e10);
      } else if (family == FAM_INDEP) {
        Type lnb1 = pois1 ? dpois(Y1(i), mu1, true)
                          : dnbinom2(Y1(i), mu1,
                                     mu1 + m1 * mu1 * mu1, true);
        Type lnb2 = pois2 ? dpois(Y2(i), mu2, true)
                          : dnbinom2(Y2(i), mu2,
                                     mu2 + m2 * mu2 * mu2, true);
        log_draw(r) = lnb1 + lnb2;
      } else {
        // Y1_int/Y2_int are cast from data, so these two flags are ordinary
        // compile-time branches that carry no parameter dependence onto the
        // tape.  Clayton selects its axis cases from them; see
        // clayton_cell_prob() for why reading them off a CDF instead made the
        // taped objective depend on the starting values.
        const bool y1_zero = (Y1_int(i) == 0);
        const bool y2_zero = (Y2_int(i) == 0);

        Type a1, a1m, b1, b1m, pmf1, pmf2;
        Type la1, la1m, lpmf1, lb1, lb1m, lpmf2;
        if (pois1) {
          pois_cdf_pair(Y1_int(i), mu1, a1, a1m, pmf1, la1, la1m, lpmf1);
        } else {
          nb2_cdf_pair(Y1_int(i), mu1, r1, a1, a1m, pmf1, la1, la1m, lpmf1);
        }
        if (pois2) {
          pois_cdf_pair(Y2_int(i), mu2, b1, b1m, pmf2, lb1, lb1m, lpmf2);
        } else {
          nb2_cdf_pair(Y2_int(i), mu2, r2, b1, b1m, pmf2, lb1, lb1m, lpmf2);
        }

        // Every copula family now builds the cell probability directly rather
        // than as a second difference of corner CDFs, so none of them can
        // return a negative probability.  See frank_log_cell_prob(),
        // clayton_cell_prob() and gaussian_cell_prob().
        //
        // Frank is the one family that returns the LOG cell probability, and
        // so is the one family that does not pass through the 1e-300 floor
        // below.  That floor is not a guard against underflow noise here: it
        // clips cell probabilities the data genuinely produces (see
        // frank_log_cell_prob()'s header for the truck observation at
        // 1.03e-300), and clipping them puts a kink in the objective that
        // costs Laplace its positive-definite inner Hessian.  Clayton and
        // Gaussian still return a linear probability and still need it.
        if (family == FAM_FRANK) {
          log_draw(r) = frank_log_cell_prob(a1, a1m, lpmf1, b1, b1m, lpmf2,
                                            theta);
          continue;
        }
        Type p_obs = Type(0);
        if (family == FAM_GAUSSIAN) {
          // Upper-tail survival, taken from the margin's own special function
          // rather than as 1 - CDF.  Only the Gaussian branch needs it (it is
          // the only family that must pass through qnorm(), singular at 1);
          // see gauss_corner_quantiles() for the full argument, and note that
          // computing it here rather than in nb2_cdf_pair()/pois_cdf_pair()
          // keeps its cost off Frank, Clayton and Famoye entirely.
          //
          // Poisson margins only -- pois1/pois2 are ordinary bools, so an NB2
          // margin puts no survival call on the tape at all.  See
          // gauss_corner_quantiles() for why the NB2 form (pbeta, whose shape
          // r = 1/m runs to 1e8 as the dispersion collapses) has to stay off
          // it: its presence alone, selected or not, cost a finite gradient.
          Type sf1 = pois1 ? pgamma(mu1, Type(Y1_int(i) + 1)) : Type(0);
          Type sf2 = pois2 ? pgamma(mu2, Type(Y2_int(i) + 1)) : Type(0);
          Type qa, qam, qb, qbm;
          gauss_corner_quantiles(a1, a1m, pmf1, sf1, y1_zero, pois1, qa, qam);
          gauss_corner_quantiles(b1, b1m, pmf2, sf2, y2_zero, pois2, qb, qbm);
          vector<Type> qcell(5);
          qcell(0) = qa;
          qcell(1) = qam;
          qcell(2) = qb;
          qcell(3) = qbm;
          qcell(4) = rho;
          p_obs = gauss_cell_vec(qcell)(0);
        } else {
          p_obs = clayton_cell_prob(la1, la1m, lpmf1, lb1, lb1m, lpmf2,
                                    theta, y1_zero, y2_zero);
        }
        p_obs = CppAD::CondExpLt(p_obs, Type(1e-300),
                                 Type(1e-300), p_obs);
        log_draw(r) = log(p_obs);
      }
    }

    if (est_method == 1) {
      Type obs_ll = log_draw(0);
      for (int j = 0; j < q1; j++)
        obs_ll += dnorm(u1(i, j), Type(0), Type(1), true);
      for (int j = 0; j < q2; j++)
        obs_ll += dnorm(u2(i, j), Type(0), Type(1), true);
      nll -= obs_ll;
    } else {
      Type max_log = log_draw(0);
      for (int r = 1; r < R; r++) {
        max_log = CppAD::CondExpGt(log_draw(r), max_log,
                                   log_draw(r), max_log);
      }
      Type scaled_sum = Type(0);
      for (int r = 0; r < R; r++) {
        scaled_sum += exp(log_draw(r) - max_log);
      }
      Type log_contribution = max_log + log(scaled_sum) - logR;
      nll -= log_contribution;
    }
  }

  // ---- REPORT derived parameters for sdreport ----
  ADREPORT(m1);
  ADREPORT(m2);
  if (family == FAM_FAMOYE) {
    ADREPORT(lam);
  } else if (family == FAM_FRANK) {
    ADREPORT(theta);
  } else if (family == FAM_GAUSSIAN) {
    ADREPORT(rho);
  } else if (family == FAM_CLAYTON) {
    ADREPORT(theta);
  }

  // ---- Kendall's tau for copula families ----
  if (family == FAM_FRANK || family == FAM_GAUSSIAN || family == FAM_CLAYTON) {
    Type tau;
    if (family == FAM_GAUSSIAN) {
      tau = Type(2.0) / Type(3.14159265358979323846) * asin(rho);
    } else if (family == FAM_CLAYTON) {
      tau = theta / (theta + Type(2.0));
    } else {  // Frank
      // Frank tau: 1 - 4/th * (1 - D1(th)) where D1 is Debye function order 1
      {
        // 20-point Gauss-Legendre quadrature on [0, theta], with a
        // small-theta series that keeps the value and derivatives smooth.
        static const double x20[10] = {0.9931285991850949, 0.9639719272779138,
          0.9122344282513259, 0.8391169718222188, 0.7463319064601508,
          0.6360536807265150, 0.5108670019508271, 0.3737060887154196,
          0.2277858511416451, 0.07652652113349733};
        static const double w20[10] = {0.01761400713915212, 0.04060142980038694,
          0.06267204833410906, 0.08327674157670475, 0.1019301198172404,
          0.1181945319615184, 0.1316886384491766, 0.1420961093183821,
          0.1491729864726037, 0.1527533871307259};
        Type signed_eps = CppAD::CondExpGe(
          theta, Type(0), Type(1e-4), Type(-1e-4)
        );
        Type quadrature_theta = CppAD::CondExpLt(
          fabs(theta), Type(1e-4), signed_eps, theta
        );
        Type D1 = 0;
        for (int qq = 0; qq < 10; qq++) {
          Type t = quadrature_theta * Type(0.5) *
            (Type(1.0) + Type(x20[qq]));
          Type f = t / stable_expm1(t);
          D1 += Type(w20[qq]) * f;
          t = quadrature_theta * Type(0.5) *
            (Type(1.0) - Type(x20[qq]));
          f = t / stable_expm1(t);
          D1 += Type(w20[qq]) * f;
        }
        D1 = D1 * Type(0.5);
        Type quadrature_tau = Type(1.0) -
          Type(4.0) / quadrature_theta * (Type(1.0) - D1);
        Type series_tau = theta / Type(9.0) -
          theta * theta * theta / Type(900.0);
        tau = CppAD::CondExpLt(
          fabs(theta), Type(1e-4), series_tau, quadrature_tau
        );
      }
    }
    ADREPORT(tau);
  }

  return nll;
}

// ---------------------------------------------------------------------------
// Combined native-routine registration for the whole package.
//
// This DLL hosts two engines: the Rcpp/OpenMP kernels (halton_parallel.cpp,
// copula_parallel.cpp, exported via RcppExports.cpp) and the TMB template
// above.  Both would otherwise want to define R_init_rpbnb, so we define it
// once here and register a single merged table.
//
// Rcpp::compileAttributes() scans every src/*.cpp except RcppExports.cpp for a
// line matching /^[^/]+R_init_rpbnb.*DllInfo.*$/ and, on finding one, suppresses
// its own CallEntries[] and R_init_rpbnb.  KEEP the definition line below
// textually intact -- the regex needs at least one non-'/' character before the
// name, which `void ` supplies.  If Rcpp ever starts emitting a second init
// anyway, the link will fail with a duplicate symbol; see the plan's fallback
// (an // [[Rcpp::init]] hook that re-registers this table).
//
// If you add or remove an // [[Rcpp::export]] function, update this table AND
// tests/testthat/test-native-registration.R.
// ---------------------------------------------------------------------------
extern "C" {

SEXP _rpbnb_pbivnorm_cpp(SEXP, SEXP, SEXP);
SEXP _rpbnb_rpbnb_copula_ll_grad_cpp(
    SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
    SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
SEXP _rpbnb_get_num_threads(void);
SEXP _rpbnb_set_rcpp_parallel_threads(SEXP);
SEXP _rpbnb_rpbnb_openmp_enabled(void);
SEXP _rpbnb_rpbnb_ll_grad_cpp(
    SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
    SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
    SEXP, SEXP, SEXP);

static const R_CallMethodDef rpbnbCallEntries[] = {
  TMB_CALLDEFS,
  {"_rpbnb_pbivnorm_cpp",              (DL_FUNC) &_rpbnb_pbivnorm_cpp,               3},
  {"_rpbnb_rpbnb_copula_ll_grad_cpp",  (DL_FUNC) &_rpbnb_rpbnb_copula_ll_grad_cpp,  23},
  {"_rpbnb_get_num_threads",           (DL_FUNC) &_rpbnb_get_num_threads,            0},
  {"_rpbnb_set_rcpp_parallel_threads", (DL_FUNC) &_rpbnb_set_rcpp_parallel_threads,  1},
  {"_rpbnb_rpbnb_openmp_enabled",      (DL_FUNC) &_rpbnb_rpbnb_openmp_enabled,       0},
  {"_rpbnb_rpbnb_ll_grad_cpp",         (DL_FUNC) &_rpbnb_rpbnb_ll_grad_cpp,         27},
  {NULL, NULL, 0}
};

void R_init_rpbnb(DllInfo *dll) {
  R_registerRoutines(dll, NULL, rpbnbCallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, (Rboolean) FALSE);
  TMB_CCALLABLES("rpbnb");
}

}  // extern "C"
