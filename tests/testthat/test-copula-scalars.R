# .copula_score_scalars must reproduce copula_grad_vec exactly: contracting the
# scalars with the design must equal copula_grad_vec's output, and copula_grad_vec
# must still match the numeric gradient of the fixed-model copula log-likelihood.

make_cop_case <- function(n = 50, seed = 4) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x1)
  X2 <- cbind(`(Intercept)` = 1, x2 = x2)
  y1 <- rnbinom(n, mu = exp(0.3 + 0.2 * x1), size = 2)
  y2 <- rnbinom(n, mu = exp(0.2 - 0.1 * x2), size = 2)
  list(y1 = y1, y2 = y2, X1 = X1, X2 = X2)
}

test_that(".copula_score_scalars contracts to copula_grad_vec output", {
  cs <- make_cop_case()
  par <- c(0.3, 0.2, 0.2, -0.1, log(0.5), log(0.6), 0.4)
  for (fam in c("frank", "normal", "kimeldorf")) {
    p1 <- ncol(cs$X1); p2 <- ncol(cs$X2)
    beta1 <- par[1:p1]; beta2 <- par[p1 + 1:p2]
    r1 <- exp(-par[p1 + p2 + 1]); r2 <- exp(-par[p1 + p2 + 2])
    z  <- par[p1 + p2 + 3]
    mu1 <- .bound_mu(cs$X1, beta1); mu2 <- .bound_mu(cs$X2, beta2)
    theta <- z_to_native(fam, z); dth <- dnative_dz(fam, z)
    sc <- .copula_score_scalars(cs$y1, cs$y2, mu1, mu2, r1, r2, theta, dth, fam)
    g_manual <- c(as.vector(t(cs$X1) %*% sc$s_eta1),
                  as.vector(t(cs$X2) %*% sc$s_eta2),
                  sum(sc$s_logm1), sum(sc$s_logm2), sum(sc$s_ztheta))
    g_ref <- copula_grad_vec(par, cs$y1, cs$y2, cs$X1, cs$X2, fam)
    expect_equal(g_manual, unname(g_ref), tolerance = 1e-10, info = fam)
  }
})

test_that("extreme-count observations do not blow up the copula score", {
  # y2=120 with mu=1 makes pnbinom(120)==pnbinom(119)==1 to machine precision:
  # the joint pmf cancels to 0 and gets floored to 1e-300. The value path takes a
  # finite log-floor penalty there, so the *score* of that clamped-constant term
  # must be 0 (not numerator/1e-300 ~ 1e279). Guards against the BFGS iter-1 stall.
  X1 <- cbind(`(Intercept)` = 1, x1 = 0)
  X2 <- cbind(`(Intercept)` = 1, x2 = 0)
  y1 <- 20L; y2 <- 120L
  for (fam in c("frank", "normal", "kimeldorf")) {
    mu1 <- 1; mu2 <- 1; r1 <- exp(-log(0.5)); r2 <- exp(-log(0.5))
    theta <- z_to_native(fam, 0); dth <- dnative_dz(fam, 0)
    sc <- .copula_score_scalars(y1, y2, mu1, mu2, r1, r2, theta, dth, fam)
    scores <- c(sc$s_eta1, sc$s_eta2, sc$s_logm1, sc$s_logm2, sc$s_ztheta)
    expect_true(all(is.finite(scores)), info = fam)
    expect_true(max(abs(scores)) < 1e6, info = fam)   # was ~1e279 before the fix
  }
})

test_that("refactored copula_grad_vec still matches numeric gradient", {
  cs <- make_cop_case()
  par <- c(0.3, 0.2, 0.2, -0.1, log(0.5), log(0.6), 0.4)
  for (fam in c("frank", "normal", "kimeldorf")) {
    g_a <- copula_grad_vec(par, cs$y1, cs$y2, cs$X1, cs$X2, fam)
    g_n <- numDeriv::grad(function(p) sum(copula_loglik_vec(p, cs$y1, cs$y2, cs$X1, cs$X2, fam)), par)
    expect_equal(unname(g_a), g_n, tolerance = 1e-5, info = fam)
  }
})
