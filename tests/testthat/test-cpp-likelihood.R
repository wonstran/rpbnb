# The multithreaded C++ likelihood must be math-identical to the R reference
# bnbr_rp_ll_and_grad / bnbr_rp_ll_fixed_bounds. The R implementation is the
# oracle; these tests fail if the port ever drifts.

skip_if_not(rpbnb_cpp_available(), "C++ likelihood not compiled")

make_case <- function(n = 60, seed = 7) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x1, x2 = x2)
  X2 <- X1
  y1 <- rnbinom(n, mu = exp(0.4 + 0.2 * x1), size = 2)
  y2 <- rnbinom(n, mu = exp(0.3 - 0.1 * x2), size = 2)
  list(y1 = y1, y2 = y2, X1 = X1, X2 = X2)
}

run_pair <- function(random_cols_1, random_cols_2, par, dists1 = NULL, dists2 = NULL,
                     signs1 = NULL, signs2 = NULL, R = 128) {
  cs <- make_case()
  rand_idx1 <- match(random_cols_1, colnames(cs$X1))
  rand_idx2 <- match(random_cols_2, colnames(cs$X2))
  q1 <- length(rand_idx1); q2 <- length(rand_idx2)
  XR1 <- if (q1) cs$X1[, rand_idx1, drop = FALSE] else NULL
  XR2 <- if (q2) cs$X2[, rand_idx2, drop = FALSE] else NULL

  if (is.null(dists1)) dists1 <- rep("normal", q1)
  if (is.null(dists2)) dists2 <- rep("normal", q2)
  if (is.null(signs1)) signs1 <- rep(1, q1)
  if (is.null(signs2)) signs2 <- rep(1, q2)

  set.seed(99)
  Z <- halton_uniform(R, q1 + q2, burn = 50)
  Z1 <- if (q1) Z[, seq_len(q1), drop = FALSE] else matrix(0, R, 0)
  Z2 <- if (q2) Z[, (q1 + 1):(q1 + q2), drop = FALSE] else matrix(0, R, 0)

  r_val <- bnbr_rp_ll_and_grad(par, cs$y1, cs$y2, cs$X1, cs$X2, XR1, XR2,
                               rand_idx1, rand_idx2, Z1, Z2,
                               dists1, dists2, signs1, signs2)
  c_val <- bnbr_rp_ll_and_grad_cpp(par, cs$y1, cs$y2, cs$X1, cs$X2, XR1, XR2,
                                   rand_idx1, rand_idx2, Z1, Z2,
                                   dists1, dists2, signs1, signs2, n_threads = 2L)
  list(r = r_val, c = c_val, cs = cs, Z1 = Z1, Z2 = Z2,
       rand_idx1 = rand_idx1, rand_idx2 = rand_idx2,
       dists1 = dists1, dists2 = dists2, signs1 = signs1, signs2 = signs2)
}

test_that("C++ value and gradient match R (all-normal random coefs)", {
  # k1=k2=3, q1=q2=3
  par <- c(0.4, 0.2, 0.0,      # beta1
           0.3, 0.0, -0.1,     # beta2
           log(0.2), log(0.2), log(0.2),   # log_sd1
           log(0.2), log(0.2), log(0.2),   # log_sd2
           log(0.5), log(0.5), 0.1)        # log_m1, log_m2, z_lambda
  out <- run_pair(c("(Intercept)", "x1", "x2"), c("(Intercept)", "x1", "x2"), par)
  expect_equal(as.numeric(out$c), as.numeric(out$r), tolerance = 1e-8)
  # maxLik ignores gradient names; compare values only (R names via colSums).
  expect_equal(attr(out$c, "gradient"), attr(out$r, "gradient"),
               tolerance = 1e-7, ignore_attr = TRUE)
})

test_that("C++ matches R with a subset of random coefficients", {
  # q1 = 1 (x1 only), q2 = 2 (Intercept, x2)
  par <- c(0.5, 0.1, -0.2,
           0.2, 0.3, 0.0,
           log(0.3),                 # log_sd1 (x1)
           log(0.25), log(0.15),     # log_sd2 (Intercept, x2)
           log(0.4), log(0.6), -0.2)
  out <- run_pair(c("x1"), c("(Intercept)", "x2"), par)
  expect_equal(as.numeric(out$c), as.numeric(out$r), tolerance = 1e-8)
  # maxLik ignores gradient names; compare values only (R names via colSums).
  expect_equal(attr(out$c, "gradient"), attr(out$r, "gradient"),
               tolerance = 1e-7, ignore_attr = TRUE)
})

test_that("C++ matches R with a lognormal random coefficient", {
  par <- c(0.4, 0.2, 0.0,
           0.3, 0.0, -0.1,
           log(0.2),                 # log_s1 for x1 (lognormal)
           log(0.2),                 # log_sd2 for x1 (normal)
           log(0.5), log(0.5), 0.05)
  out <- run_pair(c("x1"), c("x1"), par,
                  dists1 = "lognormal", dists2 = "normal",
                  signs1 = 1, signs2 = 1)
  expect_equal(as.numeric(out$c), as.numeric(out$r), tolerance = 1e-8)
  # maxLik ignores gradient names; compare values only (R names via colSums).
  expect_equal(attr(out$c, "gradient"), attr(out$r, "gradient"),
               tolerance = 1e-7, ignore_attr = TRUE)
})

test_that("OPG per-observation scores sum to the analytic gradient", {
  # colSums(scores) must equal the total gradient (which is verified vs R).
  par <- c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1,
           log(0.25), log(0.2),          # log_sd1 (Intercept, x1)
           log(0.3),                      # log_sd2 (x2)
           log(0.5), log(0.5), 0.1)
  out <- run_pair(c("(Intercept)", "x1"), c("x2"), par)
  S <- bnbr_rp_scores_cpp(par, out$cs$y1, out$cs$y2, out$cs$X1, out$cs$X2,
                          out$cs$X1[, out$rand_idx1, drop = FALSE],
                          out$cs$X2[, out$rand_idx2, drop = FALSE],
                          out$rand_idx1, out$rand_idx2, out$Z1, out$Z2,
                          out$dists1, out$dists2, out$signs1, out$signs2,
                          n_threads = 2L)
  expect_equal(nrow(S), length(out$cs$y1))
  expect_equal(colSums(S), attr(out$r, "gradient"),
               tolerance = 1e-7, ignore_attr = TRUE)
})

test_that("OPG vcov is symmetric positive-(semi)definite", {
  par <- c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1,
           log(0.2), log(0.2), log(0.2),
           log(0.2), log(0.2), log(0.2),
           log(0.5), log(0.5), 0.1)
  out <- run_pair(c("(Intercept)", "x1", "x2"), c("(Intercept)", "x1", "x2"), par)
  S <- bnbr_rp_scores_cpp(par, out$cs$y1, out$cs$y2, out$cs$X1, out$cs$X2,
                          out$cs$X1[, out$rand_idx1, drop = FALSE],
                          out$cs$X2[, out$rand_idx2, drop = FALSE],
                          out$rand_idx1, out$rand_idx2, out$Z1, out$Z2,
                          out$dists1, out$dists2, out$signs1, out$signs2,
                          n_threads = 2L)
  vc <- opg_vcov(S, paste0("p", seq_len(ncol(S))))$vcov
  expect_equal(vc, t(vc), tolerance = 1e-10)
  expect_true(all(diag(vc) >= 0))
})

test_that("fixed-bounds C++ value matches R Hessian objective", {
  par <- c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1,
           log(0.2), log(0.2), log(0.2),
           log(0.2), log(0.2), log(0.2),
           log(0.5), log(0.5), 0.1)
  out <- run_pair(c("(Intercept)", "x1", "x2"), c("(Intercept)", "x1", "x2"), par)
  # Recover bounds from the full C++ call for a valid (lamLo, lamHi).
  cs <- out$cs
  lamLo <- -0.5; lamHi <- 0.5
  r_fb <- bnbr_rp_ll_fixed_bounds(par, cs$y1, cs$y2, cs$X1, cs$X2,
                                  cs$X1[, out$rand_idx1, drop = FALSE],
                                  cs$X2[, out$rand_idx2, drop = FALSE],
                                  out$rand_idx1, out$rand_idx2, out$Z1, out$Z2,
                                  lamLo, lamHi,
                                  out$dists1, out$dists2, out$signs1, out$signs2)
  c_fb <- bnbr_rp_ll_fixed_bounds_cpp(par, cs$y1, cs$y2, cs$X1, cs$X2,
                                      cs$X1[, out$rand_idx1, drop = FALSE],
                                      cs$X2[, out$rand_idx2, drop = FALSE],
                                      out$rand_idx1, out$rand_idx2, out$Z1, out$Z2,
                                      lamLo, lamHi,
                                      out$dists1, out$dists2, out$signs1, out$signs2,
                                      n_threads = 2L)
  expect_equal(c_fb, r_fb, tolerance = 1e-9)
})

test_that("scores and gradient agree under a SUPPLIED interval, and the argument matters", {
  skip_on_cran()
  skip_if_not(rpbnb:::rpbnb_cpp_available(), "C++ core unavailable")
  # The existing identity test calls both wrappers with lam_bounds = NULL, so it
  # validates only the recomputed-bound default and would pass with the
  # fit-level wiring removed. This one supplies the frozen interval to both, as
  # fit_rpbnb() does, on a fixture whose support bound genuinely moves.
  #
  # A uniform random coefficient makes the bound depend on beta, m and s.
  set.seed(31)
  n <- 40L
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 2)
  XR1 <- X1[, 2, drop = FALSE]; XR2 <- X2[, 2, drop = FALSE]
  U1 <- rpbnb:::halton_uniform(50, 1, burn = 50)
  U2 <- rpbnb:::halton_uniform(50, 1, burn = 50)
  # "start" and "evaluation point" deliberately differ, so the frozen interval
  # is not the one the evaluation point would produce.
  p_start <- c(0.2, 0.1, 0.15, -0.1, log(0.30), log(0.30), log(0.4), log(0.4), 0.1)
  p_eval  <- c(0.5, 0.4, 0.05, -0.4, log(0.85), log(0.85), log(0.9), log(0.9), 0.6)
  frozen <- rpbnb:::.rp_support_bounds(p_start, X1, X2, 2L, 2L,
                                       "uniform", "uniform", 1, 1)
  at_eval <- rpbnb:::.rp_support_bounds(p_eval, X1, X2, 2L, 2L,
                                        "uniform", "uniform", 1, 1)
  # Premise: the two intervals differ. Without this the test is vacuous.
  expect_false(isTRUE(all.equal(unname(frozen), unname(at_eval))))

  g <- attr(bnbr_rp_ll_and_grad_cpp(p_eval, y1, y2, X1, X2, XR1, XR2,
                                    2L, 2L, U1, U2, "uniform", "uniform", 1, 1,
                                    n_threads = 1L, lam_bounds = frozen),
            "gradient")
  S <- bnbr_rp_scores_cpp(p_eval, y1, y2, X1, X2, XR1, XR2,
                          2L, 2L, U1, U2, "uniform", "uniform", 1, 1,
                          n_threads = 1L, lam_bounds = frozen)
  expect_equal(colSums(S), as.numeric(g), tolerance = 1e-7, ignore_attr = TRUE)

  # Sensitivity control: dropping lam_bounds from the score wrapper -- exactly
  # the silent-default path that caused the defect -- must break the identity.
  # Without this the test could not distinguish the fixed wiring from the
  # broken one.
  S_default <- bnbr_rp_scores_cpp(p_eval, y1, y2, X1, X2, XR1, XR2,
                                  2L, 2L, U1, U2, "uniform", "uniform", 1, 1,
                                  n_threads = 1L)
  expect_gt(max(abs(colSums(S_default) - as.numeric(g))), 1e-6)
})
