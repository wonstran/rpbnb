# Tests for the EXACT m = 0 (true Poisson) branch that replaces the fixed
# POISSON_M = 1e-6 approximation for poisson_1/poisson_2 restricted margins.
# The Poisson limit of the NB2 core: log-pmf -> dpois, c(mu,m) -> exp(-d*mu),
# CDF -> ppois, variance -> mu. These must be EXACT (full precision), not the
# old m*mu-governed approximation, at any fitted mean.

test_that("c_val(mu, m = 0) is the exact Poisson limit exp(-d*mu)", {
  d  <- 1 - exp(-1)
  mu <- c(0.5, 2, 100, 1e4, 1e6)
  expect_equal(rpbnb:::c_val(mu, 0), exp(-d * mu))
})

test_that("nb_logpmf_y_mu_r(y, mu, r = Inf) is the exact Poisson log-pmf", {
  y  <- c(0, 1, 2, 3, 10)
  mu <- c(2, 2, 2, 5, 8)
  expect_equal(rpbnb:::nb_logpmf_y_mu_r(y, mu, Inf),
               stats::dpois(y, mu, log = TRUE))
})

test_that("exact Poisson log-pmf matches dpois even at very large means", {
  # The whole point: the old fixed-m stand-in diverged by ~0.35/obs at mu = 1e6.
  # The exact branch must agree with dpois to full double precision there.
  expect_equal(rpbnb:::nb_logpmf_y_mu_r(1e6, 1e6, Inf),
               stats::dpois(1e6, 1e6, log = TRUE))
  expect_equal(rpbnb:::nb_logpmf_y_mu_r(0, 1e4, Inf),
               stats::dpois(0, 1e4, log = TRUE))
})

# ---- Independent reference for a Famoye per-obs logLik with a Poisson margin --
# Built ONLY from base-R dpois/dnbinom and exp(-d*mu), NOT from the package's own
# (now Poisson-aware) c_val/nb_logpmf, so it is a genuine oracle.
.famoye_ref_ll <- function(par, y1, y2, X1, X2, pois1, pois2) {
  d  <- 1 - exp(-1)
  p1 <- ncol(X1); p2 <- ncol(X2)
  beta1 <- par[seq_len(p1)]; beta2 <- par[p1 + seq_len(p2)]
  m1 <- exp(par[p1 + p2 + 1]); m2 <- exp(par[p1 + p2 + 2]); zlam <- par[p1 + p2 + 3]
  mu1 <- pmin(pmax(as.vector(exp(X1 %*% beta1)), 1e-300), 1e15)
  mu2 <- pmin(pmax(as.vector(exp(X2 %*% beta2)), 1e-300), 1e15)
  c1 <- if (pois1) exp(-d * mu1) else (1 + d * m1 * mu1)^(-1 / m1)
  c2 <- if (pois2) exp(-d * mu2) else (1 + d * m2 * mu2)^(-1 / m2)
  bnds <- rpbnb:::lambda_bounds_vec(c1, c2); eps <- 1e-6
  lam  <- bnds[1] + (bnds[2] - bnds[1]) * (eps + (1 - 2 * eps) * plogis(zlam))
  l1 <- if (pois1) stats::dpois(y1, mu1, log = TRUE)
        else stats::dnbinom(y1, size = 1 / m1, mu = mu1, log = TRUE)
  l2 <- if (pois2) stats::dpois(y2, mu2, log = TRUE)
        else stats::dnbinom(y2, size = 1 / m2, mu = mu2, log = TRUE)
  dep <- pmax(1 + lam * (exp(-y1) - c1) * (exp(-y2) - c2), 1e-300)
  l1 + l2 + log(dep)
}

test_that("bnb_loglik_vec(pois1=TRUE) equals the exact Poisson-margin reference at large means", {
  set.seed(11)
  n <- 40
  X <- cbind(1, rnorm(n))
  # Large mean on margin 1 (where the old approximation broke down).
  beta1 <- c(9, 0.1); beta2 <- c(0.3, -0.2)
  y1 <- rpois(n, exp(X %*% beta1)); y2 <- rnbinom(n, size = 2, mu = exp(X %*% beta2))
  par <- c(beta1, beta2, log(rpbnb:::POISSON_M), log(0.5), 0.3)
  got <- rpbnb:::bnb_loglik_vec(par, y1, y2, X, X, pois1 = TRUE, pois2 = FALSE)
  ref <- .famoye_ref_ll(par, y1, y2, X, X, pois1 = TRUE, pois2 = FALSE)
  expect_equal(got, ref)
})

test_that("bnb_loglik_vec(pois2=TRUE) and both-Poisson match the reference", {
  set.seed(12)
  n <- 30
  X <- cbind(1, rnorm(n))
  par <- c(0.4, 0.2, 8, 0.05, log(0.5), log(rpbnb:::POISSON_M), -0.4)
  y1 <- rnbinom(n, size = 2, mu = exp(X %*% par[1:2]))
  y2 <- rpois(n, exp(X %*% par[3:4]))
  expect_equal(rpbnb:::bnb_loglik_vec(par, y1, y2, X, X, pois2 = TRUE),
               .famoye_ref_ll(par, y1, y2, X, X, pois1 = FALSE, pois2 = TRUE))

  par2 <- c(6, 0.1, 7, -0.1, log(rpbnb:::POISSON_M), log(rpbnb:::POISSON_M), 0.2)
  y1b <- rpois(n, exp(X %*% par2[1:2])); y2b <- rpois(n, exp(X %*% par2[3:4]))
  expect_equal(rpbnb:::bnb_loglik_vec(par2, y1b, y2b, X, X, pois1 = TRUE, pois2 = TRUE),
               .famoye_ref_ll(par2, y1b, y2b, X, X, pois1 = TRUE, pois2 = TRUE))
})

test_that("fit_bnb famoye Poisson fit stores the EXACT Poisson-margin logLik", {
  set.seed(21)
  n <- 400
  d <- data.frame(x = rnorm(n))
  d$y1 <- rpois(n, exp(0.5 + 0.3 * d$x))
  d$y2 <- rnbinom(n, size = 1.5, mu = exp(0.2 - 0.1 * d$x))
  fit <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE)
  X <- model.matrix(~ x, d)
  # The stored logLik must equal the exact Poisson-margin objective at the coef.
  ref <- sum(rpbnb:::bnb_loglik_vec(fit$coef, d$y1, d$y2, X, X, pois1 = TRUE))
  expect_equal(as.numeric(logLik(fit)), ref)
})

test_that("large-mean Poisson famoye fit is exact and does NOT warn (P2 regression, updated)", {
  set.seed(22)
  n <- 300
  d <- data.frame(x = rnorm(n))
  d$y1 <- rpois(n, lambda = exp(8 + 0.05 * d$x))    # mean ~ 3000: old code warned
  d$y2 <- rpois(n, lambda = exp(0.2))
  # No POISSON_M mean-range warning should fire: the fit is exact Poisson now.
  w <- testthat::capture_warnings(
    fit <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE))
  expect_false(any(grepl("POISSON_M", w)))
  X <- model.matrix(~ x, d)
  ref <- sum(rpbnb:::bnb_loglik_vec(fit$coef, d$y1, d$y2, X, X, pois1 = TRUE))
  expect_equal(as.numeric(logLik(fit)), ref)
})

test_that("famoye Poisson-fit SEs agree across numeric and analytic Hessian", {
  set.seed(23)
  n <- 300
  d <- data.frame(x = rnorm(n))
  d$y1 <- rpois(n, exp(0.6 + 0.25 * d$x))
  d$y2 <- rnbinom(n, size = 1.5, mu = exp(0.2 - 0.1 * d$x))
  f_num <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE,
                   control = rpbnb_control(hessian = "numeric"))
  f_an  <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE,
                   control = rpbnb_control(hessian = "analytic"))
  free <- c("b1:(Intercept)", "b1:x", "b2:(Intercept)", "b2:x", "log_m2", "z_lambda")
  expect_equal(unname(f_num$se[free]), unname(f_an$se[free]), tolerance = 1e-3)
  expect_true(is.na(f_num$se[["log_m1"]]))          # fixed -> no SE
  expect_true(is.na(f_an$se[["log_m1"]]))
})

test_that("bnb_grad_vec(pois1=TRUE) matches numeric gradient of the exact objective (free params)", {
  set.seed(13)
  n <- 50
  X <- cbind(1, rnorm(n))
  beta1 <- c(7, 0.15); beta2 <- c(0.3, -0.2)
  y1 <- rpois(n, exp(X %*% beta1)); y2 <- rnbinom(n, size = 2, mu = exp(X %*% beta2))
  par <- c(beta1, beta2, log(rpbnb:::POISSON_M), log(0.5), 0.3)
  g_an <- rpbnb:::bnb_grad_vec(par, y1, y2, X, X, pois1 = TRUE, pois2 = FALSE)
  f <- function(p) sum(.famoye_ref_ll(p, y1, y2, X, X, pois1 = TRUE, pois2 = FALSE))
  g_num <- numDeriv::grad(f, par)
  free <- c(1, 2, 3, 4, 6, 7)                 # all but log_m1 (index 5, pinned)
  expect_equal(unname(g_an[free]), g_num[free], tolerance = 1e-5)
  # The pinned-margin dispersion score must be a finite placeholder (not NaN/Inf),
  # since maxLik consumes the full gradient vector even for fixed coordinates.
  expect_true(is.finite(g_an[5]))
})

# ============================================================================
# RP simulated-likelihood Poisson (m = 0) branch: R fallback, C++ core, Hessian.
# ============================================================================

# Build one RP test case with a random coefficient on x1 in each equation and a
# large mean on margin 1 (where a fixed-POISSON_M approximation degrades).
.rp_pois_case <- function(n = 60, R = 96L, big = TRUE, seed = 31) {
  set.seed(seed)
  x1 <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x1); X2 <- X1
  b1_int <- if (big) 7 else 0.4
  y1 <- rpois(n, exp(b1_int + 0.2 * x1))
  y2 <- rnbinom(n, mu = exp(0.3 - 0.1 * x1), size = 2)
  rand_idx1 <- 2L; rand_idx2 <- 2L
  XR1 <- X1[, rand_idx1, drop = FALSE]; XR2 <- X2[, rand_idx2, drop = FALSE]
  set.seed(99); Z <- halton_uniform(R, 2, burn = 50)
  Z1 <- Z[, 1, drop = FALSE]; Z2 <- Z[, 2, drop = FALSE]
  # par: beta1(2), beta2(2), log_sd1(1), log_sd2(1), log_m1, log_m2, z_lambda
  par <- c(b1_int, 0.2, 0.3, -0.1, log(0.15), log(0.2),
           log(rpbnb:::POISSON_M), log(0.5), 0.1)
  list(y1 = y1, y2 = y2, X1 = X1, X2 = X2, XR1 = XR1, XR2 = XR2,
       rand_idx1 = rand_idx1, rand_idx2 = rand_idx2, Z1 = Z1, Z2 = Z2, par = par,
       dist1 = "normal", dist2 = "normal", sign1 = 1, sign2 = 1)
}

test_that("R RP likelihood with no random coefs and pois1=TRUE equals the exact famoye reference", {
  set.seed(41); n <- 40
  X <- cbind(`(Intercept)` = 1, x = rnorm(n))
  beta1 <- c(7, 0.2); beta2 <- c(0.3, -0.1)
  y1 <- rpois(n, exp(X %*% beta1)); y2 <- rnbinom(n, size = 2, mu = exp(X %*% beta2))
  par <- c(beta1, beta2, log(rpbnb:::POISSON_M), log(0.5), 0.2)
  val <- rpbnb:::bnbr_rp_ll_and_grad(par, y1, y2, X, X, NULL, NULL,
                                     integer(0), integer(0),
                                     matrix(0, 1, 0), matrix(0, 1, 0),
                                     pois1 = TRUE, pois2 = FALSE)
  expect_equal(as.numeric(val), sum(.famoye_ref_ll(par, y1, y2, X, X, TRUE, FALSE)))
})

test_that("C++ RP matches R RP with pois1=TRUE at large means (value + gradient)", {
  skip_if_not(rpbnb_cpp_available(), "C++ likelihood not compiled")
  cs <- .rp_pois_case()
  r_val <- rpbnb:::bnbr_rp_ll_and_grad(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                                       cs$rand_idx1, cs$rand_idx2, cs$Z1, cs$Z2,
                                       cs$dist1, cs$dist2, cs$sign1, cs$sign2,
                                       pois1 = TRUE, pois2 = FALSE)
  c_val <- rpbnb:::bnbr_rp_ll_and_grad_cpp(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                                           cs$rand_idx1, cs$rand_idx2, cs$Z1, cs$Z2,
                                           cs$dist1, cs$dist2, cs$sign1, cs$sign2,
                                           n_threads = 2L, pois1 = TRUE, pois2 = FALSE)
  expect_equal(as.numeric(c_val), as.numeric(r_val), tolerance = 1e-8)
  expect_equal(attr(c_val, "gradient"), attr(r_val, "gradient"),
               tolerance = 1e-7, ignore_attr = TRUE)
  # log_m1 gradient (fixed Poisson margin) must be exactly 0, not a tiny-m artefact.
  expect_equal(attr(c_val, "gradient")[7], 0)
})

# Frozen lambda-bounds at `par` (mirrors fit_rpbnb's rebuild_bounds), with the
# Poisson margin's m forced to 0. The analytic RP gradient freezes these bounds,
# so the honest numeric check differentiates the FROZEN-bounds objective (the
# unfrozen objective's extra bounds-position derivative is a known analytic-vs-
# numeric gap unrelated to the Poisson branch).
.rp_frozen_bounds <- function(cs, par, pois1 = FALSE, pois2 = FALSE) {
  k1 <- ncol(cs$X1); k2 <- ncol(cs$X2)
  beta1 <- par[1:k1]; beta2 <- par[(k1 + 1):(k1 + k2)]
  m1 <- if (pois1) 0 else exp(par[k1 + k2 + 3]); m2 <- if (pois2) 0 else exp(par[k1 + k2 + 4])
  sd1 <- exp(par[k1 + k2 + 1]); sd2 <- exp(par[k1 + k2 + 2])
  dev1 <- rpbnb:::rand_realize(cs$Z1, cs$dist1, cs$sign1, beta1[cs$rand_idx1], sd1)$dev
  dev2 <- rpbnb:::rand_realize(cs$Z2, cs$dist2, cs$sign2, beta2[cs$rand_idx2], sd2)$dev
  xb1 <- as.vector(cs$X1 %*% beta1); xb2 <- as.vector(cs$X2 %*% beta2)
  lamLo <- -Inf; lamHi <- Inf
  for (r in seq_len(nrow(cs$Z1))) {
    mu1 <- pmin(exp(xb1 + as.vector(cs$XR1 %*% dev1[r, ])), 1e15)
    mu2 <- pmin(exp(xb2 + as.vector(cs$XR2 %*% dev2[r, ])), 1e15)
    b <- rpbnb:::lambda_bounds_vec(rpbnb:::c_val(mu1, m1), rpbnb:::c_val(mu2, m2))
    lamLo <- max(lamLo, b[1]); lamHi <- min(lamHi, b[2])
  }
  c(lamLo, lamHi)
}

test_that("RP analytic gradient with pois1=TRUE matches numeric gradient of the frozen-bounds objective", {
  cs <- .rp_pois_case(big = FALSE)             # moderate mean for stable finite diff
  g <- attr(rpbnb:::bnbr_rp_ll_and_grad(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                                        cs$rand_idx1, cs$rand_idx2, cs$Z1, cs$Z2,
                                        cs$dist1, cs$dist2, cs$sign1, cs$sign2,
                                        pois1 = TRUE, pois2 = FALSE), "gradient")
  lb <- .rp_frozen_bounds(cs, cs$par, pois1 = TRUE)
  f <- function(p) rpbnb:::bnbr_rp_ll_fixed_bounds(
    p, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2, cs$rand_idx1, cs$rand_idx2,
    cs$Z1, cs$Z2, lb[1], lb[2], cs$dist1, cs$dist2, cs$sign1, cs$sign2,
    pois1 = TRUE, pois2 = FALSE)
  gnum <- numDeriv::grad(f, cs$par)
  free <- c(1, 2, 3, 4, 5, 6, 8, 9)            # all but log_m1 (index 7)
  expect_equal(unname(g[free]), gnum[free], tolerance = 1e-4)
})

test_that("RP fixed-bounds value: R and C++ agree with pois1=TRUE", {
  skip_if_not(rpbnb_cpp_available(), "C++ likelihood not compiled")
  cs <- .rp_pois_case()
  lamLo <- -0.4; lamHi <- 0.4
  r_fb <- rpbnb:::bnbr_rp_ll_fixed_bounds(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                                          cs$rand_idx1, cs$rand_idx2, cs$Z1, cs$Z2,
                                          lamLo, lamHi, cs$dist1, cs$dist2, cs$sign1, cs$sign2,
                                          pois1 = TRUE, pois2 = FALSE)
  c_fb <- rpbnb:::bnbr_rp_ll_fixed_bounds_cpp(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                                              cs$rand_idx1, cs$rand_idx2, cs$Z1, cs$Z2,
                                              lamLo, lamHi, cs$dist1, cs$dist2, cs$sign1, cs$sign2,
                                              n_threads = 2L, pois1 = TRUE, pois2 = FALSE)
  expect_equal(c_fb, r_fb, tolerance = 1e-9)
})

test_that("RP scores with pois1=TRUE sum to the gradient and zero the log_m1 column", {
  skip_if_not(rpbnb_cpp_available(), "C++ likelihood not compiled")
  cs <- .rp_pois_case()
  S <- rpbnb:::bnbr_rp_scores_cpp(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                                  cs$rand_idx1, cs$rand_idx2, cs$Z1, cs$Z2,
                                  cs$dist1, cs$dist2, cs$sign1, cs$sign2,
                                  n_threads = 2L, pois1 = TRUE, pois2 = FALSE)
  g <- attr(rpbnb:::bnbr_rp_ll_and_grad_cpp(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                                            cs$rand_idx1, cs$rand_idx2, cs$Z1, cs$Z2,
                                            cs$dist1, cs$dist2, cs$sign1, cs$sign2,
                                            n_threads = 2L, pois1 = TRUE, pois2 = FALSE), "gradient")
  expect_equal(colSums(S), g, tolerance = 1e-7, ignore_attr = TRUE)
  expect_true(all(S[, 7] == 0))                # log_m1 column all zero
})

test_that("fit_rpbnb Poisson fit uses the exact m=0 branch: no POISSON_M warning, exact logLik, correct df/SE", {
  sim <- simulate_rpbnb(n = 250,
    beta1 = c("(Intercept)" = 6, x1 = 0.3),      # large mean: old approx warned
    beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 3)
  # compute_se = FALSE: at this extreme mean the numeric Hessian is near-singular
  # for BOTH the NB and Poisson fits (weak identification of the random slope), a
  # pre-existing large-mean artefact unrelated to the Poisson branch; SEs are
  # covered cleanly at moderate mean in the next test.
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)
  w <- testthat::capture_warnings(
    fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                     draws = 80, seed = 1, control = ctrl, poisson_1 = TRUE))
  expect_false(any(grepl("POISSON_M", w)))     # exact Poisson: no mean-range warning

  # Stored logLik equals the exact m = 0 objective recomputed on the fit's draws.
  X <- stats::model.matrix(~ x1, sim$data)
  XR1 <- X[, fit$rand_idx1, drop = FALSE]
  ref <- rpbnb:::bnbr_rp_ll_and_grad(fit$coef, sim$data$y1, sim$data$y2, X, X,
                                     XR1, NULL, fit$rand_idx1, integer(0),
                                     fit$rp_meta$Z1, fit$rp_meta$Z2,
                                     "normal", NULL, 1, NULL, pois1 = TRUE)
  expect_equal(as.numeric(logLik(fit)), as.numeric(ref), tolerance = 1e-6)

  # Nesting: one fewer free parameter than the full NB model. par = b1(2) + b2(2)
  # + log_sd1(1) + log_m1 + log_m2 + z_lambda = 8; pinning log_m1 leaves 7.
  expect_equal(fit$npar, 7L)
})

test_that("fit_rpbnb Poisson SE methods (numeric/opg/analytic) all succeed with finite free-param SEs", {
  sim <- simulate_rpbnb(n = 200,
    beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 5)
  for (sem in c("numeric", "opg", "analytic")) {
    ctrl <- rpbnb_control(print_level = 0, se_method = sem)
    fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                     draws = 60, seed = 1, control = ctrl, poisson_1 = TRUE)
    expect_true(is.na(fit$se[["log_m1"]]), info = sem)
    expect_true(all(is.finite(fit$se[c("b1:(Intercept)", "b1:x1", "log_m2", "z_lambda")])),
                info = sem)
  }
})

test_that("RP analytic Hessian with pois1=TRUE matches numeric Hessian over free params", {
  cs <- .rp_pois_case(big = FALSE)
  lamLo <- -0.4; lamHi <- 0.4
  H_an <- rpbnb:::bnbr_rp_hessian(cs$par, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2,
                                  cs$rand_idx1, cs$rand_idx2, cs$Z1, cs$Z2,
                                  cs$dist1, cs$dist2, cs$sign1, cs$sign2,
                                  lamLo = lamLo, lamHi = lamHi, pois1 = TRUE, pois2 = FALSE)
  f <- function(p) rpbnb:::bnbr_rp_ll_fixed_bounds(
    p, cs$y1, cs$y2, cs$X1, cs$X2, cs$XR1, cs$XR2, cs$rand_idx1, cs$rand_idx2,
    cs$Z1, cs$Z2, lamLo, lamHi, cs$dist1, cs$dist2, cs$sign1, cs$sign2,
    pois1 = TRUE, pois2 = FALSE)
  H_num <- numDeriv::hessian(f, cs$par)
  free <- c(1, 2, 3, 4, 5, 6, 8, 9)            # drop log_m1 (index 7)
  expect_equal(H_an[free, free], H_num[free, free], tolerance = 1e-3)
  # log_m1 row/col of the analytic Hessian is zeroed (fixed parameter).
  expect_true(all(H_an[7, ] == 0) && all(H_an[, 7] == 0))
})

# ============================================================================
# Residuals & diagnostics: a Poisson-restricted margin uses exact Poisson
# CDF (ppois) and variance (mu), not NB2 at r = 1/POISSON_M.
# ============================================================================

.pois_dev_resid <- function(y, mu) {
  ty <- ifelse(y == 0, 0, y * log(y / mu))
  sign(y - mu) * sqrt(pmax(2 * (ty - (y - mu)), 0))
}

test_that("bnb_fit residuals use exact Poisson variance and CDF for a restricted margin", {
  set.seed(61); n <- 300
  d <- data.frame(x = rnorm(n))
  d$y1 <- rpois(n, exp(0.5 + 0.2 * d$x))
  d$y2 <- rnbinom(n, size = 1.5, mu = exp(0.2 - 0.1 * d$x))
  fit <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", poisson_1 = TRUE)
  # Pearson: Poisson variance is mu (no m*mu^2 term).
  pr <- residuals(fit, type = "pearson", margin = "y1")
  expect_equal(pr, (fit$Y1 - fit$mu1) / sqrt(fit$mu1))
  # Deviance: exact Poisson deviance residual.
  dv <- residuals(fit, type = "deviance", margin = "y1")
  expect_equal(dv, .pois_dev_resid(fit$Y1, fit$mu1))
  # Quantile: CDF corners are ppois, not pnbinom(size = 1/POISSON_M).
  rq <- residuals(fit, type = "quantile", margin = "y1", seed = 7)
  Fhi <- stats::ppois(fit$Y1, fit$mu1)
  Flo <- ifelse(fit$Y1 > 0, stats::ppois(fit$Y1 - 1, fit$mu1), 0)
  set.seed(7); u <- runif(length(fit$Y1)); ref <- qnorm(Flo + u * (Fhi - Flo))
  expect_equal(rq, ref)
  # The NB margin (y2) is unchanged (still uses m2 > 0).
  pr2 <- residuals(fit, type = "pearson", margin = "y2")
  m2 <- exp(fit$coef[["log_m2"]])
  expect_equal(pr2, (fit$Y2 - fit$mu2) / sqrt(fit$mu2 + m2 * fit$mu2^2))
})

test_that("rpbnb_fit stores poisson flags and its mixture var/CDF use Poisson for a restricted margin", {
  sim <- simulate_rpbnb(n = 200,
    beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 8)
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 60, seed = 1, control = ctrl, poisson_1 = TRUE)
  expect_true(isTRUE(fit$poisson_1))
  expect_false(isTRUE(fit$poisson_2))
  # The stored natural-scale dispersion is exactly 0 for the Poisson margin, not
  # the POISSON_M = 1e-6 placeholder.
  expect_equal(fit$m1, 0)
  expect_gt(fit$m2, 0)

  # Mixture variance: Poisson law of total variance E_r[mu] + Var_r[mu] (no m*mu^2).
  p  <- rpbnb:::.rp_margin_parts(fit, 1L)
  expect_equal(p$m, 0)                                # Poisson margin -> m = 0
  mu <- rpbnb:::.rp_margin_mu_draws(p)
  v_ref <- rowMeans(mu) + (rowMeans(mu^2) - rowMeans(mu)^2)
  expect_equal(rpbnb:::.rp_mixture_var(fit, 1L), v_ref)

  # Mixture CDF: averages ppois over draws (must not segfault via pnbinom size=Inf).
  cc <- rpbnb:::.rp_mixture_cdf(fit, 1L)
  Rn <- ncol(mu)
  Fhi_ref <- rowMeans(vapply(seq_len(Rn),
    function(r) stats::ppois(fit$Y1, mu[, r]), numeric(nrow(mu))))
  expect_equal(cc$Fhi, Fhi_ref)
  expect_true(all(cc$Fhi >= 0 & cc$Fhi <= 1))
})
