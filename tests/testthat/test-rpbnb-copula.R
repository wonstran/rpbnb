# With no random coefficients (q=0), the RP copula simulated log-likelihood must
# reduce EXACTLY to the fixed-model discrete-copula log-likelihood, for every
# family. Layouts coincide when q=0: (beta1, beta2, log_m1, log_m2, z_theta).

make_fixed_case <- function(n = 60, seed = 3) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x1); X2 <- cbind(`(Intercept)` = 1, x2 = x2)
  y1 <- rnbinom(n, mu = exp(0.3 + 0.2 * x1), size = 2)
  y2 <- rnbinom(n, mu = exp(0.2 - 0.1 * x2), size = 2)
  list(y1 = y1, y2 = y2, X1 = X1, X2 = X2)
}

test_that("RP copula LL reduces to fixed-model copula LL when q=0", {
  cs <- make_fixed_case()
  # layout: b1(2), b2(2), log_m1, log_m2, z_theta
  par <- c(0.3, 0.2, 0.2, -0.1, log(0.5), log(0.6), 0.4)
  Z0 <- matrix(0, 1, 0)
  for (fam in c("frank", "normal", "kimeldorf")) {
    rp <- bnbr_rp_copula_ll(par, cs$y1, cs$y2, cs$X1, cs$X2, NULL, NULL,
                            integer(0), integer(0), Z0, Z0, family = fam)
    fx <- sum(copula_loglik_vec(par, cs$y1, cs$y2, cs$X1, cs$X2, family = fam))
    expect_equal(rp, fx, tolerance = 1e-9, info = fam)
  }
})

test_that("per-draw copula pmf sums to ~1 over a count grid", {
  # single obs, single draw, moderate params -> rectangle pmf over grid sums to 1
  theta <- 5
  r1 <- 1 / 0.5; r2 <- 1 / 0.6; mu1 <- 2.0; mu2 <- 1.5
  grid <- expand.grid(y1 = 0:80, y2 = 0:80)
  a  <- pnbinom(grid$y1,     size = r1, mu = mu1)
  am <- ifelse(grid$y1 > 0, pnbinom(grid$y1 - 1, size = r1, mu = mu1), 0)
  b  <- pnbinom(grid$y2,     size = r2, mu = mu2)
  bm <- ifelse(grid$y2 > 0, pnbinom(grid$y2 - 1, size = r2, mu = mu2), 0)
  p <- frank_cdf(a, b, theta) - frank_cdf(am, b, theta) -
       frank_cdf(a, bm, theta) + frank_cdf(am, bm, theta)
  expect_equal(sum(p), 1, tolerance = 1e-3)
})

test_that("copula simulator produces correct marginals and dependence sign", {
  sim <- simulate_rpbnb_copula(
    n = 4000,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.6),   # rho = 0.6 -> positive dependence
    seed = 11
  )
  expect_setequal(names(sim$data), c("y1", "y2", "x1"))
  expect_equal(nrow(sim$data), 4000)
  # marginal means approximately match the model means
  expect_equal(mean(sim$data$y1), mean(sim$mu$mu1), tolerance = 0.15)
  expect_equal(mean(sim$data$y2), mean(sim$mu$mu2), tolerance = 0.15)
  # positive dependence built in -> positive Spearman correlation
  expect_gt(cor(sim$data$y1, sim$data$y2, method = "spearman"), 0.15)
})

test_that("copula simulator with rho=0 yields near-independent margins", {
  sim <- simulate_rpbnb_copula(
    n = 4000,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.0), seed = 12
  )
  expect_lt(abs(cor(sim$data$y1, sim$data$y2, method = "spearman")), 0.05)
})

test_that("fit_rpbnb dispatches to the copula path and returns a copula fit", {
  sim <- simulate_rpbnb_copula(
    n = 800,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.5), seed = 21)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   dependence = copula("normal"),
                   draws = 50, seed = 1,
                   control = rpbnb_control(compute_se = FALSE))
  expect_s3_class(fit, "rpbnb_fit")
  expect_identical(fit$cop_family, "normal")
  expect_true("z_theta" %in% names(fit$coef))
  expect_true(is.null(fit$lambda))
  # print() must use the copula branch (native param + Kendall's tau)
  out <- paste(capture.output(print(fit)), collapse = "\n")
  expect_match(out, "rho|tau|Gaussian")
})

test_that("fit_rpbnb default dependence is unchanged (famoye) and has z_lambda", {
  sim <- simulate_rpbnb_copula(
    n = 400, beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.3), seed = 31)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, draws = 50, seed = 1,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true("z_lambda" %in% names(fit$coef))
  expect_null(fit$cop_family)
})

test_that("predict() on a copula fit errors without newdata but works with it", {
  sim <- simulate_rpbnb_copula(
    n = 300,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.5), seed = 41)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   dependence = copula("normal"),
                   draws = 40, seed = 1,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true(is.null(fit$mu1))
  expect_true(is.null(fit$mu2))

  expect_error(predict(fit), "copula")

  pred <- predict(fit, newdata = sim$data)
  expect_s3_class(pred, "data.frame")
  expect_true(all(c("mu1", "mu2") %in% names(pred)))
  expect_equal(nrow(pred), nrow(sim$data))
})

recover_copula <- function(fam, par, n = 2000, draws = 200, seed = 7) {
  sim <- simulate_rpbnb_copula(
    n = n,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    random_1 = list(x1 = list(sd = 0.3)),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula(fam, par = par), seed = seed)
  # compute_se = FALSE: these tests assert only the point estimate (est vs true);
  # the numeric Hessian for SEs is unused here and dominates runtime.
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   random_1 = "x1", random_2 = "x1",
                   dependence = copula(fam),
                   draws = draws, seed = 1,
                   control = rpbnb_control(compute_se = FALSE, print_level = 0))
  z <- fit$coef[["z_theta"]]
  list(true = par, est = z_to_native(fam, z), sim_tau = sim$true$tau)
}

test_that("Gaussian copula parameter is recovered", {
  r <- recover_copula("normal", par = 0.5)
  expect_equal(r$est, r$true, tolerance = 0.12)
})

test_that("Frank copula parameter is recovered", {
  r <- recover_copula("frank", par = 4)
  expect_equal(r$est, r$true, tolerance = 1.2)   # theta on its native (wide) scale
})

test_that("Clayton copula parameter is recovered", {
  r <- recover_copula("kimeldorf", par = 1.0)
  expect_equal(r$est, r$true, tolerance = 0.5)
})

grad_case <- function(random_cols_1, random_cols_2, par, fam,
                      dists1 = NULL, dists2 = NULL, signs1 = NULL, signs2 = NULL,
                      n = 60, R = 96, seed = 5) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x1, x2 = x2); X2 <- X1
  y1 <- rnbinom(n, mu = exp(0.4 + 0.2 * x1), size = 2)
  y2 <- rnbinom(n, mu = exp(0.3 - 0.1 * x2), size = 2)
  ri1 <- match(random_cols_1, colnames(X1)); ri2 <- match(random_cols_2, colnames(X2))
  q1 <- length(ri1); q2 <- length(ri2)
  XR1 <- if (q1) X1[, ri1, drop = FALSE] else NULL
  XR2 <- if (q2) X2[, ri2, drop = FALSE] else NULL
  if (is.null(dists1)) dists1 <- rep("normal", q1)
  if (is.null(dists2)) dists2 <- rep("normal", q2)
  if (is.null(signs1)) signs1 <- rep(1, q1)
  if (is.null(signs2)) signs2 <- rep(1, q2)
  set.seed(77); Z <- halton_uniform(R, q1 + q2, burn = 40)
  Z1 <- if (q1) Z[, seq_len(q1), drop = FALSE] else matrix(0, R, 0)
  Z2 <- if (q2) Z[, (q1 + 1):(q1 + q2), drop = FALSE] else matrix(0, R, 0)
  ana <- bnbr_rp_copula_ll_grad(par, y1, y2, X1, X2, XR1, XR2, ri1, ri2, Z1, Z2,
                                fam, dists1, dists2, signs1, signs2, want_scores = TRUE)
  num <- numDeriv::grad(function(p)
    bnbr_rp_copula_ll(p, y1, y2, X1, X2, XR1, XR2, ri1, ri2, Z1, Z2,
                      fam, dists1, dists2, signs1, signs2), par)
  list(ana = ana, num = num)
}

test_that("copula RP analytic gradient matches numeric (all families, q>0)", {
  par <- c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1, log(0.25), log(0.2), log(0.5), log(0.6), 0.3)
  for (fam in c("frank", "normal", "kimeldorf")) {
    r <- grad_case(c("x1"), c("x2"), par, fam)
    expect_equal(as.numeric(attr(r$ana, "gradient")), r$num, tolerance = 1e-6, info = fam)
    expect_equal(colSums(attr(r$ana, "scores")), as.numeric(attr(r$ana, "gradient")),
                 tolerance = 1e-8, ignore_attr = TRUE, info = fam)
  }
})

test_that("copula RP analytic gradient matches numeric (q=0)", {
  par <- c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1, log(0.5), log(0.6), 0.2)
  r <- grad_case(character(0), character(0), par, "normal")
  expect_equal(as.numeric(attr(r$ana, "gradient")), r$num, tolerance = 1e-6)
})

test_that("copula RP analytic gradient matches numeric (lognormal random coef)", {
  par <- c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1, log(0.2), log(0.2), log(0.5), log(0.6), 0.25)
  r <- grad_case(c("x1"), c("x1"), par, "frank",
                 dists1 = "lognormal", dists2 = "normal", signs1 = 1, signs2 = 1)
  expect_equal(as.numeric(attr(r$ana, "gradient")), r$num, tolerance = 1e-6)
})

test_that("copula fit with se_method='opg' gives finite SEs; 'analytic' errors", {
  sim <- simulate_rpbnb_copula(
    n = 600, beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    random_1 = list(x1 = list(sd = 0.3)),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("normal", par = 0.4), seed = 41)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   random_1 = "x1", random_2 = "x1", dependence = copula("normal"),
                   draws = 80, seed = 1,
                   control = rpbnb_control(se_method = "opg"))
  expect_true(is.finite(fit$se[["z_theta"]]))
  expect_true(all(is.finite(diag(fit$vcov))))
  expect_error(
    fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1", random_2 = "x1",
              dependence = copula("normal"), draws = 40, seed = 1,
              control = rpbnb_control(se_method = "analytic")),
    "opg|numeric")
})
