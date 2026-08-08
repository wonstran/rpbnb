test_that("RP-BNB simulated-LL analytic gradient matches numDeriv", {
  set.seed(11)
  n  <- 30
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 2)
  XR1 <- X1[, 2, drop = FALSE]; XR2 <- X2[, 2, drop = FALSE]
  rand_idx1 <- 2L; rand_idx2 <- 2L
  Z <- rpbnb:::halton_normal(60, 2, burn = 50)
  Z1 <- Z[, 1, drop = FALSE]; Z2 <- Z[, 2, drop = FALSE]
  par <- c(0.2, 0.1, 0.15, -0.1, log(0.3), log(0.3), log(0.4), log(0.4), 0.1)

  # The analytic gradient treats the lambda bounds as FROZEN -- exactly as the
  # frozen-bounds Hessian objective does. Differentiating an objective whose
  # bounds move with the parameters would differ by the omitted
  # d(bound)/d(theta) term, so the correct reference is the frozen-bounds
  # objective with bounds pinned at `par`.
  #
  # Those bounds are the coefficients' SUPPORT bound, taken from the package so
  # this cannot drift from what the objective uses. (For two unbounded normal
  # margins it is the constant [-1, 1], so the omitted term is exactly zero
  # there and the comparison is sharper than it used to be.)
  bnds <- rpbnb:::.rp_support_bounds(par, X1, X2, rand_idx1, rand_idx2,
                                     NULL, NULL, NULL, NULL)


  obj <- function(p) rpbnb:::bnbr_rp_ll_fixed_bounds(
    p, y1, y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2, Z1, Z2, bnds[1], bnds[2])
  v   <- rpbnb:::bnbr_rp_ll_and_grad(par, y1, y2, X1, X2, XR1, XR2,
                                     rand_idx1, rand_idx2, Z1, Z2)
  ana <- attr(v, "gradient")
  num <- numDeriv::grad(obj, par)
  expect_equal(ana, num, tolerance = 1e-3)
})

test_that("fit_rpbnb recovers simulated parameters (small case)", {
  skip_on_cran()
  sim <- simulate_rpbnb(
    n = 1500,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    random_2 = NULL,
    dispersion = c(m1 = 0.4, m2 = 0.5),
    lambda = 0,
    seed = 99
  )
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   random_1 = "x1", draws = 200, seed = 123,
                   control = rpbnb_control(compute_se = FALSE))
  expect_s3_class(fit, "rpbnb_fit")
  cf <- coef(fit)
  # Absolute closeness (the small-magnitude intercept makes relative
  # tolerance misleading; the estimate is ~0.034 from truth).
  expect_lt(abs(unname(cf[["b1:(Intercept)"]]) - 0.2), 0.15)
  expect_lt(abs(unname(cf[["b1:x1"]])          - 0.4), 0.15)
  # coef is estimation-scale: sd recovered via exp(log_sd)
  sd_hat <- exp(unname(cf[["log_sd1:x1"]]))
  expect_lt(abs(sd_hat - 0.5), 0.25)
  # natural-scale m stored as a field
  expect_lt(abs(unname(fit$m1) - 0.4), 0.2)
})

test_that("fit_rpbnb with sd->0 approaches fixed-parameter BNB", {
  skip_on_cran()
  sim <- simulate_rpbnb(
    n = 800,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = 0.1),
    random_1 = list(x1 = list(sd = 1e-4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 5
  )
  rp <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                  draws = 150, seed = 1, control = rpbnb_control(compute_se = FALSE))
  bn <- fit_bnb(y1 ~ x1, y2 ~ x1, data = sim$data, dependence = "famoye")
  expect_equal(unname(coef(rp)[["b1:x1"]]),
               unname(coef(bn)[["b1:x1"]]), tolerance = 0.1)
})

test_that("fit_rpbnb honours the seed (different seeds -> different draws/estimates)", {
  skip_on_cran()
  sim <- simulate_rpbnb(n = 400,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.4), seed = 6)
  ctl <- rpbnb_control(compute_se = FALSE)
  f1  <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 80, seed = 1, control = ctl)
  f1b <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 80, seed = 1, control = ctl)
  f2  <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 80, seed = 2, control = ctl)
  expect_equal(unname(coef(f1)), unname(coef(f1b)), tolerance = 1e-10)
  expect_false(isTRUE(all.equal(unname(coef(f1)), unname(coef(f2)), tolerance = 1e-6)))
})

test_that("predict.rpbnb_fit on newdata reproduces the stored draw-averaged means", {
  skip_on_cran()
  sim <- simulate_rpbnb(n = 500,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.4), seed = 5)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 300, seed = 1, control = rpbnb_control(compute_se = FALSE))
  p_stored <- predict(fit)                       # draw-averaged unconditional means
  p_new    <- predict(fit, newdata = sim$data)   # closed-form unconditional means
  expect_equal(p_new$mu1, p_stored$mu1, tolerance = 0.02)
  expect_equal(p_new$mu2, p_stored$mu2, tolerance = 0.02)

  # Independent oracle: the closed-form unconditional mean must differ
  # MATERIALLY from the naive exp(Xb) (no heterogeneity correction) on the
  # random equation -- i.e. the 0.5*sd^2*x^2 lift is actually applied. With
  # sd ~ 0.5 the average lift is well above 5%.
  b1 <- coef(fit)[grep("^b1:", names(coef(fit)))]
  names(b1) <- sub("^b1:", "", names(b1))
  X1 <- stats::model.matrix(~ x1, sim$data)
  mu_plain <- as.vector(exp(X1[, names(b1)] %*% b1))
  rel_gap <- mean(abs(p_new$mu1 - mu_plain)) / mean(p_new$mu1)
  expect_gt(rel_gap, 0.05)
})

test_that("predict honours a dot formula (y ~ .) end-to-end", {
  skip_on_cran()
  sim <- simulate_rpbnb(n = 200, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                        dispersion = c(m1 = 0.4, m2 = 0.4), seed = 9)
  d <- sim$data[, c("y1", "x1")]   # only y1 and x1 so `.` is unambiguous
  fit <- fit_bnb(y1 ~ ., y2 ~ x1,
                 data = cbind(d, y2 = sim$data$y2), dependence = "famoye")
  expect_s3_class(fit, "bnb_fit")
  expect_true("b1:x1" %in% names(coef(fit)))
})

test_that("fit_rpbnb errors on unknown random name", {
  sim <- simulate_rpbnb(n = 100, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                        beta2 = c("(Intercept)" = 0.1, x1 = 0.1),
                        dispersion = c(m1 = 0.4, m2 = 0.4), seed = 1)
  expect_error(
    fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "nope",
              draws = 50, control = rpbnb_control(compute_se = FALSE)),
    "nope"
  )
})

test_that("fit_rpbnb compute_se=TRUE yields a finite covariance and SEs", {
  skip_on_cran()
  sim <- simulate_rpbnb(n = 400,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 7)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 80, seed = 3,
                   control = rpbnb_control(compute_se = TRUE, draws_hessian = 40))
  V <- vcov(fit)
  expect_true(is.matrix(V))
  expect_equal(nrow(V), length(coef(fit)))
  expect_true(all(is.finite(fit$se)))
  expect_true(all(diag(V) >= 0))
})

test_that("numeric RP SEs use the optimization draws (independent of draws_hessian)", {
  skip_on_cran()
  sim <- simulate_rpbnb(n = 300,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 9)
  base <- function(dh) fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
    draws = 120, seed = 4,
    control = rpbnb_control(se_method = "numeric", draws_hessian = dh, print_level = 0))
  f_small <- base(30)
  f_large <- base(300)
  # The numeric Hessian is now taken with the SAME optimization draws that
  # produced the estimate, so it no longer depends on draws_hessian.
  expect_equal(unname(f_small$se), unname(f_large$se), tolerance = 1e-8)
})

test_that("fit_rpbnb parallel path matches sequential and respects workers", {
  skip_on_cran()
  skip_if_not_installed("parallel")
  sim <- simulate_rpbnb(n = 300,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 8)
  seq_fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                       draws = 60, seed = 2,
                       control = rpbnb_control(compute_se = FALSE, n_cores = 1))
  par_fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                       draws = 60, seed = 2,
                       control = rpbnb_control(compute_se = FALSE, n_cores = 2))
  # Same draws + seed => statistically identical estimates (parallel only splits the draw loop)
  expect_equal(unname(coef(seq_fit)), unname(coef(par_fit)), tolerance = 1e-6)
})
