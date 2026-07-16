test_that("at the optimum the analytic (frozen-bounds) gradient agrees with the true objective gradient", {
  # The optimizer's objective uses data-adaptive lambda bounds while the analytic
  # gradient treats them as frozen. This regression guard confirms the resulting
  # optimum is a genuine stationary point of the ACTUAL objective: the numerical
  # gradient of sum(bnb_loglik_vec) at the solution is ~0 and matches the analytic
  # gradient. (Codex review flagged a 3.71 mismatch -- but that was at an arbitrary
  # point far from the optimum, not at the solution.)
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
  fit <- fit_bnb(docvis ~ outwork + kids, hospvis ~ outwork + kids,
                 data = d, dependence = "famoye")
  ph <- fit$coef
  obj <- function(p) sum(rpbnb:::bnb_loglik_vec(p, fit$Y1, fit$Y2, fit$X1, fit$X2))
  g_true <- numDeriv::grad(obj, ph)
  g_anal <- rpbnb:::bnb_grad_vec(ph, fit$Y1, fit$Y2, fit$X1, fit$X2)
  expect_lt(max(abs(g_true)), 0.05)         # true objective gradient ~ 0 at optimum
  expect_lt(max(abs(g_true - g_anal)), 0.01) # analytic gradient agrees there
})

test_that("bnb_loglik_vec equals two NB2 logs + dependence term at lambda interior", {
  set.seed(7)
  n  <- 50
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 2)
  beta1 <- c(0.3, 0.1); beta2 <- c(0.2, -0.1)
  log_m1 <- log(0.5); log_m2 <- log(0.6); zlam <- 0
  par <- c(beta1, beta2, log_m1, log_m2, zlam)

  got <- rpbnb:::bnb_loglik_vec(par, y1, y2, X1, X2)

  m1 <- exp(log_m1); m2 <- exp(log_m2); r1 <- 1/m1; r2 <- 1/m2
  mu1 <- as.vector(exp(X1 %*% beta1)); mu2 <- as.vector(exp(X2 %*% beta2))
  c1 <- (1 + (1-exp(-1))*m1*mu1)^(-1/m1)
  c2 <- (1 + (1-exp(-1))*m2*mu2)^(-1/m2)
  b  <- rpbnb:::lambda_bounds_vec(c1, c2)
  lam <- b[1] + (b[2]-b[1]) * (1e-6 + (1-2e-6)*plogis(zlam))
  exp_ll <- dnbinom(y1, size = r1, mu = mu1, log = TRUE) +
            dnbinom(y2, size = r2, mu = mu2, log = TRUE) +
            log(pmax(1 + lam*(exp(-y1)-c1)*(exp(-y2)-c2), 1e-300))
  expect_equal(got, exp_ll, tolerance = 1e-10)
})

test_that("analytic gradient matches numDeriv", {
  set.seed(8)
  n  <- 40
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 3)
  par <- c(0.2, 0.1, 0.1, -0.05, log(0.5), log(0.5), 0.2)
  # The analytic score treats the data-adaptive lambda-bounds as a frozen
  # feasibility region (they are taken from max/min over observations and are
  # not differentiated). This is by design: BFGS and the numeric Hessian both
  # use this frozen-bounds objective. So verify the gradient against the
  # frozen-bounds log-likelihood, with bounds frozen at `par`.
  beta1 <- par[1:2]; beta2 <- par[3:4]
  m1 <- exp(par[5]); m2 <- exp(par[6])
  mu1 <- as.vector(exp(X1 %*% beta1)); mu2 <- as.vector(exp(X2 %*% beta2))
  c1 <- rpbnb:::c_val(mu1, m1); c2 <- rpbnb:::c_val(mu2, m2)
  b  <- rpbnb:::lambda_bounds_vec(c1, c2)
  f  <- function(p) rpbnb:::bnbr_loglik_fixed_bounds(p, y1, y2, X1, X2, b[1], b[2])
  num <- numDeriv::grad(f, par)
  ana <- rpbnb:::bnb_grad_vec(par, y1, y2, X1, X2)
  expect_equal(unname(ana), num, tolerance = 1e-4)
})

test_that("analytic Hessian matches numDeriv hessian of the frozen-bounds loglik", {
  # The analytic Hessian differentiates the SAME frozen-bounds objective that the
  # numeric Hessian (numDeriv) does. Verify they agree at a feasible interior
  # point with the lambda-bounds frozen at that point.
  set.seed(11)
  n  <- 60
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 2)
  par <- c(0.2, 0.1, 0.15, -0.05, log(0.6), log(0.7), 0.3)

  beta1 <- par[1:2]; beta2 <- par[3:4]; m1 <- exp(par[5]); m2 <- exp(par[6])
  mu1 <- as.vector(exp(X1 %*% beta1)); mu2 <- as.vector(exp(X2 %*% beta2))
  c1  <- rpbnb:::c_val(mu1, m1); c2 <- rpbnb:::c_val(mu2, m2)
  b   <- rpbnb:::lambda_bounds_vec(c1, c2)

  f   <- function(p) rpbnb:::bnbr_loglik_fixed_bounds(p, y1, y2, X1, X2, b[1], b[2])
  num <- numDeriv::hessian(f, par)
  ana <- rpbnb:::bnb_hessian_fixed_bounds(par, y1, y2, X1, X2, b[1], b[2])

  expect_equal(dim(ana), dim(num))
  expect_equal(unname(ana), unname(num), tolerance = 1e-4)
})

test_that("famoye analytic and numeric Hessian give matching standard errors", {
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  d  <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
  fn <- fit_bnb(docvis ~ outwork + kids, hospvis ~ outwork + kids, data = d,
                dependence = "famoye", control = rpbnb_control(hessian = "numeric"))
  fa <- fit_bnb(docvis ~ outwork + kids, hospvis ~ outwork + kids, data = d,
                dependence = "famoye", control = rpbnb_control(hessian = "analytic"))
  expect_equal(unname(fa$coef), unname(fn$coef), tolerance = 1e-6)
  expect_equal(unname(fa$se),   unname(fn$se),   tolerance = 1e-2)
  expect_true(all(is.finite(fa$se)))
})

sample_data <- function() {
  read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
}

test_that("fit_bnb famoye reproduces legacy bnbr_v2-4 estimates on rwm1984", {
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  d <- sample_data()
  fit <- fit_bnb(docvis ~ outwork + kids, hospvis ~ outwork + kids,
                 data = d, dependence = "famoye")
  expect_s3_class(fit, "bnb_fit")
  cf <- coef(fit)
  expect_true(all(c("b1:(Intercept)", "b2:(Intercept)") %in% names(cf)))
  expect_gt(cf[["b1:(Intercept)"]], 0)
  expect_true(is.finite(logLik(fit)))
  # p1 = p2 = 3 (intercept + outwork + kids) -> 6 betas + log_m1 + log_m2 +
  # z_lambda = 9 estimation-scale parameters.
  expect_equal(length(coef(fit)), 6 + 3)

  # Reference values pin the port against future drift (acceptance criterion 6).
  # These were re-pinned after correcting the Sarmanov lower lambda bound to use
  # the max((1-c1)(1-c2), c1*c2) corner (see famoye_core.R). The lambda interval
  # is reparameterized as lamLo + (lamHi-lamLo)*plogis(zlam), so tightening
  # lamLo shifts every estimate in the 4th-5th decimal versus the legacy
  # estimator inst/legacy/bnbr_v2-4.R, which still uses the old (too permissive)
  # bound. The log-likelihood is essentially unchanged (flat near the optimum).
  expect_equal(unname(cf[["b1:(Intercept)"]]),  1.056657961791, tolerance = 1e-4)
  expect_equal(unname(cf[["b1:outwork"]]),       0.515786142444, tolerance = 1e-4)
  expect_equal(unname(cf[["b1:kids"]]),         -0.313820736204, tolerance = 1e-4)
  expect_equal(unname(cf[["b2:outwork"]]),       0.322090316901, tolerance = 1e-4)
  expect_equal(unname(cf[["log_m1"]]),           0.850571235532, tolerance = 1e-4)
  expect_equal(as.numeric(logLik(fit)),      -9642.61529867,    tolerance = 1e-3)
})

test_that("famoye multi-start keeps the better of zero and glm.nb starts", {
  skip_on_cran()
  # High-mean DGP where the zero start converges to a worse optimum than the
  # marginal glm.nb start (see inst/validation/start_sensitivity_famoye.R).
  sim <- simulate_bnb(3000, c("(Intercept)" = 1.4, x = 0.2),
                      c("(Intercept)" = 1.2, x = -0.2),
                      dispersion = c(m1 = 0.5, m2 = 0.5), lambda = 0.8, seed = 202)
  d <- sim$data
  ctl <- rpbnb_control(compute_se = FALSE, print_level = 0)   # logLik only
  f_default <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", control = ctl)
  z <- c(0, 0, 0, 0, log(0.5), log(0.5), 0)          # zero start only
  f_zero <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye", start = z, control = ctl)
  # Multi-start must reach a strictly better objective than the zero start alone.
  expect_gt(as.numeric(logLik(f_default)), as.numeric(logLik(f_zero)) + 1)
})

test_that("fit_bnb independence equals two univariate NB2 fits", {
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  skip_if_not_installed("MASS")
  d <- sample_data()
  fit <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                 dependence = "independence")
  expect_s3_class(fit, "bnb_fit")
  g1 <- MASS::glm.nb(docvis ~ outwork, data = d)
  expect_equal(unname(coef(fit)[["b1:(Intercept)"]]),
               unname(coef(g1)[["(Intercept)"]]), tolerance = 1e-3)
  expect_equal(unname(coef(fit)[["b1:outwork"]]),
               unname(coef(g1)[["outwork"]]), tolerance = 1e-3)
})

test_that("fit_bnb famoye log-lik >= independence log-lik on the same data", {
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  d <- sample_data()
  fi <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d, dependence = "independence")
  ff <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d, dependence = "famoye")
  expect_gte(as.numeric(logLik(ff)) + 1e-6, as.numeric(logLik(fi)))
})

test_that("fit_bnb errors on negative counts", {
  d <- data.frame(y1 = c(1, -1, 2), y2 = c(0, 1, 2), x = rnorm(3))
  expect_error(fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye"), "non-negative")
})
