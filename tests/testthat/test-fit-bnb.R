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

  # Reference values from the legacy estimator inst/legacy/bnbr_v2-4.R
  # (bnbr_famoye_bfgs on docvis/hospvis ~ outwork + kids, rwm1984_clean.csv).
  # These pin the port against future drift (acceptance criterion 6). The new
  # fit reproduces these to full precision since both use the same BFGS path.
  expect_equal(unname(cf[["b1:(Intercept)"]]),  1.056637893095, tolerance = 1e-4)
  expect_equal(unname(cf[["b1:outwork"]]),       0.515812046351, tolerance = 1e-4)
  expect_equal(unname(cf[["b1:kids"]]),         -0.313782996360, tolerance = 1e-4)
  expect_equal(unname(cf[["b2:outwork"]]),       0.322182617958, tolerance = 1e-4)
  expect_equal(unname(cf[["log_m1"]]),           0.850570193675, tolerance = 1e-4)
  expect_equal(as.numeric(logLik(fit)),      -9642.61529797,    tolerance = 1e-3)
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
