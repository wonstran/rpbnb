make_small_bnb <- function() {
  set.seed(21)
  d <- data.frame(x = rnorm(400))
  d$y1 <- rnbinom(400, size = 2, mu = exp(0.3 + 0.2 * d$x))
  d$y2 <- rnbinom(400, size = 2, mu = exp(0.1 - 0.1 * d$x))
  fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye")
}

test_that("famoye length(coef) == npar; independence drops the fake z_lambda", {
  ff <- make_small_bnb()
  expect_equal(length(coef(ff)), ff$npar)                 # famoye: betas + log_m1/2 + z_lambda
  expect_true("z_lambda" %in% names(coef(ff)))

  set.seed(21)
  d <- data.frame(x = rnorm(400))
  d$y1 <- rnbinom(400, size = 2, mu = exp(0.3 + 0.2 * d$x))
  d$y2 <- rnbinom(400, size = 2, mu = exp(0.1 - 0.1 * d$x))
  fi <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "independence")
  expect_false("z_lambda" %in% names(coef(fi)))
  expect_equal(length(coef(fi)), fi$npar)                 # independence: no lambda
  expect_equal(attr(logLik(fi), "df"), fi$npar)
  expect_equal(nrow(vcov(fi)), length(coef(fi)))
})

test_that("summary surfaces natural-scale dispersion and dependence with finite SEs", {
  ff <- make_small_bnb()                       # famoye, default control -> SEs computed
  s <- summary(ff)
  expect_false(is.null(s$natural))
  expect_true(any(grepl("^m1", s$natural$Parameter)))
  lam_row <- s$natural[grepl("lambda", s$natural$Parameter), ]
  m1_row  <- s$natural[grepl("^m1", s$natural$Parameter), ]
  expect_true(is.finite(lam_row$Estimate))     # natural-scale lambda, not z_lambda
  expect_true(is.finite(lam_row$StdErr))       # delta-method SE actually computed
  expect_true(is.finite(m1_row$StdErr))
})

test_that("exact-Poisson natural-scale reports m = 0, not the 1e-6 placeholder", {
  # A poisson_1 = TRUE fit pins log_m1 at log(POISSON_M) = log(1e-6) only as a
  # display placeholder; the public contract is that the margin is exactly m = 0.
  # The natural-scale table must report 0 for the restricted margin (with an NA
  # SE, a fixed parameter) and leave the unrestricted margin untouched.
  cf <- c("b1:(Intercept)" = 0.3, "b2:(Intercept)" = 0.1,
          log_m1 = log(1e-6), log_m2 = log(0.5), z_lambda = 0)
  se <- c("b1:(Intercept)" = 0.1, "b2:(Intercept)" = 0.1,
          log_m1 = NA_real_, log_m2 = 0.2, z_lambda = 0.3)
  fit <- structure(list(coef = cf, se = se, poisson_1 = TRUE, poisson_2 = FALSE,
                        lambda = 0, bounds = c(-1, 1)),
                   class = "bnb_fit")
  nat <- rpbnb:::.natural_scale_flat(fit)
  m1  <- nat[nat$Parameter == "m1 (dispersion)", ]
  m2  <- nat[nat$Parameter == "m2 (dispersion)", ]
  expect_equal(m1$Estimate, 0)                       # exact Poisson, not 1e-6
  expect_true(is.na(m1$StdErr))                      # fixed parameter -> NA SE
  expect_equal(m2$Estimate, 0.5, tolerance = 1e-12)  # unrestricted margin unchanged
})

test_that("raw summary()$coefficients suppresses Wald tests for log-scale/dispersion", {
  ff  <- make_small_bnb()
  cm  <- summary(ff)$coefficients
  m1  <- cm[cm$Parameter == "log_m1", ]
  b1x <- cm[cm$Parameter == "b1:x", ]
  # log_m1 = exp-transformed dispersion: no zero-null Wald test
  expect_true(is.na(m1$z) && is.na(m1$p))
  # regression coefficient keeps its test
  expect_true(is.finite(b1x$z) && is.finite(b1x$p))
})

test_that("natural-scale scale/dispersion rows carry no Wald test; lambda keeps one", {
  ff  <- make_small_bnb()
  nat <- summary(ff)$natural
  m1_row  <- nat[grepl("^m1", nat$Parameter), ]
  lam_row <- nat[grepl("lambda", nat$Parameter), ]
  # m = exp(log_m) is a positive scale: z = est/SE reduces to 1/SE(log_m) and is
  # not a Wald test of m = 0 (m = 0 is log_m = -Inf, a boundary). No z/p/stars.
  expect_true(is.na(m1_row$z))
  expect_true(is.na(m1_row$p))
  expect_true(is.finite(m1_row$Estimate) && is.finite(m1_row$StdErr))  # est/SE stay
  # lambda has an interior zero (independence), so a Wald test remains valid.
  expect_true(is.finite(lam_row$z) && is.finite(lam_row$p))
})

test_that("independence summary shows no lambda row (no dependence parameter)", {
  set.seed(21)
  d <- data.frame(x = rnorm(400))
  d$y1 <- rnbinom(400, size = 2, mu = exp(0.3 + 0.2 * d$x))
  d$y2 <- rnbinom(400, size = 2, mu = exp(0.1 - 0.1 * d$x))
  fi <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "independence")
  nat <- summary(fi)$natural
  expect_false(any(grepl("lambda", nat$Parameter)))
  expect_true(any(grepl("^m1", nat$Parameter)))
})

test_that("coef/vcov/logLik/AIC/BIC are consistent", {
  fit <- make_small_bnb()
  cf <- coef(fit); V <- vcov(fit)
  expect_true(is.numeric(cf) && length(cf) > 0)
  expect_equal(nrow(V), ncol(V))
  expect_equal(nrow(V), length(cf))
  expect_s3_class(logLik(fit), "logLik")
  expect_equal(AIC(fit), -2 * as.numeric(logLik(fit)) + 2 * attr(logLik(fit), "df"))
  expect_true(is.finite(BIC(fit)))
})

test_that("predict returns one mean per row per outcome", {
  fit <- make_small_bnb()
  p <- predict(fit)
  expect_equal(nrow(p), fit$nobs)
  expect_true(all(c("mu1", "mu2") %in% names(p)))
  expect_true(all(p$mu1 > 0))
})

test_that("predict works with newdata", {
  fit <- make_small_bnb()
  nd <- data.frame(x = c(-1, 0, 1))
  p <- predict(fit, newdata = nd)
  expect_equal(nrow(p), 3)
  expect_true(all(p$mu1 > 0))
})

test_that("summary and print run without error", {
  fit <- make_small_bnb()
  expect_output(print(fit))
  s <- summary(fit)
  expect_s3_class(s, "summary.bnb_fit")
  expect_output(print(s))
})

test_that("rpbnb_fit methods work, incl. NA vcov under compute_se=FALSE", {
  sim <- simulate_rpbnb(n = 300, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                        random_1 = list(x1 = list(sd = 0.4)),
                        dispersion = c(m1 = 0.4, m2 = 0.4), seed = 5)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 60, seed = 1, control = rpbnb_control(compute_se = FALSE))
  expect_s3_class(fit, "rpbnb_fit")
  expect_true(is.numeric(coef(fit)))
  V <- vcov(fit)
  expect_true(is.matrix(V))
  expect_equal(nrow(V), length(coef(fit)))
  expect_true(all(is.na(V)))
  expect_s3_class(logLik(fit), "logLik")
  expect_output(print(fit))         # must not error despite NA se / NA vcov
  expect_output(print(summary(fit)))
  p <- predict(fit)
  expect_equal(nrow(p), fit$nobs)
})
