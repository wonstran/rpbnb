make_small_bnb <- function() {
  set.seed(21)
  d <- data.frame(x = rnorm(400))
  d$y1 <- rnbinom(400, size = 2, mu = exp(0.3 + 0.2 * d$x))
  d$y2 <- rnbinom(400, size = 2, mu = exp(0.1 - 0.1 * d$x))
  fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye")
}

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

test_that("rpbnb_fit methods work, incl. NULL vcov under compute_se=FALSE", {
  sim <- simulate_rpbnb(n = 300, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                        beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
                        random_1 = list(x1 = list(sd = 0.4)),
                        dispersion = c(m1 = 0.4, m2 = 0.4), seed = 5)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 60, seed = 1, control = rpbnb_control(compute_se = FALSE))
  expect_s3_class(fit, "rpbnb_fit")
  expect_true(is.numeric(coef(fit)))
  expect_null(vcov(fit))
  expect_s3_class(logLik(fit), "logLik")
  expect_output(print(fit))         # must not error despite NA se / NULL vcov
  expect_output(print(summary(fit)))
  p <- predict(fit)
  expect_equal(nrow(p), fit$nobs)
})
