# Residual-diagnostics tests. Fast tier builds synthetic fits (no optimization);
# a slow-gated tier exercises real fits. See helper-slow.R::skip_slow.

# ---- fixtures --------------------------------------------------------------
make_bnb_resid_fixture <- function() {
  set.seed(11)
  n  <- 200
  x  <- rnorm(n)
  X  <- stats::model.matrix(~ x)
  b1 <- c(0.3, 0.2); b2 <- c(0.1, -0.1)
  mu1 <- as.vector(exp(X %*% b1)); mu2 <- as.vector(exp(X %*% b2))
  m1 <- 0.4; m2 <- 0.5
  y1 <- stats::rnbinom(n, size = 1 / m1, mu = mu1)
  y2 <- stats::rnbinom(n, size = 1 / m2, mu = mu2)
  coef <- c(b1, b2, log(m1), log(m2))
  names(coef) <- c("b1:(Intercept)", "b1:x", "b2:(Intercept)", "b2:x",
                   "log_m1", "log_m2")
  structure(list(coef = coef, mu1 = mu1, mu2 = mu2, Y1 = y1, Y2 = y2,
                 X1 = X, X2 = X, dependence = "famoye"), class = "bnb_fit")
}

test_that("bnb Pearson residual matches (y-mu)/sqrt(mu + m*mu^2)", {
  f  <- make_bnb_resid_fixture()
  pr <- residuals(f, type = "pearson", margin = "y1")
  m1 <- exp(f$coef[["log_m1"]])
  expect_equal(pr, (f$Y1 - f$mu1) / sqrt(f$mu1 + m1 * f$mu1^2), tolerance = 1e-12)
})

test_that("bnb deviance residual: sign matches y-mu, ~0 when y==mu, finite at y=0", {
  f  <- make_bnb_resid_fixture()
  dv <- residuals(f, type = "deviance", margin = "y1")
  expect_equal(sign(dv), sign(f$Y1 - f$mu1))
  expect_true(all(is.finite(dv)))
  # a point with y exactly equal to mu has ~0 deviance residual
  m1 <- exp(f$coef[["log_m1"]]); r1 <- 1 / m1
  y0 <- 3; mu0 <- 3
  d0 <- rpbnb:::.nb2_deviance_resid(y0, mu0, m1)
  expect_equal(d0, 0, tolerance = 1e-8)
})

test_that("bnb RQR lies in (qnorm(F(y-1)), qnorm(F(y))) and is seed-reproducible", {
  f   <- make_bnb_resid_fixture()
  rq1 <- residuals(f, type = "quantile", margin = "y1", seed = 42)
  rq2 <- residuals(f, type = "quantile", margin = "y1", seed = 42)
  expect_equal(rq1, rq2)                         # reproducible
  m1 <- exp(f$coef[["log_m1"]]); r1 <- 1 / m1
  Fhi <- stats::pnbinom(f$Y1,     size = r1, mu = f$mu1)
  Flo <- ifelse(f$Y1 > 0, stats::pnbinom(f$Y1 - 1, size = r1, mu = f$mu1), 0)
  expect_true(all(rq1 >= stats::qnorm(Flo) - 1e-9 & rq1 <= stats::qnorm(Fhi) + 1e-9))
  rq3 <- residuals(f, type = "quantile", margin = "y1", seed = 7)
  expect_false(isTRUE(all.equal(rq1, rq3)))      # different seed differs
})

test_that("bnb residuals(margin='both') returns a two-column data frame", {
  f <- make_bnb_resid_fixture()
  d <- residuals(f, type = "pearson", margin = "both")
  expect_s3_class(d, "data.frame")
  expect_named(d, c("y1", "y2"))
  expect_equal(nrow(d), length(f$Y1))
})

test_that("residuals(seed=) does not disturb the caller's RNG stream", {
  f <- make_bnb_resid_fixture()
  set.seed(99); a <- runif(1)
  set.seed(99); invisible(residuals(f, type = "quantile", margin = "both", seed = 5)); b <- runif(1)
  expect_equal(a, b)
})
