# Tests for lr_test(): likelihood-ratio comparison of two nested fits.

# Minimal stand-in for a fitted model: lr_test() only needs logLik() with a
# "df" attribute, which both bnb_fit and rpbnb_fit provide.
fake_fit <- function(ll, df) {
  structure(list(logLik = ll, npar = df), class = "fake_lr_fit")
}
logLik.fake_lr_fit <- function(object, ...) {
  structure(as.numeric(object$logLik), df = object$npar, class = "logLik")
}
# Register so stats::logLik dispatches to it from inside the package namespace.
.S3method("logLik", "fake_lr_fit", logLik.fake_lr_fit)

test_that("interior chi-square test matches pchisq on the raw statistic", {
  full <- fake_fit(-1234.56, df = 12)
  rest <- fake_fit(-1240.11, df = 10)
  res  <- lr_test(rest, full)                 # boundary = FALSE default

  stat <- 2 * (-1234.56 - (-1240.11))
  expect_equal(res$statistic, stat)
  expect_equal(res$df, 2)
  expect_equal(res$p.value, stats::pchisq(stat, 2, lower.tail = FALSE))
  expect_false(res$boundary)
})

test_that("boundary correction halves the df=1 interior p-value", {
  full <- fake_fit(-100, df = 6)
  rest <- fake_fit(-102, df = 5)              # one boundary parameter
  stat <- 2 * (-100 - (-102))                 # = 4

  interior <- lr_test(rest, full, boundary = FALSE)
  bnd      <- lr_test(rest, full, boundary = TRUE)

  # df = 1: mixture is 0.5*chisq(1) + 0.5*point-mass-at-0 -> exactly half.
  expect_equal(bnd$p.value, 0.5 * interior$p.value)
  expect_true(bnd$boundary)
})

test_that("boundary correction with df=2 is the 50:50 chisq(2)/chisq(1) mixture", {
  full <- fake_fit(-100, df = 7)
  rest <- fake_fit(-105, df = 5)
  stat <- 2 * (-100 - (-105))                 # = 10, df = 2
  bnd  <- lr_test(rest, full, boundary = TRUE)

  expected <- 0.5 * stats::pchisq(stat, 2, lower.tail = FALSE) +
              0.5 * stats::pchisq(stat, 1, lower.tail = FALSE)
  expect_equal(bnd$p.value, expected)
})

test_that("df must be strictly positive (full has more parameters)", {
  full <- fake_fit(-100, df = 5)
  rest <- fake_fit(-102, df = 5)              # same df -> not nested this way
  expect_error(lr_test(rest, full), "more parameter")

  swapped <- fake_fit(-100, df = 5)
  bigger  <- fake_fit(-102, df = 7)
  expect_error(lr_test(bigger, swapped), "more parameter")  # args swapped
})

test_that("negative statistic warns and clamps to a conservative p = 1", {
  # restricted fits BETTER than full -> stat < 0 (convergence/nesting problem)
  full <- fake_fit(-105, df = 7)
  rest <- fake_fit(-100, df = 5)
  expect_warning(res <- lr_test(rest, full), "log-likelihood")
  expect_equal(res$statistic, 0)
  expect_equal(res$p.value, 1)
})

test_that("print method reports both fits, the statistic, and the mixture note", {
  full <- fake_fit(-100, df = 7)
  rest <- fake_fit(-105, df = 5)
  out <- capture.output(print(lr_test(rest, full, boundary = TRUE)))
  txt <- paste(out, collapse = "\n")
  expect_match(txt, "Likelihood-ratio test")
  expect_match(txt, "LR statistic")
  expect_match(txt, "mixture")                # boundary note present
})

test_that("end-to-end: real rpbnb_fit with vs without a random coefficient", {
  skip_slow()
  sim <- simulate_rpbnb(n = 600,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
  ctrl <- rpbnb_control(compute_se = FALSE)
  full <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                    draws = 100, control = ctrl)
  rest <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                    draws = 100, control = ctrl)   # no random coefficient

  res <- lr_test(rest, full, boundary = TRUE)
  expect_equal(res$df, full$npar - rest$npar)      # one extra SD parameter
  expect_true(is.finite(res$statistic) && res$statistic >= 0)
  expect_true(res$p.value >= 0 && res$p.value <= 1)
})
