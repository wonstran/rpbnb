# Tests for shared response/design preparation: row alignment and count validation.

test_that("fit_bnb aligns rows when NAs fall in different rows across the two formulas", {
  d <- data.frame(
    y1 = c(1, 2, 3, 4, 5),
    y2 = c(2, 3, 4, 5, 6),
    x1 = c(0.1, NA, 0.3, 0.4, 0.5),  # NA row 2 -> eq1 would drop row 2
    x2 = c(0.5, 0.4, NA, 0.2, 0.1)   # NA row 3 -> eq2 would drop row 3
  )
  fit <- fit_bnb(y1 ~ x1, y2 ~ x2, data = d, dependence = "famoye")
  # Complete cases across BOTH formulas = rows 1, 4, 5
  expect_equal(fit$nobs, 3L)
  expect_equal(unname(fit$Y1), c(1L, 4L, 5L))
  expect_equal(unname(fit$Y2), c(2L, 5L, 6L))
})

test_that("fit_rpbnb aligns rows across formulas with differing NAs", {
  d <- data.frame(
    y1 = c(1, 0, 2, 3, 1, 0),
    y2 = c(0, 1, 1, 2, 0, 1),
    x1 = c(0.1, NA, 0.3, 0.4, 0.5, 0.6),
    x2 = c(0.5, 0.4, NA, 0.2, 0.1, 0.3)
  )
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x2, data = d, random_1 = "x1",
                   draws = 30, control = rpbnb_control(compute_se = FALSE))
  expect_equal(fit$nobs, 4L)               # rows 1, 4, 5, 6
  expect_equal(unname(fit$Y1), c(1L, 3L, 1L, 0L))
  expect_equal(unname(fit$Y2), c(0L, 2L, 0L, 1L))
})

test_that("a transformation that creates NA aligns both equations (eq1 drops)", {
  # log(-1) = NaN appears AFTER the raw variables are complete, so eq1 alone would
  # drop row 2. The common valid-row mask must drop it from BOTH equations, or the
  # two outcomes would be paired from mismatched rows (and reach C++ unequal).
  d <- data.frame(y1 = c(1, 2, 3, 4), y2 = c(2, 3, 4, 5),
                  x = c(1, -1, 2, 3), z = c(0.1, 0.2, 0.3, 0.4))
  # log(-1) legitimately warns "NaNs produced" (the transform is evaluated on the
  # raw column); the row is then dropped by the common-case mask. That warning is
  # inherent to the data, not the behavior under test (row alignment).
  prep <- suppressWarnings(rpbnb:::.prepare_bnb_data(y1 ~ log(x), y2 ~ z, d))
  expect_equal(length(prep$Y1), length(prep$Y2))
  expect_equal(nrow(prep$X1), nrow(prep$X2))
  expect_equal(unname(prep$Y1), c(1L, 3L, 4L))   # row 2 (log(-1)) dropped
  expect_equal(unname(prep$Y2), c(2L, 4L, 5L))   # and dropped from eq2 too
})

test_that("a transformation that creates NA aligns both equations (eq2 drops)", {
  # Symmetric direction: the NaN is in equation 2's transform.
  d <- data.frame(y1 = c(1, 2, 3, 4), y2 = c(2, 3, 4, 5),
                  z = c(0.1, 0.2, 0.3, 0.4), x = c(1, 2, -1, 3))
  prep <- suppressWarnings(rpbnb:::.prepare_bnb_data(y1 ~ z, y2 ~ log(x), d))
  expect_equal(length(prep$Y1), length(prep$Y2))
  expect_equal(nrow(prep$X1), nrow(prep$X2))
  expect_equal(unname(prep$Y1), c(1L, 2L, 4L))   # row 3 (log(-1)) dropped from both
  expect_equal(unname(prep$Y2), c(2L, 3L, 5L))
})

test_that("fit_bnb rejects non-integer responses instead of silently truncating", {
  d <- data.frame(y1 = c(1.5, 2, 3), y2 = c(0, 1, 2), x = c(0.1, 0.2, 0.3))
  expect_error(
    fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye"),
    "integer"
  )
})

test_that("fit_bnb rejects non-finite responses via the finiteness check", {
  d <- data.frame(y1 = c(1, Inf, 3), y2 = c(0, 1, 2), x = c(0.1, 0.2, 0.3))
  expect_error(
    fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye"),
    "non-finite"
  )
})

test_that("fit_rpbnb rejects non-integer responses", {
  d <- data.frame(y1 = c(1.2, 2, 3, 0), y2 = c(0, 1, 2, 1), x = c(0.1, 0.2, 0.3, 0.4))
  expect_error(
    fit_rpbnb(y1 ~ x, y2 ~ x, data = d, random_1 = "x", draws = 20,
              control = rpbnb_control(compute_se = FALSE)),
    "integer"
  )
})

test_that("fit_bnb independence path also uses the aligned complete-case rows", {
  set.seed(4)
  n <- 60
  x1 <- rnorm(n); x2 <- rnorm(n)
  d <- data.frame(
    y1 = rnbinom(n, size = 2, mu = exp(0.3 + 0.2 * x1)),
    y2 = rnbinom(n, size = 2, mu = exp(0.1 - 0.1 * x2)),
    x1 = x1, x2 = x2
  )
  d$x1[2] <- NA   # eq1 alone would drop row 2
  d$x2[5] <- NA   # eq2 alone would drop row 5
  fit <- suppressWarnings(
    fit_bnb(y1 ~ x1, y2 ~ x2, data = d, dependence = "independence"))
  expect_equal(fit$nobs, n - 2L)   # rows 2 and 5 dropped from BOTH equations
  expect_equal(length(fit$Y1), length(fit$Y2))
})

# ---- stateful transformations are re-evaluated on the retained sample --------

test_that("stateful poly() basis is re-evaluated on the retained rows", {
  # Row 2 is rejected because equation 2 has log(z) = log(0) = -Inf; its x value
  # is large, so a poly() basis computed over the FULL sample differs from the
  # basis over the retained rows. The stored design must use the retained basis.
  d <- data.frame(y1 = c(3, 5, 2, 4, 6, 1, 2, 3, 4, 5),
                  y2 = c(1, 2, 1, 0, 2, 1, 3, 2, 1, 0),
                  x  = c(1, 50, 2, 3, 4, 5, 6, 7, 8, 9),
                  z  = c(2, 0, 3, 4, 5, 6, 7, 8, 9, 10))
  prep <- rpbnb:::.prepare_bnb_data(y1 ~ poly(x, 2), y2 ~ log(z), d)
  X_ref <- stats::model.matrix(y1 ~ poly(x, 2), droplevels(d[-2, ]))
  expect_equal(nrow(prep$X1), 9L)
  expect_equal(unname(prep$X1), unname(X_ref), tolerance = 1e-10)
})

test_that("a rejected-only factor level is dropped cleanly when >=2 levels remain", {
  set.seed(3); n <- 90
  d <- data.frame(y1 = rpois(n, 2), y2 = rpois(n, 2),
                  f = factor(rep(c("a", "b", "c"), length.out = n)), z = 1)
  d$z[d$f == "c"] <- 0        # every "c" row is rejected via log(z) = -Inf
  prep <- rpbnb:::.prepare_bnb_data(y1 ~ f, y2 ~ log(z), d)
  expect_equal(prep$cn1, c("(Intercept)", "fb"))   # no all-zero 'fc' column
  expect_true(all(is.finite(prep$X1)))
  expect_false("c" %in% prep$xlevels1$f)            # xlevels reflect observed only
})

test_that("a factor left with <2 observed levels errors clearly (not rank-deficient)", {
  set.seed(4); n <- 60
  d <- data.frame(y1 = rpois(n, 2), y2 = rpois(n, 2),
                  f = factor(rep(c("a", "b"), length.out = n)), z = 1)
  d$z[d$f == "b"] <- 0        # every "b" row rejected -> only "a" remains
  expect_error(rpbnb:::.prepare_bnb_data(y1 ~ f, y2 ~ log(z), d),
               "two observed levels|design")
})

test_that("predict(fit) equals predict(fit, retained_data) with a stateful poly() term", {
  set.seed(7); n <- 200
  d <- data.frame(x = rnorm(n), z = c(0, runif(n - 1, 0.5, 3)))  # z[1]=0 rejects row 1
  d$y1 <- rpois(n, exp(0.2 + 0.1 * d$x))
  d$y2 <- rpois(n, exp(0.1 + 0.2 * log(pmax(d$z, 0.5))))
  retained <- droplevels(d[-1, ])                                # row 1 (log(0)) dropped

  # independence (bnb_fit predict path)
  fi <- suppressWarnings(fit_bnb(y1 ~ poly(x, 2), y2 ~ log(z), data = d,
                                 dependence = "independence"))
  p0 <- predict(fi); p1 <- predict(fi, newdata = retained)
  expect_equal(p0$mu1, p1$mu1, tolerance = 1e-8)
  expect_equal(p0$mu2, p1$mu2, tolerance = 1e-8)

  # RP (rpbnb_fit predict path)
  fr <- suppressWarnings(fit_rpbnb(y1 ~ poly(x, 2), y2 ~ log(z), data = d,
                                   random_1 = "poly(x, 2)1", draws = 40,
                                   control = rpbnb_control(compute_se = FALSE)))
  q0 <- suppressWarnings(predict(fr)); q1 <- suppressWarnings(predict(fr, newdata = retained))
  expect_equal(q0$mu1, q1$mu1, tolerance = 1e-8)
  expect_equal(q0$mu2, q1$mu2, tolerance = 1e-8)
})

# ---- stateful terms tolerate a raw NA (pre-mask); nested invalids diagnose ----

test_that("poly() with a raw NA input removes that row (historical complete-case)", {
  # poly() aborts on a missing input, so a raw NA must be dropped BEFORE any term
  # is evaluated (Stage A). Previously fine under complete.cases; must stay fine.
  d <- data.frame(y1 = c(3, 5, 2, 4, 6, 1, 2, 3, 4),
                  y2 = c(1, 2, 1, 0, 2, 1, 3, 2, 1),
                  x  = c(1, NA, 2, 3, 4, 5, 6, 7, 8),
                  z  = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
  prep <- rpbnb:::.prepare_bnb_data(y1 ~ poly(x, 2), y2 ~ z, d)
  expect_equal(nrow(prep$X1), 8L)                 # the NA-x row (2) removed
  expect_equal(unname(prep$Y1), c(3L, 2L, 4L, 6L, 1L, 2L, 3L, 4L))
  expect_true(all(is.finite(prep$X1)))
  # And a full fit works end-to-end (independence path re-evaluates poly on data).
  fit <- suppressWarnings(fit_bnb(y1 ~ poly(x, 2), y2 ~ z, data = d,
                                  dependence = "independence"))
  expect_equal(fit$nobs, 8L)
})

test_that("a nested transform whose outer term rejects a NaN gives a clear error", {
  # poly(log(x), 2) with x < 0: log(x) = NaN, and poly() rejects it. This failed
  # before too (not a regression), but must now diagnose rather than surface the
  # cryptic "missing values are not allowed in 'poly'".
  d <- data.frame(y1 = c(1, 2, 3, 4, 5), y2 = c(0, 1, 2, 1, 0),
                  x = c(1, -1, 2, 3, 4), z = c(0.1, 0.2, 0.3, 0.4, 0.5))
  # log(-1) also warns "NaNs produced"; that is inherent to the data, not the
  # behavior under test (the clear diagnostic error).
  expect_error(
    suppressWarnings(rpbnb:::.prepare_bnb_data(y1 ~ poly(log(x), 2), y2 ~ z, d)),
    "precompute or clean|inner transformation")
})
