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
