# Fast, fixture-based tests of predict() semantics -- no model fitting. These
# cover the prediction logic (one-row shape, no-newdata/newdata consistency,
# lognormal analytic infinities, estimand metadata) that the slow end-to-end
# tests in test-predict-dist.R also exercise. See helper-slow.R::make_rp_fixture.

test_that("predict handles a one-row newdata (no fitting)", {
  f  <- make_rp_fixture("uniform")
  pr <- predict(f, newdata = data.frame(x1 = 0.5))
  expect_equal(nrow(pr), 1L)
  expect_true(is.finite(pr$mu1) && is.finite(pr$mu2))
})

test_that("predict(fit) and predict(fit, newdata = training) agree (no fitting)", {
  for (d in c("normal", "uniform", "triangular")) {
    f  <- make_rp_fixture(d)
    p0 <- predict(f)                              # recomputed from stored X1
    p1 <- predict(f, newdata = f$xtrain)          # explicit training newdata
    expect_equal(p0$mu1, p1$mu1, tolerance = 1e-12, info = d)
    expect_true(all(is.finite(p0$mu1)), info = d)
  }
})

test_that("lognormal predictions are Inf exactly where sign*covariate > 0 (both signs)", {
  for (sgn in c(1, -1)) {
    f  <- make_rp_fixture("lognormal", sign1 = sgn)
    nd <- f$xtrain
    pr <- suppressWarnings(predict(f, newdata = nd))
    inf_rows <- (sgn * nd$x1) > 0
    expect_true(all(is.infinite(pr$mu1[inf_rows])), info = paste("sign", sgn))
    expect_true(all(is.finite(pr$mu1[!inf_rows])), info = paste("sign", sgn))
    # and the no-newdata branch agrees
    p0 <- suppressWarnings(predict(f))
    expect_equal(p0$mu1, pr$mu1, info = paste("sign", sgn))
  }
})

test_that("lognormal Inf prediction emits a warning", {
  f <- make_rp_fixture("lognormal", sign1 = 1)
  expect_warning(predict(f, newdata = f$xtrain), "infinite")
})

test_that("predict output carries estimand / draw metadata", {
  f  <- make_rp_fixture("triangular")
  pr <- predict(f, newdata = data.frame(x1 = c(0.1, 0.2)))
  expect_true(grepl("population mean", attr(pr, "estimand")))
  expect_equal(attr(pr, "n_draws"), 64L)
  expect_true(is.finite(attr(pr, "per_draw_cap")))
})
