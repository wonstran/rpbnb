# Curvature diagnostics and non-silent Hessian repair (P1e of
# comments/review_2026-07-15-17-32-20.md).

test_that(".observed_info_vcov records clean diagnostics for a PD information", {
  info <- diag(c(2, 4))
  r <- rpbnb:::.observed_info_vcov(info, c("a", "b"))
  expect_true(r$diag$positive_definite)
  expect_false(r$diag$repaired)
  expect_equal(r$diag$ridge, 0)
  expect_equal(r$diag$condition, 2)          # max/min eigenvalue = 4/2
  expect_equal(unname(r$se), c(sqrt(1/2), sqrt(1/4)), tolerance = 1e-10)
})

test_that(".observed_info_vcov warns and records a ridge for a non-PD information", {
  info <- matrix(c(2, 0, 0, -1), 2, 2)        # one negative eigenvalue
  expect_warning(
    r <- rpbnb:::.observed_info_vcov(info, c("a", "b"), label = "test model"),
    "not positive definite"
  )
  expect_false(r$diag$positive_definite)
  expect_true(r$diag$repaired)
  expect_gt(r$diag$ridge, 0)
  expect_true(all(is.finite(r$se)))
})

test_that("fit_bnb famoye records hessian diagnostics on the fit object", {
  set.seed(21)
  d <- data.frame(x = rnorm(400))
  d$y1 <- rnbinom(400, size = 2, mu = exp(0.3 + 0.2 * d$x))
  d$y2 <- rnbinom(400, size = 2, mu = exp(0.1 - 0.1 * d$x))
  ff <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye")
  expect_false(is.null(ff$hessian_diag))
  expect_true(ff$hessian_diag$positive_definite)
  expect_false(ff$hessian_diag$repaired)
  expect_true(is.finite(ff$hessian_diag$condition))
})
