test_that("row_log_sum_exp matches log(rowSums(exp(.)))", {
  set.seed(1)
  M <- matrix(rnorm(20, sd = 5), nrow = 4)
  expect_equal(
    rpbnb:::row_log_sum_exp(M),
    log(rowSums(exp(M))),
    tolerance = 1e-10
  )
})

test_that("row_log_sum_exp is stable for large magnitudes", {
  M <- matrix(c(1000, 1001, 1002, 1000), nrow = 2)
  out <- rpbnb:::row_log_sum_exp(M)
  expect_true(all(is.finite(out)))
})

test_that("accepted_ll_trace keeps only improving values", {
  tr <- rpbnb:::accepted_ll_trace(c(-10, -8, -9, -7, -7.0000001))
  expect_equal(tr$logLik, c(-10, -8, -7), tolerance = 1e-6)
})
