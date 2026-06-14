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

test_that("signif_stars maps p-values to the expected codes", {
  expect_equal(rpbnb:::signif_stars(c(0.0001, 0.02, 0.2)),
               c("***", "*", " "))
})
