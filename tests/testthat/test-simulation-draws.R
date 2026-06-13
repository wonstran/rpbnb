test_that("halton_normal returns the requested shape", {
  Z <- rpbnb:::halton_normal(n_draws = 200, d = 3, burn = 50, skip = 10)
  expect_equal(dim(Z), c(200L, 3L))
  expect_true(all(is.finite(Z)))
})

test_that("halton_normal columns are approximately standard normal", {
  Z <- rpbnb:::halton_normal(n_draws = 2000, d = 2, burn = 100, skip = 10)
  expect_lt(abs(mean(Z[, 1])), 0.1)
  expect_lt(abs(sd(Z[, 1]) - 1), 0.1)
})

test_that("d = 0 yields a 0-column matrix with n_draws rows", {
  Z <- rpbnb:::halton_normal(n_draws = 10, d = 0)
  expect_equal(dim(Z), c(10L, 0L))
})
