test_that("halton_normal returns the requested shape", {
  Z <- rpbnb:::halton_normal(n_draws = 200, d = 3, burn = 50)
  expect_equal(dim(Z), c(200L, 3L))
  expect_true(all(is.finite(Z)))
})

test_that("halton_normal columns are approximately standard normal", {
  set.seed(1)
  Z <- rpbnb:::halton_normal(n_draws = 2000, d = 2, burn = 100)
  expect_lt(abs(mean(Z[, 1])), 0.1)
  expect_lt(abs(sd(Z[, 1]) - 1), 0.1)
})

test_that("d = 0 yields a 0-column matrix with n_draws rows", {
  Z <- rpbnb:::halton_normal(n_draws = 10, d = 0)
  expect_equal(dim(Z), c(10L, 0L))
})

test_that("halton_normal is reproducible for a fixed seed", {
  set.seed(123); a <- rpbnb:::halton_normal(n_draws = 100, d = 2, burn = 50)
  set.seed(123); b <- rpbnb:::halton_normal(n_draws = 100, d = 2, burn = 50)
  expect_identical(a, b)
})

test_that("halton_normal draws differ across seeds (seed is honoured)", {
  set.seed(1); a <- rpbnb:::halton_normal(n_draws = 100, d = 2, burn = 50)
  set.seed(2); b <- rpbnb:::halton_normal(n_draws = 100, d = 2, burn = 50)
  expect_false(isTRUE(all.equal(a, b)))
})
