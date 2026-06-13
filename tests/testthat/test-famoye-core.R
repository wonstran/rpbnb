test_that("nb_logpmf_y_mu_r matches stats::dnbinom", {
  y  <- 0:10
  mu <- 2.3
  m  <- 0.7
  r  <- 1 / m
  expect_equal(
    rpbnb:::nb_logpmf_y_mu_r(y, mu, r),
    stats::dnbinom(y, size = r, mu = mu, log = TRUE),
    tolerance = 1e-10
  )
})

test_that("c_val equals E[exp(-Y)] on a truncated NB2 grid", {
  mu <- 1.5; m <- 0.8; r <- 1 / m
  y  <- 0:2000
  approx <- sum(exp(-y) * stats::dnbinom(y, size = r, mu = mu))
  expect_equal(rpbnb:::c_val(mu, m), approx, tolerance = 1e-8)
})

test_that("lambda_bounds_vec returns a valid open interval", {
  c1 <- c(0.2, 0.4, 0.6); c2 <- c(0.3, 0.5, 0.1)
  b  <- rpbnb:::lambda_bounds_vec(c1, c2)
  expect_length(b, 2)
  expect_true(b[1] < 0 && b[2] > 0 && b[1] < b[2])
})

test_that("d_const is Famoye's 1 - exp(-1)", {
  expect_equal(rpbnb:::d_const(), 1 - exp(-1), tolerance = 1e-12)
})
