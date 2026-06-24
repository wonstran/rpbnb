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

test_that("lambda_bounds_vec returns correct numeric bounds", {
  c1 <- c(0.2, 0.4, 0.6); c2 <- c(0.3, 0.5, 0.1)
  b  <- rpbnb:::lambda_bounds_vec(c1, c2)
  expect_length(b, 2)
  # obs 1 is binding lower bound: -1/((1-0.2)*(1-0.3))
  expect_equal(b[1], -1 / ((1 - 0.2) * (1 - 0.3)), tolerance = 1e-10)
  # obs 3 is binding upper bound: 1/(0.6*(1-0.1))
  expect_equal(b[2], 1 / (0.6 * (1 - 0.1)), tolerance = 1e-10)
  expect_true(b[1] < b[2])
})

test_that("d_const is Famoye's 1 - exp(-1)", {
  expect_equal(rpbnb:::d_const(), 1 - exp(-1), tolerance = 1e-12)
})

test_that("d2ct_dm2 matches numDeriv second derivative of c_val wrt m", {
  mu <- 1.7; m <- 0.6
  f   <- function(mm) rpbnb:::c_val(mu, mm)
  num <- numDeriv::hessian(f, m)[1, 1]
  ana <- rpbnb:::d2ct_dm2(mu, m, rpbnb:::c_val(mu, m))
  expect_equal(ana, num, tolerance = 1e-6)
})

test_that("d2ct_dbeta2_factor matches numDeriv second derivative of c_val wrt beta", {
  x <- 0.8; m <- 0.6; beta <- 0.3
  f   <- function(b) rpbnb:::c_val(exp(x * b), m)   # mu = exp(x*beta)
  num <- numDeriv::hessian(f, beta)[1, 1]
  mu  <- exp(x * beta)
  b   <- rpbnb:::d2ct_dbeta2_factor(mu, m, rpbnb:::c_val(mu, m))
  expect_equal(b * x * x, num, tolerance = 1e-6)   # d2c/dbj dbs = b * x_j * x_s
})

test_that("d2ct_dmdbeta_factor matches numDeriv cross derivative of c_val", {
  x <- 0.8; m0 <- 0.6; beta0 <- 0.3
  f   <- function(p) rpbnb:::c_val(exp(x * p[2]), p[1])  # p = (m, beta)
  num <- numDeriv::hessian(f, c(m0, beta0))[1, 2]
  mu  <- exp(x * beta0)
  e   <- rpbnb:::d2ct_dmdbeta_factor(mu, m0, rpbnb:::c_val(mu, m0))
  expect_equal(e * x, num, tolerance = 1e-6)          # d2c/dm dbj = e * x_j
})
