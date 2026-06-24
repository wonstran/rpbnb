test_that("tri_icdf maps endpoints and midpoint correctly", {
  expect_equal(tri_icdf(0.5), 0)
  expect_equal(tri_icdf(1e-12), -1, tolerance = 1e-5)
  expect_equal(tri_icdf(1 - 1e-12), 1, tolerance = 1e-5)
  u <- seq(0.01, 0.99, length.out = 50)
  expect_true(all(diff(tri_icdf(u)) > 0))  # monotone increasing
})

test_that("registry transforms have the right large-sample moments", {
  set.seed(1)
  z <- stats::qnorm(stats::runif(2e5)); u <- stats::runif(2e5)
  # normal: mean b, sd s
  cn <- rand_dist_registry$normal$coef(0.5, 0.8, z, 1)
  expect_equal(mean(cn), 0.5, tolerance = 0.02)
  expect_equal(stats::sd(cn), 0.8, tolerance = 0.02)
  # uniform on [b-w, b+w]: var = w^2/3
  cu <- rand_dist_registry$uniform$coef(0.5, 0.9,
          rand_dist_registry$uniform$u_to_base(u), 1)
  expect_equal(mean(cu), 0.5, tolerance = 0.02)
  expect_equal(stats::var(cu), 0.9^2 / 3, tolerance = 0.02)
  expect_true(all(cu >= 0.5 - 0.9 - 1e-8 & cu <= 0.5 + 0.9 + 1e-8))
  # triangular: var = w^2/6
  ct <- rand_dist_registry$triangular$coef(0.0, 0.9,
          rand_dist_registry$triangular$u_to_base(u), 1)
  expect_equal(stats::var(ct), 0.9^2 / 6, tolerance = 0.02)
  # lognormal negative sign => strictly negative
  cl <- rand_dist_registry$lognormal$coef(-0.2, 0.5, z, -1)
  expect_true(all(cl < 0))
})

test_that("lognormal gradient factors match definitions", {
  z <- c(-1, 0, 1.5); b <- 0.3; s <- 0.4
  coef <- rand_dist_registry$lognormal$coef(b, s, z, -1)
  expect_equal(rand_dist_registry$lognormal$dloc_factor(b, s, z, coef), coef)
  expect_equal(rand_dist_registry$lognormal$dscale(b, s, z, coef), coef * z * s)
})

test_that("parse_rand_spec handles char vector, named list, and errors", {
  expect_equal(parse_rand_spec(NULL)$names, character(0))
  cv <- parse_rand_spec(c("x1", "x2"))
  expect_equal(cv$dist, c("normal", "normal"))
  expect_equal(cv$sign, c(1, 1))
  nl <- parse_rand_spec(list(x1 = "uniform",
                             p = list(dist = "lognormal", sign = -1)))
  expect_equal(nl$names, c("x1", "p"))
  expect_equal(nl$dist, c("uniform", "lognormal"))
  expect_equal(nl$sign, c(1, -1))
  expect_error(parse_rand_spec(list(x1 = "weibull")), "unknown distribution")
  expect_error(parse_rand_spec(list(x1 = list(dist = "normal", sign = -1))),
               "only meaningful for lognormal")
  expect_error(parse_rand_spec(list(x1 = list(dist = "lognormal", sign = 2))),
               "must be -1 or 1")
  expect_error(parse_rand_spec(list("normal")), "named list")
})

test_that("parse_rand_spec reads scale via scale or sd alias", {
  s <- parse_rand_spec(list(x1 = list(dist = "uniform", scale = 0.7),
                            x2 = list(sd = 0.5)))
  expect_equal(s$scale, c(0.7, 0.5))
  expect_equal(s$dist, c("uniform", "normal"))
})

test_that("rand_realize returns aligned matrices", {
  set.seed(2)
  U <- matrix(stats::runif(20), nrow = 10, ncol = 2)
  out <- rand_realize(U, dist = c("normal", "lognormal"),
                      sign = c(1, -1), b = c(0.1, 0.2), s = c(0.3, 0.4))
  expect_equal(dim(out$coef), c(10, 2))
  expect_true(all(out$coef[, 2] < 0))            # lognormal neg sign
  expect_equal(out$dloc[, 1], rep(1, 10))        # normal location factor
  expect_equal(out$dloc[, 2], out$coef[, 2])     # lognormal location factor
})
