# Tests for rpbnb_boundary_tests(): merged boundary LR tests (sd*/m*) for an
# rpbnb_fit, correctly handling multiple random coefficients per equation.

test_that(".build_rand_spec round-trips through parse_rand_spec", {
  # multiple coefficients, mixed distributions
  spec <- rpbnb:::.build_rand_spec(c("x1", "x2"),
                                   c("normal", "lognormal"), c(1, -1))
  p <- rpbnb:::parse_rand_spec(spec)
  expect_equal(p$names, c("x1", "x2"))
  expect_equal(p$dist,  c("normal", "lognormal"))
  expect_equal(p$sign,  c(1, -1))

  # single coefficient
  p1 <- rpbnb:::parse_rand_spec(rpbnb:::.build_rand_spec("x1", "normal", 1))
  expect_equal(p1$names, "x1")

  # empty -> NULL (all-fixed equation)
  expect_null(rpbnb:::.build_rand_spec(character(0), character(0), numeric(0)))
})

test_that("scale labels match the natural-scale table prefixes", {
  expect_equal(rpbnb:::.sd_label("normal", 1, "hhninc"),    "sd1:hhninc")
  expect_equal(rpbnb:::.sd_label("uniform", 2, "age"),      "w2:age")
  expect_equal(rpbnb:::.sd_label("lognormal", 1, "educ"),   "s1:educ")
})

test_that("errors on non-rpbnb_fit and on copula fits", {
  expect_error(rpbnb_boundary_tests(list(), data.frame(x = 1)), "rpbnb_fit")

  fake_cop <- structure(list(cop_family = "frank"), class = "rpbnb_fit")
  expect_error(rpbnb_boundary_tests(fake_cop, data.frame(x = 1)), "copula")
})

test_that("end-to-end: multiple random coefficients are tested one at a time", {
  skip_slow()
  sim <- simulate_rpbnb(n = 700,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4, x2 = 0.3),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5), x2 = list(sd = 0.4)),
    random_2 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 3)
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)
  fit <- fit_rpbnb(y1 ~ x1 + x2, y2 ~ x1, data = sim$data,
                   random_1 = c("x1", "x2"), random_2 = "x1",
                   draws = 200, seed = 3, control = ctrl)

  bt <- rpbnb_boundary_tests(fit, sim$data, control = ctrl)

  expect_s3_class(bt, "rpbnb_boundary_tests")
  expect_equal(bt$Parameter, c("sd1:x1", "sd1:x2", "sd2:x1", "m1", "m2"))
  expect_true(all(bt$df == 1))                       # each drops ONE parameter
  expect_true(all(bt$p.value >= 0 & bt$p.value <= 1))

  # The sd1:x1 restriction drops only x1: a fit keeping x2 has npar = full - 1,
  # and its logLik equals the restricted logLik implied by the sd1:x1 LR stat.
  rest_keep_x2 <- fit_rpbnb(y1 ~ x1 + x2, y2 ~ x1, data = sim$data,
                            random_1 = "x2", random_2 = "x1",
                            draws = 200, seed = 3, control = ctrl)
  expect_equal(rest_keep_x2$npar, fit$npar - 1L)
  ll_rest_implied <- as.numeric(logLik(fit)) - bt$LR[bt$Parameter == "sd1:x1"] / 2
  expect_equal(as.numeric(logLik(rest_keep_x2)), ll_rest_implied, tolerance = 1e-6)
})

test_that("which = 'dispersion' returns only m1, m2", {
  skip_slow()
  sim <- simulate_rpbnb(n = 500,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 4)
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 150, seed = 4, control = ctrl)
  bt <- rpbnb_boundary_tests(fit, sim$data, control = ctrl, which = "dispersion")
  expect_equal(bt$Parameter, c("m1", "m2"))
})
