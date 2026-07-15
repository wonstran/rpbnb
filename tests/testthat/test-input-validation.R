# Input validation for copula domains, RP dependence dispatch, and the copula
# simulator (P1c of comments/review_2026-07-15-17-32-20.md).

test_that("copula() rejects out-of-domain native parameters when par is supplied", {
  expect_error(copula("normal", par = 1.2), "rho|-1|1", ignore.case = TRUE)
  expect_error(copula("normal", par = -1), "rho|-1|1", ignore.case = TRUE)
  expect_error(copula("kimeldorf", par = -1), "positive|theta|> 0", ignore.case = TRUE)
  expect_error(copula("kimeldorf", par = 0), "positive|theta|> 0", ignore.case = TRUE)
  expect_error(copula("frank", par = Inf), "finite", ignore.case = TRUE)
  expect_error(copula("normal", par = c(0.1, 0.2)), "single|length", ignore.case = TRUE)
})

test_that("copula() accepts in-domain parameters and NULL par", {
  expect_s3_class(copula("normal", par = 0.3), "rpbnb_copula")
  expect_s3_class(copula("normal", par = -0.9), "rpbnb_copula")
  expect_s3_class(copula("kimeldorf", par = 2), "rpbnb_copula")
  expect_s3_class(copula("frank", par = -5), "rpbnb_copula")
  expect_s3_class(copula("frank"), "rpbnb_copula")   # par = NULL for estimation
})

test_that("fit_rpbnb rejects a dependence string that is not 'famoye'", {
  d <- data.frame(y1 = rpois(30, 1), y2 = rpois(30, 1), x = rnorm(30))
  expect_error(
    fit_rpbnb(y1 ~ x, y2 ~ x, data = d, dependence = "typo"),
    "famoye|copula", ignore.case = TRUE
  )
})

test_that("simulate_rpbnb_copula validates dispersion, random names, and scales", {
  b1 <- c("(Intercept)" = 0.2, x1 = 0.3)
  b2 <- c("(Intercept)" = 0.1, x1 = 0.2)
  cop <- copula("frank", par = 2)

  # character random spec with no scale must error, not return NA counts
  expect_error(
    simulate_rpbnb_copula(50, b1, b2, random_1 = "x1", copula = cop, seed = 1),
    "scale|sd", ignore.case = TRUE
  )
  # dispersion missing required names
  expect_error(
    simulate_rpbnb_copula(50, b1, b2, dispersion = c(a = 0.5, b = 0.5),
                          copula = cop, seed = 1),
    "m1|m2|dispersion", ignore.case = TRUE
  )
  # random name not present in beta (validly-formed spec, absent coefficient)
  expect_error(
    simulate_rpbnb_copula(50, b1, b2,
                          random_1 = list(nope = list(dist = "normal", sd = 0.5)),
                          copula = cop, seed = 1),
    "not in|random name", ignore.case = TRUE
  )
})
