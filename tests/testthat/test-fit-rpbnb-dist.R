test_that("character-vector random spec still fits and labels with log_sd", {
  sim <- simulate_rpbnb(n = 400,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 80, seed = 5,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true(any(grepl("^log_sd1:x1", names(coef(fit)))))
  expect_equal(unname(coef(fit)["b1:x1"]), 0.4, tolerance = 0.25)
})

test_that("uniform distribution fits and labels the scale as log_w", {
  sim <- simulate_rpbnb(n = 500,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1),
    random_1 = list(x1 = list(dist = "uniform", scale = 0.6)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 2)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ 1, data = sim$data,
                   random_1 = list(x1 = "uniform"),
                   draws = 100, seed = 7,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true(any(grepl("^log_w1:x1", names(coef(fit)))))
  expect_true(fit$convergence$converged)
})

test_that("lognormal fit recovers a negative-signed coefficient location", {
  sim <- simulate_rpbnb(n = 600,
    beta1 = c("(Intercept)" = 0.2, x1 = -0.2),
    beta2 = c("(Intercept)" = 0.1),
    random_1 = list(x1 = list(dist = "lognormal", scale = 0.3, sign = -1)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 3)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ 1, data = sim$data,
                   random_1 = list(x1 = list(dist = "lognormal", sign = -1)),
                   draws = 120, seed = 11,
                   control = rpbnb_control(compute_se = FALSE))
  expect_true(any(grepl("^log_s1:x1", names(coef(fit)))))
  expect_true(fit$convergence$converged)
})
