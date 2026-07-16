# Distribution-aware RP prediction and natural-scale reporting (P1a of
# comments/review_2026-07-15-17-32-20.md).

test_that("predict integrated means match stored draw-averaged means (uniform RP)", {
  skip_on_cran()
  sim <- simulate_rpbnb(400,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.5),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(dist = "uniform", scale = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 12)
  d <- sim$data
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = d,
    random_1 = list(x1 = list(dist = "uniform")), random_2 = NULL,
    draws = 150, seed = 5,
    control = rpbnb_control(compute_se = FALSE, print_level = 0))
  pr <- predict(fit, newdata = d)
  # The integrated (population) mean is the draw average of exp(x'beta); on the
  # training design it must reproduce the stored fitted means. The old normal-
  # only correction ignored the uniform (log_w) scale, so mu1 was just exp(x'b).
  expect_equal(pr$mu1, fit$mu1, tolerance = 1e-6)
  expect_equal(pr$mu2, fit$mu2, tolerance = 1e-6)
})

test_that("natural-scale summary includes a uniform (log_w) random-scale row", {
  skip_on_cran()
  sim <- simulate_rpbnb(300,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.5),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(dist = "uniform", scale = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 13)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
    random_1 = list(x1 = list(dist = "uniform")), random_2 = NULL,
    draws = 120, seed = 6,
    control = rpbnb_control(compute_se = TRUE, print_level = 0))
  nat <- summary(fit)$natural
  expect_true(any(grepl("^w1:x1", nat$Parameter)))   # uniform half-width scale
})
