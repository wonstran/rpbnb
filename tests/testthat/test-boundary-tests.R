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

test_that("errors on non-rpbnb_fit", {
  expect_error(rpbnb_boundary_tests(list(), data.frame(x = 1)), "rpbnb_fit")
})

test_that("rpbnb_boundary_tests rejects a non-converged full fit", {
  fake <- structure(
    list(cop_family = NULL,
         convergence = list(converged = FALSE, code = 1L,
                            message = "iteration limit exceeded")),
    class = "rpbnb_fit")
  expect_error(rpbnb_boundary_tests(fake, data.frame(x = 1)), "did not converge")
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

  # The m1/m2 dispersion refits use Poisson-limit margins; a tail draw can trip
  # the mean-range guard (exercised separately). Muffle only that diagnostic, so a
  # non-convergence or negative-LR warning would still surface.
  bt <- suppress_poisson_warning(rpbnb_boundary_tests(fit, sim$data, control = ctrl))

  expect_s3_class(bt, "rpbnb_boundary_tests")
  expect_equal(bt$Parameter, c("sd1:x1", "sd1:x2", "sd2:x1", "m1", "m2"))
  expect_true(all(bt$df == 1))                       # each drops ONE parameter
  expect_true(all(bt$p.value >= 0 & bt$p.value <= 1))

  # Common random numbers: reusing the full fit's stored draws with nothing
  # pinned must reproduce the full fit exactly (identical draw columns, same
  # optimum) -- this is the guarantee every restricted refit relies on.
  crn <- list(Z1 = fit$rp_meta$Z1, Z2 = fit$rp_meta$Z2)
  rest_crn <- fit_rpbnb(y1 ~ x1 + x2, y2 ~ x1, data = sim$data,
                        random_1 = c("x1", "x2"), random_2 = "x1",
                        draws = 200, seed = 3, control = ctrl, .opt_draws = crn)
  expect_equal(rest_crn$rp_meta$Z1, fit$rp_meta$Z1)   # exact same draw columns
  expect_equal(rest_crn$rp_meta$Z2, fit$rp_meta$Z2)
  expect_equal(as.numeric(logLik(rest_crn)), as.numeric(logLik(fit)),
               tolerance = 1e-6)

  # The sd1:x1 restriction zeroes x1's draw column (exact SD-zero null: base 0 on
  # every draw) and pins its now-inert log-scale, keeping x2 random. npar = full -
  # 1, and its logLik equals the restricted logLik implied by the sd1:x1 LR stat.
  z <- crn; z$Z1[, 1] <- 0.5                            # median draw -> base 0
  pin <- stats::setNames(fit$coef[["log_sd1:x1"]], "log_sd1:x1")
  rest_pin <- fit_rpbnb(y1 ~ x1 + x2, y2 ~ x1, data = sim$data,
                        random_1 = c("x1", "x2"), random_2 = "x1",
                        draws = 200, seed = 3, control = ctrl,
                        start = fit$coef, .fixed = pin, .opt_draws = z)
  expect_equal(rest_pin$npar, fit$npar - 1L)
  ll_rest_implied <- as.numeric(logLik(fit)) - bt$LR[bt$Parameter == "sd1:x1"] / 2
  expect_equal(as.numeric(logLik(rest_pin)), ll_rest_implied, tolerance = 1e-6)
})

test_that("median draw zeroes a coefficient's deviation exactly (any scale)", {
  # Exact SD-zero null: the median draw u = 0.5 gives a zero deviation on every
  # draw for ANY scale (normal/lognormal/triangular map u = 0.5 to base 0; uniform
  # uses the centered value 2 * 0.5 - 1 = 0) -- so normal/uniform/triangular
  # collapse to the fixed coefficient b, and lognormal to sign*exp(b).
  Z <- matrix(0.5, 5L, 1L)
  for (d in c("normal", "uniform", "triangular")) {
    rr <- rpbnb:::rand_realize(Z, d, 1, b = 0.4, s = 1e9)   # huge scale, still exact
    expect_true(all(rr$dev[, 1] == 0))
    expect_true(all(rr$coef[, 1] == 0.4))
  }
  rl <- rpbnb:::rand_realize(Z, "lognormal", 1, b = 0.4, s = 1e9)
  expect_equal(rl$coef[, 1], rep(exp(0.4), 5))              # sign*exp(b), not b = 0.4
})

test_that("exact SD-zero null is covariate-scale invariant (P1 large-covariate regression)", {
  # Reviewer counterexample: with x = 1e6 and native scale 1e-6 the OLD finite
  # proxy left a material random mixture (eta perturbation +/-1, multiplier ~1.54).
  # Base-zeroing leaves the deviation constant across draws, so a huge covariate
  # induces NO residual mixture -- the exact scale-zero null.
  R <- 8L
  Zzero <- matrix(0.5, R, 1L)
  x <- 1e6
  for (d in c("normal", "lognormal")) {
    dev <- rpbnb:::rand_realize(Zzero, d, 1, b = 0.4, s = 0.5)$dev[, 1]
    expect_true(all(x * (dev - dev[1]) == 0))              # constant eta -> no mixture
  }
  # sanity: the OLD proxy (real draws, native scale 1e-6) DID perturb eta materially
  Zreal   <- matrix(seq(0.05, 0.95, length.out = R), ncol = 1L)
  dev_old <- rpbnb:::rand_realize(Zreal, "normal", 1, b = 0.4, s = 1e-6)$dev[, 1]
  expect_gt(diff(range(x * dev_old)), 1)                   # the failure the fix removes
})

test_that("fit_rpbnb rejects a mis-shaped .opt_draws before fitting", {
  d <- data.frame(y1 = c(0L, 1L, 2L, 1L, 0L),
                  y2 = c(1L, 0L, 2L, 1L, 3L),
                  x1 = c(-1, 0, 1, 0.5, -0.5))
  expect_error(
    fit_rpbnb(y1 ~ x1, y2 ~ x1, data = d, random_1 = "x1", draws = 5,
              .opt_draws = list(Z1 = matrix(0.5, 5, 2),          # wrong ncol (q1=1)
                                Z2 = matrix(0, 5, 0))),
    "opt_draws")
})

test_that("fit_rpbnb .fixed rejects unknown parameter names", {
  d <- data.frame(y1 = c(0L, 1L, 2L, 1L, 0L),
                  y2 = c(1L, 0L, 2L, 1L, 3L),
                  x1 = c(-1, 0, 1, 0.5, -0.5))
  expect_error(
    fit_rpbnb(y1 ~ x1, y2 ~ x1, data = d, random_1 = "x1", draws = 5,
              .fixed = c(not_a_param = 0)),
    "not in the model")
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
  bt <- suppress_poisson_warning(
    rpbnb_boundary_tests(fit, sim$data, control = ctrl, which = "dispersion"))
  expect_equal(bt$Parameter, c("m1", "m2"))
})

test_that("boundary tests report NA (never a p-value) for a non-converged refit", {
  skip_slow()
  sim <- simulate_rpbnb(n = 400,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 2)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 120, seed = 2,
                   control = rpbnb_control(print_level = 0, compute_se = FALSE))
  expect_true(fit$convergence$converged)                 # full fit converged

  # Force every restricted refit to hit the iteration limit (maxLik code 1).
  ctrl1 <- rpbnb_control(print_level = 0, compute_se = FALSE, iterlim = 1)
  w  <- character(0)
  bt <- withCallingHandlers(
    rpbnb_boundary_tests(fit, sim$data, control = ctrl1),
    warning = function(cnd) { w <<- c(w, conditionMessage(cnd)); invokeRestart("muffleWarning") })

  expect_true(all(is.na(bt$LR)))                          # no LR statistic
  expect_true(all(is.na(bt$p.value)))                     # no p-value reported
  conv_w <- grep("did not converge", w, value = TRUE)     # one per parameter row
  expect_equal(length(conv_w), nrow(bt))
})

test_that("restricted refits warm-start from the full fit (non-default start reproduced)", {
  skip_slow()
  sim <- simulate_rpbnb(n = 500,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 6)
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)
  # Full fit from an explicit NON-default named start.
  st <- c("b1:(Intercept)" = 0.1, "b1:x1" = 0.3, "b2:(Intercept)" = 0.05,
          "b2:x1" = -0.2, "log_sd1:x1" = log(0.4),
          "log_m1" = log(0.3), "log_m2" = log(0.4), "z_lambda" = 0)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 150, seed = 6, start = st, control = ctrl)
  # An unpinned exact-draw refit warm-started from fit$coef must reproduce the
  # optimum -- this is what every restricted refit relies on to avoid landing at
  # an inferior local optimum of the start-sensitive simulated objective.
  crn  <- list(Z1 = fit$rp_meta$Z1, Z2 = fit$rp_meta$Z2)
  rest <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                    draws = 150, seed = 6, start = fit$coef, control = ctrl,
                    .opt_draws = crn)
  expect_equal(as.numeric(logLik(rest)), as.numeric(logLik(fit)), tolerance = 1e-6)
  expect_equal(unname(rest$coef), unname(fit$coef), tolerance = 1e-4)
})
