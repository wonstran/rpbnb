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

test_that("a near-zero scale does not produce a clamped negative LR statistic", {
  skip_slow()
  # sd = 0.02 puts the full fit on the flat part of the simulated likelihood,
  # where its single BFGS run used to stop just below the warm-started
  # restricted refit -- the "Restricted model has the higher log-likelihood"
  # clamp. The full-model polish in rpbnb_boundary_tests() removes it.
  sim <- simulate_rpbnb(n = 600,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.02)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 11)
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 150, seed = 11, control = ctrl)
  expect_no_warning(
    bt <- rpbnb_boundary_tests(fit, sim$data, control = ctrl, which = "sd"))
  expect_true(all(bt$LR >= 0))
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

# ---- Group switch and the dependence (association) test ---------------------

test_that("the boundary-test group switch normalizes as documented", {
  n <- rpbnb:::.normalize_boundary_tests
  expect_equal(n(FALSE), character(0))
  expect_equal(n(NULL), character(0))
  expect_equal(n("none"), character(0))
  # TRUE keeps its historical meaning and does NOT silently grow a third refit.
  expect_equal(n(TRUE), c("sd", "dispersion"))
  expect_equal(n("all"), c("sd", "dispersion", "dependence"))
  expect_equal(n("dispersion"), "dispersion")
  # canonical order, regardless of how it was written
  expect_equal(n(c("dependence", "sd")), c("sd", "dependence"))
  expect_error(n("bogus"), "unknown group")
  expect_error(n(c(TRUE, FALSE)), "TRUE/FALSE")
  expect_error(n(1L), "TRUE/FALSE")
})

test_that("dependence family, label, and boundary flag are read off the fit", {
  fam <- rpbnb:::.fit_dep_family
  # classic: cop_family, or famoye when it is absent
  expect_equal(fam(list(cop_family = "frank")), "frank")
  expect_equal(fam(list(cop_family = NULL, coef = 1)), "famoye")
  # tmb: the `dependence` field, character or copula()
  expect_equal(fam(list(dependence = "independence")), "independence")
  expect_equal(fam(list(dependence = copula("kimeldorf"))), "kimeldorf")

  expect_equal(rpbnb:::.dep_boundary_param("famoye"), "lam")
  expect_equal(rpbnb:::.dep_boundary_param("frank"), "theta")
  expect_equal(rpbnb:::.dep_boundary_param("kimeldorf"), "theta")
  expect_equal(rpbnb:::.dep_boundary_param("normal"), "rho")
  expect_null(rpbnb:::.dep_boundary_param("independence"))

  # Only Clayton/Kimeldorf constrains theta > 0, so only it has a boundary null
  # at independence. Blanket-applying the 50:50 mixture would halve the other
  # three families' p-values.
  expect_true(rpbnb:::.dep_null_is_boundary("kimeldorf"))
  expect_false(rpbnb:::.dep_null_is_boundary("famoye"))
  expect_false(rpbnb:::.dep_null_is_boundary("frank"))
  expect_false(rpbnb:::.dep_null_is_boundary("normal"))
})

test_that(".famoye_indep_z inverts the lambda map to exactly zero", {
  fit <- make_rp_fixture()
  z0 <- rpbnb:::.famoye_indep_z(fit)
  expect_true(is.finite(z0))
  # The contract: fed back through the forward map on the SAME interval the
  # restricted refit freezes, the pin must give lambda = 0 -- independence.
  b <- rpbnb:::.rp_support_bounds(
    fit$coef, fit$X1, fit$X2, fit$rand_idx1, fit$rand_idx2,
    fit$rp_meta$dist1, fit$rp_meta$dist2, fit$rp_meta$sign1, fit$rp_meta$sign2)
  expect_equal(rpbnb:::famoye_lam_from_z(c(b[["lower"]], b[["upper"]]), z0), 0,
               tolerance = 1e-12)
})

test_that("the Clayton independence pin lands inside the product-copula branch", {
  # Both the R reference (kimeldorf_cdf) and the C++ kernel switch to u*v below
  # theta = 1e-10. The pin must clear that threshold, or the four-corner cell
  # probability becomes a difference of numbers agreeing to ~1e-9.
  theta <- exp(rpbnb:::.CLAYTON_INDEP_Z)
  expect_lt(theta, 1e-10)
  expect_equal(rpbnb:::kimeldorf_cdf(0.3, 0.7, theta), 0.3 * 0.7)
})

test_that("end-to-end: which = 'dependence' tests the association parameter", {
  skip_slow()
  sim <- simulate_rpbnb(n = 600,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5), seed = 11)
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 150, seed = 11, control = ctrl)
  bt <- rpbnb_boundary_tests(fit, sim$data, control = ctrl,
                             which = "dependence")
  expect_equal(bt$Parameter, "lam")          # famoye
  expect_equal(bt$df, 1L)                    # z_lambda pinned out
  expect_true(is.na(bt$LR) || (bt$LR >= 0 && bt$p.value >= 0 && bt$p.value <= 1))

  # summary() shows the LR test on the dependence row in place of its Wald z.
  fit$boundary_tests <- bt
  nat <- rpbnb:::.natural_scale_table(fit)
  lam_row <- nat$dispersion[nat$dispersion$Parameter == "lambda (dependence)", ]
  expect_equal(nrow(lam_row), 1L)
  expect_equal(lam_row$df, bt$df)
  expect_true(is.na(lam_row$z))
})

test_that("the printed header describes the tests actually in the table", {
  # Scales/dispersions only: the historical boundary-mixture header.
  bt_b <- structure(
    data.frame(Parameter = c("sd1:x1", "m1"), LR = c(3, 4), df = c(1L, 1L),
               p.value = c(0.04, 0.02), Signif = c("*", "*"),
               stringsAsFactors = FALSE),
    class = c("rpbnb_boundary_tests", "data.frame"))
  out_b <- capture.output(print(bt_b))
  expect_true(any(grepl("Boundary-parameter LR tests", out_b, fixed = TRUE)))

  # With a dependence row the mixture is NOT universal, so the header must not
  # claim it is -- this table is the standalone report people read.
  bt_d <- structure(
    data.frame(Parameter = c("m1", "theta"), LR = c(4, 9), df = c(1L, 1L),
               p.value = c(0.02, 0.003), Signif = c("*", "**"),
               stringsAsFactors = FALSE),
    class = c("rpbnb_boundary_tests", "data.frame"))
  out_d <- capture.output(print(bt_d))
  expect_false(any(grepl("Boundary-parameter LR tests", out_d, fixed = TRUE)))
  expect_true(any(grepl("no association", out_d, fixed = TRUE)))
  expect_true(any(grepl("Clayton/Kimeldorf", out_d, fixed = TRUE)))
})

test_that("end-to-end: the copula dependence pin is family-correct", {
  skip_slow()
  sim <- simulate_rpbnb_copula(n = 400,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    dispersion = c(m1 = 0.4, m2 = 0.5),
    copula = copula("frank", par = 3), seed = 9)
  ctrl <- rpbnb_control(print_level = 0, compute_se = FALSE)

  expected <- c(frank = "theta", normal = "rho", kimeldorf = "theta")
  for (fam in names(expected)) {
    fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                     draws = 100, seed = 9, control = ctrl,
                     dependence = copula(fam))
    bt <- rpbnb_boundary_tests(fit, sim$data, control = ctrl,
                              which = "dependence")
    expect_identical(bt$Parameter, unname(expected[[fam]]), info = fam)
    expect_identical(bt$df, 1L, info = fam)          # z_theta pinned out
    expect_true(is.finite(bt$LR) && bt$LR > 0, info = fam)
    # The boundary correction is applied per family, not blanket: only Clayton
    # constrains theta > 0, so only it halves the chi-square(1) tail.
    plain <- stats::pchisq(bt$LR, 1L, lower.tail = FALSE)
    expect_equal(bt$p.value, if (fam == "kimeldorf") plain / 2 else plain,
                 tolerance = 1e-12, info = fam)
  }
})
