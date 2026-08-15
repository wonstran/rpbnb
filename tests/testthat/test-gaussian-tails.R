# The Gaussian copula is the only family whose margins must pass through
# qnorm(), which is singular at 0 and 1, so it is the only one whose cell can
# be destroyed by a marginal CDF saturating. gauss_corner_quantiles() (in
# src/rpbnb_tmb.cpp) takes each cell corner from whichever tail is still
# representable -- the CDF when F <= 1/2, the directly-computed survival
# otherwise -- instead of clamping the CDF to [1e-15, 1-1e-15].
#
# These tests pin the behaviour that clamp used to break. Reproduced from the
# truck data's m1 boundary LR test, where pinning margin 1 to Poisson against
# counts running to 242 collapsed 2.69% of observation-draw cells to zero
# width, each contributing the 1e-300 floor with EXACTLY ZERO gradient. The
# objective became a step function (1,285 nats of swing over parameter steps of
# 2e-4) whose AD gradient disagreed with a finite difference by ~100%, and
# nlminb stopped with "false convergence (8)".

# Counts far enough into the upper tail of a Poisson margin to saturate its
# CDF: mean ~4 with a heavy contaminated tail, so pinning m = 0 forces mu far
# below the largest counts and F(y) rounds to 1 in double precision.
gauss_tail_data <- function(n = 120, seed = 13) {
  set.seed(seed)
  x1 <- rnorm(n)
  y1 <- rpois(n, exp(1.2 + 0.3 * x1))
  y1[seq_len(8)] <- c(150L, 180L, 210L, 240L, 130L, 160L, 190L, 220L)
  data.frame(y1 = y1, y2 = rpois(n, exp(0.2 - 0.2 * x1)), x1 = x1)
}

test_that("a saturated Poisson margin no longer collapses the Gaussian cell", {
  skip_on_cran()
  d <- gauss_tail_data()
  # poisson_1 = TRUE is the m = 0 restriction the dispersion boundary test
  # applies; against these counts it is exactly the regime that used to
  # collapse.
  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d,
                       dependence = copula("normal"), draws = 20, seed = 4,
                       poisson_1 = TRUE,
                       control = rpbnb_tmb_control(n_cores = 1L, restarts = 0L))

  expect_true(is.finite(fit$optimizer$objective))
  expect_true(is.finite(fit$optimizer$max_abs_gradient))
  # The floor is 1e-300 per cell (690.8 nats). A fit whose objective is still
  # dominated by floored cells shows up as an implausibly large nll for n = 120.
  expect_lt(fit$optimizer$objective, 120 * 690)
})

test_that("the Gaussian objective's AD gradient matches finite differences in the tail regime", {
  skip_on_cran()
  # The decisive test, and the one that actually failed before the fix: a
  # clamped cell contributes zero derivative but a non-zero function change, so
  # AD and FD part company. Coordinates that do not move the saturated margin
  # agreed even then -- so this must check the ones that DO.
  d <- gauss_tail_data()
  cap <- new.env()
  testthat::local_mocked_bindings(
    nlminb = function(start, objective, gradient, control, ...) {
      cap$start <- start; cap$fn <- objective; cap$gr <- gradient
      stop("captured-for-test")
    },
    .package = "stats"
  )
  suppressWarnings(try(
    fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d,
                  dependence = copula("normal"), draws = 20, seed = 4,
                  poisson_1 = TRUE, inference = "none",
                  control = rpbnb_tmb_control(n_cores = 1L, print_level = 0L)),
    silent = TRUE))

  p0 <- cap$start
  expect_true(is.finite(cap$fn(p0)))
  g <- cap$gr(p0)
  expect_true(all(is.finite(g)))

  h <- 1e-5
  # Coordinates 1:3 are eq-1 betas -- they move mu1, hence the Poisson margin
  # whose CDF saturates. The last coordinate is z_dep.
  for (j in c(1L, 2L, 3L, length(p0))) {
    pp <- p0; pp[j] <- pp[j] + h
    pm <- p0; pm[j] <- pm[j] - h
    fd <- (cap$fn(pp) - cap$fn(pm)) / (2 * h)
    expect_equal(g[j], fd, tolerance = 1e-3,
                 info = paste("coordinate", j))
  }
})

test_that("an unrestricted NB2 margin keeps a finite gradient as its dispersion collapses", {
  skip_on_cran()
  # The counterpart risk, and the one that actually bit: the NB2 survival is
  # pbeta(mu/(mu+r), y+1, r), whose shape r = 1/m runs away precisely when the
  # data wants a Poisson margin. Because CppAD::CondExp evaluates both of its
  # branches, merely HAVING that call on the tape -- selected or not -- left the
  # polished fit with a non-finite max|gradient| and stopped the dispersion
  # collapsing on Poisson data. NB2 margins therefore keep the historical path
  # and put no survival call on the tape at all.
  #
  # Poisson-generated data with a free dispersion drives m toward 0 (r toward
  # its 4.85e8 clamp), which is exactly that regime.
  set.seed(31)
  n <- 150
  x1 <- rnorm(n)
  d <- data.frame(y1 = rpois(n, exp(0.4 + 0.5 * x1)),
                  y2 = rpois(n, exp(0.2 - 0.3 * x1)), x1 = x1)

  fit <- suppressWarnings(
    fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d,
                  dependence = copula("normal"), draws = 20, seed = 5,
                  control = rpbnb_tmb_control(n_cores = 1L)))

  # The finite gradient is the distinctive symptom and the whole point of this
  # test: with the NB2 survival on the tape it came back NaN.
  expect_true(is.finite(fit$optimizer$max_abs_gradient))
  expect_true(is.finite(fit$optimizer$objective))
  expect_true(all(is.finite(fit$optimizer$par)))
  # The regression's OTHER symptom -- the dispersion no longer collapsing on
  # Poisson data -- is deliberately not re-asserted here. It needs a sample
  # large enough for the MLE to actually reach the limit (n = 150 with 20 draws
  # lands around m = 0.04 from sampling noise alone, fix or no fix), and
  # test-convergence-polish.R already covers it at n = 400 where it is a real
  # signal rather than a coin flip.
})
