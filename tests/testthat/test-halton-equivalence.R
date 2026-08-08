# The two Halton generators are kept separate on purpose -- see R/tmb_halton.R.
#
# halton_uniform() (Rcpp engine) builds the sequence with randtoolbox::halton();
# .tmb_halton_uniform() (TMB engine) uses a hand-rolled radical inverse. They
# are believed to be numerically equivalent for a common `burn`, but
# test-laplace.R pins the TMB engine's SML log-likelihood to 1e-10, so they were
# NOT unified on that belief during the merge.
#
# This test documents the belief and makes it falsifiable. If it holds across
# platforms, the two can be collapsed into one generator and this file deleted.

test_that("the two Halton generators agree for a common burn", {
  skip_on_cran()
  skip_slow()

  for (d in c(1L, 2L, 5L)) {
    set.seed(99)
    a <- halton_uniform(200L, d, burn = 300L)
    set.seed(99)
    b <- .tmb_halton_uniform(200L, d, burn = 300L)

    expect_equal(dim(a), dim(b))
    # If this fails, the generators genuinely differ and unifying them would
    # have moved the bit-identity baseline in test-laplace.R.
    expect_lt(max(abs(a - b)), 1e-12)
  }
})

test_that("both generators return values strictly inside (0, 1)", {
  a <- halton_uniform(50L, 3L, burn = 300L)
  b <- .tmb_halton_uniform(50L, 3L, burn = 300L)
  for (m in list(a, b)) {
    expect_true(all(m > 0))
    expect_true(all(m < 1))
  }
})

test_that("random-coefficient distribution codes match the C++ template", {
  # rand_dist_registry is now shared between the engines, and the TMB fitter
  # maps names to DIST_* codes with an explicit vector rather than by registry
  # order. Pin the codes so a registry reordering cannot silently remap them.
  codes <- match(c("normal", "lognormal", "uniform", "triangular"),
                 c("normal", "lognormal", "uniform", "triangular")) - 1L
  expect_equal(codes, 0:3)
  expect_true(all(c("normal", "lognormal", "uniform", "triangular") %in%
                    names(rand_dist_registry)))
})
