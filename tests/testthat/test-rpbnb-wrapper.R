# rpbnb() is a dispatcher, not an estimator. Everything here is validation --
# no model is fitted, so this stays in the fast tier.

toy <- function(n = 30L) {
  set.seed(11)
  data.frame(y1 = rnbinom(n, mu = 2, size = 2),
             y2 = rnbinom(n, mu = 2, size = 2),
             x1 = rnorm(n))
}

test_that("engine-specific arguments are rejected by the other engine", {
  d <- toy()
  # tmb-only arguments under engine = "classic"
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic", inference = "none"),
    'only accepted by engine = "tmb"'
  )
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic", keep = "full"),
    'only accepted by engine = "tmb"'
  )
  # classic-only arguments under engine = "tmb"
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", draw_type = "halton"),
    'only accepted by engine = "classic"'
  )
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", .opt_draws = 10),
    'only accepted by engine = "classic"'
  )
})

test_that("tmb-only method/force_parallel_gaussian are ignored (warning) under classic", {
  d <- toy()
  # dependence = "independence" errors under classic immediately AFTER the
  # dots validation, so reaching it proves the tmb knobs were dropped (rather
  # than erroring) without ever fitting a model.
  expect_error(
    expect_warning(
      rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic",
            method = "laplace", dependence = "independence"),
      "`method` ignored: tmb-only"
    ),
    'does not implement dependence = "independence"'
  )
  expect_error(
    expect_warning(
      rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic",
            method = "laplace", force_parallel_gaussian = TRUE,
            dependence = "independence"),
      "`method`, `force_parallel_gaussian` ignored: tmb-only"
    ),
    'does not implement dependence = "independence"'
  )
  # The ignore list is one-directional: classic-only names under tmb keep
  # their hard error.
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", draw_type = "halton"),
    'only accepted by engine = "classic"'
  )
  # And the remaining tmb-only names keep the hard error under classic.
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic", inference = "none"),
    'only accepted by engine = "tmb"'
  )
})

test_that("unknown argument names are an error, not a silent no-op", {
  d <- toy()
  # The whole point of the wrapper: a typo must not vanish into `...`.
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic", drawtype = "halton"),
    "not an argument of either engine"
  )
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", scaling = list(x1 = 1)),
    "not an argument of either engine"
  )
})

test_that("unnamed extra arguments are rejected", {
  d <- toy()
  # An unnamed argument only reaches `...` once every preceding formal is
  # supplied by name -- otherwise R positionally binds it to the first unfilled
  # formal (here `random_1`, which then fails with "random name(s) not found").
  # So fill them all, or this asserts the wrong error.
  expect_error(
    rpbnb(formula_1 = y1 ~ x1, formula_2 = y2 ~ x1, data = d, engine = "classic",
          random_1 = NULL, random_2 = NULL, draws = 400, seed = 1234,
          start = NULL, dependence = "famoye",
          poisson_1 = FALSE, poisson_2 = FALSE,
          standardize = FALSE, continuous_vars = NULL,
          boundary_tests = FALSE, boundary_draws = NULL, control = NULL,
          "halton"),
    "must be named"
  )
})

test_that("one control object serves both engines", {
  # The two constructors were merged in 0.4.1: either name builds the same
  # object and either engine accepts it. What is NOT translated is the meaning
  # of a shared field -- see the resolution test below for the defaults.
  expect_s3_class(rpbnb_tmb_control(), "rpbnb_control")
  expect_s3_class(rpbnb_control(), "rpbnb_tmb_control")
  expect_identical(unclass(rpbnb_tmb_control(n_cores = 2L)),
                   unclass(rpbnb_control(n_cores = 2L)))

  # A foreign object is still rejected, by the resolver rather than by rpbnb().
  expect_error(
    rpbnb:::.resolve_control(list(iterlim = 10), "tmb"),
    "must be an `rpbnb_control` object"
  )
})

test_that("estimator-dependent control defaults resolve per engine", {
  ctl <- rpbnb_control()
  expect_null(ctl$iterlim)
  expect_null(ctl$print_level)

  cl <- rpbnb:::.resolve_control(ctl, "classic")
  expect_identical(cl$iterlim, 300L)
  expect_identical(cl$print_level, 2L)

  tm <- rpbnb:::.resolve_control(ctl, "tmb")
  expect_identical(tm$iterlim, 500L)
  expect_identical(tm$print_level, 0L)

  # An explicit value is honored by every engine.
  set <- rpbnb_control(iterlim = 77L, print_level = 3L)
  expect_identical(rpbnb:::.resolve_control(set, "classic")$iterlim, 77L)
  expect_identical(rpbnb:::.resolve_control(set, "tmb")$iterlim, 77L)
  expect_identical(rpbnb:::.resolve_control(set, "tmb")$print_level, 3L)
})

test_that("inapplicable control settings are recorded, not rejected", {
  ctl <- rpbnb_control(se_method = "opg", gradtol = 1e-6, draws_hessian = 50L)
  expect_setequal(attr(ctl, "supplied"),
                  c("se_method", "gradtol", "draws_hessian"))
  # Only what the caller actually supplied is reported -- an untouched field
  # this engine does not read is not "ignored", it was never asked for.
  expect_setequal(attr(rpbnb:::.resolve_control(ctl, "tmb"), "ignored"),
                  c("se_method", "draws_hessian"))
  expect_setequal(attr(rpbnb:::.resolve_control(ctl, "classic"), "ignored"),
                  c("gradtol", "draws_hessian"))
  expect_setequal(attr(rpbnb:::.resolve_control(ctl, "bnb"), "ignored"),
                  c("se_method", "gradtol", "draws_hessian"))
  # Nothing supplied -> nothing reported, for any engine.
  expect_length(attr(rpbnb:::.resolve_control(rpbnb_control(), "tmb"),
                     "ignored"), 0L)
})

test_that("dependence structures unsupported by an engine are rejected", {
  d <- toy()
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic",
          dependence = "independence"),
    'engine = "classic" does not implement'
  )
  # copula(par=) is simulation-only; rpbnb() rejects it for BOTH engines even
  # though fit_rpbnb() would accept and ignore it.
  for (eng in c("classic", "tmb")) {
    expect_error(
      rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = eng,
            dependence = copula("frank", par = 2)),
      "simulation-only"
    )
  }
})

test_that("rpbnb() returns exactly what the underlying fitter returns", {
  skip_on_cran()
  skip_slow()
  d <- toy(60L)
  ctl <- rpbnb_control(compute_se = FALSE, print_level = 0)

  a <- rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic",
             draws = 30, seed = 7, control = ctl)
  b <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = d,
                 draws = 30, seed = 7, control = ctl)
  expect_s3_class(a, "rpbnb_fit")
  expect_equal(coef(a), coef(b))
  expect_equal(as.numeric(logLik(a)), as.numeric(logLik(b)))

  p <- rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb",
             draws = 30, seed = 7, inference = "none")
  q <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d,
                     draws = 30, seed = 7, inference = "none")
  expect_s3_class(p, "rpbnb_tmb_fit")
  expect_equal(coef(p), coef(q))
  expect_equal(as.numeric(logLik(p)), as.numeric(logLik(q)))
})
